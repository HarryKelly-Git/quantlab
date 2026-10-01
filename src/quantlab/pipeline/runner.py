"""Persistent Alpaca PAPER runner: one long-lived process that paper-trades the daily-bar strategies
through the EXISTING daily pipeline and every one of its gates. It adds scheduling, broker
monitoring and reconciliation; it never adds a way around a gate.

Timeline for a decision session D (all times US/Eastern, from the broker's market calendar):

  16:00 D        D closes: the information cutoff for D (ARCHITECTURE.md section 2).
  >= 19:05 D     ``paper.runner.process_after_et``: ingest D's bars/corporate actions, reconcile with
                 the broker, then run ``DailyPipeline(D)``: validate -> fills/marks -> exits ->
                 features/strategies -> candidates -> EV/no-trade/portfolio/risk -> orders. Orders
                 are Alpaca ``opg`` (market-on-open) orders for D+1's opening auction, matching the
                 strategies' "signal at close D -> fill at open D+1" rule. Alpaca rejects opg orders
                 submitted 09:28-19:00 ET, hence 19:05.
  < 09:25 D+1    ``paper.runner.order_cutoff_et``: the last moment an order for D+1's open may be
                 submitted. A session processed later (runner was down) still records candidates,
                 decisions and shadow outcomes, but every order is refused ("execution window
                 missed"): filling it at a later open would break the strategy's execution rule.
  09:30 D+1      the opening auction fills; fills arrive on the trade_updates stream (REST polling
                 is the fallback) and are applied to the ledger.

Idempotency: one ``paper_session_jobs`` row per (book, D) holds the pipeline run_id. A restart
RESUMES that run (same persisted candidates -> same deterministic client_order_ids), and the
execution service never submits a client_order_id the broker already knows. So a restart, a crash
or a reconnect cannot create a duplicate order.

Kill switch: SYSTEM_PAUSED stays authoritative (the execution service reads it before every
order). This runner PAUSES the system on systemic problems -- data stale/invalid, broker not
verifiable, reconciliation mismatch, unknown broker order, pipeline failure, any unexpected error --
and keeps running so existing orders/positions stay monitored. Per-candidate problems (strategy not
paper-eligible, EV, no-trade, portfolio, risk) are rejections recorded by the decision chain.
"""
from __future__ import annotations

import math
import os
import queue
import socket
import threading
import time as _time
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.config import PAPER_TRADING_BASE_URL
from quantlab.context import AppContext
from quantlab.core.types import Book, OrderStatus, TradePlan, new_id
from quantlab.data.panel import DataBundle
from quantlab.db.database import from_json, open_db, to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, LiveTradingForbidden
from quantlab.execution.ledger import Ledger, bind_book, book_binding
from quantlab.execution.preflight import PreflightResult, run_preflight
from quantlab.execution.reconcile import Reconciler
from quantlab.execution.service import PaperExecutionService
from quantlab.logging_setup import get_logger, log_event
from quantlab.monitoring.killswitch import KillSwitch

log = get_logger(__name__)

ET = ZoneInfo("America/New_York")
OPG_REJECT_FROM = time(9, 28)     # Alpaca rejects opg orders submitted 09:28-19:00 ET
OPG_REJECT_UNTIL = time(19, 0)
TEST_ORDER_PREFIX = "qltest-"     # connectivity-test orders (never strategy orders, never filled)
BOOK = Book.BOT.value
BROKER_NAME = "alpaca_paper"


_ES_CONTINUOUS, _ES_SYSTEM_REQUIRED, _ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def keep_awake_flags(now: datetime, submit_after: time, open_until: time, process_after: time,
                     process_minutes: int = 90, display: bool = True) -> int:
    """SetThreadExecutionState flags the runner holds at ``now`` (Windows).

    While the runner runs the system never idle-sleeps (ES_SYSTEM_REQUIRED). On weekdays, if
    ``display``, the display is also held on (ES_DISPLAY_REQUIRED) through the two windows where a
    missed tick costs a whole session: from 15 min before the pre-open submit until ``open_until``,
    and from 5 min before the evening processing time for ``process_minutes``. On Modern Standby
    laptops the screen timing out starts standby; a display request is what reliably holds it off."""
    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    et = now.astimezone(ET)
    if display and et.weekday() < 5:
        m = et.hour * 60 + et.minute
        a = submit_after.hour * 60 + submit_after.minute - 15
        b = open_until.hour * 60 + open_until.minute
        c = process_after.hour * 60 + process_after.minute - 5
        if a <= m < b or c <= m < c + 5 + process_minutes:
            flags |= _ES_DISPLAY_REQUIRED
    return flags


class RunnerRefused(RuntimeError):
    """The runner refused to start (preflight, another live runner, ledger binding, dirty account)."""


def _parse_hhmm(text: str) -> time:
    h, m = str(text).split(":")
    return time(int(h), int(m))


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


# ------------------------------------------------------------------------------------------------
# market calendar
# ------------------------------------------------------------------------------------------------
class MarketCalendar:
    """Regular sessions with their open/close (ET). Source: the broker's /v2/calendar (includes
    early closes); if that is unreachable, the rule-based NYSE calendar in ``data/audit.py`` with a
    16:00 close. The two are cross-checked and any disagreement is reported."""

    def __init__(self, sessions: dict[date, tuple[time, time]], source: str, mismatches: list[str] | None = None):
        self.sessions = dict(sorted(sessions.items()))
        self.source = source
        self.mismatches = mismatches or []

    @classmethod
    def load(cls, broker: Any, today: date, past_days: int = 40, future_days: int = 40) -> "MarketCalendar":
        from quantlab.data.audit import expected_sessions
        start, end = today - timedelta(days=past_days), today + timedelta(days=future_days)
        rules = {d.date(): (time(9, 30), time(16, 0)) for d in expected_sessions(str(start), str(end))}
        try:
            rows = broker.calendar(str(start), str(end))
            sessions = {date.fromisoformat(r["date"]): (_parse_hhmm(r["open"]), _parse_hhmm(r["close"]))
                        for r in rows if r.get("date") and r.get("open") and r.get("close")}
            if not sessions:
                raise BrokerError("empty calendar")
        except (BrokerError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return cls(rules, "rules", [f"broker calendar unavailable ({type(exc).__name__}); using NYSE rules"])
        diff = sorted(set(sessions) ^ set(rules))
        mism = [f"{d}: {'broker only' if d in sessions else 'rules only'}" for d in diff]
        return cls(sessions, "alpaca", mism)

    def close_dt(self, d: date) -> datetime:
        return datetime.combine(d, self.sessions[d][1], ET)

    def open_dt(self, d: date) -> datetime:
        return datetime.combine(d, self.sessions[d][0], ET)

    def last_closed(self, now: datetime) -> date | None:
        done = [d for d in self.sessions if self.close_dt(d) <= now]
        return done[-1] if done else None

    def next_after(self, d: date) -> date | None:
        later = [x for x in self.sessions if x > d]
        return later[0] if later else None

    def is_open(self, now: datetime) -> bool:
        d = now.astimezone(ET).date()
        return d in self.sessions and self.open_dt(d) <= now < self.close_dt(d)

    def current_session(self, now: datetime) -> date | None:
        """Today's session if today trades (from its open), else the last closed session."""
        d = now.astimezone(ET).date()
        if d in self.sessions and now >= self.open_dt(d):
            return d
        return self.last_closed(now)


@dataclass
class SessionPlan:
    session: date                 # D: decision session (its close is the information cutoff)
    next_session: date            # D+1: execution session (opening auction)
    process_at: datetime          # earliest processing time for D
    order_deadline: datetime      # last moment an order for D+1's open may be submitted
    due: bool
    orders_allowed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("session", "next_session"):
            d[k] = str(d[k])
        for k in ("process_at", "order_deadline"):
            d[k] = _iso(d[k])
        return d


def order_window(session: date, next_session: date, process_after: time, order_cutoff: time) -> tuple[datetime, datetime]:
    """[start, end) in which next-open (opg) orders decided at ``session`` may be submitted."""
    start = datetime.combine(session, max(process_after, OPG_REJECT_UNTIL), ET)
    end = datetime.combine(next_session, min(order_cutoff, OPG_REJECT_FROM), ET)
    return start, end


def order_window_reason(now: datetime, session: date, next_session: date, process_after: time,
                        order_cutoff: time) -> str | None:
    start, end = order_window(session, next_session, process_after, order_cutoff)
    if now < start:
        return f"before the order window for the {next_session} open (opens {start:%Y-%m-%d %H:%M} ET)"
    if now >= end:
        return (f"execution window missed: orders for the {next_session} open had to be submitted before "
                f"{end:%Y-%m-%d %H:%M} ET (next-session execution rule)")
    return None


def plan_session(cal: MarketCalendar, now: datetime, process_after: time, order_cutoff: time) -> SessionPlan | None:
    d = cal.last_closed(now)
    if d is None:
        return None
    nxt = cal.next_after(d)
    if nxt is None:
        return None
    process_at = datetime.combine(d, process_after, ET)
    start, deadline = order_window(d, nxt, process_after, order_cutoff)
    due = now >= process_at
    allowed = start <= now < deadline
    if not due:
        reason = f"waiting: {d} is processed from {process_at:%Y-%m-%d %H:%M} ET"
    elif allowed:
        reason = f"{d} due; orders for the {nxt} open allowed until {deadline:%H:%M} ET"
    else:
        reason = f"{d} due; order window for the {nxt} open has passed: decisions only, no orders"
    return SessionPlan(d, nxt, process_at, deadline, due, allowed, reason)


# ------------------------------------------------------------------------------------------------
# default data hooks (real data)
# ------------------------------------------------------------------------------------------------
def default_ingest(ctx: AppContext, session: date) -> dict[str, Any]:
    """Incremental real-data ingest for the decision session (bars + corporate actions)."""
    from quantlab.data.ingest import IngestionService
    lookback = int(ctx.config.get("paper.runner.ingest_lookback_days", 10))
    run_id = ctx.start_run("ingest", notes=f"paper runner: incremental ingest for {session}")
    try:
        rep = IngestionService(ctx.config, ctx.store, ctx.db).ingest_all(
            session - timedelta(days=lookback), session, None, run_id=run_id,
            kinds=("reference", "bars", "corporate_actions"), bar_chunk=100_000)
        ctx.finish_run(run_id, "succeeded" if rep.ok else "failed")
        return rep.summary()
    except Exception as exc:
        ctx.finish_run(run_id, "failed", repr(exc))
        raise


def default_bundle_loader(ctx: AppContext) -> DataBundle:
    """Real-data bundle for one session: recent bars (``paper.runner.bundle_bars_days``, enough for
    every feature, universe rule and exit), ~150 days of news and ~3 years of facts. Loading less
    history never changes what is known at D; it keeps catalyst history within this machine's RAM."""
    from quantlab.discovery.catalyst_research import research_bundle
    today = pd.Timestamp.now().normalize()
    days = int(ctx.config.get("paper.runner.bundle_bars_days", 900))
    return research_bundle(ctx, str((today - pd.Timedelta(days=days)).date()), str(today.date()),
                           news_since=str((today - pd.Timedelta(days=150)).date()),
                           facts_since=str((today - pd.Timedelta(days=3 * 366)).date()))


def default_catalyst_refresh(ctx: AppContext, session: date, scope: str = "daily",
                             symbols: list[str] | None = None) -> dict[str, Any]:
    """Recent news / SEC filings / new fundamentals (best effort; see data/catalyst_refresh.py).
    ``scope="preopen"``: news + SEC for the current candidates only (fast)."""
    from quantlab.data.catalyst_refresh import refresh_catalysts
    if not bool(ctx.config.get("paper.runner.catalyst_refresh", True)):
        return {"skipped": "paper.runner.catalyst_refresh is false"}
    days = int(ctx.config.get("paper.runner.catalyst_sec_days", 10))
    parts = ("news", "sec") if scope == "preopen" else ("news", "sec", "facts")
    return refresh_catalysts(ctx, pd.Timestamp(session), symbols=symbols, parts=parts, sec_days=days)


def stored_last_bar_date(ctx: AppContext, synthetic: bool = False) -> date | None:
    row = ctx.db.fetchone("SELECT MAX(end_date) AS d FROM datasets WHERE kind='bars' AND is_synthetic=?",
                          (int(synthetic),))
    return date.fromisoformat(str(row["d"])[:10]) if row and row["d"] else None


def paper_eligible_strategies(db) -> list[dict[str, Any]]:
    """(strategy, version) allowed to place BOT paper orders: status ACTIVE at stage PAPER/PROMOTED."""
    return db.fetchall("SELECT strategy_id, version, status, stage FROM strategies "
                       "WHERE status='ACTIVE' AND stage IN ('PAPER','PROMOTED') ORDER BY strategy_id")


# ------------------------------------------------------------------------------------------------
# runner
# ------------------------------------------------------------------------------------------------
class PaperRunner:
    def __init__(self, ctx: AppContext, *, broker: Any = None, stream_factory: Callable[..., Any] | None = None,
                 now: Callable[[], datetime] | None = None, ingest: Callable[[AppContext, date], dict] | None = None,
                 bundle_loader: Callable[[AppContext], DataBundle] | None = None, env: dict[str, str] | None = None,
                 heartbeat_thread: bool = True, allow_synthetic: bool = False,
                 catalyst_refresh: Callable[..., dict] | None = None):
        self.ctx, self.db, self.cfg = ctx, ctx.db, ctx.config
        if broker is None:
            from quantlab.execution.alpaca_paper import AlpacaPaperBroker
            broker = AlpacaPaperBroker(ctx.config)
        if getattr(broker, "name", None) != BROKER_NAME:
            raise RunnerRefused(f"the paper runner only drives the Alpaca PAPER broker, got {getattr(broker, 'name', None)!r}")
        self.broker = broker
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._ingest = ingest or default_ingest
        self._bundle_loader = bundle_loader or default_bundle_loader
        # network refresh of catalyst data: real data only (tests / synthetic worlds never call out)
        self._catalyst_refresh = catalyst_refresh or (None if allow_synthetic else default_catalyst_refresh)
        self._preopen_done: date | None = None
        self._stream_factory = stream_factory
        self.env = env
        self.allow_synthetic = allow_synthetic
        self._use_hb_thread = heartbeat_thread

        g = lambda k, d: self.cfg.get(f"paper.runner.{k}", d)   # noqa: E731
        self.process_after = _parse_hhmm(g("process_after_et", "19:05"))
        self.order_cutoff = _parse_hhmm(g("order_cutoff_et", "09:25"))
        if self.process_after < OPG_REJECT_UNTIL:
            raise RunnerRefused(f"paper.runner.process_after_et must be >= 19:00 ET (Alpaca rejects opg orders "
                                f"09:28-19:00), got {self.process_after}")
        if self.order_cutoff >= OPG_REJECT_FROM:
            raise RunnerRefused(f"paper.runner.order_cutoff_et must be < 09:28 ET, got {self.order_cutoff}")
        self.tick_seconds = float(g("tick_seconds", 2))
        self.heartbeat_seconds = float(g("heartbeat_seconds", 10))
        self.account_poll_seconds = float(g("account_poll_seconds", 60))
        self.order_poll_seconds = float(g("order_poll_seconds", 60))
        self.broker_fresh_seconds = float(g("broker_fresh_seconds", 300))
        self.broker_unverified_pause_seconds = float(g("broker_unverified_pause_seconds", 900))
        self.stale_heartbeat_seconds = float(g("stale_heartbeat_seconds", 120))
        self.data_retry_seconds = float(g("data_retry_seconds", 900))
        # an opg order that does not execute in the opening cross is EXPIRED by Alpaca, unfilled.
        # One market-day fallback per decision then takes the position shortly after the open.
        self.open_fill_fallback = bool(g("open_fill_fallback", True))
        self.fallback_until = _parse_hhmm(g("open_fill_fallback_until_et", "15:45"))
        # keep the PC awake while the runner runs (and the display on in the critical windows)
        self.keep_awake = bool(g("keep_awake", True))
        self.keep_display_on = bool(g("keep_display_on_in_critical_windows", True))
        self.awake_submit_after = _parse_hhmm(str(self.cfg.get("exploration.submit_after_et", "08:30")))
        self._awake_flags: int | None = None
        self.fallback_poll_seconds = float(g("open_fill_fallback_poll_seconds", 30))
        self._last_fallback_scan: datetime | None = None
        self.max_job_attempts = int(g("max_job_attempts", 3))
        # a mismatch is re-checked once after this delay before pausing (fills racing the snapshot)
        self.reconcile_retry_seconds = float(g("reconcile_retry_seconds", 3))

        self.killswitch = KillSwitch(self.db)
        self.queue: queue.Queue = queue.Queue()
        self.session_id: str | None = None
        self.calendar: MarketCalendar | None = None
        self._calendar_day: date | None = None
        self.stream: Any = None
        self.stream_state = "stopped"
        self.ledger: Ledger | None = None
        self.exec: PaperExecutionService | None = None
        self.broker_ok_at: datetime | None = None
        self.broker_fail_since: datetime | None = None
        self.broker_snapshot: dict[str, Any] = {}
        self.reconciled_ok = False
        self.phase = "starting"
        self.plan: SessionPlan | None = None
        self.last_job: dict[str, Any] | None = None
        self._last_account_poll: datetime | None = None
        self._last_order_poll: datetime | None = None
        self._last_heartbeat: datetime | None = None
        self._last_tick_at: str | None = None
        self._reconcile_due: str | None = None
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    # -- audit ------------------------------------------------------------------------------------
    def event(self, level: str, kind: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.db.insert("paper_runner_events", {"session_id": self.session_id, "at": utcnow_iso(), "level": level,
                                               "kind": kind, "message": message[:2000],
                                               "details_json": to_json(details or {})})
        log_event(log, f"paper runner {kind}: {message}", level=40 if level in ("ERROR", "CRITICAL") else 20)

    def pause(self, reason: str, trigger: str, details: dict[str, Any] | None = None) -> None:
        changed = self.killswitch.pause(reason, trigger=trigger, details=details, actor="system:paper_runner")
        self.event("CRITICAL", "killswitch", f"SYSTEM_PAUSED ({trigger}): {reason}" + ("" if changed else " [already paused]"),
                   details)

    # -- lifecycle --------------------------------------------------------------------------------
    def start(self) -> PreflightResult:
        """Preflight -> single-instance session -> ledger binding -> calendar -> reconcile -> stream."""
        self.session_id = new_id("paper")
        pre = run_preflight(self.cfg, self.db, self.broker, session_id=self.session_id, env=self.env)
        if not pre.ok:
            self.event("CRITICAL", "preflight", f"paper preflight REFUSED: {pre.reason}", {"preflight_id": pre.preflight_id})
            raise RunnerRefused(f"paper preflight failed: {pre.reason}")
        self._acquire_session()
        self.event("INFO", "preflight", "paper preflight passed on the PAPER endpoint",
                   {"preflight_id": pre.preflight_id, "account": pre.account_ref, "status": pre.account_status,
                    "equity": pre.equity, "cash": pre.cash, "buying_power": pre.buying_power,
                    "positions": len(pre.positions), "open_orders": pre.open_orders})
        self.broker_snapshot = {"equity": pre.equity, "cash": pre.cash, "buying_power": pre.buying_power,
                                "positions": pre.positions, "account_status": pre.account_status, "at": pre.at,
                                "trading_blocked": False}
        self.broker_ok_at = self._now()
        self._bind_and_seed(pre)
        from quantlab.strategies.registry import build_strategies, register_strategies
        register_strategies(self.db, build_strategies(self.cfg))
        self._refresh_calendar(self._now(), force=True)
        self._report_eligibility()
        self.reconcile("startup")
        self._start_stream()
        self._heartbeat(force=True)
        if self._use_hb_thread:
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="paper-heartbeat", daemon=True)
            self._hb_thread.start()
        return pre

    def run_forever(self, max_ticks: int | None = None) -> str:
        """Blocking main loop. Returns the stop reason. Only LiveTradingForbidden escapes."""
        reason = "stopped"
        try:
            self.start()
        except RunnerRefused as exc:
            self.shutdown(str(exc), status="REFUSED")
            raise
        except Exception as exc:
            self.shutdown(f"startup failed: {type(exc).__name__}: {exc}", status="CRASHED")
            raise
        try:
            n = 0
            while True:
                r = self.tick()
                if r:
                    reason = r
                    break
                n += 1
                if max_ticks is not None and n >= max_ticks:
                    reason = f"max_ticks={max_ticks}"
                    break
                self._wait(self.tick_seconds)
        except KeyboardInterrupt:
            reason = "keyboard interrupt"
        except LiveTradingForbidden as exc:
            reason = f"LIVE TRADING FORBIDDEN: {exc}"
            if self.session_id:
                self.pause(reason, "live_trading_forbidden")
            self.shutdown(reason, status="CRASHED")
            raise
        self.shutdown(reason)
        return reason

    def _set_execution_state(self, flags: int) -> bool:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))

    def _keep_awake(self, now: datetime) -> None:
        """Hold the system awake while running; hold the display on in the critical windows. Only
        calls Windows when the required state changes; records each change as an event."""
        if not self.keep_awake or os.name != "nt":
            return
        flags = keep_awake_flags(now, self.awake_submit_after, time(10, 0), self.process_after,
                                 display=self.keep_display_on)
        if flags == self._awake_flags:
            return
        ok = self._set_execution_state(flags)
        self._awake_flags = flags
        what = ("system awake + display on (pre-open/open or evening-processing window)"
                if flags & _ES_DISPLAY_REQUIRED else "system awake; display may turn off")
        self.event("INFO" if ok else "WARN", "power", f"keep-awake: {what}" + ("" if ok else ": SetThreadExecutionState FAILED"),
                   {"flags": hex(flags)})

    def shutdown(self, reason: str, status: str = "STOPPED") -> None:
        if self._awake_flags is not None and os.name == "nt":
            try:
                self._set_execution_state(_ES_CONTINUOUS)      # release: normal power policy again
            except Exception:
                pass
            self._awake_flags = None
        if self.session_id is None:
            return
        self._hb_stop.set()
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass
        if self.exec is not None:
            self._drain()      # apply anything the stream delivered before it stopped
        row = self.db.fetchone("SELECT status FROM paper_runner_sessions WHERE session_id=?", (self.session_id,))
        if row and row["status"] == "RUNNING":
            self.db.execute("UPDATE paper_runner_sessions SET status=?, stopped_at=?, phase=?, stream_status=?, "
                            "last_heartbeat_at=?, error=? WHERE session_id=?",
                            (status, utcnow_iso(), "stopped", "stopped", utcnow_iso(),
                             None if status == "STOPPED" else reason, self.session_id))
            self.event("INFO" if status == "STOPPED" else "CRITICAL", "lifecycle", f"runner {status.lower()}: {reason}")
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=5)

    def _acquire_session(self) -> None:
        now = datetime.now(timezone.utc)      # process liveness is wall-clock time, not market time
        with self.db.transaction():
            for r in self.db.fetchall("SELECT session_id, last_heartbeat_at FROM paper_runner_sessions WHERE status='RUNNING'"):
                hb = pd.Timestamp(r["last_heartbeat_at"]) if r["last_heartbeat_at"] else None
                age = (pd.Timestamp(now) - hb).total_seconds() if hb is not None else None
                if age is not None and age < self.stale_heartbeat_seconds:
                    raise RunnerRefused(f"another paper runner is RUNNING (session {r['session_id']}, heartbeat "
                                        f"{age:.0f}s ago). Stop it first: quantlab paper stop")
                self.db.execute("UPDATE paper_runner_sessions SET status='CRASHED', stopped_at=?, error=? WHERE session_id=?",
                                (utcnow_iso(), f"heartbeat stale ({age}s); process presumed dead", r["session_id"]))
            self.db.insert("paper_runner_sessions", {
                "session_id": self.session_id, "book": BOOK, "mode": "PAPER", "endpoint": PAPER_TRADING_BASE_URL,
                "broker": BROKER_NAME, "pid": os.getpid(), "host": socket.gethostname(), "started_at": utcnow_iso(),
                "last_heartbeat_at": utcnow_iso(), "status": "RUNNING", "phase": "starting",
                "stream_status": "stopped", "detail_json": to_json({}),
            })

    def _bind_and_seed(self, pre: PreflightResult) -> None:
        """Bind the BOT book to the Alpaca paper account. First use seeds the ledger's cash from the
        broker (only when both the book and the paper account are clean)."""
        bound = book_binding(self.db, BOOK)
        if bound is None:
            n_cash = self.db.fetchone("SELECT COUNT(*) AS n FROM ledger_cash_events WHERE book=?", (BOOK,))["n"]
            n_orders = self.db.fetchone("SELECT COUNT(*) AS n FROM orders WHERE book=?", (BOOK,))["n"]
            if n_cash or n_orders:
                raise RunnerRefused("the BOT book in this database already has simulated history; the Alpaca paper "
                                    "runner needs a fresh BOT book (use a separate project.db_path)")
            if pre.positions or pre.open_orders:
                raise RunnerRefused(f"the Alpaca paper account is not clean ({len(pre.positions)} position(s), "
                                    f"{pre.open_orders} open order(s)); QuantLab needs a dedicated paper account")
            if pre.cash is None or pre.cash <= 0:
                raise RunnerRefused(f"cannot seed the BOT ledger: broker cash is {pre.cash!r}")
            bind_book(self.db, BOOK, BROKER_NAME, {"seeded_cash": pre.cash, "session_id": self.session_id,
                                                  "account": pre.account_ref})
            Ledger(self.db, BOOK, starting_cash=float(pre.cash), config=self.cfg, broker_name=BROKER_NAME)
            self.event("INFO", "ledger", f"BOT ledger bound to the Alpaca paper account and seeded with its cash "
                       f"(${pre.cash:,.2f})", {"account": pre.account_ref})
        elif bound != BROKER_NAME:
            raise RunnerRefused(f"the BOT book in this database is bound to {bound!r}, not {BROKER_NAME!r}; "
                                "use a separate project.db_path for the Alpaca paper runner")
        self.ledger = Ledger(self.db, BOOK, config=self.cfg, broker_name=BROKER_NAME)
        self.exec = PaperExecutionService(self.db, self.cfg, BOOK, self.broker, self.ledger)
        # the ONLY order this service submits is the open-fill fallback; the session pipeline builds
        # its own service with the order-window guard
        self.exec.submission_guard = self._fallback_guard()

    def _report_eligibility(self) -> list[dict[str, Any]]:
        elig = paper_eligible_strategies(self.db)
        if not elig:
            self.event("WARN", "strategies", "NO PAPER-ELIGIBLE STRATEGY: every strategy is SHADOW/research. The runner "
                       "records candidates and decisions (all rejected at the STRATEGY gate) and places no orders.")
        else:
            self.event("INFO", "strategies", "paper-eligible strategies: " +
                       ", ".join(f"{r['strategy_id']}@{r['version']} ({r['stage']})" for r in elig))
        return elig

    # -- stream -----------------------------------------------------------------------------------
    def _start_stream(self) -> None:
        on_event = lambda data: self.queue.put(("update", data))          # noqa: E731
        on_state = lambda state, detail: self.queue.put(("state", state, detail))   # noqa: E731
        if self._stream_factory is not None:
            self.stream = self._stream_factory(on_event, on_state)
        else:
            from quantlab.execution.trade_stream import TradeUpdateStream
            self.stream = TradeUpdateStream(self.broker.stream_url(), self.broker.stream_auth_message,
                                            on_event, on_state)
        self.stream.start()

    def _wait(self, seconds: float) -> None:
        deadline = _time.monotonic() + seconds
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return
            try:
                item = self.queue.get(timeout=remaining)
            except queue.Empty:
                return
            self._handle(item)

    def _drain(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return
            self._handle(item)

    def _handle(self, item: tuple) -> None:
        try:
            if item[0] == "update":
                self.on_trade_update(item[1])
            elif item[0] == "state":
                self.on_stream_state(item[1], item[2])
        except LiveTradingForbidden:
            raise
        except Exception as exc:
            self._critical("stream item handling", exc)

    def on_stream_state(self, state: str, detail: str) -> None:
        prev, self.stream_state = self.stream_state, state
        if state == "connected":
            self.event("INFO", "stream", f"trade_updates stream connected ({detail})")
            if getattr(self.stream, "connects", 1) > 1:
                self.reconcile("stream reconnect")     # anything missed while disconnected
        elif state == "disconnected":
            self.event("WARN", "stream", f"trade_updates stream disconnected: {detail}; reconnecting (REST polling covers the gap)")
        elif state == "unauthorized":
            self.pause(f"broker connection cannot be verified: trade_updates stream unauthorized ({detail})",
                       "broker_unverified")
        elif state == "stopped" and prev != "stopped":
            self.event("INFO", "stream", "trade_updates stream stopped")

    def on_trade_update(self, data: dict[str, Any]) -> None:
        order = data.get("order") or {}
        bo = self.broker.parse_order(order)
        cid = bo.client_order_id
        local = self.db.fetchone("SELECT order_id FROM orders WHERE client_order_id=? AND book=?", (cid, BOOK))
        self.record_update("stream", data.get("event", "?"), bo, local["order_id"] if local else None, data)
        if local is None:
            if cid.startswith(TEST_ORDER_PREFIX):
                return
            self.pause(f"state reconciliation failed: broker order {cid or bo.broker_order_id} ({bo.symbol}) is not in "
                       "the QuantLab ledger", "reconciliation", {"client_order_id": cid, "event": data.get("event")})
            return
        assert self.exec is not None
        res = self.exec.apply_broker_order(bo, self._session_for(self._now()))
        if res.filled:
            self.event("INFO", "fill", f"{data.get('event')}: {bo.symbol} {res.filled[0]['qty']:g} @ "
                       f"{res.filled[0]['price']}", {"order_id": local["order_id"], "fills": res.filled})
            self._reconcile_due = f"after {data.get('event')} {bo.symbol}"
        if res.unknown:
            self.pause("broker reported an UNKNOWN order state", "broker_unknown_state", {"orders": res.unknown})
        if res.rejected or res.canceled or res.expired:
            self.event("WARN", "order", f"{data.get('event')}: {bo.symbol} {bo.status.value}"
                       + (f" ({bo.reason})" if bo.reason else ""), {"order_id": local["order_id"]})

    def record_update(self, source: str, event: str, bo, order_id: str | None, raw: dict[str, Any]) -> None:
        def f(x):
            try:
                return float(x) if x not in (None, "") else None
            except (TypeError, ValueError):
                return None
        self.db.insert("broker_order_updates", {
            "session_id": self.session_id, "received_at": utcnow_iso(), "source": source, "event": event,
            "order_id": order_id, "client_order_id": bo.client_order_id, "broker_order_id": bo.broker_order_id,
            "execution_id": raw.get("execution_id"), "status": bo.status.value, "event_qty": f(raw.get("qty")),
            "event_price": f(raw.get("price")), "filled_qty": bo.filled_qty, "filled_avg_price": bo.filled_avg_price,
            "event_at": raw.get("timestamp") or (bo.raw or {}).get("updated_at"), "raw_json": to_json(raw),
        }, or_ignore=True)

    # -- reconciliation ---------------------------------------------------------------------------
    def reconcile(self, why: str) -> bool:
        """Poll every open local order, look for broker orders QuantLab does not know, compare
        cash/positions with the ledger. A mismatch is re-checked once (a fill can land between the
        order poll and the account snapshot); if it persists the system is PAUSED for human review."""
        problems = self._reconcile_once(why)
        if problems and self.reconcile_retry_seconds >= 0:
            _time.sleep(self.reconcile_retry_seconds)
            problems = self._reconcile_once(why)
        self.reconciled_ok = not problems
        if problems:
            self.pause(f"state reconciliation failed ({why}): " + " | ".join(problems), "reconciliation",
                       {"problems": problems})
        else:
            self.event("INFO", "reconcile", f"reconciliation ok ({why})")
        return self.reconciled_ok

    def _reconcile_once(self, why: str) -> list[str]:
        assert self.exec is not None and self.ledger is not None
        problems: list[str] = []
        try:
            sync = self.exec.sync(self._session_for(self._now()))
            for kind in ("filled", "rejected", "expired", "canceled"):
                for x in getattr(sync, kind):
                    self.event("INFO", "order", f"reconcile ({why}): {kind} {x}")
            if sync.unknown:
                problems.append(f"{len(sync.unknown)} order(s) in UNKNOWN broker state")
            known = {r["client_order_id"] for r in self.db.fetchall(
                "SELECT client_order_id FROM orders WHERE book=?", (BOOK,))}
            orphans = [o.client_order_id for o in self.broker.list_orders("open")
                       if o.client_order_id not in known and not o.client_order_id.startswith(TEST_ORDER_PREFIX)]
            if orphans:
                problems.append(f"broker has {len(orphans)} open order(s) unknown to QuantLab: {orphans[:5]}")
            stuck = self.db.fetchall("SELECT order_id FROM orders WHERE book=? AND status=?",
                                     (BOOK, OrderStatus.UNKNOWN.value))
            if stuck:
                problems.append(f"{len(stuck)} local order(s) still UNKNOWN")
            rec = Reconciler(self.ledger, self.broker).reconcile(run_id=None)
            if rec.status == "broker_unavailable":
                problems.append("broker unavailable for reconciliation")
            elif not rec.ok:
                problems.append("ledger/broker mismatch: " + "; ".join(
                    f"{d['field']} broker={d['broker']} ledger={d['internal']}" for d in rec.diffs[:5]))
        except BrokerError as exc:
            problems.append(f"broker error during reconciliation: {exc}")
        return problems

    # -- the loop ---------------------------------------------------------------------------------
    def tick(self) -> str | None:
        """One loop iteration. Returns a stop reason, or None to keep running."""
        now = self._now()
        self._last_tick_at = now.isoformat()
        self._drain()
        if self._reconcile_due:
            why, self._reconcile_due = self._reconcile_due, None
            try:
                self.reconcile(why)
            except LiveTradingForbidden:
                raise
            except Exception as exc:
                self._critical("reconcile", exc)
        if self._stop_requested():
            return "stop requested (quantlab paper stop)"
        for name, fn in (("calendar", self._refresh_calendar), ("account", self._poll_account),
                         ("orders", self._poll_orders), ("open_fill", self._maybe_fill_after_open),
                         ("session", self._maybe_process), ("exploration", self._maybe_explore),
                         ("keep_awake", self._keep_awake)):
            try:
                fn(now)
            except LiveTradingForbidden:
                raise
            except Exception as exc:
                self._critical(name, exc)
        self._heartbeat()      # detail (job, broker snapshot, plan); the thread only keeps liveness fresh
        return None

    def _critical(self, where: str, exc: Exception) -> None:
        tb = traceback.format_exc()[-2500:]
        self.event("CRITICAL", "error", f"{where}: {type(exc).__name__}: {exc}", {"traceback": tb})
        self.pause(f"critical system error in paper runner ({where}): {type(exc).__name__}: {exc}", "runner_error")

    def _stop_requested(self) -> bool:
        row = self.db.fetchone("SELECT stop_requested_at FROM paper_runner_sessions WHERE session_id=?", (self.session_id,))
        return bool(row and row["stop_requested_at"])

    def _refresh_calendar(self, now: datetime, force: bool = False) -> None:
        today = now.astimezone(ET).date()
        if not force and self._calendar_day == today:
            return
        self.calendar = MarketCalendar.load(self.broker, today)
        self._calendar_day = today
        self.event("INFO" if self.calendar.source == "alpaca" else "WARN", "calendar",
                   f"market calendar from {self.calendar.source}: {len(self.calendar.sessions)} sessions around {today}"
                   + (f"; cross-check notes: {self.calendar.mismatches[:5]}" if self.calendar.mismatches else ""))

    def _session_for(self, now: datetime) -> date:
        d = self.calendar.current_session(now) if self.calendar else None
        return d or now.astimezone(ET).date()

    def _poll_account(self, now: datetime, force: bool = False) -> bool:
        if not force and self._last_account_poll and (now - self._last_account_poll).total_seconds() < self.account_poll_seconds:
            return self.broker_ok_at is not None
        self._last_account_poll = now
        try:
            acct = self.broker.account()
            positions = self.broker.positions()
        except BrokerError as exc:
            self.broker_fail_since = self.broker_fail_since or now
            down = (now - self.broker_fail_since).total_seconds()
            self.event("WARN", "broker", f"broker check failed ({type(exc).__name__}); unverified for {down:.0f}s")
            if down >= self.broker_unverified_pause_seconds:
                self.pause(f"broker connection cannot be verified for {down / 60:.0f} min: {exc}", "broker_unverified")
            return False
        self.broker_fail_since = None
        self.broker_ok_at = now
        self.broker_snapshot = {
            "at": now.isoformat(), "equity": acct.equity, "cash": acct.cash, "buying_power": acct.buying_power,
            "account_status": acct.status, "trading_blocked": acct.trading_blocked,
            "positions": [{"symbol": p.symbol, "qty": p.qty, "avg_entry_price": p.avg_entry_price,
                           "market_value": p.market_value, "current_price": p.current_price,
                           "unrealized_pl": (p.raw or {}).get("unrealized_pl")} for p in positions],
        }
        if acct.trading_blocked or acct.status != "ACTIVE":
            self.pause(f"broker account not tradable (status={acct.status}, trading_blocked={acct.trading_blocked})",
                       "broker_unverified")
        return True

    def _poll_orders(self, now: datetime) -> None:
        if self._last_order_poll and (now - self._last_order_poll).total_seconds() < self.order_poll_seconds:
            return
        self._last_order_poll = now
        n = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM orders WHERE book=? AND status IN (?,?,?,?,?)",
            (BOOK, OrderStatus.PENDING_SUBMIT.value, OrderStatus.ACCEPTED.value, OrderStatus.NEW.value,
             OrderStatus.PARTIALLY_FILLED.value, OrderStatus.UNKNOWN.value))["n"]
        if n:
            self.reconcile("order poll")

    # -- session processing -----------------------------------------------------------------------
    def _job(self, d: date) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM paper_session_jobs WHERE book=? AND as_of_date=?", (BOOK, str(d)))

    def _save_job(self, d: date, plan: SessionPlan, **fields: Any) -> None:
        cur = self._job(d) or {}
        now = self._now().isoformat()
        row = {"book": BOOK, "as_of_date": str(d), "next_session": str(plan.next_session),
               "run_id": cur.get("run_id"), "status": cur.get("status", "running"),
               "orders_allowed": int(plan.orders_allowed), "reason": cur.get("reason"),
               "attempts": int(cur.get("attempts") or 0), "session_id": self.session_id,
               "created_at": cur.get("created_at", now), "updated_at": now}
        row.update(fields)
        self.db.upsert("paper_session_jobs", row, ["book", "as_of_date"])
        self.last_job = self._job(d)

    def _maybe_process(self, now: datetime) -> None:
        if self.calendar is None:
            return
        plan = plan_session(self.calendar, now, self.process_after, self.order_cutoff)
        self.plan = plan
        if plan is None:
            self.phase = "no session in calendar window"
            return
        if not plan.due:
            self.phase = f"monitoring; {plan.reason}"
            return
        job = self._job(plan.session)
        if job is not None:
            self.last_job = job
            if job["status"] == "succeeded":
                self.phase = (f"monitoring; {plan.session} processed"
                              + ("" if job["orders_allowed"] else " (decisions only)")
                              + (f"; market open" if self.calendar.is_open(now) else ""))
                return
            if job["status"] == "failed" and int(job["attempts"]) >= self.max_job_attempts:
                self.phase = f"monitoring; processing {plan.session} failed {job['attempts']}x (system paused)"
                return
            if job["status"] in ("stale_data", "failed"):
                last = pd.Timestamp(job["updated_at"])
                if (pd.Timestamp(now) - last).total_seconds() < self.data_retry_seconds:
                    self.phase = f"monitoring; retrying {plan.session} after {job['reason']}"
                    return
        self.process_session(plan)

    def process_session(self, plan: SessionPlan) -> dict[str, Any]:
        d = plan.session
        self.phase = f"processing {d} (orders {'allowed' if plan.orders_allowed else 'NOT allowed'})"
        self._heartbeat(force=True)
        job = self._job(d)
        attempts = int(job["attempts"]) + 1 if job else 1
        self._save_job(d, plan, status="running", attempts=attempts, reason=plan.reason)
        self.event("INFO", "job", f"processing session {d}: {plan.reason}", plan.to_dict())

        # 1. data for D (ingest if the store does not reach D yet)
        last = stored_last_bar_date(self.ctx, synthetic=self.allow_synthetic)
        if last is None or last < d:
            self.phase = f"processing {d}: ingesting bars"
            self._heartbeat(force=True)
            summary = self._ingest(self.ctx, d)
            self.event("INFO", "data", f"ingest for {d}: {to_json(summary)[:500]}")
        bundle = self._bundle_loader(self.ctx)
        if bundle.is_synthetic and not self.allow_synthetic:
            raise RunnerRefused("the paper runner refuses SYNTHETIC market data")
        if pd.Timestamp(d) not in bundle.panel.dates:
            reason = f"data stale: no bars for {d} yet (latest {bundle.panel.dates[-1].date()})"
            self._save_job(d, plan, status="stale_data", reason=reason)
            self.event("WARN", "data", reason)
            if self._now() >= plan.order_deadline:
                self.pause(f"{reason}; order window for {plan.next_session} has passed", "data_stale")
            return {"status": "stale_data"}

        # 2. catalyst context. AFTER the staleness check above: when D's bars have not arrived yet the
        # session is retried every ``data_retry_seconds``, and refreshing first re-crawled SEC/news on
        # every retry (measured: ~12 min of requests per attempt, 4 attempts for one session).
        if self._catalyst_refresh is not None:
            self.phase = f"processing {d}: catalyst refresh"
            self._heartbeat(force=True)
            try:
                cr = self._catalyst_refresh(self.ctx, d, "daily")
                self.event("INFO", "data", f"catalyst refresh for {d}: " + ", ".join(
                    f"{k} {v.get('rows', v.get('error', ''))}" for k, v in cr.items() if isinstance(v, dict)), cr)
            except Exception as exc:          # catalysts are discovery context: never block trading
                self.event("WARN", "data", f"catalyst refresh for {d} failed: {exc!r}"[:500])

        # 3. broker verified + state reconciled, immediately before deciding
        self._poll_account(self._now(), force=True)
        self.reconcile(f"before processing {d}")

        # 4. the existing pipeline, with the runner's order gate
        from quantlab.pipeline.daily import DailyPipeline
        run_id = (job or {}).get("run_id") or self.ctx.start_run("pipeline", mode="BOT_PAPER", as_of_date=str(d),
                                                                  notes=f"paper runner session {self.session_id}")
        self._save_job(d, plan, run_id=run_id)
        pipe = DailyPipeline(self.ctx, synthetic=bundle.is_synthetic, broker=self.broker, bundle=bundle,
                             order_guard=self._order_guard(plan), exploration_submit="preopen")
        self.phase = f"processing {d}: pipeline run {run_id}"
        self._heartbeat(force=True)
        res = pipe.run(pd.Timestamp(d), resume_run_id=run_id)
        counts = {k: v for k, v in res.counts.items() if k.split(".")[0] in ("research", "decide", "orders", "execution", "exits")}
        if res.errors:
            self._save_job(d, plan, status="failed", reason="; ".join(res.errors)[:1000])
            self.pause(f"pipeline for {d} failed: {'; '.join(res.errors)[:300]}", "pipeline_error", {"run_id": run_id})
        else:
            self._save_job(d, plan, status="succeeded",
                           reason=plan.reason if plan.orders_allowed else "decisions only: order window missed")
        self.event("INFO" if not res.errors else "ERROR", "job",
                   f"session {d} pipeline {'failed' if res.errors else 'succeeded'}: "
                   f"{counts.get('research.candidates', 0)} candidates, {counts.get('decide.trade', 0)} TRADE, "
                   f"{counts.get('orders.orders_placed', 0)} order(s) placed, {counts.get('orders.orders_refused', 0)} refused",
                   {"run_id": run_id, "counts": counts, "steps": res.steps, "report": res.report_path})
        if not paper_eligible_strategies(self.db):
            self.event("WARN", "strategies", f"NO PAPER-ELIGIBLE STRATEGY for {d}: candidates recorded, no orders")
        self.phase = f"monitoring; {d} processed"
        return {"status": "failed" if res.errors else "succeeded", "run_id": run_id, "counts": counts}

    def _maybe_explore(self, now: datetime) -> dict[str, Any] | None:
        """PRE-OPEN step, from ``exploration.submit_after_et`` on the next session's date until the order
        cutoff (once per session): refresh overnight catalysts (real data), attach them to the
        next-session run, run the pre-open recheck, then -- in EXPLORATION mode only -- revalidate the
        planned exploratory entries and submit them through the same execution service and order guard
        as strict orders. STRICT mode submits nothing (selections stay SHADOW)."""
        from quantlab.exploration import ExplorationPolicy, paper_mode, preopen_submit
        if self.plan is None or self.calendar is None:
            return None
        plan = self.plan
        job = self._job(plan.session)
        if not job or job["status"] != "succeeded" or not plan.orders_allowed:
            return None
        pol = ExplorationPolicy.from_config(self.ctx.config)
        h, m = (int(x) for x in pol.submit_after_et.split(":"))
        start = datetime.combine(plan.next_session, time(h, m), tzinfo=ET).astimezone(timezone.utc)
        if now < start or order_window_reason(now, plan.session, plan.next_session, self.process_after, self.order_cutoff):
            return None
        if self._preopen_done != plan.session:
            self._preopen_done = plan.session
            self._preopen_information(plan, now)
        if paper_mode(self.ctx.config) != "EXPLORATION":
            return None
        from quantlab.execution.ledger import Ledger
        from quantlab.execution.service import PaperExecutionService
        book = Book.BOT.value
        svc = PaperExecutionService(self.db, self.ctx.config, book, self.broker,
                                    Ledger(self.db, book, config=self.ctx.config, broker_name=self.broker.name))
        svc.submission_guard = self._order_guard(plan)
        res = preopen_submit(self.ctx, svc, now=pd.Timestamp(now), session=str(plan.session))
        if res.get("submitted") or res.get("cancelled") or res.get("refused"):
            self.event("INFO", "exploration", f"pre-open exploration for {plan.next_session}: {res.get('submitted', 0)} "
                       f"submitted, {res.get('cancelled', 0)} cancelled, {res.get('refused', 0)} refused",
                       {k: v for k, v in res.items() if k != "details"})
        return res

    def _fallback_guard(self) -> Callable[[str, str], str | None]:
        """Gate for orders the RUNNER itself submits. Same broker/state conditions as the session
        guard (reconciliation passed, broker verified recently, trading not blocked), plus the
        fallback's own window. Anything other than the fallback is refused outright: this service is
        not the path for session decisions."""
        def guard(purpose: str, symbol: str) -> str | None:
            now = self._now()
            if purpose not in ("entry_fallback", "exit_fallback"):
                return (f"the runner's execution service only submits the open-fill fallback, not {purpose!r}")
            if self.calendar is None or not self.calendar.is_open(now):
                return "the market is not open: no open-fill fallback"
            if now.astimezone(ET).time() >= self.fallback_until:
                return f"after the open-fill fallback cutoff ({self.fallback_until:%H:%M} ET)"
            if not self.reconciled_ok:
                return "state reconciliation has not passed: no new orders"
            if self.broker_ok_at is None or (now - self.broker_ok_at).total_seconds() > self.broker_fresh_seconds:
                return "broker connection not verified recently: no new orders"
            if self.broker_snapshot.get("trading_blocked"):
                return "broker reports trading_blocked"
            return None
        return guard

    def _maybe_fill_after_open(self, now: datetime) -> dict[str, Any] | None:
        """An ``opg`` entry that expired unfilled in the opening auction gets ONE market-day order,
        so the decision actually takes a position instead of silently producing nothing.

        Bounded: regular session only, before ``open_fill_fallback_until_et``, only for entries that
        expired during this session with nothing filled, and only when no live entry order exists for
        that symbol/decision session (the execution service's duplicate guard re-checks this, and the
        fallback's client_order_id is deterministic, so a restart can never double-submit)."""
        if not self.open_fill_fallback or self.calendar is None or self.exec is None:
            return None
        if self._last_fallback_scan and (now - self._last_fallback_scan).total_seconds() < self.fallback_poll_seconds:
            return None
        self._last_fallback_scan = now
        if not self.calendar.is_open(now):
            return None
        session = self.calendar.current_session(now)
        if session is None or now.astimezone(ET).time() >= self.fallback_until:
            return None
        opened_at = self.calendar.open_dt(session)
        # exits are checked independently of entries: either can expire in the same auction
        out = {"considered": 0, "submitted": 0, "refused": 0}
        out.update(self._exit_fallbacks(opened_at))
        rows = self.db.fetchall(
            "SELECT o.order_id, o.symbol, o.qty, o.candidate_id, o.decision_id, o.human_decision_id, i.session_date, "
            "i.intent_json FROM orders o JOIN order_intents i ON i.order_id = o.order_id "
            "WHERE o.book=? AND o.purpose='entry' AND o.status=? AND o.time_in_force='opg' AND o.filled_qty=0 "
            "AND o.last_update_at >= ? ORDER BY o.created_at",
            (BOOK, OrderStatus.EXPIRED.value, opened_at.astimezone(timezone.utc).isoformat()))
        if not rows:
            return out if out["exits_considered"] else None
        out["considered"] = len(rows)
        for r in rows:
            live = self.db.fetchone(
                "SELECT 1 FROM orders o JOIN order_intents i ON i.order_id=o.order_id WHERE o.book=? AND o.symbol=? "
                "AND o.purpose='entry' AND i.session_date=? AND o.status NOT IN (?,?,?)",
                (BOOK, r["symbol"], r["session_date"], OrderStatus.REJECTED.value, OrderStatus.CANCELED.value,
                 OrderStatus.EXPIRED.value))
            if live:
                continue          # already replaced (or filled): nothing to do
            # carry the ORIGINAL plan through: a position without its stop and holding period would
            # not be managed by the exit engine
            intent = from_json(r["intent_json"], {}) or {}
            pl = intent.get("plan") or {}
            plan = TradePlan(entry_ref_price=pl.get("entry_ref_price"), stop_price=pl.get("stop_price"),
                             target_price=pl.get("target_price"),
                             holding_sessions=int(pl.get("holding_sessions", 20) or 20),
                             invalidation=str(pl.get("invalidation") or "")) if pl else None
            # The setup is only valid near the reference close: every plan states it is invalidated by
            # an opening gap of more than 1 ATR. A market fallback would chase a gapped-up open and
            # take exactly the trade the plan disowns, on a stop sized for the old price. Bound it at
            # ref + 1 ATR instead, so that case is a no-fill rather than a bad fill. The stop is
            # ref - 2*ATR, which is where the ATR comes from.
            style, limit = "market_day", None
            bound = self._fallback_limit(pl)
            if bound is not None:
                style, limit = "limit_day", bound
            res = self.exec.submit_entry(
                r["symbol"], float(r["qty"]), candidate_id=r["candidate_id"], decision_id=r["decision_id"],
                human_decision_id=r["human_decision_id"], session_date=r["session_date"], entry_style=style,
                limit_price=limit,
                plan=plan, strategy_id=intent.get("strategy_id"), strategy_version=intent.get("strategy_version"),
                journal={**(intent.get("journal") or {}), "fallback_for_order_id": r["order_id"],
                         "fallback_reason": "the opg order expired unfilled in the opening auction",
                         "fallback_limit_price": limit})
            if res.get("refused"):
                out["refused"] += 1
                self.event("WARN", "order", f"open-fill fallback refused for {r['symbol']}: {res.get('reason')}",
                           {"order_id": r["order_id"]})
            else:
                out["submitted"] += 1
                how = f"limit {limit:g} (ref + 1 ATR)" if limit is not None else "market-day (no ATR in the plan)"
                self.event("INFO", "order", f"open-fill fallback: {r['symbol']} {r['qty']:g} {how} after the "
                           f"opg order expired unfilled", {"expired_order_id": r["order_id"],
                                                            "order_id": res.get("order_id"),
                                                            "limit_price": limit})
        return out

    def _fallback_limit(self, plan: dict[str, Any] | None) -> float | None:
        """The highest price at which the fallback entry is still the planned trade: ref + 1 ATR.

        Exploratory stops are ``ref - stop_atr * ATR``, so the ATR is recoverable from the plan
        itself and no extra price data is needed at the open. Returns None when the plan does not
        carry a usable reference and stop, in which case the caller keeps the old market fallback
        rather than skipping the trade."""
        if not plan:
            return None
        try:
            ref = float(plan.get("entry_ref_price"))
            stop = float(plan.get("stop_price"))
            mult = float(plan.get("stop_atr") or self.cfg.get("exploration.stop_atr", 2.0))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(ref) and math.isfinite(stop) and math.isfinite(mult)):
            return None
        if not (0 < stop < ref) or mult <= 0:
            return None
        atr = (ref - stop) / mult
        limit = round(ref + atr, 2)
        return limit if limit > 0 else None

    def _exit_fallbacks(self, opened_at: datetime) -> dict[str, Any]:
        """Same fallback for EXITS: an opg exit that expired unfilled would leave the position open
        with its stop already breached. Only for trades that are still OPEN."""
        assert self.exec is not None
        rows = self.db.fetchall(
            "SELECT o.order_id, o.trade_id, i.session_date, i.intent_json FROM orders o "
            "JOIN order_intents i ON i.order_id = o.order_id JOIN trades t ON t.trade_id = o.trade_id "
            "WHERE o.book=? AND o.purpose='exit' AND o.status=? AND o.time_in_force='opg' AND o.filled_qty=0 "
            "AND o.last_update_at >= ? AND t.status='OPEN' ORDER BY o.created_at",
            (BOOK, OrderStatus.EXPIRED.value, opened_at.astimezone(timezone.utc).isoformat()))
        out = {"exits_considered": len(rows), "exits_submitted": 0, "exits_refused": 0}
        for r in rows:
            intent = from_json(r["intent_json"], {}) or {}
            res = self.exec.submit_exit(r["trade_id"], str(intent.get("reason") or "stop"),
                                        detail={**{k: v for k, v in intent.items() if k not in ("trade_id", "reason")},
                                                "fallback_for_order_id": r["order_id"],
                                                "fallback_reason": "the opg exit expired unfilled"},
                                        session_date=r["session_date"], exit_style="market_day")
            if res.get("refused"):
                out["exits_refused"] += 1
                self.event("WARN", "order", f"exit fallback refused for trade {r['trade_id']}: {res.get('reason')}",
                           {"order_id": r["order_id"]})
            else:
                out["exits_submitted"] += 1
                self.event("INFO", "order", f"exit fallback: trade {r['trade_id']} market-day after the opg exit "
                           f"expired unfilled", {"expired_order_id": r["order_id"], "order_id": res.get("order_id")})
        return out

    def _preopen_information(self, plan: SessionPlan, now: datetime) -> None:
        """Overnight catalysts -> next-session run -> pre-open recheck. Best effort, logged."""
        from quantlab.discovery import nextsession as ns
        run = ns.latest_run(self.db)
        if run is None or str(run["as_of_date"]) != str(plan.session):
            self.event("WARN", "preopen", f"no end-of-day discovery run for {plan.session}: pre-open recheck skipped")
            return
        if self._catalyst_refresh is not None and not run["is_synthetic"]:
            syms = [r["symbol"] for r in self.db.fetchall("SELECT symbol FROM discovery_candidates WHERE discovery_run_id=?",
                                                          (run["discovery_run_id"],))]
            try:
                self._catalyst_refresh(self.ctx, plan.session, "preopen", symbols=syms)
            except Exception as exc:
                self.event("WARN", "preopen", f"pre-open catalyst refresh failed: {exc!r}"[:500])
        try:
            r1 = ns.overnight_refresh(self.ctx, pd.Timestamp(now))
            r2 = ns.preopen_recheck(self.ctx, pd.Timestamp(now))
            self.event("INFO", "preopen", f"pre-open for {plan.next_session}: {r1.get('recorded', 0)} overnight item(s), "
                       f"{r1.get('new_candidates', 0)} new candidate(s), {r2.get('checked', 0)} rechecked",
                       {"overnight": r1, "transitions": r2.get("transitions")})
        except Exception as exc:
            self.event("WARN", "preopen", f"pre-open recheck failed: {exc!r}"[:500])

    def _order_guard(self, plan: SessionPlan) -> Callable[[str, str], str | None]:
        def guard(purpose: str, symbol: str) -> str | None:
            now = self._now()
            if purpose == "entry_fallback":
                # replaces an expired opening order during the regular session, not a new decision
                if self.calendar is None or not self.calendar.is_open(now):
                    return "the market is not open: no open-fill fallback"
                if now.astimezone(ET).time() >= self.fallback_until:
                    return f"after the open-fill fallback cutoff ({self.fallback_until:%H:%M} ET)"
            else:
                reason = order_window_reason(now, plan.session, plan.next_session, self.process_after,
                                             self.order_cutoff)
                if reason:
                    return reason
            if not self.reconciled_ok:
                return "state reconciliation has not passed: no new orders"
            if self.broker_ok_at is None or (now - self.broker_ok_at).total_seconds() > self.broker_fresh_seconds:
                return "broker connection not verified recently: no new orders"
            if self.broker_snapshot.get("trading_blocked"):
                return "broker reports trading_blocked"
            return None
        return guard

    # -- heartbeat --------------------------------------------------------------------------------
    def detail(self) -> dict[str, Any]:
        elig = paper_eligible_strategies(self.db)
        return {
            "broker": self.broker_snapshot, "broker_ok_at": _iso(self.broker_ok_at),
            "calendar_source": self.calendar.source if self.calendar else None,
            "market_open": self.calendar.is_open(self._now()) if self.calendar else None,
            "plan": self.plan.to_dict() if self.plan else None, "last_job": self.last_job,
            "eligible_strategies": elig, "no_paper_eligible_strategy": not elig,
            "reconciled_ok": self.reconciled_ok, "last_tick_at": self._last_tick_at,
            "stream_connects": getattr(self.stream, "connects", None),
            "process_after_et": self.process_after.strftime("%H:%M"),
            "order_cutoff_et": self.order_cutoff.strftime("%H:%M"),
        }

    def _heartbeat(self, force: bool = False, db=None) -> None:
        now = self._now()
        if not force and self._last_heartbeat and (now - self._last_heartbeat).total_seconds() < self.heartbeat_seconds:
            return
        self._last_heartbeat = now
        (db or self.db).execute(
            "UPDATE paper_runner_sessions SET last_heartbeat_at=?, phase=?, stream_status=?, detail_json=? "
            "WHERE session_id=? AND status='RUNNING'",
            (utcnow_iso(), self.phase[:500], self.stream_state, to_json(self.detail()), self.session_id))

    def _heartbeat_loop(self) -> None:
        """Separate connection + thread, so a long pipeline run never looks like a dead runner."""
        db = open_db(self.db.path, migrate=False) if str(self.db.path) != ":memory:" else self.db
        try:
            while not self._hb_stop.wait(self.heartbeat_seconds):
                try:
                    state = getattr(self.stream, "state", None) or self.stream_state   # live even mid-pipeline
                    db.execute("UPDATE paper_runner_sessions SET last_heartbeat_at=?, phase=?, stream_status=? "
                               "WHERE session_id=? AND status='RUNNING'",
                               (utcnow_iso(), self.phase[:500], state, self.session_id))
                except Exception as exc:   # pragma: no cover - heartbeat must never crash the runner
                    log_event(log, "heartbeat write failed", error=repr(exc))
        finally:
            if db is not self.db:
                db.close()


# ------------------------------------------------------------------------------------------------
# control + status (used by the CLI and the dashboard)
# ------------------------------------------------------------------------------------------------
def request_stop(db, reason: str = "operator stop") -> list[str]:
    rows = db.fetchall("SELECT session_id FROM paper_runner_sessions WHERE status='RUNNING'")
    for r in rows:
        db.execute("UPDATE paper_runner_sessions SET stop_requested_at=?, stop_reason=? WHERE session_id=?",
                   (utcnow_iso(), reason, r["session_id"]))
        db.insert("paper_runner_events", {"session_id": r["session_id"], "at": utcnow_iso(), "level": "INFO",
                                          "kind": "lifecycle", "message": f"stop requested: {reason}",
                                          "details_json": to_json({})})
    return [r["session_id"] for r in rows]


def runner_status(db, stale_after: float = 120.0, now: datetime | None = None) -> dict[str, Any]:
    """RUNNING (fresh heartbeat) | STALE (marked running, heartbeat old: process died) | STOPPED |
    CRASHED | NEVER_STARTED, plus the latest session row."""
    row = db.fetchone("SELECT * FROM paper_runner_sessions ORDER BY started_at DESC LIMIT 1")
    if row is None:
        return {"state": "NEVER_STARTED", "session": None}
    now = now or datetime.now(timezone.utc)
    age = None
    if row["last_heartbeat_at"]:
        age = (pd.Timestamp(now) - pd.Timestamp(row["last_heartbeat_at"])).total_seconds()
    state = row["status"]
    if state == "RUNNING" and (age is None or age > stale_after):
        state = "STALE"
    row = dict(row)
    row["detail"] = from_json(row.pop("detail_json"), {})
    return {"state": state, "heartbeat_age_seconds": age, "session": row}


# ------------------------------------------------------------------------------------------------
# connectivity test (NOT a strategy order)
# ------------------------------------------------------------------------------------------------
def connectivity_order_test(ctx: AppContext, symbol: str = "SPY", *, broker: Any = None, stream_factory=None,
                            env: dict[str, str] | None = None, wait_seconds: float = 45.0,
                            limit_fraction: float = 0.5) -> dict[str, Any]:
    """Prove the order path against the PAPER account without trading: submit ONE share as a DAY
    limit BUY at ``limit_fraction`` x the last stored close (non-marketable, cannot fill), watch the
    trade_updates stream, cancel it, and record every broker event. It is not a strategy order, is
    never written to the ledger, and is refused while SYSTEM_PAUSED."""
    from quantlab.core.types import Side
    from quantlab.execution.broker import OrderRequest

    if broker is None:
        from quantlab.execution.alpaca_paper import AlpacaPaperBroker
        broker = AlpacaPaperBroker(ctx.config)
    pre = run_preflight(ctx.config, ctx.db, broker, env=env)
    if not pre.ok:
        raise RunnerRefused(f"paper preflight failed: {pre.reason}")
    if KillSwitch(ctx.db).is_paused():
        raise RunnerRefused("SYSTEM_PAUSED: the connectivity test places a paper order and is refused while paused")
    bars = ctx.store.load("bars", ctx.store.dataset_ids("bars", synthetic=False))
    px = bars.loc[bars["symbol"] == symbol].sort_values("date")["close"]
    if px.empty:
        raise RunnerRefused(f"no stored real close for {symbol}")
    limit = round(float(px.iloc[-1]) * limit_fraction, 2)
    q: queue.Queue = queue.Queue()
    events: list[dict[str, Any]] = []
    on_event = lambda data: q.put(("update", data))          # noqa: E731
    on_state = lambda s, d: q.put(("state", s, d))           # noqa: E731
    if stream_factory is not None:
        stream = stream_factory(on_event, on_state)
    else:
        from quantlab.execution.trade_stream import TradeUpdateStream
        stream = TradeUpdateStream(broker.stream_url(), broker.stream_auth_message, on_event, on_state)
    stream.start()
    cid = f"{TEST_ORDER_PREFIX}{new_id('o')[2:18]}"
    states: list[str] = []

    def pump(until: Callable[[], bool], seconds: float) -> None:
        end = _time.monotonic() + seconds
        while not until() and _time.monotonic() < end:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item[0] == "state":
                states.append(item[1])
            else:
                data = item[1]
                bo = broker.parse_order(data.get("order") or {})
                if bo.client_order_id == cid:
                    events.append({"event": data.get("event"), "status": bo.status.value,
                                   "broker_order_id": bo.broker_order_id, "at": data.get("timestamp")})
                    ctx.db.insert("broker_order_updates", {
                        "session_id": None, "received_at": utcnow_iso(), "source": "test", "event": data.get("event", "?"),
                        "order_id": None, "client_order_id": cid, "broker_order_id": bo.broker_order_id,
                        "execution_id": data.get("execution_id"), "status": bo.status.value, "event_qty": None,
                        "event_price": None, "filled_qty": bo.filled_qty, "filled_avg_price": bo.filled_avg_price,
                        "event_at": data.get("timestamp"), "raw_json": to_json(data)}, or_ignore=True)

    try:
        pump(lambda: "connected" in states or "unauthorized" in states, wait_seconds)
        if "connected" not in states:
            raise RunnerRefused(f"trade_updates stream did not connect (states: {states})")
        req = OrderRequest(client_order_id=cid, symbol=symbol, side=Side.BUY, qty=1, order_type="limit",
                           time_in_force="day", limit_price=limit)
        submitted = broker.submit_order(req)
        pump(lambda: any(e["status"] in ("accepted", "new", "rejected") for e in events), wait_seconds)
        canceled = None
        if submitted.status.value not in ("rejected", "filled"):
            canceled = broker.cancel_order(cid)
            pump(lambda: any(e["status"] in ("canceled", "filled", "expired") for e in events), wait_seconds)
        final = broker.get_order_by_client_id(cid)
    finally:
        stream.stop()
    out = {"client_order_id": cid, "symbol": symbol, "limit_price": limit, "last_close": float(px.iloc[-1]),
           "submitted_status": submitted.status.value, "broker_order_id": submitted.broker_order_id,
           "cancel_requested": canceled is not None, "final_status": final.status.value if final else None,
           "final_filled_qty": final.filled_qty if final else None, "stream_states": states,
           "stream_events": events}
    ctx.db.insert("paper_runner_events", {"session_id": None, "at": utcnow_iso(),
                                          "level": "INFO" if events else "WARN", "kind": "order_test",
                                          "message": f"connectivity order test {symbol}: submitted "
                                          f"{out['submitted_status']}, final {out['final_status']}, "
                                          f"{len(events)} stream event(s)", "details_json": to_json(out)})
    return out


def round_trip_order_test(ctx: AppContext, symbol: str = "SPY", qty: float = 1, *, broker: Any = None,
                          stream_factory=None, env: dict[str, str] | None = None,
                          wait_seconds: float = 60.0) -> dict[str, Any]:
    """Prove that orders FILL and that a position can be SOLD on the PAPER account: market BUY
    ``qty`` (DAY), wait for the fill on the trade_updates stream, market SELL the same quantity,
    wait for that fill, and check the position is flat again. Needs regular market hours.

    Not a strategy trade and never written to a ledger. Because it moves the broker's cash by the
    round-trip spread, it is refused once the BOT book is bound to the Alpaca paper account (the
    ledger would no longer reconcile) and while SYSTEM_PAUSED."""
    from quantlab.core.types import Side
    from quantlab.execution.broker import OrderRequest

    if book_binding(ctx.db, BOOK) == BROKER_NAME:
        raise RunnerRefused("round-trip test refused: the BOT ledger is already bound to this paper account and the "
                            "test's cash change would break reconciliation")
    if broker is None:
        from quantlab.execution.alpaca_paper import AlpacaPaperBroker
        broker = AlpacaPaperBroker(ctx.config)
    pre = run_preflight(ctx.config, ctx.db, broker, env=env)
    if not pre.ok:
        raise RunnerRefused(f"paper preflight failed: {pre.reason}")
    if KillSwitch(ctx.db).is_paused():
        raise RunnerRefused("SYSTEM_PAUSED: the round-trip test places paper orders and is refused while paused")
    if any(p["symbol"] == symbol for p in pre.positions):
        raise RunnerRefused(f"the paper account already holds {symbol}; the test needs a flat position")
    clock = broker.clock()
    if not clock.get("is_open"):
        raise RunnerRefused(f"market closed (next open {clock.get('next_open')}): market orders would not fill now")

    q: queue.Queue = queue.Queue()
    on_event = lambda data: q.put(("update", data))          # noqa: E731
    on_state = lambda s, d: q.put(("state", s, d))           # noqa: E731
    if stream_factory is not None:
        stream = stream_factory(on_event, on_state)
    else:
        from quantlab.execution.trade_stream import TradeUpdateStream
        stream = TradeUpdateStream(broker.stream_url(), broker.stream_auth_message, on_event, on_state)
    base = new_id("o")[2:14]
    legs = {"buy": f"{TEST_ORDER_PREFIX}{base}-b", "sell": f"{TEST_ORDER_PREFIX}{base}-s"}
    states: list[str] = []
    events: list[dict[str, Any]] = []

    def pump(until: Callable[[], bool], seconds: float) -> None:
        end = _time.monotonic() + seconds
        while not until() and _time.monotonic() < end:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item[0] == "state":
                states.append(item[1])
                continue
            data = item[1]
            bo = broker.parse_order(data.get("order") or {})
            if bo.client_order_id not in legs.values():
                continue
            events.append({"leg": "buy" if bo.client_order_id == legs["buy"] else "sell", "event": data.get("event"),
                           "status": bo.status.value, "price": data.get("price"), "qty": data.get("qty"),
                           "filled_qty": bo.filled_qty, "at": data.get("timestamp")})
            ctx.db.insert("broker_order_updates", {
                "session_id": None, "received_at": utcnow_iso(), "source": "test", "event": data.get("event", "?"),
                "order_id": None, "client_order_id": bo.client_order_id, "broker_order_id": bo.broker_order_id,
                "execution_id": data.get("execution_id"), "status": bo.status.value,
                "event_qty": float(data["qty"]) if data.get("qty") else None,
                "event_price": float(data["price"]) if data.get("price") else None, "filled_qty": bo.filled_qty,
                "filled_avg_price": bo.filled_avg_price, "event_at": data.get("timestamp"),
                "raw_json": to_json(data)}, or_ignore=True)

    def done(leg: str) -> bool:
        return any(e["leg"] == leg and e["status"] in ("filled", "rejected", "canceled", "expired") for e in events)

    stream.start()
    result: dict[str, Any] = {"symbol": symbol, "qty": qty, "client_order_ids": legs}
    try:
        pump(lambda: "connected" in states or "unauthorized" in states, wait_seconds)
        if "connected" not in states:
            raise RunnerRefused(f"trade_updates stream did not connect (states: {states})")
        for leg, side in (("buy", Side.BUY), ("sell", Side.SELL)):
            sub = broker.submit_order(OrderRequest(client_order_id=legs[leg], symbol=symbol, side=side, qty=qty,
                                                   order_type="market", time_in_force="day"))
            pump(lambda leg=leg: done(leg), wait_seconds)
            final = broker.get_order_by_client_id(legs[leg])
            result[leg] = {"submitted_status": sub.status.value, "broker_order_id": sub.broker_order_id,
                           "final_status": final.status.value if final else None,
                           "filled_qty": final.filled_qty if final else None,
                           "filled_avg_price": final.filled_avg_price if final else None,
                           "filled_at": final.filled_at if final else None,
                           "fill_seen_on_stream": any(e["leg"] == leg and e["event"] in ("fill", "partial_fill")
                                                      for e in events)}
            if not final or final.status.value != "filled":
                if final is not None and final.is_open:
                    broker.cancel_order(legs[leg])
                raise RunnerRefused(f"{leg} leg did not fill within {wait_seconds:.0f}s: {result[leg]}")
    finally:
        stream.stop()
        after = [p for p in broker.positions() if p.symbol == symbol]
        result["position_after"] = after[0].qty if after else 0.0
        result["stream_states"] = states
        result["stream_events"] = events
        b, s = result.get("buy") or {}, result.get("sell") or {}
        if b.get("filled_avg_price") and s.get("filled_avg_price"):
            result["round_trip_pnl"] = round((s["filled_avg_price"] - b["filled_avg_price"]) * qty, 4)
        ctx.db.insert("paper_runner_events", {
            "session_id": None, "at": utcnow_iso(), "level": "INFO" if result["position_after"] == 0 else "ERROR",
            "kind": "order_test", "message": f"round-trip test {symbol}: buy {b.get('final_status')} "
            f"@ {b.get('filled_avg_price')}, sell {s.get('final_status')} @ {s.get('filled_avg_price')}, "
            f"position after {result['position_after']:g}", "details_json": to_json(result)})
    return result


__all__ = ["MarketCalendar", "PaperRunner", "RunnerRefused", "SessionPlan", "connectivity_order_test",
           "order_window", "order_window_reason", "paper_eligible_strategies", "plan_session", "request_stop",
           "runner_status"]
