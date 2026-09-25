"""Persistent PAPER runner on SYNTHETIC data with an in-memory Alpaca-paper stand-in:
schedule / next-session execution, SHADOW -> no orders, eligible strategy -> opg order + stream
fills, restart idempotency (no duplicate orders), partial fills, rejections, reconnect
reconciliation, stale data, kill switch, EV rejection, single instance, clean shutdown/restart,
book binding, and the dashboard live view."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.context import AppContext
from quantlab.core.types import TradePlan
from quantlab.dashboard.app import create_app
from quantlab.execution.ledger import BookBindingError, Ledger, bind_book
from quantlab.execution.service import PaperExecutionService
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.pipeline.daily import DailyPipeline
from quantlab.pipeline.runner import (
    MarketCalendar, PaperRunner, RunnerRefused, order_window_reason, plan_session, request_stop, runner_status,
)

from .fake_alpaca import PAPER_ENV, FakeAlpacaBroker, FakeStream
from .test_daily import _activate, world  # noqa: F401  (fixture re-export)

ET = ZoneInfo("America/New_York")


def at(d, hh, mm=0) -> datetime:
    return datetime.combine(pd.Timestamp(d).date(), time(hh, mm), ET).astimezone(timezone.utc)


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def make_runner(ctx, broker, clock, bundle=None, **kw):
    streams = []

    def factory(on_event, on_state):
        streams.append(FakeStream(on_event, on_state))
        return streams[-1]
    b = bundle if bundle is not None else ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    r = PaperRunner(ctx, broker=broker, stream_factory=factory, now=clock, ingest=lambda c, d: {"skipped": True},
                    bundle_loader=kw.pop("bundle_loader", lambda c: b), env=kw.pop("env", PAPER_ENV),
                    heartbeat_thread=False, allow_synthetic=True, **kw)
    r.streams = streams
    r.reconcile_retry_seconds = 0.0
    return r, b


@pytest.fixture
def setup(world):  # noqa: F811
    bundle = world.store.load_bundle(world.config.section("benchmarks"), synthetic=True)
    sessions = [d.date() for d in bundle.panel.dates]
    return world, bundle, sessions


# -- schedule (pure) ------------------------------------------------------------------------------
def test_schedule_and_next_session_order_window():
    days = [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8)]  # 4th = holiday
    cal = MarketCalendar({d: (time(9, 30), time(16, 0)) for d in days}, "test")
    pa, cut = time(19, 5), time(9, 25)
    p = plan_session(cal, at("2024-07-02", 17, 0), pa, cut)
    assert p.session == date(2024, 7, 2) and not p.due                    # closed, but before 19:05
    p = plan_session(cal, at("2024-07-02", 19, 10), pa, cut)
    assert p.due and p.orders_allowed and p.next_session == date(2024, 7, 3)
    p = plan_session(cal, at("2024-07-03", 19, 10), pa, cut)
    assert p.next_session == date(2024, 7, 5)                             # skips the holiday
    p = plan_session(cal, at("2024-07-05", 9, 20), pa, cut)
    assert p.session == date(2024, 7, 3) and p.orders_allowed             # still before the cutoff
    p = plan_session(cal, at("2024-07-05", 9, 26), pa, cut)
    assert p.due and not p.orders_allowed                                 # the next open is too close/past
    p = plan_session(cal, at("2024-07-05", 20, 0), pa, cut)
    assert p.session == date(2024, 7, 5) and p.next_session == date(2024, 7, 8)   # Friday -> Monday
    assert "missed" in order_window_reason(at("2024-07-03", 9, 30), date(2024, 7, 2), date(2024, 7, 3), pa, cut)
    assert "before the order window" in order_window_reason(at("2024-07-02", 18, 0), date(2024, 7, 2),
                                                            date(2024, 7, 3), pa, cut)
    assert order_window_reason(at("2024-07-03", 8, 0), date(2024, 7, 2), date(2024, 7, 3), pa, cut) is None


def test_config_cannot_move_orders_into_alpacas_opg_rejection_window(setup):
    ctx, bundle, sessions = setup
    for bad in ({"process_after_et": "18:30"}, {"order_cutoff_et": "09:29"}):
        c = AppContext.create(ctx.config.with_overrides({"paper": {"runner": bad}}), init_logging=False)
        with pytest.raises(RunnerRefused):
            make_runner(c, FakeAlpacaBroker(sessions), Clock(at(sessions[-5], 20)))


# -- start-up refusals ------------------------------------------------------------------------------
def test_refuses_to_start_in_live_mode(setup):
    ctx, bundle, sessions = setup
    r, _ = make_runner(ctx, FakeAlpacaBroker(sessions), Clock(at(sessions[-5], 20)),
                       env={**PAPER_ENV, "LIVE_TRADING": "true"})
    with pytest.raises(RunnerRefused, match="LIVE_TRADING"):
        r.run_forever(max_ticks=1)
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM paper_runner_sessions WHERE status='RUNNING'")["n"] == 0
    assert ctx.db.fetchone("SELECT ok FROM paper_preflights")["ok"] == 0


def test_refuses_a_dirty_paper_account_and_a_sim_bound_book(setup):
    ctx, bundle, sessions = setup
    dirty = FakeAlpacaBroker(sessions)
    dirty.pos["ORCL"] = [1.0, 100.0]
    r, _ = make_runner(ctx, dirty, Clock(at(sessions[-5], 20)))
    with pytest.raises(RunnerRefused, match="dedicated paper account"):
        r.run_forever(max_ticks=1)
    assert runner_status(ctx.db)["session"]["status"] == "REFUSED"
    # a BOT book that already ran on the simulated broker can never be continued by Alpaca paper
    DailyPipeline(ctx, synthetic=True)
    r2, _ = make_runner(ctx, FakeAlpacaBroker(sessions), Clock(at(sessions[-5], 20)))
    with pytest.raises(RunnerRefused, match="simulated|bound"):
        r2.run_forever(max_ticks=1)


def test_book_binding_is_permanent(db):
    bind_book(db, "BOT", "alpaca_paper")
    with pytest.raises(BookBindingError):
        bind_book(db, "BOT", "sim")
    assert bind_book(db, "BOT", "alpaca_paper") == "alpaca_paper"


# -- SHADOW only ----------------------------------------------------------------------------------------
def test_shadow_strategies_record_candidates_and_place_no_orders(setup):
    ctx, bundle, sessions = setup
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _ = make_runner(ctx, broker, clock)
    r.start()
    assert r.tick() is None
    job = ctx.db.fetchone("SELECT * FROM paper_session_jobs")
    assert job["status"] == "succeeded" and job["as_of_date"] == str(sessions[-30]) and job["orders_allowed"] == 1
    n = ctx.db.fetchone("SELECT COUNT(*) AS n FROM candidates")["n"]
    assert n > 0 and ctx.db.fetchone("SELECT COUNT(*) AS n FROM decisions")["n"] == n
    stages = {x["reject_stage"] for x in ctx.db.fetchall("SELECT reject_stage FROM decisions")}
    assert "NONE" not in stages and "STRATEGY" in stages
    assert broker.submits == [] and ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0
    msgs = [e["message"] for e in ctx.db.fetchall("SELECT message FROM paper_runner_events")]
    assert any("NO PAPER-ELIGIBLE STRATEGY" in m for m in msgs)
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE"
    # the ledger was seeded from the broker's cash and reconciles
    assert Ledger(ctx.db, "BOT", broker_name="alpaca_paper").cash() == pytest.approx(100_000.0)
    # a second tick the same evening does not re-process the session
    r.tick()
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM runs WHERE kind='pipeline'")["n"] == 1
    r.shutdown("test")


# -- eligible strategy: next-session order, stream fills, restart idempotency ------------------------
def _run_until_order(ctx, broker, sessions, start_idx=-60, span=35):
    """Advance day by day (19:10 ET) until the runner places an entry order."""
    clock = Clock(at(sessions[start_idx], 19, 10))
    r, b = make_runner(ctx, broker, clock)
    r.start()
    for i in range(start_idx, start_idx + span):
        clock.t = at(sessions[i], 19, 10)
        r.tick()
        if broker.submits:
            return r, clock, sessions[i]
    pytest.fail("no entry order within the scanned sessions")


@pytest.mark.slow
def test_eligible_strategy_orders_fill_via_stream_and_restart_never_duplicates(setup):
    ctx, bundle, sessions = setup
    _activate(ctx)
    broker = FakeAlpacaBroker(sessions)
    r, clock, d = _run_until_order(ctx, broker, sessions)
    orders = ctx.db.fetchall("SELECT o.*, i.session_date FROM orders o JOIN order_intents i ON i.order_id=o.order_id")
    assert orders and all(o["time_in_force"] == "opg" and o["order_type"] == "market" for o in orders)
    assert all(o["session_date"] == str(d) for o in orders)                       # decided at D's close
    assert all(o["broker_order_id"] and o["status"] == "accepted" for o in orders)  # broker id persisted
    n_submits = len(broker.submits)

    # restart: the job is marked as interrupted and a NEW runner process resumes it
    r.shutdown("simulated crash")
    ctx.db.execute("UPDATE paper_session_jobs SET status='running' WHERE as_of_date=?", (str(d),))
    r2, _ = make_runner(ctx, broker, clock)
    r2.start()
    r2.tick()
    assert len(broker.submits) == n_submits, "a restart must never resubmit an order"
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == len(orders)
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM runs WHERE kind='pipeline' AND as_of_date=?", (str(d),))["n"] == 1

    # D+1 open: fills arrive on the stream (one partial fill, then the rest)
    nxt = sessions[sessions.index(d) + 1]
    clock.t = at(nxt, 9, 31)
    ts = at(nxt, 9, 30).isoformat()
    o = orders[0]
    px = float(bundle.panel.open.at[pd.Timestamp(nxt), o["symbol"]])
    half = max(1.0, float(int(o["qty"] // 2)))
    stream = r2.streams[-1]
    stream.push(broker.fill(o["client_order_id"], half, px, ts))
    if o["qty"] - half > 0:
        stream.push(broker.fill(o["client_order_id"], o["qty"] - half, px + 0.10, ts))
    for other in orders[1:]:
        p2 = float(bundle.panel.open.at[pd.Timestamp(nxt), other["symbol"]])
        stream.push(broker.fill(other["client_order_id"], other["qty"], p2, ts))
    r2.tick()
    row = ctx.db.fetchone("SELECT * FROM orders WHERE order_id=?", (o["order_id"],))
    assert row["status"] == "filled" and row["filled_qty"] == pytest.approx(o["qty"])
    fills = ctx.db.fetchall("SELECT * FROM fills WHERE order_id=? ORDER BY filled_at, fill_id", (o["order_id"],))
    assert sum(f["qty"] for f in fills) == pytest.approx(o["qty"])
    assert {f["session_date"] for f in fills} == {str(nxt)}                        # executed at D+1, not D
    trade = ctx.db.fetchone("SELECT * FROM trades WHERE symbol=? AND book='BOT'", (o["symbol"],))
    assert trade["status"] == "OPEN" and trade["entry_date"] == str(nxt)
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE", KillSwitch(ctx.db).state()[1]
    # replayed events after a reconnect change nothing (cumulative filled_qty -> no delta)
    n_fills = ctx.db.fetchone("SELECT COUNT(*) AS n FROM fills")["n"]
    stream.reconnect()
    stream.push(broker.event(o["client_order_id"], "fill", ts))
    r2.tick()
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM fills")["n"] == n_fills
    events = [e["message"] for e in ctx.db.fetchall("SELECT message FROM paper_runner_events")]
    assert any("reconciliation ok (stream reconnect)" in m for m in events)

    # the dashboard live view shows the runner, the order with its broker id, and the position
    c = TestClient(create_app(ctx))
    live = c.get("/api/live").json()
    assert live["system"]["session"]["session_id"] == r2.session_id
    assert any(x["broker_order_id"] == row["broker_order_id"] for x in live["orders"])
    assert o["symbol"] in {p["symbol"] for p in live["portfolio"]["positions"]}
    r2.shutdown("test")


# -- idempotency at the service level ---------------------------------------------------------------
def _alpaca_service(ctx, broker):
    bind_book(ctx.db, "BOT", "alpaca_paper")
    ledger = Ledger(ctx.db, "BOT", starting_cash=100_000.0, broker_name="alpaca_paper")
    return PaperExecutionService(ctx.db, ctx.config, "BOT", broker, ledger), ledger


def test_crash_between_broker_accept_and_local_update_is_adopted_not_resubmitted(ctx):
    broker = FakeAlpacaBroker([])
    svc, ledger = _alpaca_service(ctx, broker)
    broker.crash_after_submit = True
    with pytest.raises(RuntimeError, match="simulated crash"):
        svc.submit_entry("AAA", 10, plan=TradePlan(entry_ref_price=10.0), session_date="2024-05-01")
    row = ctx.db.fetchone("SELECT * FROM orders")
    assert row["status"] == "pending_submit" and row["broker_order_id"] is None
    again = svc.submit_entry("AAA", 10, plan=TradePlan(entry_ref_price=10.0), session_date="2024-05-01")
    assert broker.submits == [row["client_order_id"]]                  # exactly one POST ever
    assert again["status"] == "accepted" and again["broker_order_id"]
    third = svc.submit_entry("AAA", 10, plan=TradePlan(entry_ref_price=10.0), session_date="2024-05-01")
    assert third.get("duplicate") and len(broker.submits) == 1
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 1


def test_partial_entry_and_partial_exit_fills(ctx):
    broker = FakeAlpacaBroker([])
    svc, ledger = _alpaca_service(ctx, broker)
    e = svc.submit_entry("AAA", 10, plan=TradePlan(entry_ref_price=10.0), session_date="2024-05-01")
    cid = e["client_order_id"]
    svc.apply_broker_order(broker.parse_order(broker.fill(cid, 4, 10.0, "2024-05-02T13:30:01Z")["order"]), "2024-05-02")
    svc.apply_broker_order(broker.parse_order(broker.fill(cid, 6, 11.0, "2024-05-02T13:30:02Z")["order"]), "2024-05-02")
    prices = sorted(f["price"] for f in ctx.db.fetchall("SELECT price FROM fills"))
    assert prices == pytest.approx([10.0, 11.0])                        # increment priced, not the cumulative avg
    assert ledger.get_position("AAA")["qty"] == pytest.approx(10)
    assert ledger.cash() == pytest.approx(100_000 - 40 - 66)
    assert ledger.cash() == pytest.approx(broker.cash)
    trade = ledger.open_trades()[0]
    x = svc.submit_exit(trade.trade_id, "TIME", session_date="2024-05-03")
    xo = x["client_order_id"]
    svc.apply_broker_order(broker.parse_order(broker.fill(xo, 3, 12.0, "2024-05-06T13:30:01Z")["order"]), "2024-05-06")
    assert ledger.journal.get_trade(trade.trade_id)["status"] == "OPEN"   # partial exit: still open
    assert ledger.get_position("AAA")["qty"] == pytest.approx(7)
    svc.apply_broker_order(broker.parse_order(broker.fill(xo, 7, 13.0, "2024-05-06T13:30:02Z")["order"]), "2024-05-06")
    t = ledger.journal.get_trade(trade.trade_id)
    assert t["status"] == "CLOSED" and t["exit_price"] == pytest.approx((3 * 12 + 7 * 13) / 10)
    assert t["net_pnl"] == pytest.approx((3 * 12 + 7 * 13) - (40 + 66))
    assert ledger.get_position("AAA") is None and ledger.cash() == pytest.approx(broker.cash)


def test_rejected_orders_leave_the_ledger_untouched(ctx):
    broker = FakeAlpacaBroker([])
    svc, ledger = _alpaca_service(ctx, broker)
    broker.reject_next = True
    r = svc.submit_entry("AAA", 10, plan=TradePlan(entry_ref_price=10.0), session_date="2024-05-01")
    assert r["status"] == "rejected"
    ok = svc.submit_entry("BBB", 5, plan=TradePlan(entry_ref_price=10.0), session_date="2024-05-01")
    bo = broker.parse_order(broker.event(ok["client_order_id"], "rejected", "t", status="rejected")["order"])
    res = svc.apply_broker_order(bo, "2024-05-02")
    assert res.rejected and ctx.db.fetchone("SELECT status FROM orders WHERE order_id=?", (ok["order_id"],))["status"] == "rejected"
    assert ledger.positions() == [] and ledger.cash() == pytest.approx(100_000.0)
    # a late/replayed "accepted" never moves a terminal status backwards
    late = broker.parse_order({**broker.orders[ok["client_order_id"]], "status": "accepted"})
    svc.apply_broker_order(late, "2024-05-02")
    assert ctx.db.fetchone("SELECT status FROM orders WHERE order_id=?", (ok["order_id"],))["status"] == "rejected"


# -- kill switch triggers ---------------------------------------------------------------------------
def test_unknown_broker_order_and_ledger_mismatch_pause_the_system(setup):
    ctx, bundle, sessions = setup
    broker = FakeAlpacaBroker(sessions)
    r, _ = make_runner(ctx, broker, Clock(at(sessions[-30], 12, 0)))
    r.start()
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE"
    foreign = {"event": "new", "timestamp": "t", "order": {"id": "x1", "client_order_id": "manual-web-order",
                                                            "symbol": "TSLA", "side": "buy", "qty": "1", "status": "new",
                                                            "filled_qty": "0"}}
    r.streams[-1].push(foreign)
    r.tick()
    st, reason, _ = KillSwitch(ctx.db).state()
    assert st.value == "SYSTEM_PAUSED" and "not in the QuantLab ledger" in reason
    KillSwitch(ctx.db).resume("reviewed in test", "human:test")
    broker.cash -= 500.0                                              # cash moved outside QuantLab
    assert r.reconcile("test") is False
    assert "ledger/broker mismatch" in KillSwitch(ctx.db).state()[1]
    r.shutdown("test")


def test_stale_data_blocks_the_session_and_pauses_after_the_deadline(setup):
    ctx, bundle, sessions = setup
    d = sessions[-30]
    stale = bundle.truncate(pd.Timestamp(sessions[-31]))
    clock = Clock(at(d, 19, 10))
    r, _ = make_runner(ctx, FakeAlpacaBroker(sessions), clock, bundle=stale)
    r.start()
    r.tick()
    job = ctx.db.fetchone("SELECT * FROM paper_session_jobs")
    assert job["status"] == "stale_data" and "no bars" in job["reason"]
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM runs WHERE kind='pipeline'")["n"] == 0
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE"            # still inside the window: retry later
    clock.t = at(sessions[-29], 9, 40)                                # deadline passed, still no data
    r.tick()
    st, reason, _ = KillSwitch(ctx.db).state()
    assert st.value == "SYSTEM_PAUSED" and "data stale" in reason
    r.shutdown("test")


@pytest.mark.slow
def test_paused_system_ev_rejection_and_missed_window_place_no_orders(setup):
    ctx, bundle, sessions = setup
    _activate(ctx)
    # 1) SYSTEM_PAUSED: decisions recorded, nothing submitted
    broker = FakeAlpacaBroker(sessions)
    KillSwitch(ctx.db).pause("test pause", trigger="manual", actor="human:test")
    clock = Clock(at(sessions[-60], 19, 10))
    r, _ = make_runner(ctx, broker, clock)
    r.start()
    for i in range(-60, -40):
        clock.t = at(sessions[i], 19, 10)
        r.tick()
    assert broker.submits == []
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM decisions")["n"] > 0
    stages = {x["reject_stage"] for x in ctx.db.fetchall("SELECT reject_stage FROM decisions")}
    assert "NONE" not in stages                                       # nothing traded while paused
    r.shutdown("test")
    KillSwitch(ctx.db).resume("test done", "human:test")

    # 2) EV gate: an impossible EV threshold rejects every candidate at the EV stage
    ev_ctx = AppContext.create(ctx.config.with_overrides({"expected_value": {"min_ev_bps_after_costs": 1e9}}),
                               init_logging=False)
    clock = Clock(at(sessions[-39], 19, 10))
    r, _ = make_runner(ev_ctx, broker, clock)
    r.start()
    before = ev_ctx.db.fetchone("SELECT COUNT(*) AS n FROM decisions")["n"]
    for i in range(-39, -30):
        clock.t = at(sessions[i], 19, 10)
        r.tick()
    new = ev_ctx.db.fetchall("SELECT d.reject_stage FROM decisions d ORDER BY d.created_at")[before:]
    assert new and broker.submits == []
    assert "NONE" not in {x["reject_stage"] for x in new}
    assert "EV" in {x["reject_stage"] for x in new}
    r.shutdown("test")

    # 3) missed window: processing D after D+1's 09:25 cutoff records decisions but orders are refused
    clock = Clock(at(sessions[-29], 11, 0))                           # D = sessions[-30]
    r, _ = make_runner(ctx, broker, clock)
    r.start()
    r.tick()
    job = ctx.db.fetchone("SELECT * FROM paper_session_jobs WHERE as_of_date=?", (str(sessions[-30]),))
    assert job["status"] == "succeeded" and job["orders_allowed"] == 0
    assert broker.submits == []
    refusals = ctx.db.fetchall("SELECT reason FROM execution_refusals")
    assert all("window" in x["reason"] or "SYSTEM_PAUSED" in x["reason"] for x in refusals)
    r.shutdown("test")


# -- lifecycle ------------------------------------------------------------------------------------------
def test_single_instance_clean_stop_and_restart(setup):
    ctx, bundle, sessions = setup
    clock = Clock(at(sessions[-30], 12, 0))
    a, _ = make_runner(ctx, FakeAlpacaBroker(sessions), clock)
    a.start()
    b, _ = make_runner(ctx, FakeAlpacaBroker(sessions), clock)
    with pytest.raises(RunnerRefused, match="another paper runner is RUNNING"):
        b.run_forever(max_ticks=1)
    c = TestClient(create_app(ctx))
    assert c.get("/api/live").json()["system"]["runner"] == "RUNNING"
    assert "NO PAPER-ELIGIBLE STRATEGY" in c.get("/live").text
    assert request_stop(ctx.db, "test stop") == [a.session_id]
    assert a.tick() == "stop requested (quantlab paper stop)"
    a.shutdown("stop requested")
    assert runner_status(ctx.db)["session"]["status"] == "STOPPED"
    assert c.get("/api/live").json()["system"]["runner"] == "STOPPED"
    assert a.streams[-1].stopped
    b2, _ = make_runner(ctx, FakeAlpacaBroker(sessions), clock)
    b2.start()                                                         # restart after a clean stop
    assert runner_status(ctx.db)["state"] == "RUNNING"
    # a runner that died without stopping is detected by its stale heartbeat
    ctx.db.execute("UPDATE paper_runner_sessions SET last_heartbeat_at=? WHERE session_id=?",
                   ((datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), b2.session_id))
    assert runner_status(ctx.db)["state"] == "STALE"
    b3, _ = make_runner(ctx, FakeAlpacaBroker(sessions), clock)
    b3.start()
    assert ctx.db.fetchone("SELECT status FROM paper_runner_sessions WHERE session_id=?", (b2.session_id,))["status"] == "CRASHED"
    b3.shutdown("test")
