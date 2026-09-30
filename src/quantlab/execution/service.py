"""The daily paper execution service: turns a decision (bot candidate or human decision) into a
broker order, and turns broker state back into ledger fills.

Refuses (no order created, but the refusal is recorded in ``execution_refusals``) when
``system_state.state == 'SYSTEM_PAUSED'`` (read directly, ARCHITECTURE.md section 10) or when
``risk.max_daily_orders`` / ``risk.max_order_notional`` would be breached, or when the optional
``submission_guard`` (set by the paper runner: execution window, broker verified, reconciliation
ok) returns a reason. BOT and HUMAN books never mix: one service instance is bound to exactly one
book, and every query it makes filters on it.

Idempotency (restarts can never duplicate an order):
  * client_order_id is deterministic (entry: the candidate/decision; exit: trade + decision session);
  * an order that already exists locally is returned, not resubmitted;
  * the local row is written as ``pending_submit`` BEFORE the broker call, and before submitting we
    ask the broker whether that client_order_id already exists (a crash between the broker call and
    the local update is adopted on restart, never resubmitted). Alpaca itself also rejects a reused
    client_order_id.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from quantlab.config import Config
from quantlab.core.calendar import to_session, to_utc
from quantlab.core.types import OrderStatus, Side, SystemState, TradePlan, new_id
from quantlab.data.panel import Panel
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, MAX_CLIENT_ORDER_ID_LEN, OrderRequest, PaperBroker
from quantlab.execution.ledger import Ledger
from quantlab.execution.sim_broker import paper_book
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.service")

_EPS = 1e-9
_OPEN_LOCAL = (OrderStatus.PENDING_SUBMIT.value, OrderStatus.ACCEPTED.value, OrderStatus.NEW.value,
               OrderStatus.PARTIALLY_FILLED.value, OrderStatus.UNKNOWN.value)
# Status can only move forward. A late/replayed "accepted" event must never overwrite "filled".
_STATUS_RANK = {OrderStatus.PENDING_SUBMIT: 0, OrderStatus.UNKNOWN: 0, OrderStatus.ACCEPTED: 1,
                OrderStatus.NEW: 2, OrderStatus.PARTIALLY_FILLED: 3, OrderStatus.FILLED: 4,
                OrderStatus.CANCELED: 4, OrderStatus.EXPIRED: 4, OrderStatus.REJECTED: 4}


def _fill_session(filled_at: str | None) -> str | None:
    """US/Eastern session date of a broker fill timestamp (Alpaca reports UTC)."""
    if not filled_at:
        return None
    try:
        return to_utc(filled_at).tz_convert("America/New_York").date().isoformat()
    except (ValueError, TypeError):
        return None


class ExecutionError(RuntimeError):
    pass


@dataclass
class SyncResult:
    session_date: str
    book: str
    filled: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    expired: list[dict[str, Any]] = field(default_factory=list)
    canceled: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[dict[str, Any]] = field(default_factory=list)


def _plan_get(plan: Any, key: str) -> Any:
    if plan is None:
        return None
    if isinstance(plan, TradePlan):
        return getattr(plan, key, None)
    if isinstance(plan, dict):
        return plan.get(key)
    return getattr(plan, key, None)


def _client_order_id(book: str, purpose: str, basis: str) -> str:
    """Deterministic id: the SAME (book, purpose, basis) always yields the SAME id, so resubmitting
    the same decision is idempotent at the broker (submit_order returns the existing order)."""
    digest = hashlib.sha1(f"{purpose}|{basis}".encode("utf-8")).hexdigest()[:20]
    cid = f"ql-{book.lower()}-{digest}"
    return cid[:MAX_CLIENT_ORDER_ID_LEN]


class PaperExecutionService:
    """Orchestrates one book's broker + ledger. One instance per (db, book, broker, ledger)."""

    def __init__(self, db: Database, config: Config | None, book: str, broker: PaperBroker, ledger: Ledger):
        self.db = db
        self.config = config
        self.book = paper_book(book)
        if ledger.book != self.book:
            raise ValueError(f"ledger is for book {ledger.book!r}, service is for {self.book!r}")
        self.broker = broker
        self.ledger = ledger
        self.journal = ledger.journal

        def cfg(key: str, default: Any) -> Any:
            return config.get(key, default) if config is not None else default

        self.max_daily_orders = int(cfg("risk.max_daily_orders", 20))
        self.max_order_notional = float(cfg("risk.max_order_notional", 15000.0))
        # (purpose, symbol) -> refusal reason or None. Set by the paper runner; None = no extra gate.
        self.submission_guard: Callable[[str, str], str | None] | None = None

    # ------------------------------------------------------------------------------------------
    # gates
    # ------------------------------------------------------------------------------------------
    def _system_paused(self) -> bool:
        row = self.db.fetchone("SELECT state FROM system_state WHERE id=1")
        return bool(row) and row["state"] == SystemState.PAUSED.value

    def _refuse(self, purpose: str, symbol: str, qty: float, reason: str, *, candidate_id: str | None,
               decision_id: str | None, human_decision_id: str | None, trade_id: str | None,
               session_date: str | None) -> dict[str, Any]:
        self.db.insert("execution_refusals", {
            "book": self.book, "purpose": purpose, "symbol": symbol, "qty": qty, "reason": reason,
            "candidate_id": candidate_id, "decision_id": decision_id, "human_decision_id": human_decision_id,
            "trade_id": trade_id, "session_date": session_date, "details_json": to_json({}),
            "created_at": utcnow_iso(),
        })
        log_event(log, "execution refused", book=self.book, purpose=purpose, symbol=symbol, reason=reason)
        return {"refused": True, "reason": reason, "symbol": symbol, "qty": qty, "purpose": purpose}

    def _check_gates(self, purpose: str, symbol: str, qty: float, *, candidate_id: str | None,
                     decision_id: str | None, human_decision_id: str | None, trade_id: str | None,
                     session_date: str | None, ref_price: float | None) -> dict[str, Any] | None:
        if self._system_paused():
            return self._refuse(purpose, symbol, qty, "SYSTEM_PAUSED", candidate_id=candidate_id,
                                decision_id=decision_id, human_decision_id=human_decision_id, trade_id=trade_id,
                                session_date=session_date)
        if session_date is not None:
            n = self.db.fetchone("SELECT COUNT(*) AS n FROM order_intents WHERE book=? AND session_date=?",
                                 (self.book, session_date))["n"]
            if n >= self.max_daily_orders:
                return self._refuse(purpose, symbol, qty,
                                    f"risk.max_daily_orders reached ({n} >= {self.max_daily_orders}) for "
                                    f"session {session_date}", candidate_id=candidate_id, decision_id=decision_id,
                                    human_decision_id=human_decision_id, trade_id=trade_id, session_date=session_date)
        if ref_price is not None:
            notional = float(qty) * float(ref_price)
            if notional > self.max_order_notional + _EPS:
                return self._refuse(purpose, symbol, qty,
                                    f"order notional {notional:.2f} > risk.max_order_notional "
                                    f"{self.max_order_notional:.2f}", candidate_id=candidate_id,
                                    decision_id=decision_id, human_decision_id=human_decision_id,
                                    trade_id=trade_id, session_date=session_date)
        return None

    def _decision_session(self, candidate_id: str | None, human_decision_id: str | None, plan: Any,
                          session_date: str | None) -> str | None:
        if session_date is not None:
            return to_session(session_date).date().isoformat()
        if candidate_id:
            row = self.db.fetchone("SELECT as_of_date FROM candidates WHERE candidate_id=?", (candidate_id,))
            if row and row["as_of_date"]:
                return row["as_of_date"]
        if human_decision_id:
            row = self.db.fetchone("SELECT ref_session_date FROM human_decisions WHERE decision_id=?",
                                   (human_decision_id,))
            if row and row["ref_session_date"]:
                return row["ref_session_date"]
        plan_signal = _plan_get(plan, "signal_date")
        if plan_signal:
            return to_session(plan_signal).date().isoformat()
        return getattr(self.broker, "last_session", None)

    # ------------------------------------------------------------------------------------------
    # entry / exit
    # ------------------------------------------------------------------------------------------
    def submit_entry(
        self,
        symbol: str,
        qty: float,
        *,
        candidate_id: str | None = None,
        decision_id: str | None = None,
        human_decision_id: str | None = None,
        plan: TradePlan | dict[str, Any] | None = None,
        journal: dict[str, Any] | None = None,
        strategy_id: str | None = None,
        strategy_version: str | None = None,
        direction: str = "LONG",
        session_date: str | None = None,
        entry_style: str = "opg",
    ) -> dict[str, Any]:
        """Submit a LONG entry order. ``entry_style``:

        * ``"opg"`` (default): market-on-open for the next session's opening auction, matching
          ARCHITECTURE.md section 6 (signal at close D -> fill at open D+1);
        * ``"market_day"``: a regular-session market order, used ONLY as the recorded fallback when
          an ``opg`` order expired unfilled in the opening cross (Alpaca expires an opg order that
          does not execute in the auction). It gets its own deterministic client_order_id, so the
          fallback is idempotent and can never be submitted twice for the same decision.

        Returns ``{"refused": True, "reason": ...}`` or the created order's summary."""
        if entry_style not in ("opg", "market_day"):
            raise ExecutionError(f"unknown entry_style {entry_style!r}")
        symbol = str(symbol).strip().upper()
        qty = float(qty)
        signal_date = self._decision_session(candidate_id, human_decision_id, plan, session_date)
        ref_price = _plan_get(plan, "entry_ref_price")
        if ref_price is None and candidate_id:
            row = self.db.fetchone("SELECT entry_ref_price FROM candidates WHERE candidate_id=?", (candidate_id,))
            ref_price = row["entry_ref_price"] if row else None

        basis = candidate_id or decision_id or human_decision_id or (f"{symbol}:{signal_date}" if signal_date else None)
        if basis and entry_style != "opg":
            basis = f"{basis}#{entry_style}"
        client_order_id = _client_order_id(self.book, "entry", basis) if basis else None
        guard_purpose = "entry" if entry_style == "opg" else "entry_fallback"
        existing = self._local_order(client_order_id) if client_order_id else None
        if existing is not None and existing["status"] != OrderStatus.PENDING_SUBMIT.value:
            return self._summary(existing, duplicate=True)   # already submitted (e.g. resumed run)

        refusal = self._check_gates("entry", symbol, qty, candidate_id=candidate_id, decision_id=decision_id,
                                    human_decision_id=human_decision_id, trade_id=None, session_date=signal_date,
                                    ref_price=ref_price)
        if refusal is None and signal_date is None:
            raise ExecutionError(
                "submit_entry: cannot determine the decision session (pass session_date, candidate_id, "
                "human_decision_id, or a plan/dict with signal_date)")
        if refusal is None:
            refusal = self._duplicate_entry_refusal(symbol, qty, signal_date, client_order_id, candidate_id,
                                                    decision_id, human_decision_id)
        if refusal is None:
            refusal = self._guard_refusal(guard_purpose, symbol, qty, candidate_id=candidate_id,
                                          decision_id=decision_id, human_decision_id=human_decision_id,
                                          trade_id=None, session_date=signal_date)
        if refusal is not None:
            if existing is not None:
                self._abandon_pending(existing, refusal["reason"])
            return refusal

        request = OrderRequest(client_order_id=client_order_id, symbol=symbol, side=Side.BUY, qty=qty,
                               order_type="market", time_in_force="opg" if entry_style == "opg" else "day",
                               decision_session=signal_date)
        if existing is not None:
            return self._send(existing["order_id"], request, purpose="entry")
        order_id = new_id("order")
        now = utcnow_iso()
        self.db.insert("order_intents", {
            "order_id": order_id, "book": self.book, "session_date": signal_date, "purpose": "entry",
            "intent_json": to_json({"candidate_id": candidate_id, "decision_id": decision_id,
                                    "human_decision_id": human_decision_id, "symbol": symbol, "qty": qty,
                                    "direction": direction, "strategy_id": strategy_id,
                                    "strategy_version": strategy_version, "plan": plan, "journal": journal}),
            "created_at": now,
        })
        self._insert_pending(order_id, request, purpose="entry", candidate_id=candidate_id, decision_id=decision_id,
                             human_decision_id=human_decision_id, trade_id=None, now=now)
        # strategy_id/version are stashed on the trade only once the trade is opened at fill time
        # (see Ledger.apply_fill); keep them retrievable via the order_intents snapshot above.
        return self._send(order_id, request, purpose="entry")

    def submit_exit(self, trade_id: str, reason: str, *, detail: dict[str, Any] | None = None,
                   session_date: str | None = None) -> dict[str, Any]:
        """Submit the exit order for an OPEN trade (full position, opg / next-open)."""
        trade = self.journal.get_trade(trade_id)
        if trade is None:
            raise ExecutionError(f"unknown trade {trade_id}")
        if trade["book"] != self.book:
            raise ExecutionError(f"trade {trade_id} belongs to book {trade['book']!r}, not {self.book!r}")
        if trade["status"] != "OPEN":
            raise ExecutionError(f"trade {trade_id} is not OPEN (status={trade['status']})")

        decision_session = to_session(session_date).date().isoformat() if session_date is not None else (
            getattr(self.broker, "last_session", None) or trade["entry_date"])
        # one exit order per (trade, decision session): a retry the next session gets a new id
        client_order_id = _client_order_id(self.book, "exit", f"{trade_id}|{decision_session}")
        existing = self._local_order(client_order_id)
        if existing is not None and existing["status"] != OrderStatus.PENDING_SUBMIT.value:
            return self._summary(existing, duplicate=True)
        side = Side.SELL if trade["direction"] == "LONG" else Side.BUY
        # after a partial exit only the remaining position is sold
        pos = self.ledger.get_position(trade["symbol"])
        qty = float(trade["qty"]) if pos is None else min(float(trade["qty"]), float(pos["qty"]))

        refusal = self._check_gates("exit", trade["symbol"], qty, candidate_id=trade["candidate_id"],
                                    decision_id=trade["decision_id"], human_decision_id=trade["human_decision_id"],
                                    trade_id=trade_id, session_date=decision_session, ref_price=None)
        if refusal is None:
            pending = self.db.fetchone(
                f"SELECT order_id FROM orders WHERE book=? AND trade_id=? AND purpose='exit' AND client_order_id<>? "
                f"AND status IN ({','.join('?' * len(_OPEN_LOCAL))})",
                (self.book, trade_id, client_order_id, *_OPEN_LOCAL))
            if pending is not None:
                refusal = self._refuse("exit", trade["symbol"], qty,
                                       f"an exit order for trade {trade_id} is still open ({pending['order_id']})",
                                       candidate_id=trade["candidate_id"], decision_id=trade["decision_id"],
                                       human_decision_id=trade["human_decision_id"], trade_id=trade_id,
                                       session_date=decision_session)
        if refusal is None:
            refusal = self._guard_refusal("exit", trade["symbol"], qty, candidate_id=trade["candidate_id"],
                                          decision_id=trade["decision_id"],
                                          human_decision_id=trade["human_decision_id"], trade_id=trade_id,
                                          session_date=decision_session)
        if refusal is not None:
            if existing is not None:
                self._abandon_pending(existing, refusal["reason"])
            return refusal

        request = OrderRequest(client_order_id=client_order_id, symbol=trade["symbol"], side=side, qty=qty,
                               order_type="market", time_in_force="opg", decision_session=decision_session)
        if existing is not None:
            return self._send(existing["order_id"], request, purpose="exit", trade_id=trade_id)
        order_id = new_id("order")
        now = utcnow_iso()
        self.db.insert("order_intents", {
            "order_id": order_id, "book": self.book, "session_date": decision_session, "purpose": "exit",
            "intent_json": to_json({"trade_id": trade_id, "reason": reason, **(detail or {})}),
            "created_at": now,
        })
        self._insert_pending(order_id, request, purpose="exit", candidate_id=trade["candidate_id"],
                             decision_id=trade["decision_id"], human_decision_id=trade["human_decision_id"],
                             trade_id=trade_id, now=now)
        self.journal.record_event(trade_id, "exit_submitted", {"order_id": order_id, "reason": reason})
        return self._send(order_id, request, purpose="exit", trade_id=trade_id)

    # ------------------------------------------------------------------------------------------
    # submission internals (idempotent)
    # ------------------------------------------------------------------------------------------
    def _local_order(self, client_order_id: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM orders WHERE client_order_id=? AND book=?",
                                (client_order_id, self.book))

    def _summary(self, row: dict[str, Any], *, duplicate: bool = False) -> dict[str, Any]:
        intent = self.db.fetchone("SELECT session_date FROM order_intents WHERE order_id=?", (row["order_id"],))
        out = {"refused": False, "order_id": row["order_id"], "client_order_id": row["client_order_id"],
               "symbol": row["symbol"], "qty": row["qty"], "status": row["status"],
               "broker_order_id": row["broker_order_id"], "session_date": intent["session_date"] if intent else None}
        if row["trade_id"] and row["purpose"] == "exit":
            out["trade_id"] = row["trade_id"]
        if duplicate:
            out["duplicate"] = True
        return out

    def _duplicate_entry_refusal(self, symbol: str, qty: float, signal_date: str, client_order_id: str | None,
                                 candidate_id, decision_id, human_decision_id) -> dict[str, Any] | None:
        """A second (different) entry order for the same symbol decided on the same session is a bug
        or a second process: refuse it."""
        row = self.db.fetchone(
            "SELECT o.order_id FROM orders o JOIN order_intents i ON i.order_id = o.order_id "
            "WHERE o.book=? AND o.symbol=? AND o.purpose='entry' AND i.session_date=? AND o.client_order_id<>? "
            "AND o.status NOT IN (?,?,?)",
            (self.book, symbol, signal_date, client_order_id or "", OrderStatus.REJECTED.value,
             OrderStatus.CANCELED.value, OrderStatus.EXPIRED.value))
        if row is None:
            return None
        return self._refuse("entry", symbol, qty, f"duplicate entry: order {row['order_id']} for {symbol} "
                            f"decided at {signal_date} already exists", candidate_id=candidate_id,
                            decision_id=decision_id, human_decision_id=human_decision_id, trade_id=None,
                            session_date=signal_date)

    def _guard_refusal(self, purpose: str, symbol: str, qty: float, **ids: Any) -> dict[str, Any] | None:
        if self.submission_guard is None:
            return None
        reason = self.submission_guard(purpose, symbol)
        return None if reason is None else self._refuse(purpose, symbol, qty, reason, **ids)

    def _insert_pending(self, order_id: str, request: OrderRequest, *, purpose: str, candidate_id, decision_id,
                        human_decision_id, trade_id, now: str) -> None:
        self.db.insert("orders", {
            "order_id": order_id, "client_order_id": request.client_order_id, "book": self.book,
            "broker": self.broker.name, "candidate_id": candidate_id, "decision_id": decision_id,
            "human_decision_id": human_decision_id, "trade_id": trade_id, "purpose": purpose,
            "symbol": request.symbol, "side": request.side.value, "qty": request.qty,
            "order_type": request.order_type, "time_in_force": request.time_in_force,
            "limit_price": request.limit_price, "created_at": now, "submitted_at": None,
            "status": OrderStatus.PENDING_SUBMIT.value, "broker_order_id": None, "filled_qty": 0.0,
            "filled_avg_price": None, "last_update_at": now, "raw_json": to_json({}),
        })

    def _abandon_pending(self, row: dict[str, Any], reason: str) -> None:
        """A pending_submit row whose (re)submission is now refused: it never reached the broker
        (checked by client_order_id), so it is closed as canceled with the refusal reason."""
        try:
            at_broker = self.broker.get_order_by_client_id(row["client_order_id"])
        except BrokerError:
            at_broker = None
            self.ledger.record_order_status(row["order_id"], OrderStatus.UNKNOWN,
                                            f"pending_submit and broker unreachable: {reason}")
            return
        if at_broker is not None:
            self._adopt(row["order_id"], at_broker, "adopted")
            return
        self.ledger.record_order_status(row["order_id"], OrderStatus.CANCELED, f"not submitted: {reason}")

    def _send(self, order_id: str, request: OrderRequest, *, purpose: str, trade_id: str | None = None) -> dict[str, Any]:
        """Submit a pending_submit order exactly once. A client_order_id the broker already knows is
        adopted (a crash happened after the broker accepted it), never sent twice."""
        try:
            prior = self.broker.get_order_by_client_id(request.client_order_id)
        except BrokerError as exc:
            self.ledger.record_order_status(order_id, OrderStatus.UNKNOWN,
                                            f"could not check the broker before submitting: {exc}")
            return self._summary(self._order_row(order_id))
        if prior is not None:
            self._adopt(order_id, prior, "adopted")
        else:
            self._adopt(order_id, self.broker.submit_order(request), "submitted")
        return self._summary(self._order_row(order_id))

    def _order_row(self, order_id: str) -> dict[str, Any]:
        return self.db.fetchone("SELECT * FROM orders WHERE order_id=? AND book=?", (order_id, self.book))

    def _adopt(self, order_id: str, bo, event: str) -> None:
        """Record the broker's acknowledgement on the local row. Fills are NOT copied here: sync /
        trade updates apply them to the ledger from the cumulative filled_qty (delta > 0 only)."""
        now = utcnow_iso()
        status = bo.status if bo.filled_qty <= _EPS else OrderStatus.ACCEPTED
        self.db.execute(
            "UPDATE orders SET status=?, broker_order_id=COALESCE(?, broker_order_id), submitted_at=COALESCE(submitted_at, ?), "
            "last_update_at=?, raw_json=? WHERE order_id=? AND book=?",
            (status.value, bo.broker_order_id, bo.submitted_at or now, now, to_json(bo.raw), order_id, self.book))
        self.db.insert("order_events", {"order_id": order_id, "event": event, "status": bo.status.value, "at": now,
                                        "raw_json": to_json({"reason": bo.reason,
                                                             "broker_order_id": bo.broker_order_id})})

    # ------------------------------------------------------------------------------------------
    # sync (broker -> ledger)
    # ------------------------------------------------------------------------------------------
    def sync(self, session_date: Any, panel: Panel | None = None) -> SyncResult:
        """Advance to ``session_date``: for a SimBroker this fills eligible pending orders at its
        open; for a real broker (no fixed session clock) this polls every locally-open order. Any
        newly observed fill/rejection/expiry/cancel/unknown state is applied to the ledger."""
        s_str = to_session(session_date).date().isoformat()
        result = SyncResult(session_date=s_str, book=self.book)

        changed = list(self.broker.process_session(session_date, panel)) if panel is not None else []
        seen = {bo.client_order_id for bo in changed}

        open_local = self.db.fetchall(
            f"SELECT order_id, client_order_id, status, filled_qty FROM orders WHERE book=? "
            f"AND status IN ({','.join('?' * len(_OPEN_LOCAL))})", (self.book, *_OPEN_LOCAL))
        for row in open_local:
            if row["client_order_id"] in seen:
                continue
            try:
                bo = self.broker.get_order_by_client_id(row["client_order_id"])
            except BrokerError as exc:
                log_event(log, "sync: broker unreachable for order; will retry next sync",
                          client_order_id=row["client_order_id"], error=str(exc))
                continue
            if bo is None:
                continue   # pending_submit never reached the broker: left for the (resumed) submitter
            if bo.status.value != row["status"] or bo.filled_qty > float(row["filled_qty"] or 0.0) + _EPS:
                changed.append(bo)

        for bo in changed:
            self._apply_broker_order(bo, s_str, result)
        return result

    def apply_broker_order(self, bo, session_date: Any) -> SyncResult:
        """Apply one broker view of an order (e.g. from a trade_updates event). Idempotent."""
        s_str = to_session(session_date).date().isoformat()
        result = SyncResult(session_date=s_str, book=self.book)
        self._apply_broker_order(bo, s_str, result)
        return result

    def _apply_broker_order(self, bo, s_str: str, result: SyncResult) -> None:
        row = self.db.fetchone("SELECT * FROM orders WHERE client_order_id=? AND book=?",
                               (bo.client_order_id, self.book))
        if row is None:
            log_event(log, "sync: broker order has no local record", client_order_id=bo.client_order_id,
                      book=self.book)
            return
        order_id = row["order_id"]
        prev_filled = float(row["filled_qty"] or 0.0)
        if bo.broker_order_id and not row["broker_order_id"]:
            self.db.execute("UPDATE orders SET broker_order_id=? WHERE order_id=?", (bo.broker_order_id, order_id))

        if bo.status is OrderStatus.UNKNOWN:
            self.ledger.record_order_status(order_id, OrderStatus.UNKNOWN, bo.reason)
            result.unknown.append({"order_id": order_id, "client_order_id": bo.client_order_id,
                                   "reason": bo.reason})
            return

        delta = bo.filled_qty - prev_filled
        if delta > _EPS:
            fill_price = bo.filled_avg_price if bo.filled_avg_price is not None else row["limit_price"]
            prev_avg = row["filled_avg_price"]
            if prev_filled > _EPS and prev_avg is not None and bo.filled_avg_price is not None:
                # the broker reports a cumulative average: price THIS increment only
                inc = (bo.filled_avg_price * bo.filled_qty - float(prev_avg) * prev_filled) / delta
                fill_price = inc if inc > 0 else bo.filled_avg_price
            intent = self.db.fetchone("SELECT session_date, intent_json FROM order_intents WHERE order_id=?",
                                      (order_id,))
            intent_data = from_json(intent["intent_json"], {}) if intent else {}
            signal_date = intent["session_date"] if intent else None
            exit_reason = intent_data.get("reason") if row["purpose"] == "exit" else None
            broker_fill_id = bo.raw.get("id") if isinstance(bo.raw, dict) else None
            fr = self.ledger.apply_fill(
                order_id=order_id, qty=delta, price=fill_price,
                session_date=bo.fill_session or _fill_session(bo.filled_at) or s_str,
                commission=bo.commission, modeled_cost=bo.modeled_cost, filled_at=bo.filled_at,
                broker_fill_id=broker_fill_id, source=self.broker.name,
                direction=intent_data.get("direction", "LONG"), signal_date=signal_date,
                exit_reason=exit_reason, held_sessions=intent_data.get("held_sessions"),
                mae=intent_data.get("mae"), mfe=intent_data.get("mfe"),
                candidate_id=row["candidate_id"], decision_id=row["decision_id"],
                human_decision_id=row["human_decision_id"], strategy_id=intent_data.get("strategy_id"),
                strategy_version=intent_data.get("strategy_version"), plan=intent_data.get("plan"),
                journal_extra=intent_data.get("journal"),
            )
            result.filled.append({"order_id": order_id, "trade_id": fr.trade_id, "symbol": bo.symbol or row["symbol"],
                                  "side": row["side"], "qty": delta, "price": fill_price})

        if bo.status is OrderStatus.REJECTED and row["status"] != OrderStatus.REJECTED.value:
            self.ledger.record_order_status(order_id, OrderStatus.REJECTED, bo.reason)
            result.rejected.append({"order_id": order_id, "reason": bo.reason})
        elif bo.status is OrderStatus.EXPIRED and row["status"] != OrderStatus.EXPIRED.value:
            self.ledger.record_order_status(order_id, OrderStatus.EXPIRED, bo.reason)
            result.expired.append({"order_id": order_id, "reason": bo.reason})
        elif bo.status is OrderStatus.CANCELED and row["status"] != OrderStatus.CANCELED.value:
            self.ledger.record_order_status(order_id, OrderStatus.CANCELED, bo.reason)
            result.canceled.append({"order_id": order_id, "reason": bo.reason})
        elif bo.status in (OrderStatus.ACCEPTED, OrderStatus.NEW) and delta <= _EPS:
            # acknowledgement progress (accepted -> new); never move a status backwards
            try:
                cur = OrderStatus(row["status"])
            except ValueError:
                cur = OrderStatus.UNKNOWN
            if _STATUS_RANK[bo.status] > _STATUS_RANK[cur]:
                self.ledger.record_order_status(order_id, bo.status, None)
