"""The daily paper execution service: turns a decision (bot candidate or human decision) into a
broker order, and turns broker state back into ledger fills.

Refuses (no order created, but the refusal is recorded in ``execution_refusals``) when
``system_state.state == 'SYSTEM_PAUSED'`` (read directly, ARCHITECTURE.md section 10) or when
``risk.max_daily_orders`` / ``risk.max_order_notional`` would be breached. BOT and HUMAN books never
mix: one service instance is bound to exactly one book, and every query it makes filters on it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.core.types import OrderStatus, Side, SystemState, TradePlan, new_id
from quantlab.data.panel import Panel
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, MAX_CLIENT_ORDER_ID_LEN, OrderRequest, PaperBroker
from quantlab.execution.ledger import Ledger
from quantlab.execution.sim_broker import paper_book
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.service")

_EPS = 1e-9


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
    ) -> dict[str, Any]:
        """Submit a LONG entry order (opg / next-open, matching ARCHITECTURE.md section 6: signal at
        close D -> fill at open D+1). Returns ``{"refused": True, "reason": ...}`` or the created
        order's summary."""
        symbol = str(symbol).strip().upper()
        qty = float(qty)
        signal_date = self._decision_session(candidate_id, human_decision_id, plan, session_date)
        ref_price = _plan_get(plan, "entry_ref_price")
        if ref_price is None and candidate_id:
            row = self.db.fetchone("SELECT entry_ref_price FROM candidates WHERE candidate_id=?", (candidate_id,))
            ref_price = row["entry_ref_price"] if row else None

        refusal = self._check_gates("entry", symbol, qty, candidate_id=candidate_id, decision_id=decision_id,
                                    human_decision_id=human_decision_id, trade_id=None, session_date=signal_date,
                                    ref_price=ref_price)
        if refusal is not None:
            return refusal
        if signal_date is None:
            raise ExecutionError(
                "submit_entry: cannot determine the decision session (pass session_date, candidate_id, "
                "human_decision_id, or a plan/dict with signal_date)")

        basis = candidate_id or decision_id or human_decision_id or f"{symbol}:{signal_date}"
        client_order_id = _client_order_id(self.book, "entry", basis)
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
        request = OrderRequest(client_order_id=client_order_id, symbol=symbol, side=Side.BUY, qty=qty,
                               order_type="market", time_in_force="opg", decision_session=signal_date)
        broker_order = self.broker.submit_order(request)
        self.db.insert("orders", {
            "order_id": order_id, "client_order_id": client_order_id, "book": self.book, "broker": self.broker.name,
            "candidate_id": candidate_id, "decision_id": decision_id, "human_decision_id": human_decision_id,
            "trade_id": None, "purpose": "entry", "symbol": symbol, "side": Side.BUY.value, "qty": qty,
            "order_type": "market", "time_in_force": "opg", "limit_price": None, "created_at": now,
            "submitted_at": now, "status": broker_order.status.value, "broker_order_id": broker_order.broker_order_id,
            "filled_qty": broker_order.filled_qty, "filled_avg_price": broker_order.filled_avg_price,
            "last_update_at": now, "raw_json": to_json(broker_order.raw),
        })
        self.db.insert("order_events", {"order_id": order_id, "event": "submitted",
                                        "status": broker_order.status.value, "at": now,
                                        "raw_json": to_json({"reason": broker_order.reason})})
        # strategy_id/version are stashed on the trade only once the trade is opened at fill time
        # (see Ledger.apply_fill); keep them retrievable via the order_intents snapshot above.
        return {"refused": False, "order_id": order_id, "client_order_id": client_order_id, "symbol": symbol,
                "qty": qty, "status": broker_order.status.value, "session_date": signal_date}

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
        refusal = self._check_gates("exit", trade["symbol"], trade["qty"], candidate_id=trade["candidate_id"],
                                    decision_id=trade["decision_id"], human_decision_id=trade["human_decision_id"],
                                    trade_id=trade_id, session_date=decision_session, ref_price=None)
        if refusal is not None:
            return refusal

        side = Side.SELL if trade["direction"] == "LONG" else Side.BUY
        qty = float(trade["qty"])
        client_order_id = _client_order_id(self.book, "exit", trade_id)
        order_id = new_id("order")
        now = utcnow_iso()
        self.db.insert("order_intents", {
            "order_id": order_id, "book": self.book, "session_date": decision_session, "purpose": "exit",
            "intent_json": to_json({"trade_id": trade_id, "reason": reason, **(detail or {})}),
            "created_at": now,
        })
        request = OrderRequest(client_order_id=client_order_id, symbol=trade["symbol"], side=side, qty=qty,
                               order_type="market", time_in_force="opg", decision_session=decision_session)
        broker_order = self.broker.submit_order(request)
        self.db.insert("orders", {
            "order_id": order_id, "client_order_id": client_order_id, "book": self.book, "broker": self.broker.name,
            "candidate_id": trade["candidate_id"], "decision_id": trade["decision_id"],
            "human_decision_id": trade["human_decision_id"], "trade_id": trade_id, "purpose": "exit",
            "symbol": trade["symbol"], "side": side.value, "qty": qty, "order_type": "market",
            "time_in_force": "opg", "limit_price": None, "created_at": now, "submitted_at": now,
            "status": broker_order.status.value, "broker_order_id": broker_order.broker_order_id,
            "filled_qty": broker_order.filled_qty, "filled_avg_price": broker_order.filled_avg_price,
            "last_update_at": now, "raw_json": to_json(broker_order.raw),
        })
        self.db.insert("order_events", {"order_id": order_id, "event": "submitted",
                                        "status": broker_order.status.value, "at": now,
                                        "raw_json": to_json({"reason": broker_order.reason})})
        self.journal.record_event(trade_id, "exit_submitted", {"order_id": order_id, "reason": reason})
        return {"refused": False, "order_id": order_id, "client_order_id": client_order_id, "trade_id": trade_id,
                "symbol": trade["symbol"], "qty": qty, "status": broker_order.status.value,
                "session_date": decision_session}

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
            "SELECT order_id, client_order_id, status, filled_qty FROM orders WHERE book=? AND status IN (?,?,?)",
            (self.book, OrderStatus.ACCEPTED.value, OrderStatus.NEW.value, OrderStatus.PARTIALLY_FILLED.value))
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
                continue
            if bo.status.value != row["status"] or bo.filled_qty > float(row["filled_qty"] or 0.0) + _EPS:
                changed.append(bo)

        for bo in changed:
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

        if bo.status is OrderStatus.UNKNOWN:
            self.ledger.record_order_status(order_id, OrderStatus.UNKNOWN, bo.reason)
            result.unknown.append({"order_id": order_id, "client_order_id": bo.client_order_id,
                                   "reason": bo.reason})
            return

        delta = bo.filled_qty - prev_filled
        if delta > _EPS:
            fill_price = bo.filled_avg_price if bo.filled_avg_price is not None else row["limit_price"]
            intent = self.db.fetchone("SELECT session_date, intent_json FROM order_intents WHERE order_id=?",
                                      (order_id,))
            intent_data = from_json(intent["intent_json"], {}) if intent else {}
            signal_date = intent["session_date"] if intent else None
            exit_reason = intent_data.get("reason") if row["purpose"] == "exit" else None
            broker_fill_id = bo.raw.get("id") if isinstance(bo.raw, dict) else None
            fr = self.ledger.apply_fill(
                order_id=order_id, qty=delta, price=fill_price, session_date=bo.fill_session or s_str,
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
