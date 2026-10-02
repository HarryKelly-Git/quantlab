"""PAPER execution for the OPT options book. GATED OFF by default (``options.paper_trading: false``).

Rules (each one a refusal recorded in ``options_refusals`` when violated):
  * PAPER only: the broker is a ``PaperBroker`` (AlpacaPaperBroker refuses any non-paper host);
  * defined-risk structures only (``OptionStructure`` cannot hold a naked short; re-checked here);
  * LIMIT + DAY orders only (also enforced by CHECK constraints in migration 055); buys at the ask
    or ``entry_mid_fraction`` of the way to the mid, sells at the bid or that fraction toward the
    mid -- NEVER better than the midpoint;
  * legging that can never be naked: a spread's LONG leg is bought first; the short leg is sold
    only after the long leg has filled, and only for the filled quantity. Closing buys the short
    leg back first, then sells the long leg. If the short leg never fills, the position is a plain
    long option (still bounded by its premium) -- the per-trade premium cap is therefore checked
    against the long leg alone;
  * caps: open structures, contracts per trade, premium at risk per trade and in total, the OPT
    allocation; the kill switch blocks every OPT order (entries and closes);
  * never held into expiry: entries are refused on/after ``close_by_date`` and ``due_for_close``
    lists positions to close ``close_before_expiry_sessions`` sessions before expiry (exercise or
    assignment would create a stock position in the shared account, which the stock
    reconciliation correctly treats as a mismatch).

Fills come only from the broker's cumulative ``filled_qty``/``filled_avg_price``; no fill is ever
simulated or assumed.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Callable

import pandas as pd

from quantlab.core.types import OrderStatus, Side, SystemState, new_id
from quantlab.data.audit import expected_sessions
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, OrderRequest
from quantlab.logging_setup import get_logger, log_event
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.options.book import (
    OPT_BOOK, TERMINAL, OptionsLedger, open_opt_orders, opt_client_order_id,
)
from quantlab.options.data import OptionQuote
from quantlab.options.settings import OptionsSettings
from quantlab.options.structures import NakedShortError, OptionStructure, assert_defined_risk

log = get_logger("options.execution")

ACTIVE_STATUSES = ("OPENING", "OPEN", "CLOSING")


def buy_limit(bid: float, ask: float, fraction: float) -> float:
    """Ask, or ``fraction`` of the way from the ask to the mid, rounded UP to the cent: >= mid."""
    if not (bid > 0 and ask >= bid) or not 0.0 <= fraction <= 1.0:
        raise ValueError("buy_limit needs a two-sided quote and 0 <= fraction <= 1")
    mid = 0.5 * (bid + ask)
    return max(math.ceil((ask - fraction * (ask - mid)) * 100 - 1e-9) / 100.0, 0.01)


def sell_limit(bid: float, ask: float, fraction: float) -> float:
    """Bid, or ``fraction`` of the way from the bid to the mid, rounded DOWN to the cent: <= mid."""
    if not (bid > 0 and ask >= bid) or not 0.0 <= fraction <= 1.0:
        raise ValueError("sell_limit needs a two-sided quote and 0 <= fraction <= 1")
    mid = 0.5 * (bid + ask)
    return math.floor((bid + fraction * (mid - bid)) * 100 + 1e-9) / 100.0


def close_by_date(expiration: date, sessions_before: int) -> date:
    """The last session on which an OPT position may still be held: ``sessions_before`` NYSE
    sessions before expiry."""
    before = expected_sessions(pd.Timestamp(expiration) - pd.Timedelta(days=40), pd.Timestamp(expiration) - pd.Timedelta(days=1))
    return before[-sessions_before].date()


class OptionsPaperExecutor:
    def __init__(self, config: Any, db: Database, broker: Any, *, settings: OptionsSettings | None = None,
                 killswitch: KillSwitch | None = None, clock: Callable[[], pd.Timestamp] | None = None):
        self.s = settings or OptionsSettings.from_config(config)
        self.db = db
        self.broker = broker
        self.killswitch = killswitch or KillSwitch(db)
        self.ledger = OptionsLedger(db)
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))

    # -- gates ----------------------------------------------------------------------------------
    def _refuse(self, reason: str, *, evaluation_id: str | None = None, structure_id: str | None = None,
                details: dict[str, Any] | None = None) -> dict[str, Any]:
        self.db.insert("options_refusals", {"evaluation_id": evaluation_id, "structure_id": structure_id,
                                            "reason": reason, "details_json": to_json(details or {}),
                                            "created_at": utcnow_iso()})
        log_event(log, "OPT order refused", reason=reason, structure_id=structure_id)
        return {"ok": False, "refused": reason}

    def _state_gate(self) -> str | None:
        st, reason, _ = self.killswitch.state()
        if st is not SystemState.ACTIVE:
            return f"SYSTEM_PAUSED: {reason}"
        bound = self.db.fetchone("SELECT broker FROM paper_book_bindings WHERE book=?", (OPT_BOOK,))
        name = getattr(self.broker, "name", "unknown")
        if bound is None:
            self.db.insert("paper_book_bindings", {"book": OPT_BOOK, "broker": name, "bound_at": utcnow_iso(),
                                                   "details_json": to_json({"note": "options paper book"})})
        elif bound["broker"] != name:
            return f"OPT book is bound to broker {bound['broker']!r}; refusing {name!r}"
        return None

    def premium_at_risk(self) -> float:
        row = self.db.fetchone("SELECT COALESCE(SUM(max_loss_usd),0) AS v FROM options_structures WHERE status IN "
                               f"({','.join('?' for _ in ACTIVE_STATUSES)})", ACTIVE_STATUSES)
        return float(row["v"])

    def open_count(self) -> int:
        return int(self.db.fetchone("SELECT COUNT(*) AS n FROM options_structures WHERE status IN "
                                    f"({','.join('?' for _ in ACTIVE_STATUSES)})", ACTIVE_STATUSES)["n"])

    # -- entry ----------------------------------------------------------------------------------
    def open_structure(self, structure: OptionStructure, qty: int, *, session_date: date,
                       evaluation_id: str | None = None, horizon_date: date | None = None) -> dict[str, Any]:
        sid_hint = getattr(structure, "structure_id", None)
        if not self.s.paper_trading:
            return self._refuse("options.paper_trading is false: OPT paper execution is gated OFF",
                                evaluation_id=evaluation_id, structure_id=sid_hint)
        if not isinstance(structure, OptionStructure):
            return self._refuse("not an OptionStructure", evaluation_id=evaluation_id)
        try:
            assert_defined_risk(structure.legs)
        except NakedShortError as exc:
            return self._refuse(f"naked short refused: {exc}", evaluation_id=evaluation_id, structure_id=sid_hint)
        gate = self._state_gate()
        if gate:
            return self._refuse(gate, evaluation_id=evaluation_id, structure_id=sid_hint)
        p = self.s.paper
        if not isinstance(qty, int) or qty < 1 or qty > p.max_contracts_per_trade:
            return self._refuse(f"qty {qty!r} outside 1..{p.max_contracts_per_trade} contracts",
                                evaluation_id=evaluation_id, structure_id=sid_hint)
        if self.open_count() >= p.max_open_positions:
            return self._refuse(f"max_open_positions {p.max_open_positions} reached", evaluation_id=evaluation_id,
                                structure_id=sid_hint)
        long = structure.long_leg
        if long.bid is None or long.ask is None or not (long.bid > 0 and long.ask >= long.bid):
            return self._refuse("long leg has no two-sided quote", evaluation_id=evaluation_id, structure_id=sid_hint)
        limit = buy_limit(long.bid, long.ask, p.entry_mid_fraction)
        mult = structure.multiplier
        worst = limit * qty * mult        # if the short leg never fills, the long leg alone is at risk
        if worst > p.max_premium_per_trade + 1e-9:
            return self._refuse(f"premium at risk {worst:.2f} > max_premium_per_trade {p.max_premium_per_trade:.2f}",
                                evaluation_id=evaluation_id, structure_id=sid_hint)
        if self.premium_at_risk() + worst > p.max_premium_total + 1e-9:
            return self._refuse(f"total premium at risk would be {self.premium_at_risk() + worst:.2f} > "
                                f"max_premium_total {p.max_premium_total:.2f}", evaluation_id=evaluation_id,
                                structure_id=sid_hint)
        self.ledger.ensure_allocation(p.allocation)
        if worst > self.ledger.cash() + 1e-9:
            return self._refuse(f"premium {worst:.2f} exceeds the OPT book's remaining cash {self.ledger.cash():.2f}",
                                evaluation_id=evaluation_id, structure_id=sid_hint)
        cbd = close_by_date(structure.expiration, p.close_before_expiry_sessions)
        if session_date >= cbd:
            return self._refuse(f"session {session_date} is on/after the close-by date {cbd} (never held into expiry)",
                                evaluation_id=evaluation_id, structure_id=sid_hint)
        sid = new_id("opt")
        now = utcnow_iso()
        self.db.insert("options_structures", {
            "structure_id": sid, "book": OPT_BOOK, "evaluation_id": evaluation_id, "kind": structure.kind.value,
            "underlying": structure.underlying, "expiration": structure.expiration.isoformat(), "multiplier": mult,
            "legs_json": to_json([lg.to_dict() for lg in structure.legs]), "qty": qty, "max_loss_usd": worst,
            "horizon_date": str(horizon_date) if horizon_date else None, "close_by_date": cbd.isoformat(),
            "status": "OPENING", "created_at": now, "updated_at": now})
        self._event(sid, "created", {"expression": structure.structure_id, "worst_case_premium": worst})
        order = self._submit_leg(sid, "long", "open", long.contract.symbol, Side.BUY, qty, limit, mult,
                                 _quote_dict(long.bid, long.ask, long.quote_time), session_date)
        return {"ok": True, "structure_id": sid, "order": order, "limit_price": limit, "max_loss_usd": worst}

    # -- orders ---------------------------------------------------------------------------------
    def _submit_leg(self, sid: str, role: str, purpose: str, symbol: str, side: Side, qty: int, limit: float,
                    mult: float, quote: dict[str, Any], session_date: date, attempt: int = 0) -> dict[str, Any]:
        cid = opt_client_order_id(sid, role, purpose, attempt)
        order_id = new_id("optord")
        now = utcnow_iso()
        row = {"order_id": order_id, "client_order_id": cid, "book": OPT_BOOK, "structure_id": sid, "leg_role": role,
               "purpose": purpose, "contract_symbol": symbol, "side": side.value, "qty": int(qty), "order_type": "limit",
               "time_in_force": "day", "limit_price": float(limit), "multiplier": float(mult),
               "quote_json": to_json(quote), "status": OrderStatus.PENDING_SUBMIT.value,
               "broker": getattr(self.broker, "name", "unknown"), "broker_order_id": None, "filled_qty": 0.0,
               "filled_avg_price": None, "created_at": now, "last_update_at": now, "raw_json": None}
        self.db.insert("options_orders", row)               # written BEFORE the broker call
        req = OrderRequest(client_order_id=cid, symbol=symbol, side=side, qty=int(qty), order_type="limit",
                           time_in_force="day", limit_price=float(limit), decision_session=str(session_date))
        try:
            bo = self.broker.submit_order(req)
        except BrokerError as exc:
            self.db.execute("UPDATE options_orders SET status=?, last_update_at=? WHERE order_id=?",
                            (OrderStatus.UNKNOWN.value, utcnow_iso(), order_id))
            self.db.insert("options_order_events", {"order_id": order_id, "event": "submit_error",
                                                    "details_json": to_json({"error": str(exc)}), "at": utcnow_iso()})
            return {**row, "status": OrderStatus.UNKNOWN.value, "error": str(exc)}
        self.ledger.apply_broker_order(row, bo, str(session_date))
        return self.db.fetchone("SELECT * FROM options_orders WHERE order_id=?", (order_id,))

    def _event(self, sid: str, event: str, details: dict[str, Any]) -> None:
        self.db.insert("options_structure_events", {"structure_id": sid, "event": event,
                                                    "details_json": to_json(details), "at": utcnow_iso()})

    def _set_status(self, sid: str, status: str, details: dict[str, Any]) -> None:
        self.db.execute("UPDATE options_structures SET status=?, updated_at=? WHERE structure_id=?",
                        (status, utcnow_iso(), sid))
        self._event(sid, status.lower(), details)

    # -- sync: fills + leg sequencing -----------------------------------------------------------
    def sync(self, session_date: date, quote_fn: Callable[[str], OptionQuote | None] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"fills": [], "transitions": [], "unknown": []}
        for o in open_opt_orders(self.db):
            bo = self.broker.get_order_by_client_id(o["client_order_id"])
            if bo is None:
                if o["status"] in (OrderStatus.PENDING_SUBMIT.value, OrderStatus.UNKNOWN.value):
                    self.db.execute("UPDATE options_orders SET status=?, last_update_at=? WHERE order_id=?",
                                    (OrderStatus.REJECTED.value, utcnow_iso(), o["order_id"]))
                    self.db.insert("options_order_events", {"order_id": o["order_id"], "event": "not_at_broker",
                                                            "details_json": to_json({"note": "never reached the broker"}),
                                                            "at": utcnow_iso()})
                else:
                    out["unknown"].append(o["client_order_id"])
                continue
            if bo.status is OrderStatus.UNKNOWN:
                out["unknown"].append(o["client_order_id"])
            out["fills"] += self.ledger.apply_broker_order(o, bo, str(session_date))
        for st in self.db.fetchall("SELECT * FROM options_structures WHERE status IN ('OPENING','CLOSING')"):
            t = self._advance(st, session_date, quote_fn)
            if t:
                out["transitions"].append(t)
        return out

    def _orders(self, sid: str) -> dict[tuple[str, str], dict[str, Any]]:
        rows = self.db.fetchall("SELECT * FROM options_orders WHERE structure_id=? ORDER BY created_at", (sid,))
        return {(r["leg_role"], r["purpose"]): r for r in rows}

    def _advance(self, st: dict[str, Any], session_date: date,
                 quote_fn: Callable[[str], OptionQuote | None] | None) -> dict[str, Any] | None:
        sid, legs = st["structure_id"], {lg["side"]: lg for lg in _legs(st)}
        orders = self._orders(sid)
        pos = self.ledger.structure_positions(sid)
        if st["status"] == "OPENING":
            lo = orders.get(("long", "open"))
            if lo is None or lo["status"] not in TERMINAL:
                return None
            filled = int(round(float(lo["filled_qty"] or 0)))
            if filled == 0:
                self._set_status(sid, "ABANDONED", {"reason": f"long leg {lo['status']} unfilled"})
                return {"structure_id": sid, "to": "ABANDONED"}
            short = legs.get("sell")
            if short is None:
                self._set_status(sid, "OPEN", {"long_filled": filled})
                return {"structure_id": sid, "to": "OPEN"}
            so = orders.get(("short", "open"))
            if so is None:
                gate = None if self.s.paper_trading else "options.paper_trading is false"
                gate = gate or self._state_gate()
                if gate:
                    self._set_status(sid, "OPEN", {"long_filled": filled, "short_leg": f"not sold: {gate}"})
                    return {"structure_id": sid, "to": "OPEN", "note": gate}
                q = _fresh(quote_fn, short["symbol"]) or (short.get("bid"), short.get("ask"), short.get("quote_time"))
                bid, ask, qt = q
                if not (bid and ask and bid > 0 and ask >= bid):
                    self._set_status(sid, "OPEN", {"long_filled": filled, "short_leg": "not sold: no two-sided quote"})
                    return {"structure_id": sid, "to": "OPEN"}
                limit = sell_limit(bid, ask, self.s.paper.entry_mid_fraction)
                if limit <= 0:
                    self._set_status(sid, "OPEN", {"long_filled": filled, "short_leg": "not sold: limit rounds to 0"})
                    return {"structure_id": sid, "to": "OPEN"}
                # covered by the FILLED long quantity only: never more short than long
                self._submit_leg(sid, "short", "open", short["symbol"], Side.SELL, filled, limit,
                                 float(st["multiplier"]), _quote_dict(bid, ask, qt), session_date)
                self._event(sid, "short_leg_submitted", {"qty": filled, "limit": limit})
                return {"structure_id": sid, "to": "OPENING", "note": "short leg submitted after long fill"}
            if so["status"] not in TERMINAL:
                return None
            self._set_status(sid, "OPEN", {"long_filled": filled, "short_filled": float(so["filled_qty"] or 0)})
            return {"structure_id": sid, "to": "OPEN"}
        if st["status"] == "CLOSING":
            if all(abs(q) < 1e-9 for q in pos.values()):
                self._set_status(sid, "CLOSED", {"positions": pos})
                return {"structure_id": sid, "to": "CLOSED"}
            open_close = [o for (role, purpose), o in orders.items() if purpose == "close" and o["status"] not in TERMINAL]
            if open_close:
                return None
            if not any(q < -1e-9 for q in pos.values()):      # short leg flat -> sell the long leg
                return self._submit_close_longs(st, pos, session_date, quote_fn)
        return None

    # -- exits ----------------------------------------------------------------------------------
    def due_for_close(self, today: date) -> list[dict[str, Any]]:
        rows = self.db.fetchall("SELECT * FROM options_structures WHERE status='OPEN'")
        return [r for r in rows if (r["horizon_date"] and r["horizon_date"] <= str(today))
                or r["close_by_date"] <= str(today)]

    def close_structure(self, structure_id: str, reason: str, session_date: date,
                        quote_fn: Callable[[str], OptionQuote | None] | None = None) -> dict[str, Any]:
        """Close an OPEN structure: buy back any short leg first, then sell the long leg. Allowed while
        ``paper_trading`` is false (closing only reduces risk) but never under SYSTEM_PAUSED."""
        st = self.db.fetchone("SELECT * FROM options_structures WHERE structure_id=?", (structure_id,))
        if st is None or st["status"] != "OPEN":
            return self._refuse(f"structure {structure_id} is not OPEN", structure_id=structure_id)
        gate = self._state_gate()
        if gate:
            return self._refuse(gate, structure_id=structure_id)
        pos = self.ledger.structure_positions(structure_id)
        self._set_status(structure_id, "CLOSING", {"reason": reason, "positions": pos})
        shorts = {s: q for s, q in pos.items() if q < -1e-9}
        if not shorts:
            return self._submit_close_longs(st, pos, session_date, quote_fn) or {"ok": True}
        legs = {lg["symbol"]: lg for lg in _legs(st)}
        sent = []
        for sym, q in shorts.items():
            bid, ask, qt = _fresh(quote_fn, sym) or (legs[sym].get("bid"), legs[sym].get("ask"), legs[sym].get("quote_time"))
            if not (bid and ask and bid > 0 and ask >= bid):
                return self._refuse(f"no two-sided quote to buy back {sym}", structure_id=structure_id)
            limit = buy_limit(bid, ask, self.s.paper.exit_mid_fraction)
            sent.append(self._submit_leg(structure_id, "short", "close", sym, Side.BUY, int(round(-q)), limit,
                                         float(st["multiplier"]), _quote_dict(bid, ask, qt), session_date))
        return {"ok": True, "orders": sent}

    def _submit_close_longs(self, st: dict[str, Any], pos: dict[str, float], session_date: date,
                            quote_fn: Callable[[str], OptionQuote | None] | None) -> dict[str, Any] | None:
        legs = {lg["symbol"]: lg for lg in _legs(st)}
        sent = []
        for sym, q in pos.items():
            if q <= 1e-9:
                continue
            bid, ask, qt = _fresh(quote_fn, sym) or (legs[sym].get("bid"), legs[sym].get("ask"), legs[sym].get("quote_time"))
            if not (bid and ask and bid > 0 and ask >= bid):
                self._event(st["structure_id"], "close_blocked", {"symbol": sym, "reason": "no two-sided quote"})
                continue
            limit = sell_limit(bid, ask, self.s.paper.exit_mid_fraction)
            if limit <= 0:
                self._event(st["structure_id"], "close_blocked", {"symbol": sym, "reason": "bid rounds to 0"})
                continue
            sent.append(self._submit_leg(st["structure_id"], "long", "close", sym, Side.SELL, int(round(q)), limit,
                                         float(st["multiplier"]), _quote_dict(bid, ask, qt), session_date))
        return {"structure_id": st["structure_id"], "to": "CLOSING", "orders": sent} if sent else None


def _legs(st: dict[str, Any]) -> list[dict[str, Any]]:
    from quantlab.db.database import from_json
    return from_json(st["legs_json"], [])


def _quote_dict(bid: Any, ask: Any, qt: Any) -> dict[str, Any]:
    return {"bid": bid, "ask": ask, "quote_time": qt, "feed": "indicative"}


def _fresh(quote_fn: Callable[[str], OptionQuote | None] | None, symbol: str) -> tuple[Any, Any, Any] | None:
    if quote_fn is None:
        return None
    q = quote_fn(symbol)
    if q is None:
        return None
    return q.bid, q.ask, q.quote_time.isoformat() if q.quote_time is not None else None
