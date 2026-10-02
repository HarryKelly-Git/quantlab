"""Recording (append-only evaluations) and the separate PAPER options book ``OPT``.

The OPT book shares the Alpaca paper ACCOUNT with the stock BOT book but nothing else:
  * its own tables (migration 055), its own premium journal and its own positions (summed from
    broker-reported fills, contracts x multiplier handled explicitly on every fill);
  * client order ids start with :data:`OPT_ORDER_PREFIX`, so the stock runner can tell OPT orders
    from orders it does not know;
  * :func:`cotenant_cash_offset` is the cash the OPT book has moved in the shared account. The stock
    reconciliation adds it to the BOT ledger's cash before comparing with the broker, and only
    compares ``us_equity`` positions (execution/reconcile.py). :func:`reconcile_options` compares
    the ``us_option`` positions with this book.
"""
from __future__ import annotations

import hashlib
from typing import Any

from quantlab.core.types import OrderStatus, new_id
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, BrokerOrder
from quantlab.execution.reconcile import OPTION_ASSET_CLASS, position_asset_class
from quantlab.logging_setup import get_logger, log_event

log = get_logger("options.book")

OPT_BOOK = "OPT"
OPT_ORDER_PREFIX = "qlopt-"
TERMINAL = frozenset({OrderStatus.FILLED.value, OrderStatus.CANCELED.value, OrderStatus.EXPIRED.value,
                      OrderStatus.REJECTED.value})
_EPS = 1e-9


def opt_client_order_id(structure_id: str, leg_role: str, purpose: str, attempt: int = 0) -> str:
    """Deterministic: the same (structure, leg, purpose, attempt) always maps to the same id, so a
    resubmission is adopted by the broker instead of creating a second order."""
    digest = hashlib.sha1(f"{structure_id}|{leg_role}|{purpose}|{attempt}".encode("utf-8")).hexdigest()[:24]
    return f"{OPT_ORDER_PREFIX}{digest}"


# -- recording -----------------------------------------------------------------------------------
def record_evaluation(db: Database, cmp: Any, *, spot_source: str | None = None, is_synthetic: bool = False,
                      evaluation_id: str | None = None) -> str:
    """Write one comparison: the run (all inputs + the choice), one row per expression, and every
    liquidity check. Append-only (triggers)."""
    eid = evaluation_id or new_id("opteval")
    now = utcnow_iso()
    t = cmp.thesis
    with db.transaction():
        db.insert("options_eval_runs", {
            "evaluation_id": eid, "created_at": now, "session_date": str(cmp.session_date), "symbol": t.symbol,
            "direction": t.direction, "horizon_sessions": int(t.horizon_sessions), "horizon_date": str(cmp.horizon_date),
            "spot": t.spot, "spot_source": spot_source, "stock_stop": t.stop_price, "feed": cmp.settings.feed,
            "expiration": str(cmp.expiration) if cmp.expiration else None, "expiry_reason": cmp.expiry_reason,
            "distribution_label": cmp.distribution.label, "distribution_json": to_json(cmp.distribution.to_dict()),
            "chosen_expression": cmp.choice, "reason": cmp.reason, "contracts_checked": len(cmp.liquidity),
            "contracts_passed": sum(1 for c in cmp.liquidity if c.passed), "grid_notes_json": to_json(cmp.grid_notes),
            "data_notes_json": to_json(cmp.data_notes), "settings_json": to_json(cmp.settings.as_dict()),
            "is_synthetic": int(bool(is_synthetic)),
        })
        for r in cmp.results:
            st = r.structure
            qtimes = sorted(lg.quote_time for lg in st.legs if lg.quote_time) if st is not None else []
            db.insert("options_evaluations", {
                "evaluation_id": eid, "session_date": str(cmp.session_date), "symbol": t.symbol,
                "expression_id": r.expression_id, "kind": r.kind,
                "expiration": st.expiration.isoformat() if st is not None else None,
                "legs_json": to_json([lg.to_dict() for lg in st.legs]) if st is not None else None,
                "feed": cmp.settings.feed if st is not None else "n/a (stock)",
                "quote_time_min": qtimes[0] if qtimes else None, "quote_time_max": qtimes[-1] if qtimes else None,
                "entry_cost": r.entry_cost, "risk_usd": r.risk_usd, "max_loss_usd": r.max_loss_usd,
                "max_gain_usd": r.max_gain_usd, "breakeven": r.breakeven, "expected_pnl_usd": r.expected_pnl_usd,
                "expected_pnl_per_risk": r.expected_pnl_per_risk, "p_profit": r.p_profit, "p_breakeven": r.p_breakeven,
                "max_loss_per_risk": r.max_loss_per_risk, "expected_loss_given_loss": r.expected_loss_given_loss,
                "eligible": int(r.eligible), "ineligible_reason": r.ineligible_reason, "chosen": int(r.chosen),
                "label": r.label, "notes_json": to_json(r.notes), "created_at": now,
            })
        for c in cmp.liquidity:
            db.insert("options_liquidity_checks", {
                "evaluation_id": eid, "contract_symbol": c.symbol, "passed": int(c.passed),
                "reasons_json": to_json(c.reasons), "bid": c.bid, "ask": c.ask, "spread_pct": c.spread_pct,
                "quote_time": c.quote_time, "quote_age_minutes": c.quote_age_minutes, "open_interest": c.open_interest,
                "oi_status": c.oi_status, "feed": c.feed, "created_at": now,
            })
    log_event(log, "options evaluation recorded", evaluation_id=eid, symbol=t.symbol, choice=cmp.choice)
    return eid


# -- OPT ledger ----------------------------------------------------------------------------------
class OptionsLedger:
    """Source of truth for the OPT book: premium journal + fills (positions are their sum)."""

    def __init__(self, db: Database):
        self.db = db

    def ensure_allocation(self, amount: float) -> None:
        if self.db.fetchone("SELECT 1 FROM options_cash_events WHERE kind='allocation'") is None:
            self.db.insert("options_cash_events", {"book": OPT_BOOK, "kind": "allocation", "amount": float(amount),
                                                   "details_json": to_json({"note": "premium budget of the OPT book; "
                                                                            "not a transfer at the broker"}),
                                                   "created_at": utcnow_iso()})

    def allocation(self) -> float:
        return float(self.db.fetchone("SELECT COALESCE(SUM(amount),0) AS v FROM options_cash_events "
                                      "WHERE kind='allocation'")["v"])

    def net_trading_cash(self) -> float:
        """USD the OPT book's fills moved in the broker account (premium received - paid)."""
        return float(self.db.fetchone("SELECT COALESCE(SUM(amount),0) AS v FROM options_cash_events "
                                      "WHERE kind<>'allocation'")["v"])

    def cash(self) -> float:
        return self.allocation() + self.net_trading_cash()

    def positions(self) -> dict[str, float]:
        rows = self.db.fetchall("SELECT contract_symbol, SUM(CASE WHEN side='buy' THEN qty ELSE -qty END) AS q "
                                "FROM options_fills GROUP BY contract_symbol")
        return {r["contract_symbol"]: float(r["q"]) for r in rows if abs(float(r["q"])) > _EPS}

    def structure_positions(self, structure_id: str) -> dict[str, float]:
        rows = self.db.fetchall("SELECT contract_symbol, SUM(CASE WHEN side='buy' THEN qty ELSE -qty END) AS q "
                                "FROM options_fills WHERE structure_id=? GROUP BY contract_symbol", (structure_id,))
        return {r["contract_symbol"]: float(r["q"]) for r in rows}

    def apply_broker_order(self, order: dict[str, Any], bo: BrokerOrder, session_date: str | None) -> list[dict[str, Any]]:
        """Apply the broker's CUMULATIVE fill state to one local OPT order (positive deltas only, so
        replays are harmless). Returns the new fills. Never invents a price: the delta's price is
        derived from the broker's cumulative filled_qty x filled_avg_price."""
        fills: list[dict[str, Any]] = []
        prev_q = float(order["filled_qty"] or 0.0)
        prev_avg = float(order["filled_avg_price"] or 0.0)
        new_q = float(bo.filled_qty or 0.0)
        now = utcnow_iso()
        with self.db.transaction():
            if new_q > prev_q + _EPS and bo.filled_avg_price is not None:
                delta = new_q - prev_q
                price = (new_q * float(bo.filled_avg_price) - prev_q * prev_avg) / delta
                mult = float(order["multiplier"])
                sign = 1.0 if order["side"] == "buy" else -1.0
                cash = -sign * delta * price * mult
                fill_id = new_id("optfill")
                self.db.insert("options_fills", {
                    "fill_id": fill_id, "order_id": order["order_id"], "structure_id": order["structure_id"],
                    "contract_symbol": order["contract_symbol"], "side": order["side"], "qty": delta, "price": price,
                    "multiplier": mult, "cash_amount": cash, "cum_filled_qty": new_q, "session_date": session_date,
                    "filled_at": bo.filled_at, "created_at": now})
                self.db.insert("options_cash_events", {
                    "book": OPT_BOOK, "kind": "premium_paid" if sign > 0 else "premium_received", "amount": cash,
                    "fill_id": fill_id, "structure_id": order["structure_id"],
                    "details_json": to_json({"contract": order["contract_symbol"], "qty": delta, "price": price,
                                             "multiplier": mult}), "created_at": now})
                fills.append({"fill_id": fill_id, "qty": delta, "price": price, "cash": cash})
                new_avg = float(bo.filled_avg_price)
            else:
                new_q, new_avg = prev_q, (order["filled_avg_price"] if prev_q > 0 else None)
            status = bo.status.value
            if status != order["status"] or fills:
                self.db.insert("options_order_events", {"order_id": order["order_id"], "event": "status" if not fills
                                                        else "fill", "details_json": to_json(
                                                            {"from": order["status"], "to": status, "fills": fills,
                                                             "reason": bo.reason}), "at": now})
            self.db.execute("UPDATE options_orders SET status=?, filled_qty=?, filled_avg_price=?, broker_order_id="
                            "COALESCE(?, broker_order_id), last_update_at=?, raw_json=? WHERE order_id=?",
                            (status, new_q, new_avg, bo.broker_order_id, now, to_json(bo.raw or {}), order["order_id"]))
        return fills


def open_opt_orders(db: Database) -> list[dict[str, Any]]:
    marks = ",".join("?" for _ in TERMINAL)
    return db.fetchall(f"SELECT * FROM options_orders WHERE status NOT IN ({marks}) ORDER BY created_at",
                       tuple(sorted(TERMINAL)))


def cotenant_cash_offset(db: Database, broker: Any = None) -> float:
    """Cash the OPT book has moved in the shared Alpaca paper account: its recorded net premium
    flow plus (when a broker is given) fills the broker already reports on still-open OPT orders
    that the OPT ledger has not applied yet. Read-only. Used by the stock reconciliation so that an
    options fill can never look like a stock-ledger cash mismatch."""
    offset = OptionsLedger(db).net_trading_cash()
    if broker is None:
        return offset
    for o in open_opt_orders(db):
        bo = broker.get_order_by_client_id(o["client_order_id"])
        if bo is None or bo.filled_avg_price is None:
            continue
        unapplied = float(bo.filled_qty) * float(bo.filled_avg_price) - float(o["filled_qty"] or 0.0) * float(
            o["filled_avg_price"] or 0.0)
        if abs(unapplied) > _EPS:
            sign = 1.0 if o["side"] == "buy" else -1.0
            offset += -sign * unapplied * float(o["multiplier"])
    return offset


def reconcile_options(db: Database, broker: Any, run_id: str | None = None) -> dict[str, Any]:
    """Compare the broker's us_option positions (and OPT-prefixed open orders) with the OPT book.
    Records a ``reconciliations`` row with book 'OPT'. Never pauses anything itself."""
    ledger = OptionsLedger(db)
    try:
        positions = broker.positions()
        open_orders = broker.list_orders("open")
    except BrokerError as exc:
        res = {"ok": False, "status": "broker_unavailable", "diffs": [{"error": str(exc)}]}
        _write_rec(db, run_id, res, None, None)
        return res
    b_pos = {p.symbol: float(p.qty) for p in positions if position_asset_class(p) == OPTION_ASSET_CLASS}
    i_pos = ledger.positions()
    diffs: list[dict[str, Any]] = []
    for sym in sorted(set(b_pos) | set(i_pos)):
        b, i = b_pos.get(sym, 0.0), i_pos.get(sym, 0.0)
        if abs(b - i) > 1e-6:
            diffs.append({"field": f"position:{sym}", "broker": b, "internal": i, "delta": b - i})
    known = {r["client_order_id"] for r in db.fetchall("SELECT client_order_id FROM options_orders")}
    for o in open_orders:
        if o.client_order_id.startswith(OPT_ORDER_PREFIX) and o.client_order_id not in known:
            diffs.append({"field": f"order:{o.client_order_id}", "broker": "open", "internal": "unknown",
                          "delta": None})
    res = {"ok": not diffs, "status": "ok" if not diffs else "mismatch", "diffs": diffs}
    _write_rec(db, run_id, res, {"option_positions": b_pos}, {"positions": i_pos, "cash": ledger.cash()})
    if diffs:
        log_event(log, "OPT reconciliation mismatch", diffs=diffs)
    return res


def _write_rec(db: Database, run_id: str | None, res: dict[str, Any], broker_snap: Any, internal: Any) -> None:
    db.insert("reconciliations", {"run_id": run_id, "book": OPT_BOOK, "at": utcnow_iso(), "status": res["status"],
                                  "broker_json": to_json(broker_snap), "internal_json": to_json(internal),
                                  "diffs_json": to_json(res["diffs"])})
