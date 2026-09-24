"""Internal per-book ledger: THE source of truth for cash, positions and realized P&L.

A broker (sim or Alpaca paper) is only an external view that gets polled and reconciled against
this ledger (see ``reconcile.py`` and ARCHITECTURE.md section 8). BOT and HUMAN books are always
kept apart by the ``book`` column; every query here filters on it (see :func:`paper_book`).

Cash is never a mutable balance: it is the SUM of the append-only ``ledger_cash_events`` journal
(migration 050), so every dollar can be traced to a fill, a commission, a dividend or the starting
deposit. Positions are a mutable STATE row per (book, symbol) with an append-only ``position_log``
history. Trades themselves (open/close, P&L, the audit snapshot) are persisted through
:class:`~quantlab.execution.journal.TradeJournal`, which this ledger owns by default, so there is
one writer for the ``trades``/``trade_plans``/``trade_events`` tables.

Realized P&L on close is reconstructed from the ACTUAL fill prices (which already bake in the
CostModel spread/slippage the broker charged) plus commission, so it is a real economic number, not
a re-derivation of ``core.tradesim``'s cost formula. The two are only approximately equal (both are
~ basis-point-sized effects); see ``tests/execution/test_equivalence.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.core.types import OrderStatus, Side, new_id
from quantlab.data.panel import Panel
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.execution.journal import TradeJournal
from quantlab.execution.sim_broker import paper_book
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.ledger")

_EPS = 1e-9


class LedgerError(RuntimeError):
    pass


@dataclass
class FillResult:
    fill_id: str
    order_id: str
    trade_id: str | None
    order_status: str
    purpose: str


class Ledger:
    """Source of truth for one book (BOT or HUMAN). One instance per (db, book)."""

    def __init__(
        self,
        db: Database,
        book: str,
        starting_cash: float | None = None,
        *,
        config: Config | None = None,
        journal: TradeJournal | None = None,
    ):
        self.db = db
        self.book = paper_book(book)
        self.journal = journal or TradeJournal(db)

        def cfg(key: str, default: Any) -> Any:
            return config.get(key, default) if config is not None else default

        self.credit_dividends = bool(cfg("execution.sim.credit_dividends", True))
        if starting_cash is None:
            starting_cash = float(cfg(f"paper.{self.book.lower()}.starting_cash", 100000.0))
        if not (math.isfinite(starting_cash) and starting_cash >= 0):
            raise ValueError("starting_cash must be a non-negative number")
        self._ensure_starting_cash(float(starting_cash))

    # ------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------
    def _ensure_starting_cash(self, starting_cash: float) -> None:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS total FROM ledger_cash_events WHERE book=?",
            (self.book,),
        )
        if row and row["n"] > 0:
            return  # already initialized (e.g. a previous pipeline run) -- never re-seed
        self.db.insert("ledger_cash_events", {
            "book": self.book, "session_date": None, "kind": "starting_cash", "amount": starting_cash,
            "symbol": None, "trade_id": None, "ref_type": "init", "ref_id": None,
            "details_json": to_json({}), "created_at": utcnow_iso(),
        })
        log_event(log, "ledger initialized", book=self.book, starting_cash=starting_cash)

    # ------------------------------------------------------------------------------------------
    # cash / positions (read)
    # ------------------------------------------------------------------------------------------
    def cash(self) -> float:
        row = self.db.fetchone("SELECT COALESCE(SUM(amount), 0) AS total FROM ledger_cash_events WHERE book=?",
                               (self.book,))
        return float(row["total"]) if row else 0.0

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM positions WHERE book=? AND symbol=?", (self.book, symbol))

    def positions(self) -> list[dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM positions WHERE book=? ORDER BY symbol", (self.book,))

    def open_trades(self) -> list["OpenTrade"]:
        """Open trades in :class:`quantlab.execution.exits.OpenTrade` form, ready for ExitEngine."""
        from quantlab.execution.exits import OpenTrade  # local import: avoid a top-level cycle

        rows = self.db.fetchall(
            "SELECT t.trade_id, t.symbol, t.entry_date, tp.signal_date, t.qty, t.stop_price, t.target_price, "
            "tp.holding_sessions, t.direction, t.book, t.entry_price, t.strategy_id, t.candidate_id, "
            "t.decision_id, t.human_decision_id FROM trades t LEFT JOIN trade_plans tp ON tp.trade_id = t.trade_id "
            "WHERE t.book=? AND t.status='OPEN' ORDER BY t.trade_id", (self.book,))
        fields = OpenTrade.__dataclass_fields__
        return [OpenTrade(**{k: r[k] for k in fields if k in r.keys()}) for r in rows]

    def state(self) -> dict[str, Any]:
        return {"book": self.book, "cash": self.cash(), "positions": self.positions(),
                "open_trades": self.db.fetchall(
                    "SELECT trade_id, symbol, qty, entry_date, entry_price, direction FROM trades "
                    "WHERE book=? AND status='OPEN' ORDER BY trade_id", (self.book,))}

    # ------------------------------------------------------------------------------------------
    # fills
    # ------------------------------------------------------------------------------------------
    def record_order_status(self, order_id: str, status: OrderStatus, reason: str | None = None) -> None:
        """Non-fill status transition (rejected / expired / canceled / unknown). Touches no cash,
        position or trade -- see :meth:`apply_fill` for that."""
        now = utcnow_iso()
        self.db.execute("UPDATE orders SET status=?, last_update_at=? WHERE order_id=? AND book=?",
                        (status.value, now, order_id, self.book))
        self.db.insert("order_events", {"order_id": order_id, "event": "status_change", "status": status.value,
                                        "at": now, "raw_json": to_json({"reason": reason})})

    def apply_fill(
        self,
        *,
        order_id: str,
        qty: float,
        price: float,
        session_date: Any,
        commission: float = 0.0,
        modeled_cost: float = 0.0,
        filled_at: str | None = None,
        broker_fill_id: str | None = None,
        source: str = "sim",
        direction: str = "LONG",
        signal_date: str | None = None,
        exit_reason: str | None = None,
        held_sessions: int | None = None,
        mae: float | None = None,
        mfe: float | None = None,
        candidate_id: str | None = None,
        decision_id: str | None = None,
        human_decision_id: str | None = None,
        strategy_id: str | None = None,
        strategy_version: str | None = None,
        plan: Any = None,
        journal_extra: dict[str, Any] | None = None,
    ) -> FillResult:
        """Apply one fill against an existing ``order_id`` (this book).

        Writes an append-only ``fills`` row, updates the order's cumulative filled_qty/avg_price/
        status (+ ``order_events``), updates ``positions`` (+ append-only ``position_log``) and
        ``ledger_cash_events``, and opens or closes the linked trade through :attr:`journal`
        (``orders.purpose`` decides which; ``orders.side`` decides the cash direction). Full-position
        exits only (matching ``core.tradesim``'s whole-position model) -- ``qty`` on an exit fill
        must equal the trade's open quantity.
        """
        qty = float(qty)
        price = float(price)
        if qty <= 0 or not math.isfinite(qty):
            raise ValueError(f"fill qty must be positive, got {qty!r}")
        if price <= 0 or not math.isfinite(price):
            raise ValueError(f"fill price must be positive, got {price!r}")
        order = self.db.fetchone("SELECT * FROM orders WHERE order_id=? AND book=?", (order_id, self.book))
        if order is None:
            raise LedgerError(f"unknown order {order_id} for book {self.book}")
        side = Side(order["side"])
        symbol = order["symbol"]
        purpose = order["purpose"]
        s_str = to_session(session_date).date().isoformat()
        filled_at = filled_at or utcnow_iso()
        now = utcnow_iso()

        with self.db.transaction():
            prev_filled_qty = float(order["filled_qty"] or 0.0)
            prev_avg = order["filled_avg_price"]
            order_qty = float(order["qty"])
            new_filled_qty = min(prev_filled_qty + qty, order_qty)
            new_avg = price if prev_filled_qty <= _EPS else (
                (prev_filled_qty * float(prev_avg) + qty * price) / (prev_filled_qty + qty))
            new_status = (OrderStatus.FILLED if new_filled_qty >= order_qty - _EPS
                         else OrderStatus.PARTIALLY_FILLED)
            self.db.execute(
                "UPDATE orders SET filled_qty=?, filled_avg_price=?, status=?, last_update_at=? WHERE order_id=?",
                (new_filled_qty, new_avg, new_status.value, now, order_id))
            self.db.insert("order_events", {
                "order_id": order_id, "event": "fill", "status": new_status.value, "at": now,
                "raw_json": to_json({"qty": qty, "price": price, "commission": commission,
                                     "modeled_cost": modeled_cost, "session_date": s_str, "source": source}),
            })

            fill_id = new_id("fill")
            self.db.insert("fills", {
                "fill_id": fill_id, "order_id": order_id, "book": self.book, "symbol": symbol, "side": side.value,
                "qty": qty, "price": price, "commission": commission, "modeled_cost": modeled_cost,
                "filled_at": filled_at, "session_date": s_str, "broker_fill_id": broker_fill_id, "source": source,
            })
            self._record_fill_cash(order["trade_id"], symbol, side, qty, price, commission, s_str, fill_id)

            trade_id = order["trade_id"]
            if purpose == "entry":
                self._apply_position_fill(symbol, side, qty, price, trade_id, s_str, filled_at)
                if trade_id is None:
                    trade_id = self.journal.open_trade(
                        book=self.book, symbol=symbol, qty=new_filled_qty, entry_date=s_str, entry_price=new_avg,
                        direction=direction, candidate_id=candidate_id, decision_id=decision_id,
                        human_decision_id=human_decision_id, strategy_id=strategy_id,
                        strategy_version=strategy_version, plan=plan, signal_date=signal_date or s_str,
                        journal=journal_extra,
                    )
                    self.db.execute("UPDATE orders SET trade_id=? WHERE order_id=?", (trade_id, order_id))
                    self.db.execute("UPDATE positions SET trade_id=? WHERE book=? AND symbol=?",
                                    (trade_id, self.book, symbol))
                else:
                    self.journal.add_entry_fill(trade_id, qty=new_filled_qty, entry_price=new_avg)
            elif purpose == "exit":
                if trade_id is None:
                    raise LedgerError(f"exit order {order_id} has no linked trade_id")
                self._apply_position_fill(symbol, side, qty, price, trade_id, s_str, filled_at)
                self._close_trade(trade_id, qty=qty, price=price, commission=commission,
                                  modeled_cost=modeled_cost, session_date=s_str, exit_reason=exit_reason,
                                  held_sessions=held_sessions, mae=mae, mfe=mfe)
            else:
                raise LedgerError(f"order {order_id} has unknown purpose {purpose!r}")

        return FillResult(fill_id=fill_id, order_id=order_id, trade_id=trade_id,
                          order_status=new_status.value, purpose=purpose)

    def _record_fill_cash(self, trade_id: str | None, symbol: str, side: Side, qty: float, price: float,
                          commission: float, session_date: str, fill_id: str) -> None:
        """The two ``ledger_cash_events`` rows a fill produces: the buy/sell notional, then
        commission (kept separate from the notional so every dollar traces to one cause)."""
        kind = "buy" if side is Side.BUY else "sell"
        amount = -(qty * price) if side is Side.BUY else (qty * price)
        self.db.insert("ledger_cash_events", {
            "book": self.book, "session_date": session_date, "kind": kind, "amount": amount, "symbol": symbol,
            "trade_id": trade_id, "ref_type": "fill", "ref_id": fill_id, "details_json": to_json({"qty": qty, "price": price}),
            "created_at": utcnow_iso(),
        })
        if abs(commission) > _EPS:
            self.db.insert("ledger_cash_events", {
                "book": self.book, "session_date": session_date, "kind": "commission", "amount": -abs(commission),
                "symbol": symbol, "trade_id": trade_id, "ref_type": "fill", "ref_id": fill_id,
                "details_json": to_json({}), "created_at": utcnow_iso(),
            })

    def _apply_position_fill(self, symbol: str, side: Side, qty: float, price: float, trade_id: str | None,
                             session_date: str, filled_at: str) -> None:
        pos = self.get_position(symbol)
        qty_before = float(pos["qty"]) if pos else 0.0
        avg_before = float(pos["avg_cost"]) if pos else None
        now = utcnow_iso()
        if side is Side.BUY:
            qty_after = qty_before + qty
            avg_after = price if qty_before <= _EPS else (qty_before * avg_before + qty * price) / qty_after
            self.db.upsert("positions", {
                "book": self.book, "symbol": symbol, "qty": qty_after, "avg_cost": avg_after,
                "trade_id": trade_id or (pos["trade_id"] if pos else None),
                "opened_at": pos["opened_at"] if pos else (filled_at or session_date), "updated_at": now,
            }, ["book", "symbol"])
        else:
            if pos is None:
                raise LedgerError(f"{self.book}/{symbol}: sell fill with no open position (no short selling)")
            if qty > qty_before + 1e-6:
                raise LedgerError(f"{self.book}/{symbol}: sell qty {qty:g} > held {qty_before:g}")
            qty_after = qty_before - qty
            avg_after = avg_before
            if qty_after <= _EPS:
                self.db.execute("DELETE FROM positions WHERE book=? AND symbol=?", (self.book, symbol))
                qty_after = 0.0
            else:
                self.db.execute("UPDATE positions SET qty=?, updated_at=? WHERE book=? AND symbol=?",
                                (qty_after, now, self.book, symbol))
        self.db.insert("position_log", {
            "book": self.book, "symbol": symbol, "trade_id": trade_id, "qty_before": qty_before,
            "qty_after": qty_after, "avg_cost_before": avg_before, "avg_cost_after": avg_after,
            "reason": "fill", "ref_id": None, "session_date": session_date, "at": now,
        })

    def _close_trade(self, trade_id: str, *, qty: float, price: float, commission: float, modeled_cost: float,
                     session_date: str, exit_reason: str | None, held_sessions: int | None,
                     mae: float | None, mfe: float | None) -> None:
        trade = self.journal.get_trade(trade_id)
        if trade is None:
            raise LedgerError(f"unknown trade {trade_id}")
        entry_costs = self.db.fetchone(
            "SELECT COALESCE(SUM(f.commission), 0) AS commission, COALESCE(SUM(f.modeled_cost), 0) AS modeled_cost "
            "FROM fills f JOIN orders o ON o.order_id = f.order_id "
            "WHERE o.trade_id=? AND o.purpose='entry' AND o.book=?", (trade_id, self.book))
        sign = 1.0 if trade["direction"] == "LONG" else -1.0
        entry_price, entry_qty = float(trade["entry_price"]), float(trade["qty"])
        entry_commission = float(entry_costs["commission"]) if entry_costs else 0.0
        entry_modeled_cost = float(entry_costs["modeled_cost"]) if entry_costs else 0.0
        # actual_pnl already nets both round-trip modeled costs (they are baked into the fill prices);
        # adding them back reconstructs a cost-free "gross" pnl, matching core.tradesim's gross_ret.
        actual_pnl = (price - entry_price) * qty * sign
        gross_pnl = actual_pnl + entry_modeled_cost + modeled_cost
        costs = entry_modeled_cost + modeled_cost + entry_commission + commission
        net_pnl = gross_pnl - costs
        notional = entry_price * entry_qty
        ret = (net_pnl / notional) if notional > _EPS else None
        self.journal.close_trade(
            trade_id, exit_date=session_date, exit_price=price, exit_reason=exit_reason or "MANUAL",
            gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl, ret=ret, holding_sessions=held_sessions,
            mae=mae, mfe=mfe,
        )

    # ------------------------------------------------------------------------------------------
    # corporate actions
    # ------------------------------------------------------------------------------------------
    def apply_corporate_actions(self, session_date: Any, panel: Panel) -> list[dict[str, Any]]:
        """Apply session ``S``'s splits/dividends to every held position. Idempotent per
        (book, symbol, session, action_type) via ``ledger_corporate_actions`` (its primary key), so
        re-running the same session twice never double-applies a split.

        Dividends: credited to cash only when ``credit_dividends`` (Alpaca paper does NOT simulate
        dividends -- see docs/EXTERNAL-SERVICES.md). When not credited the action is still recorded
        (``credited=0``) so a comparison against a broker book can explain the gap.
        """
        s = to_session(session_date)
        s_str = s.date().isoformat()
        applied: list[dict[str, Any]] = []
        with self.db.transaction():
            for pos in self.positions():
                sym = pos["symbol"]
                if sym not in panel.symbols or s not in panel.dates:
                    continue
                ratio = float(panel.split_ratio.at[s, sym])
                div = float(panel.dividend.at[s, sym])
                qty_before = float(pos["qty"])

                if math.isfinite(div) and div > 1e-12:
                    if not self.db.fetchone(
                        "SELECT 1 FROM ledger_corporate_actions WHERE book=? AND symbol=? AND session_date=? "
                        "AND action_type='cash_dividend'", (self.book, sym, s_str)):
                        amount = qty_before * div
                        credited = self.credit_dividends
                        if credited:
                            self.db.insert("ledger_cash_events", {
                                "book": self.book, "session_date": s_str, "kind": "dividend", "amount": amount,
                                "symbol": sym, "trade_id": pos["trade_id"], "ref_type": "corporate_action",
                                "ref_id": None, "details_json": to_json({"per_share": div}),
                                "created_at": utcnow_iso(),
                            })
                        self.db.insert("ledger_corporate_actions", {
                            "book": self.book, "symbol": sym, "session_date": s_str, "action_type": "cash_dividend",
                            "ratio": None, "amount": div, "qty_before": qty_before, "qty_after": qty_before,
                            "cash_amount": amount if credited else 0.0, "credited": int(credited),
                            "trade_id": pos["trade_id"], "created_at": utcnow_iso(),
                        })
                        applied.append({"symbol": sym, "action_type": "cash_dividend", "amount": div,
                                        "cash_amount": amount if credited else 0.0, "credited": credited})

                if math.isfinite(ratio) and ratio > 0 and abs(ratio - 1.0) > _EPS:
                    if not self.db.fetchone(
                        "SELECT 1 FROM ledger_corporate_actions WHERE book=? AND symbol=? AND session_date=? "
                        "AND action_type='split'", (self.book, sym, s_str)):
                        qty_after = qty_before * ratio
                        avg_before = float(pos["avg_cost"])
                        avg_after = avg_before / ratio
                        now = utcnow_iso()
                        self.db.execute("UPDATE positions SET qty=?, avg_cost=?, updated_at=? WHERE book=? AND symbol=?",
                                        (qty_after, avg_after, now, self.book, sym))
                        self.db.insert("position_log", {
                            "book": self.book, "symbol": sym, "trade_id": pos["trade_id"], "qty_before": qty_before,
                            "qty_after": qty_after, "avg_cost_before": avg_before, "avg_cost_after": avg_after,
                            "reason": "split", "ref_id": None, "session_date": s_str, "at": now,
                        })
                        self.db.insert("ledger_corporate_actions", {
                            "book": self.book, "symbol": sym, "session_date": s_str, "action_type": "split",
                            "ratio": ratio, "amount": None, "qty_before": qty_before, "qty_after": qty_after,
                            "cash_amount": 0.0, "credited": 1, "trade_id": pos["trade_id"], "created_at": now,
                        })
                        applied.append({"symbol": sym, "action_type": "split", "ratio": ratio,
                                        "qty_before": qty_before, "qty_after": qty_after})
        return applied

    # ------------------------------------------------------------------------------------------
    # mark to market
    # ------------------------------------------------------------------------------------------
    def mark_to_market(self, session_date: Any, panel: Panel) -> dict[str, Any]:
        """Snapshot equity/exposure/drawdown/open-risk for ``session_date`` into
        ``portfolio_snapshots`` (upserted: idempotent per (book, as_of_date))."""
        s = to_session(session_date)
        s_str = s.date().isoformat()
        cash = self.cash()
        positions = self.positions()
        details_positions = []
        market_value_total = 0.0
        for pos in positions:
            sym = pos["symbol"]
            px = None
            if sym in panel.symbols:
                valid = panel.close[sym].loc[:s].dropna()
                if len(valid):
                    px = float(valid.iloc[-1])
            mv = pos["qty"] * px if px is not None else pos["qty"] * pos["avg_cost"]
            market_value_total += mv
            details_positions.append({"symbol": sym, "qty": pos["qty"], "price": px, "market_value": mv,
                                      "priced": px is not None})
        equity = cash + market_value_total
        gross_exposure = (sum(abs(d["market_value"]) for d in details_positions) / equity) if equity > _EPS else 0.0

        prev = self.db.fetchone("SELECT peak_equity FROM portfolio_snapshots WHERE book=? AND as_of_date<? "
                                "ORDER BY as_of_date DESC LIMIT 1", (self.book, s_str))
        peak_equity = max(equity, float(prev["peak_equity"])) if prev else equity
        drawdown = (equity / peak_equity - 1.0) if peak_equity > _EPS else 0.0

        open_trades = self.db.fetchall(
            "SELECT symbol, qty, entry_price, stop_price, direction FROM trades WHERE book=? AND status='OPEN'",
            (self.book,))
        risk_dollars = 0.0
        for t in open_trades:
            if t["stop_price"] is None or t["entry_price"] is None:
                continue
            sign = 1.0 if t["direction"] == "LONG" else -1.0
            risk_dollars += max(0.0, (t["entry_price"] - t["stop_price"]) * sign) * t["qty"]
        open_risk = (risk_dollars / equity) if equity > _EPS else None

        row = {
            "book": self.book, "as_of_date": s_str, "cash": cash, "equity": equity,
            "gross_exposure": gross_exposure, "positions_count": len(positions), "peak_equity": peak_equity,
            "drawdown": drawdown, "open_risk": open_risk,
            "details_json": to_json({"positions": details_positions}), "created_at": utcnow_iso(),
        }
        self.db.upsert("portfolio_snapshots", row, ["book", "as_of_date"])
        return row
