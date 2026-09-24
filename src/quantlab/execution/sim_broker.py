"""Deterministic simulated PAPER broker (the default broker for both the BOT and the HUMAN book).

Fill model: IDENTICAL to ``core.tradesim.simulate_plan``, so bot, human and backtest results are
comparable:
  * An order decided at the close of session D fills at the RAW open of the first later session
    on which the symbol trades. If that session's open is missing but its close exists (a data
    issue), it fills at that close, which is what ``tradesim.next_exec`` does.
  * Price = ``CostModel.fill_price(side, ref, adv)``, where adv is the median dollar volume of the
    20 sessions ending at the decision session (the window tradesim uses), plus
    ``CostModel.commission(qty)``.
  * A symbol that has had no bar for ``execution.delisting_missing_sessions`` sessions while the
    market kept trading counts as delisted. Pending sells settle at last close x
    (1 + costs.delisting_return), the same convention as tradesim. Pending buys are rejected.
    (Tradesim knows the symbol never trades again because it can see the future; a live system
    cannot, so this is the PIT-safe version of the rule.)
  * If there is no bar for another reason (halt, missing data), the order stays pending, and after
    ``execution.sim.max_pending_sessions`` sessions it EXPIRES. It is never filled at an invented
    price.
  * Limit orders are checked only against the reference open. Intraday fills are not simulated
    (conservative), so a limit that is not marketable at the open EXPIRES.
  * No margin: a buy whose cost exceeds cash is REJECTED at fill time. No shorting: a sell larger
    than the position is REJECTED.
  * Corporate actions with ex-date S apply BEFORE fills at S's open. Dividends go to shares held
    at the close of S-1, on the pre-split share count (the schema's basis). Splits scale position
    qty and avg cost, and they scale the qty and limit of orders still pending from before S. The
    ledger applies the same order of operations.

Point-in-time: ``process_session(S, panel)`` truncates the panel at S before looking at it, so it
gives the same result whether it gets the full history or ``panel.truncate(S)``, which the tests
check.

State (cash, positions, orders) is kept in memory. When a Database is given it is also persisted
in ``sim_broker_state`` (one JSON document per book), and every action is appended to
``sim_broker_events``, so the broker survives separate daily pipeline runs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, time
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import MARKET_TZ, to_session
from quantlab.core.costs import CostModel
from quantlab.core.types import Book, OrderStatus, Side
from quantlab.data.panel import Panel
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.execution.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    PaperBroker,
)
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

PAPER_BOOKS = (Book.BOT.value, Book.HUMAN.value)
_EPS = 1e-9
ADV_WINDOW = 20                       # sessions; identical to core.tradesim


def paper_book(book: Book | str) -> str:
    """Normalize and validate a PAPER book name (BOT or HUMAN)."""
    value = Book(book).value if not isinstance(book, Book) else book.value
    if value not in PAPER_BOOKS:
        raise ValueError(f"{value} is not a paper book (allowed: {PAPER_BOOKS})")
    return value


def _session_str(d) -> str:
    return to_session(d).date().isoformat()


def _session_ts_utc(session: pd.Timestamp, t: time) -> str:
    local = pd.Timestamp(datetime.combine(session.date(), t)).tz_localize(MARKET_TZ)
    return local.tz_convert("UTC").isoformat()


@dataclass
class _SimOrder:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    time_in_force: str
    limit_price: float | None
    decision_session: str | None
    eligible_after: str | None          # fills only at sessions strictly after this one
    status: str
    seq: int
    submitted_at: str
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    filled_at: str | None = None
    fill_session: str | None = None
    reason: str | None = None
    commission: float = 0.0
    modeled_cost: float = 0.0
    missed_sessions: int = 0
    split_factor: float = 1.0
    delisting_settlement: bool = False

    @property
    def is_pending(self) -> bool:
        return self.status in (OrderStatus.ACCEPTED.value, OrderStatus.NEW.value)

    def to_broker_order(self) -> BrokerOrder:
        return BrokerOrder(
            client_order_id=self.client_order_id, broker_order_id=self.broker_order_id,
            status=OrderStatus(self.status), filled_qty=self.filled_qty, filled_avg_price=self.filled_avg_price,
            submitted_at=self.submitted_at, raw=asdict(self), symbol=self.symbol, side=Side(self.side),
            qty=self.qty, order_type=self.order_type, time_in_force=self.time_in_force,
            limit_price=self.limit_price, filled_at=self.filled_at, reason=self.reason,
            commission=self.commission, modeled_cost=self.modeled_cost, fill_session=self.fill_session,
            delisting_settlement=self.delisting_settlement,
        )


class SimBroker(PaperBroker):
    """Deterministic internal paper broker for one book. See module docstring for the fill model."""

    name = "sim"

    def __init__(
        self,
        book: Book | str,
        cost_model: CostModel,
        price_source: Panel | Callable[[], Panel] | None = None,
        *,
        starting_cash: float | None = None,
        config: Config | None = None,
        db: Database | None = None,
        max_pending_sessions: int | None = None,
        delisting_missing_sessions: int | None = None,
        credit_dividends: bool | None = None,
    ):
        self.book = paper_book(book)
        self.cost_model = cost_model
        self.price_source = price_source
        self.db = db

        def cfg(key: str, default: Any) -> Any:
            return config.get(key, default) if config is not None else default

        self.max_pending_sessions = int(max_pending_sessions if max_pending_sessions is not None
                                        else cfg("execution.sim.max_pending_sessions", 5))
        self.delisting_missing_sessions = int(delisting_missing_sessions if delisting_missing_sessions is not None
                                              else cfg("execution.delisting_missing_sessions", 5))
        self.credit_dividends = bool(credit_dividends if credit_dividends is not None
                                     else cfg("execution.sim.credit_dividends", True))
        if starting_cash is None:
            starting_cash = float(cfg(f"paper.{self.book.lower()}.starting_cash", 100000.0))
        if not (math.isfinite(starting_cash) and starting_cash >= 0):
            raise ValueError("starting_cash must be a non-negative number")
        if self.max_pending_sessions < 1 or self.delisting_missing_sessions < 1:
            raise ValueError("max_pending_sessions and delisting_missing_sessions must be >= 1")

        self.events: list[dict[str, Any]] = []
        self._init_state(float(starting_cash))

    # ------------------------------------------------------------------------------------------
    # state & persistence
    # ------------------------------------------------------------------------------------------
    def _init_state(self, starting_cash: float) -> None:
        row = self.db.fetchone("SELECT state_json FROM sim_broker_state WHERE book=?", (self.book,)) if self.db else None
        if row is not None:
            st = from_json(row["state_json"])
            self.starting_cash = float(st["starting_cash"])
            self.cash = float(st["cash"])
            self.positions_state: dict[str, dict[str, float]] = st["positions"]
            self.orders: dict[str, _SimOrder] = {k: _SimOrder(**v) for k, v in st["orders"].items()}
            self.last_session: str | None = st["last_session"]
            self.seq = int(st["seq"])
            self.last_prices: dict[str, float] = st.get("last_prices", {})
            if abs(self.starting_cash - starting_cash) > _EPS:
                log_event(log, "sim broker state already exists; configured starting cash ignored",
                          book=self.book, persisted=self.starting_cash, configured=starting_cash)
            return
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions_state = {}
        self.orders = {}
        self.last_session = None
        self.seq = 0
        self.last_prices = {}
        self._save()

    def _state(self) -> dict[str, Any]:
        return {
            "book": self.book, "starting_cash": self.starting_cash, "cash": self.cash,
            "positions": self.positions_state, "orders": {k: asdict(v) for k, v in self.orders.items()},
            "last_session": self.last_session, "seq": self.seq, "last_prices": self.last_prices,
        }

    def _save(self) -> None:
        if self.db is None:
            return
        self.db.upsert("sim_broker_state", {"book": self.book, "state_json": to_json(self._state()),
                                             "updated_at": utcnow_iso()}, ["book"])

    def _event(self, event: str, session: str | None, cid: str | None = None, symbol: str | None = None,
               **details: Any) -> None:
        rec = {"book": self.book, "session_date": session, "event": event, "client_order_id": cid,
               "symbol": symbol, "details": details}
        self.events.append(rec)
        if self.db is not None:
            self.db.insert("sim_broker_events", {"book": self.book, "session_date": session, "event": event,
                                                 "client_order_id": cid, "symbol": symbol,
                                                 "details_json": to_json(details), "at": utcnow_iso()})

    # ------------------------------------------------------------------------------------------
    # PaperBroker interface
    # ------------------------------------------------------------------------------------------
    def is_available(self) -> bool:
        return True

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        existing = self.orders.get(request.client_order_id)
        if existing is not None:
            # Idempotent: a client_order_id is one order, forever.
            return existing.to_broker_order()
        self.seq += 1
        eligible_after = request.decision_session or self.last_session
        order = _SimOrder(
            client_order_id=request.client_order_id, broker_order_id=f"sim-{self.book.lower()}-{self.seq:06d}",
            symbol=request.symbol, side=request.side.value, qty=float(request.qty), order_type=request.order_type,
            time_in_force=request.time_in_force, limit_price=request.limit_price,
            decision_session=request.decision_session, eligible_after=eligible_after,
            status=OrderStatus.ACCEPTED.value, seq=self.seq, submitted_at=utcnow_iso(),
        )
        if eligible_after is not None and self.last_session is not None and eligible_after < self.last_session:
            # The broker already processed a later session: filling now would silently execute
            # later than the decision implied. Refuse instead of pretending.
            order.status = OrderStatus.REJECTED.value
            order.reason = (f"stale decision session {eligible_after}: broker already processed "
                            f"{self.last_session}")
        self.orders[order.client_order_id] = order
        self._event("submitted" if order.status == OrderStatus.ACCEPTED.value else "rejected", eligible_after,
                    order.client_order_id, order.symbol, side=order.side, qty=order.qty, reason=order.reason)
        self._save()
        return order.to_broker_order()

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        o = self.orders.get(client_order_id)
        return o.to_broker_order() if o is not None else None

    def list_orders(self, status: str = "open") -> list[BrokerOrder]:
        if status not in ("open", "closed", "all"):
            raise ValueError("status must be open | closed | all")
        out = []
        for o in sorted(self.orders.values(), key=lambda x: x.seq):
            terminal = OrderStatus(o.status).is_terminal
            if status == "all" or (status == "open" and not terminal) or (status == "closed" and terminal):
                out.append(o.to_broker_order())
        return out

    def cancel_order(self, client_order_id: str) -> BrokerOrder | None:
        o = self.orders.get(client_order_id)
        if o is None:
            return None
        if o.is_pending:
            o.status = OrderStatus.CANCELED.value
            o.reason = "canceled by client"
            self._event("canceled", self.last_session, o.client_order_id, o.symbol)
            self._save()
        return o.to_broker_order()

    def account(self) -> BrokerAccount:
        equity: float | None = self.cash
        for sym, pos in self.positions_state.items():
            px = self._mark_price(sym)
            if px is None:
                equity = None
                break
            equity += pos["qty"] * px
        return BrokerAccount(cash=self.cash, equity=equity, buying_power=self.cash, status="ACTIVE",
                             raw={"book": self.book, "last_session": self.last_session, "simulated": True})

    def positions(self) -> list[BrokerPosition]:
        out = []
        for sym in sorted(self.positions_state):
            pos = self.positions_state[sym]
            px = self._mark_price(sym)
            out.append(BrokerPosition(symbol=sym, qty=pos["qty"], avg_entry_price=pos["avg_cost"],
                                      market_value=None if px is None else pos["qty"] * px, current_price=px))
        return out

    # ------------------------------------------------------------------------------------------
    # session processing
    # ------------------------------------------------------------------------------------------
    def process_session(self, session_date, panel: Panel) -> list[BrokerOrder]:
        """Advance to ``session_date``: apply its corporate actions, then fill eligible pending
        orders at its open. Idempotent for the last processed session; sessions must advance."""
        s = to_session(session_date)
        s_str = s.date().isoformat()
        if self.last_session is not None:
            if s_str == self.last_session:
                return []
            if s_str < self.last_session:
                raise ValueError(f"sessions must be processed in order: {s_str} < {self.last_session}")
        if s not in panel.dates:
            raise ValueError(f"session {s_str} is not in the panel; cannot simulate fills without data")
        p = panel.truncate(s)          # PIT: nothing after S is ever looked at
        changed: dict[str, _SimOrder] = {}

        self._apply_corporate_actions(s, s_str, p, changed)
        pending = [o for o in self.orders.values() if o.is_pending
                   and (o.eligible_after is None or o.eligible_after < s_str)]
        # Sells first (they free cash), then buys; ties in submission order -> deterministic.
        for o in sorted(pending, key=lambda x: (x.side != Side.SELL.value, x.seq)):
            if self._try_fill(o, s, s_str, p):
                changed[o.client_order_id] = o

        for sym in list(self.positions_state):
            px = self._last_close(sym, p)
            if px is not None:
                self.last_prices[sym] = px
        self.last_session = s_str
        self._save()
        return [o.to_broker_order() for o in sorted(changed.values(), key=lambda x: x.seq)]

    def _apply_corporate_actions(self, s: pd.Timestamp, s_str: str, p: Panel, changed: dict) -> None:
        for sym in sorted(self.positions_state):
            if sym not in p.symbols:
                continue
            ratio = float(p.split_ratio.at[s, sym])
            div = float(p.dividend.at[s, sym])
            pos = self.positions_state[sym]
            if math.isfinite(div) and div > 0 and self.credit_dividends:
                amount = pos["qty"] * div          # pre-split share count (schema basis)
                self.cash += amount
                self._event("dividend", s_str, None, sym, per_share=div, qty=pos["qty"], amount=amount)
            if math.isfinite(ratio) and ratio > 0 and abs(ratio - 1.0) > _EPS:
                before = pos["qty"]
                pos["qty"] = before * ratio
                pos["avg_cost"] = pos["avg_cost"] / ratio
                self._event("split", s_str, None, sym, ratio=ratio, qty_before=before, qty_after=pos["qty"])
        # Pending orders sized before the split are rescaled so an exit still closes the position.
        for o in self.orders.values():
            if not o.is_pending or o.symbol not in p.symbols:
                continue
            if o.eligible_after is not None and o.eligible_after >= s_str:
                continue
            ratio = float(p.split_ratio.at[s, o.symbol])
            if math.isfinite(ratio) and ratio > 0 and abs(ratio - 1.0) > _EPS:
                o.qty *= ratio
                o.split_factor *= ratio
                if o.limit_price is not None:
                    o.limit_price /= ratio
                self._event("order_split_adjusted", s_str, o.client_order_id, o.symbol, ratio=ratio, qty=o.qty)
                changed[o.client_order_id] = o

    def _last_close(self, sym: str, p: Panel) -> float | None:
        if sym not in p.symbols:
            return None
        valid = p.close[sym].dropna()
        return float(valid.iloc[-1]) if len(valid) else None

    def delisting_status(self, sym: str, s: pd.Timestamp, p: Panel) -> tuple[bool, float | None, str | None]:
        """(is_delisted, last raw close, last trade date) using data up to S only."""
        if sym not in p.symbols:
            return False, None, None
        closes = p.close[sym].loc[:s]
        valid = closes.dropna()
        if valid.empty:
            return False, None, None
        last_date = valid.index[-1]
        dates = p.dates
        since = int(((dates > last_date) & (dates <= s)).sum())
        return since >= self.delisting_missing_sessions, float(valid.iloc[-1]), last_date.date().isoformat()

    def _adv(self, sym: str, eligible_after: str | None, s: pd.Timestamp, p: Panel) -> float | None:
        end = pd.Timestamp(eligible_after) if eligible_after is not None else None
        if end is None or end not in p.dates:
            prev = p.dates[p.dates < s]
            if len(prev) == 0:
                return None
            end = prev[-1]
        dv = p.dollar_volume[sym].loc[:end].iloc[-ADV_WINDOW:]
        return float(dv.median()) if dv.notna().any() else None

    def _reject(self, o: _SimOrder, s_str: str, reason: str) -> None:
        o.status = OrderStatus.REJECTED.value
        o.reason = reason
        self._event("rejected", s_str, o.client_order_id, o.symbol, reason=reason)
        log_event(log, "sim order rejected", book=self.book, client_order_id=o.client_order_id, reason=reason)

    def _expire(self, o: _SimOrder, s_str: str, reason: str) -> None:
        o.status = OrderStatus.EXPIRED.value
        o.reason = reason
        self._event("expired", s_str, o.client_order_id, o.symbol, reason=reason)

    def _try_fill(self, o: _SimOrder, s: pd.Timestamp, s_str: str, p: Panel) -> bool:
        """Returns True when the order's state changed."""
        side = Side(o.side)
        has = o.symbol in p.symbols
        op = float(p.open.at[s, o.symbol]) if has else float("nan")
        cl = float(p.close.at[s, o.symbol]) if has else float("nan")
        settlement = False
        fill_time = time(9, 30)
        if math.isfinite(op) and op > 0:
            ref = op
        elif math.isfinite(cl) and cl > 0:
            ref, fill_time = cl, time(16, 0)          # open missing (data issue): tradesim transacts at close
        else:
            gone, last_close, last_date = self.delisting_status(o.symbol, s, p)
            if gone and last_close is not None:
                if side is Side.BUY:
                    self._reject(o, s_str, f"{o.symbol} has stopped trading (no bar since {last_date})")
                    return True
                ref = last_close * (1.0 + self.cost_model.delisting_return)
                settlement, fill_time = True, time(16, 0)
            else:
                o.missed_sessions += 1
                if o.missed_sessions >= self.max_pending_sessions:
                    self._expire(o, s_str, f"no bar for {o.symbol} in {o.missed_sessions} eligible sessions")
                    return True
                return False

        if o.order_type == "limit" and not settlement:
            marketable = ref <= o.limit_price + _EPS if side is Side.BUY else ref >= o.limit_price - _EPS
            if not marketable:
                self._expire(o, s_str, f"limit {o.limit_price:.4f} not marketable at reference {ref:.4f} "
                                       "(intraday fills are not simulated)")
                return True

        adv = self._adv(o.symbol, o.eligible_after, s, p)
        price = self.cost_model.fill_price(side, ref, adv)
        qty = o.qty
        commission = self.cost_model.commission(qty)
        if side is Side.BUY:
            need = qty * price + commission
            if need > self.cash + _EPS:
                self._reject(o, s_str, f"insufficient cash: need {need:.2f}, have {self.cash:.2f} (no margin)")
                return True
            pos = self.positions_state.get(o.symbol)
            if pos is None:
                self.positions_state[o.symbol] = {"qty": qty, "avg_cost": price}
            else:
                new_qty = pos["qty"] + qty
                pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + qty * price) / new_qty
                pos["qty"] = new_qty
            self.cash -= need
        else:
            pos = self.positions_state.get(o.symbol)
            held = pos["qty"] if pos else 0.0
            if qty > held + 1e-6:
                self._reject(o, s_str, f"insufficient position: sell {qty:g} > held {held:g} (no short selling)")
                return True
            qty = min(qty, held)
            self.cash += qty * price - commission
            pos["qty"] = held - qty
            if pos["qty"] <= _EPS:
                del self.positions_state[o.symbol]
                self.last_prices.pop(o.symbol, None)

        o.status = OrderStatus.FILLED.value
        o.filled_qty = qty
        o.filled_avg_price = price
        o.fill_session = s_str
        o.filled_at = _session_ts_utc(s, fill_time)
        o.commission = commission
        o.modeled_cost = abs(price - ref) * qty
        o.delisting_settlement = settlement
        self._event("filled", s_str, o.client_order_id, o.symbol, side=o.side, qty=qty, price=price, ref=ref,
                    adv=adv, commission=commission, delisting_settlement=settlement)
        return True

    # ------------------------------------------------------------------------------------------
    def _mark_price(self, sym: str) -> float | None:
        if self.price_source is not None and self.last_session is not None:
            panel = self.price_source() if callable(self.price_source) else self.price_source
            if sym in panel.symbols:
                valid = panel.close[sym].loc[:pd.Timestamp(self.last_session)].dropna()
                if len(valid):
                    return float(valid.iloc[-1])
        px = self.last_prices.get(sym)
        return float(px) if px is not None and np.isfinite(px) else None
