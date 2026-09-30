"""In-memory stand-in for the Alpaca PAPER broker + trade_updates stream (offline tests only).

Speaks the same objects as ``AlpacaPaperBroker`` (BrokerOrder/BrokerAccount/...), keeps its own
cash/positions like a real broker, refuses a reused client_order_id like Alpaca (HTTP 422), and
produces trade_updates events in Alpaca's JSON shape so the runner's stream path is exercised."""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from quantlab.config import PAPER_TRADING_BASE_URL
from quantlab.execution.alpaca_paper import PAPER_STREAM_URL, AlpacaPaperBroker
from quantlab.execution.broker import (
    BrokerAccount, BrokerError, BrokerOrder, BrokerPosition, BrokerUnavailable, OrderRequest, PaperBroker,
)

PAPER_ENV = {"TRADING_MODE": "PAPER", "LIVE_TRADING": "false", "ALPACA_PAPER_KEY_ID": "PKTESTKEY000",
             "ALPACA_PAPER_SECRET_KEY": "test-secret-value-never-logged"}


class FakeAlpacaBroker(PaperBroker):
    name = "alpaca_paper"
    trading_base_url = PAPER_TRADING_BASE_URL

    def __init__(self, sessions: list[date], cash: float = 100_000.0):
        self.sessions = sorted(sessions)
        self.cash = float(cash)
        self.pos: dict[str, list[float]] = {}          # symbol -> [qty, avg]
        self.orders: dict[str, dict[str, Any]] = {}    # client_order_id -> Alpaca order JSON
        self.submits: list[str] = []                   # every POST /v2/orders (client ids)
        self.available = True
        self.reject_next = False
        self.crash_after_submit = False                # store the order, then raise (crash simulation)
        self.calls = 0
        self.market_open = True
        self.auto_fill_market: float | None = None     # fill market orders at this price on submit
        self.listeners: list = []                      # trade_updates subscribers (see FakeStream)
        self.cancel_stuck = False                      # cancels stay 'pending_cancel' (never confirmed)

    # -- helpers --------------------------------------------------------------------------------
    def _check(self) -> None:
        self.calls += 1
        if not self.available:
            raise BrokerUnavailable("fake broker unavailable")

    def parse_order(self, raw: dict[str, Any]) -> BrokerOrder:
        return AlpacaPaperBroker._parse_order(self, raw)

    # -- PaperBroker ------------------------------------------------------------------------------
    def account(self) -> BrokerAccount:
        self._check()
        mv = sum(q * a for q, a in self.pos.values())
        return BrokerAccount(cash=self.cash, equity=self.cash + mv, buying_power=self.cash * 2, status="ACTIVE",
                             currency="USD", trading_blocked=False,
                             raw={"account_number": "PA1234567XY", "account_blocked": False})

    def positions(self) -> list[BrokerPosition]:
        self._check()
        return [BrokerPosition(symbol=s, qty=q, avg_entry_price=a, market_value=q * a, current_price=a,
                               raw={"unrealized_pl": "0"}) for s, (q, a) in self.pos.items() if q]

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        self._check()
        self.submits.append(request.client_order_id)
        if request.client_order_id in self.orders:
            raise BrokerError("alpaca_paper: HTTP 422 at /v2/orders: client_order_id must be unique")
        status = "rejected" if self.reject_next else "accepted"
        reason = "insufficient buying power" if status == "rejected" else None
        self.reject_next = False
        if status == "accepted" and request.side.value == "sell":
            # like Alpaca: shares reserved by open sell orders (e.g. a resting stop) are not available
            term = {"filled", "canceled", "expired", "rejected"}
            held = sum(float(o["qty"]) - float(o["filled_qty"]) for o in self.orders.values()
                       if o["symbol"] == request.symbol and o["side"] == "sell" and o["status"] not in term)
            have = self.pos.get(request.symbol, [0.0, 0.0])[0]
            if request.qty > have - held + 1e-9:
                status = "rejected"
                reason = f"insufficient qty available for order (requested: {request.qty:g}, available: {have - held:g})"
        o = {"id": str(uuid.uuid4()), "client_order_id": request.client_order_id, "symbol": request.symbol,
             "side": request.side.value, "qty": str(request.qty), "type": request.order_type,
             "time_in_force": request.time_in_force, "limit_price": request.limit_price,
             "stop_price": request.stop_price, "status": status,
             "filled_qty": "0", "filled_avg_price": None, "submitted_at": "2000-01-01T00:00:00Z", "filled_at": None,
             "reject_reason": reason}
        self.orders[request.client_order_id] = o
        if status == "accepted" and request.order_type == "market" and self.auto_fill_market is not None:
            for listener in list(self.listeners):
                listener(self.event(request.client_order_id, "new", "2026-01-02T15:00:00Z", status="new"))
            ev = self.fill(request.client_order_id, request.qty, self.auto_fill_market, "2026-01-02T15:00:01Z")
            for listener in list(self.listeners):
                listener(ev)
        if self.crash_after_submit:
            self.crash_after_submit = False
            raise RuntimeError("simulated crash after the broker accepted the order")
        return self.parse_order(o)

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        self._check()
        o = self.orders.get(client_order_id)
        return self.parse_order(o) if o else None

    def list_orders(self, status: str = "open") -> list[BrokerOrder]:
        self._check()
        term = {"filled", "canceled", "expired", "rejected"}
        return [self.parse_order(o) for o in self.orders.values()
                if status == "all" or (status == "open") == (o["status"] not in term)]

    def cancel_order(self, client_order_id: str) -> BrokerOrder | None:
        self._check()
        o = self.orders.get(client_order_id)
        if o and o["status"] not in ("filled", "canceled", "rejected", "expired"):
            o["status"] = "pending_cancel" if self.cancel_stuck else "canceled"
        return self.parse_order(o) if o else None

    def is_available(self) -> bool:
        return self.available

    def clock(self) -> dict[str, Any]:
        self._check()
        return {"timestamp": "t", "is_open": self.market_open, "next_open": "n", "next_close": "c"}

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        self._check()
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        return [{"date": str(d), "open": "09:30", "close": "16:00"} for d in self.sessions if s <= d <= e]

    def stream_url(self) -> str:
        return PAPER_STREAM_URL

    def stream_auth_message(self) -> dict[str, str]:
        return {"action": "auth", "key": "k", "secret": "s"}

    # -- broker-side events ---------------------------------------------------------------------
    def fill(self, client_order_id: str, qty: float, price: float, at: str) -> dict[str, Any]:
        """Fill ``qty`` more of the order at ``price`` and return the trade_updates event."""
        o = self.orders[client_order_id]
        prev_q = float(o["filled_qty"])
        prev_avg = float(o["filled_avg_price"] or 0.0)
        new_q = prev_q + qty
        o["filled_avg_price"] = str((prev_q * prev_avg + qty * price) / new_q)
        o["filled_qty"] = str(new_q)
        o["filled_at"] = at
        o["status"] = "filled" if new_q >= float(o["qty"]) - 1e-9 else "partially_filled"
        q, a = self.pos.get(o["symbol"], [0.0, 0.0])
        if o["side"] == "buy":
            self.pos[o["symbol"]] = [q + qty, (q * a + qty * price) / (q + qty)]
            self.cash -= qty * price
        else:
            self.pos[o["symbol"]] = [q - qty, a]
            self.cash += qty * price
            if self.pos[o["symbol"]][0] <= 1e-9:
                del self.pos[o["symbol"]]
        return {"event": "fill" if o["status"] == "filled" else "partial_fill", "execution_id": str(uuid.uuid4()),
                "order": dict(o), "price": str(price), "qty": str(qty), "timestamp": at,
                "position_qty": str(self.pos.get(o["symbol"], [0.0])[0])}

    def event(self, client_order_id: str, event: str, at: str, status: str | None = None) -> dict[str, Any]:
        o = self.orders[client_order_id]
        if status:
            o["status"] = status
        return {"event": event, "order": dict(o), "timestamp": at}


class FakeStream:
    """Stands in for TradeUpdateStream: the test drives connect/disconnect and pushes events."""

    def __init__(self, on_event, on_state):
        self.on_event, self.on_state = on_event, on_state
        self.connects = 0
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True
        self.connects += 1
        self.on_state("connected", f"connection #{self.connects}")

    def reconnect(self) -> None:
        self.on_state("disconnected", "ConnectionClosedError: simulated drop")
        self.connects += 1
        self.on_state("connected", f"connection #{self.connects}")

    def push(self, event: dict[str, Any]) -> None:
        self.on_event(event)

    def stop(self) -> None:
        self.stopped = True
        self.on_state("stopped", "")
