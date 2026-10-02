"""In-memory paper broker for OPT tests: records every OrderRequest, fills only when the test says
so (never on its own), keeps option positions with Alpaca's asset_class and cash x multiplier."""
from __future__ import annotations

import uuid
from typing import Any

from quantlab.core.types import OrderStatus
from quantlab.execution.broker import BrokerAccount, BrokerOrder, BrokerPosition, OrderRequest, PaperBroker


class FakeOptionsBroker(PaperBroker):
    name = "alpaca_paper"

    def __init__(self, cash: float = 100_000.0, multiplier: float = 100.0):
        self.cash = cash
        self.mult = multiplier
        self.requests: list[OrderRequest] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.pos: dict[str, float] = {}
        self.stock_pos: dict[str, float] = {}

    def account(self) -> BrokerAccount:
        return BrokerAccount(cash=self.cash, equity=self.cash, status="ACTIVE")

    def positions(self) -> list[BrokerPosition]:
        out = [BrokerPosition(symbol=s, qty=q, raw={"asset_class": "us_option"}) for s, q in self.pos.items() if q]
        out += [BrokerPosition(symbol=s, qty=q, raw={"asset_class": "us_equity"}) for s, q in self.stock_pos.items() if q]
        return out

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        self.requests.append(request)
        if request.client_order_id in self.orders:
            return self._bo(self.orders[request.client_order_id])
        o = {"id": str(uuid.uuid4()), "cid": request.client_order_id, "symbol": request.symbol, "side": request.side,
             "qty": request.qty, "type": request.order_type, "tif": request.time_in_force,
             "limit": request.limit_price, "status": OrderStatus.ACCEPTED, "filled": 0.0, "avg": None}
        self.orders[request.client_order_id] = o
        return self._bo(o)

    def fill(self, cid: str, qty: float, price: float, final: OrderStatus | None = None) -> None:
        o = self.orders[cid]
        prev = o["filled"] * (o["avg"] or 0.0)
        o["filled"] += qty
        o["avg"] = (prev + qty * price) / o["filled"]
        sign = 1 if o["side"].value == "buy" else -1
        self.pos[o["symbol"]] = self.pos.get(o["symbol"], 0.0) + sign * qty
        self.cash -= sign * qty * price * self.mult
        o["status"] = final or (OrderStatus.FILLED if o["filled"] >= o["qty"] - 1e-9 else OrderStatus.PARTIALLY_FILLED)

    def end_day(self, cid: str) -> None:
        o = self.orders[cid]
        if o["status"] not in (OrderStatus.FILLED,):
            o["status"] = OrderStatus.EXPIRED

    def _bo(self, o: dict[str, Any]) -> BrokerOrder:
        return BrokerOrder(client_order_id=o["cid"], broker_order_id=o["id"], status=o["status"], filled_qty=o["filled"],
                           filled_avg_price=o["avg"], symbol=o["symbol"], side=o["side"], qty=o["qty"],
                           order_type=o["type"], time_in_force=o["tif"], limit_price=o["limit"], raw={})

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        o = self.orders.get(client_order_id)
        return self._bo(o) if o else None

    def list_orders(self, status: str = "open") -> list[BrokerOrder]:
        return [self._bo(o) for o in self.orders.values()
                if status == "all" or (status == "open") == (not o["status"].is_terminal)]

    def cancel_order(self, client_order_id: str) -> BrokerOrder | None:
        o = self.orders.get(client_order_id)
        if o and not o["status"].is_terminal:
            o["status"] = OrderStatus.CANCELED
        return self._bo(o) if o else None

    def is_available(self) -> bool:
        return True
