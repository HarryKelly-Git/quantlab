"""Broker abstraction shared by the simulated broker and the Alpaca PAPER broker.

Why an abstraction: the internal ledger (``execution/ledger.py``) is the source of truth for each
book, and a broker is only an *external view* that is polled and reconciled against it. Both
implementations speak the same small vocabulary, so the execution service, reconciliation and the
daily pipeline never branch on which broker is configured.

Safety: there is no live broker. :class:`LiveTradingForbidden` is raised whenever anything tries to
reach a non-paper endpoint. It deliberately does NOT inherit from :class:`BrokerError`, so generic
``except BrokerError`` handlers can never swallow it.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from quantlab.core.types import OrderStatus, Side

ORDER_TYPES = frozenset({"market", "limit"})
TIME_IN_FORCE = frozenset({"opg", "day"})
ORDER_LIST_STATUSES = frozenset({"open", "closed", "all"})
MAX_CLIENT_ORDER_ID_LEN = 48          # Alpaca allows 128; we keep ids short and readable
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_QTY_EPS = 1e-9


class LiveTradingForbidden(RuntimeError):
    """Raised when code attempts to use any broker endpoint other than the paper endpoint.

    Not a BrokerError on purpose: it must propagate and stop the process, never be handled as an
    ordinary broker failure."""


class BrokerError(RuntimeError):
    """A broker operation failed. The caller must treat broker state as unknown."""


class BrokerUnavailable(BrokerError):
    """The broker cannot be reached / did not answer usefully (network, 5xx, rate limit)."""


class BrokerNotConfigured(BrokerUnavailable):
    """Credentials or settings are missing (e.g. no paper API key in the environment)."""


class BrokerAuthError(BrokerUnavailable):
    """The broker rejected our credentials (HTTP 401). Systemic: the pipeline should pause."""


def is_whole(qty: float) -> bool:
    return abs(qty - round(qty)) <= _QTY_EPS


@dataclass(frozen=True)
class OrderRequest:
    """One order as QuantLab intends it.

    ``decision_session`` is the session whose close produced the decision (ISO date). The simulated
    broker fills only at a session strictly AFTER it (signal at close D -> fill at open D+1). It is
    metadata for real brokers (Alpaca uses the wall clock) but is always recorded for audit.
    """

    client_order_id: str
    symbol: str
    side: Side
    qty: float
    order_type: str = "market"
    time_in_force: str = "opg"
    limit_price: float | None = None
    decision_session: str | None = None

    def __post_init__(self) -> None:
        if not self.client_order_id or len(self.client_order_id) > MAX_CLIENT_ORDER_ID_LEN:
            raise ValueError(f"client_order_id must be 1..{MAX_CLIENT_ORDER_ID_LEN} chars")
        if not _CLIENT_ID_RE.match(self.client_order_id):
            raise ValueError(f"client_order_id has invalid characters: {self.client_order_id!r}")
        if not self.symbol or self.symbol != self.symbol.strip().upper():
            raise ValueError(f"symbol must be a non-empty upper-case ticker: {self.symbol!r}")
        if not isinstance(self.side, Side):
            raise ValueError(f"side must be a Side enum, got {self.side!r}")
        if not (isinstance(self.qty, (int, float)) and math.isfinite(self.qty) and self.qty > 0):
            raise ValueError(f"qty must be a positive finite number, got {self.qty!r}")
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type must be one of {sorted(ORDER_TYPES)}")
        if self.time_in_force not in TIME_IN_FORCE:
            raise ValueError(f"time_in_force must be one of {sorted(TIME_IN_FORCE)}")
        if self.order_type == "limit":
            if self.limit_price is None or not math.isfinite(self.limit_price) or self.limit_price <= 0:
                raise ValueError("limit orders need a positive limit_price")
        elif self.limit_price is not None:
            raise ValueError("market orders must not carry a limit_price")
        # Alpaca: fractional orders are DAY-only (no opg). Enforced here so the simulated and the
        # Alpaca book accept exactly the same orders.
        if self.time_in_force == "opg" and not is_whole(self.qty):
            raise ValueError("opg orders must be whole shares (fractional orders are DAY-only)")


@dataclass
class BrokerOrder:
    """The broker's view of one order. ``filled_qty``/``filled_avg_price`` are cumulative."""

    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    symbol: str | None = None
    side: Side | None = None
    qty: float | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    limit_price: float | None = None
    filled_at: str | None = None
    reason: str | None = None                 # rejection / expiry / unknown-state explanation
    commission: float = 0.0                   # cumulative commission (simulated broker only)
    modeled_cost: float = 0.0                 # cumulative modeled spread+slippage (simulated broker only)
    fill_session: str | None = None           # session date of the (last) fill
    delisting_settlement: bool = False        # sim: settled at last close x (1 + delisting_return)

    @property
    def is_open(self) -> bool:
        return not self.status.is_terminal and self.status is not OrderStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["side"] = self.side.value if self.side is not None else None
        return d


@dataclass
class BrokerPosition:
    symbol: str
    qty: float                                # signed: long > 0
    avg_entry_price: float | None = None
    market_value: float | None = None
    current_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerAccount:
    cash: float
    equity: float | None = None
    buying_power: float | None = None
    status: str | None = None
    currency: str = "USD"
    trading_blocked: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class PaperBroker(ABC):
    """Paper broker interface. Implementations must never contact a live-trading endpoint."""

    name: str = "abstract"

    @abstractmethod
    def account(self) -> BrokerAccount:
        """Cash/equity as the broker sees it. Raises BrokerError if unavailable."""

    @abstractmethod
    def positions(self) -> list[BrokerPosition]:
        """Open positions as the broker sees them. Raises BrokerError if unavailable."""

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        """Submit once. Must be idempotent on ``client_order_id``: re-submitting an id the broker
        already knows returns that order instead of creating a second one. Ambiguous outcomes
        (timeout / connection loss / 5xx) come back as ``OrderStatus.UNKNOWN`` -- never guessed."""

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """The order, or None when the broker says it does not exist. Raises BrokerError when the
        broker could not answer."""

    @abstractmethod
    def list_orders(self, status: str = "open") -> list[BrokerOrder]:
        """Orders with status 'open' | 'closed' | 'all'."""

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> BrokerOrder | None:
        """Request cancellation; returns the order's state afterwards (None if unknown id)."""

    @abstractmethod
    def is_available(self) -> bool:
        """True when the broker is configured, reachable and allowed to trade."""

    def process_session(self, session_date, panel) -> list[BrokerOrder]:
        """Advance a simulated broker to ``session_date`` (fills at its open). Real brokers fill on
        their own clock, so the default is a no-op."""
        return []
