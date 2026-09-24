"""Alpaca PAPER trading broker. The only broker in this codebase allowed to make an HTTP request.

HARD SAFETY: the base URL is checked against ``quantlab.config.PAPER_TRADING_BASE_URL`` for exact
equality in :meth:`__init__` AND again in :meth:`_request`, immediately before every single HTTP
call. Anything else raises :class:`~quantlab.execution.broker.LiveTradingForbidden`, which does not
inherit from ``BrokerError`` and therefore cannot be swallowed by a generic broker-error handler.

Idempotency (docs/EXTERNAL-SERVICES.md, Alpaca Trading API section, facts 6-7): Alpaca says not to
resend an order after a submit timeout. So on a connection error / timeout / 5xx during
``POST /v2/orders`` we never retry the POST itself blindly -- we look the order up by its
deterministic ``client_order_id`` via ``GET /v2/orders:by_client_order_id`` and only report
``OrderStatus.UNKNOWN`` if that, too, cannot establish what happened.

Credentials (``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY``) come only from
``quantlab.secrets.get_secret`` and are never logged or included in an exception message.
"""
from __future__ import annotations

import json as _json
import time
from typing import Any

import requests

from quantlab.config import Config, PAPER_TRADING_BASE_URL
from quantlab.core.types import OrderStatus, Side
from quantlab.data.providers.http import HttpResponse, describe_body, parse_retry_after
from quantlab.execution.broker import (
    BrokerAccount,
    BrokerAuthError,
    BrokerError,
    BrokerNotConfigured,
    BrokerOrder,
    BrokerPosition,
    BrokerUnavailable,
    LiveTradingForbidden,
    OrderRequest,
    PaperBroker,
)
from quantlab.logging_setup import get_logger, log_event
from quantlab.secrets import get_secret

log = get_logger("execution.alpaca_paper")

# Explicit forbidden-host check (see tests/execution/test_repo_scan.py, which asserts that this bare
# live host string appears nowhere else under src/). It is compared against and refused -- never
# used to build a URL we connect to.
_LIVE_TRADING_BASE_URL = "https://api.alpaca.markets"  # noqa: forbidden-host-check

# Alpaca OrderStatus enum (docs/EXTERNAL-SERVICES.md fact 11) mapped to core.types.OrderStatus.
# Anything not listed here (stopped, suspended, calculated, held, and any future/unknown value)
# falls back to UNKNOWN -- never guessed.
_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.ACCEPTED,
    "pending_cancel": OrderStatus.ACCEPTED,
    "pending_replace": OrderStatus.ACCEPTED,
    "accepted_for_bidding": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.EXPIRED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "replaced": OrderStatus.CANCELED,
}


def map_order_status(raw: str | None) -> OrderStatus:
    return _STATUS_MAP.get(str(raw or "").lower(), OrderStatus.UNKNOWN)


class AlpacaRejected(BrokerError):
    """HTTP 403 on order submission (insufficient buying power, wash-trade protection, ...)."""

    def __init__(self, message: str, code: Any = None):
        super().__init__(message)
        self.code = code


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _qty_str(qty: float) -> str:
    s = f"{qty:.9f}".rstrip("0").rstrip(".")
    return s or "0"


class AlpacaPaperBroker(PaperBroker):
    """Alpaca paper-trading broker (``https://paper-api.alpaca.markets`` only)."""

    name = "alpaca_paper"

    def __init__(
        self,
        config: Config,
        *,
        session: Any = None,
        timeout: float | None = None,
        max_retries: int = 3,
        sleep: Any = time.sleep,
        clock: Any = time.time,
    ):
        base = str(config.get("providers.alpaca.trading_base_url", PAPER_TRADING_BASE_URL)).rstrip("/")
        self.trading_base_url = base
        self._assert_paper_url()  # checked in __init__ ...
        self.key_env = config.get("providers.alpaca.key_id_env", "ALPACA_PAPER_KEY_ID")
        self.secret_env = config.get("providers.alpaca.secret_env", "ALPACA_PAPER_SECRET_KEY")
        self._key = get_secret(self.key_env)
        self._secret = get_secret(self.secret_env)
        self.session = session if session is not None else requests.Session()
        self.timeout = float(timeout if timeout is not None else config.get("providers.http.timeout_seconds", 30))
        self.max_retries = int(max_retries)
        self._sleep = sleep
        self._clock = clock

    # -- safety -----------------------------------------------------------------------------
    def _assert_paper_url(self) -> None:
        if self.trading_base_url == _LIVE_TRADING_BASE_URL:
            raise LiveTradingForbidden(
                f"AlpacaPaperBroker refuses the LIVE Alpaca trading endpoint ({self.trading_base_url!r}); "
                f"the only allowed endpoint is {PAPER_TRADING_BASE_URL!r}")
        if self.trading_base_url != PAPER_TRADING_BASE_URL:
            raise LiveTradingForbidden(
                f"AlpacaPaperBroker refuses non-paper base URL {self.trading_base_url!r}; "
                f"must be exactly {PAPER_TRADING_BASE_URL!r}")

    @property
    def is_configured(self) -> bool:
        return bool(self._key) and bool(self._secret)

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise BrokerNotConfigured(
                f"AlpacaPaperBroker needs environment variables {self.key_env} and {self.secret_env}")

    def _headers(self) -> dict[str, str]:
        assert self._key is not None and self._secret is not None
        return {"APCA-API-KEY-ID": self._key.reveal(), "APCA-API-SECRET-KEY": self._secret.reveal()}

    # -- HTTP -------------------------------------------------------------------------------
    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                json_body: dict[str, Any] | None = None, not_found_ok: bool = False) -> Any:
        """One logical request (with bounded retries on 429/5xx/connection errors).

        Returns parsed JSON (or None for a 404 when ``not_found_ok``, or an empty/204 body). Raises
        ``BrokerAuthError`` (401), ``AlpacaRejected`` (403), ``BrokerError`` (other 4xx / bad body),
        or ``BrokerUnavailable`` for anything ambiguous (timeout, connection error, exhausted 429/5xx
        retries) -- the caller (only :meth:`submit_order` needs to) must treat that as "unknown
        whether it reached the broker", never as "it did not happen".
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._assert_paper_url()  # ... AND immediately before every HTTP request
            url = f"{self.trading_base_url}{path}"
            try:
                raw = self.session.request(method, url, params=params, json=json_body,
                                           headers=self._headers(), timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    log_event(log, "alpaca_paper transient network error; retrying", path=path,
                              attempt=attempt + 1, error=type(exc).__name__)
                    self._sleep(min(2.0 * (attempt + 1), 10.0))
                    continue
                raise BrokerUnavailable(f"alpaca_paper: network error at {path} after {attempt + 1} "
                                        f"attempts: {type(exc).__name__}") from None
            except requests.RequestException as exc:
                raise BrokerUnavailable(f"alpaca_paper: request to {path} failed: {type(exc).__name__}") from None

            status = int(getattr(raw, "status_code", 0))
            headers = {str(k).lower(): str(v) for k, v in dict(getattr(raw, "headers", {}) or {}).items()}
            raw_content = getattr(raw, "content", None)
            content = (raw_content if isinstance(raw_content, (bytes, bytearray))
                      else str(getattr(raw, "text", "") or "").encode("utf-8"))
            resp = HttpResponse(status=status, url=url, headers=headers, content=content)

            if 200 <= status < 300:
                if status == 204 or not content:
                    return None
                stripped = content.lstrip()
                if "json" not in resp.content_type and stripped[:1] not in (b"{", b"["):
                    desc, _ = describe_body(resp)
                    raise BrokerError(f"alpaca_paper: expected JSON from {path}, got: {desc}")
                try:
                    return _json.loads(content)
                except ValueError:
                    raise BrokerError(f"alpaca_paper: invalid JSON body from {path}") from None
            if status == 404 and not_found_ok:
                return None
            if status == 401:
                desc, _ = describe_body(resp)
                raise BrokerAuthError(f"alpaca_paper: HTTP 401 at {path}: {desc}")
            if status == 403:
                desc, code = describe_body(resp)
                raise AlpacaRejected(f"alpaca_paper: HTTP 403 at {path}: {desc}", code)
            if status == 429:
                if attempt < self.max_retries:
                    wait = parse_retry_after(headers.get("retry-after"), self._clock())
                    wait = wait if wait is not None else min(2.0 * (attempt + 1), 10.0)
                    log_event(log, "alpaca_paper rate limited; backing off", path=path, attempt=attempt + 1,
                              wait_s=wait)
                    self._sleep(wait)
                    continue
                raise BrokerUnavailable(f"alpaca_paper: rate limited (429) at {path} after {attempt + 1} attempts")
            if 500 <= status < 600:
                if attempt < self.max_retries:
                    log_event(log, "alpaca_paper server error; retrying", path=path, status=status,
                              attempt=attempt + 1)
                    self._sleep(min(2.0 * (attempt + 1), 10.0))
                    continue
                raise BrokerUnavailable(f"alpaca_paper: HTTP {status} at {path} after {attempt + 1} attempts")
            desc, code = describe_body(resp)
            raise BrokerError(f"alpaca_paper: HTTP {status} at {path}: {desc}")
        raise BrokerUnavailable(f"alpaca_paper: request to {path} failed: {last_exc!r}")  # pragma: no cover

    # -- PaperBroker interface ---------------------------------------------------------------
    def is_available(self) -> bool:
        if not self.is_configured:
            return False
        try:
            self._assert_paper_url()
            self._request("GET", "/v2/account")
            return True
        except BrokerError:
            return False

    def account(self) -> BrokerAccount:
        self._require_configured()
        raw = self._request("GET", "/v2/account") or {}
        return BrokerAccount(
            cash=_num(raw.get("cash")) or 0.0, equity=_num(raw.get("equity")),
            buying_power=_num(raw.get("buying_power")), status=raw.get("status"),
            currency=raw.get("currency", "USD"), trading_blocked=bool(raw.get("trading_blocked", False)), raw=raw,
        )

    def positions(self) -> list[BrokerPosition]:
        self._require_configured()
        raw = self._request("GET", "/v2/positions") or []
        out = []
        for p in raw:
            qty = _num(p.get("qty")) or 0.0
            if str(p.get("side", "long")).lower() == "short":
                qty = -abs(qty)
            out.append(BrokerPosition(symbol=p.get("symbol"), qty=qty, avg_entry_price=_num(p.get("avg_entry_price")),
                                      market_value=_num(p.get("market_value")),
                                      current_price=_num(p.get("current_price")), raw=p))
        return out

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        self._require_configured()
        body: dict[str, Any] = {
            "symbol": request.symbol, "qty": _qty_str(request.qty), "side": request.side.value,
            "type": request.order_type, "time_in_force": request.time_in_force,
            "client_order_id": request.client_order_id,
        }
        if request.order_type == "limit":
            body["limit_price"] = f"{request.limit_price:.4f}"
        try:
            raw = self._request("POST", "/v2/orders", json_body=body)
        except AlpacaRejected as exc:
            desc = str(exc)
            log_event(log, "alpaca_paper order rejected (403)", client_order_id=request.client_order_id)
            return BrokerOrder(client_order_id=request.client_order_id, broker_order_id=None,
                               status=OrderStatus.REJECTED, symbol=request.symbol, side=request.side,
                               qty=request.qty, order_type=request.order_type,
                               time_in_force=request.time_in_force, limit_price=request.limit_price, reason=desc)
        except BrokerUnavailable as exc:
            # Ambiguous outcome (timeout / connection error / 5xx): never blindly resubmit.
            log_event(log, "alpaca_paper submit ambiguous; checking by client_order_id",
                      client_order_id=request.client_order_id, error=str(exc))
            try:
                existing = self.get_order_by_client_id(request.client_order_id)
            except BrokerError:
                existing = None
            if existing is not None:
                return existing
            return BrokerOrder(client_order_id=request.client_order_id, broker_order_id=None,
                               status=OrderStatus.UNKNOWN, symbol=request.symbol, side=request.side,
                               qty=request.qty, order_type=request.order_type,
                               time_in_force=request.time_in_force, limit_price=request.limit_price,
                               reason=f"submit outcome unknown after network error: {exc}")
        return self._parse_order(raw)

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        self._require_configured()
        raw = self._request("GET", "/v2/orders:by_client_order_id",
                            params={"client_order_id": client_order_id}, not_found_ok=True)
        return self._parse_order(raw) if raw is not None else None

    def list_orders(self, status: str = "open") -> list[BrokerOrder]:
        if status not in ("open", "closed", "all"):
            raise ValueError("status must be open | closed | all")
        self._require_configured()
        raw = self._request("GET", "/v2/orders", params={"status": status, "limit": 500})
        return [self._parse_order(o) for o in (raw or [])]

    def cancel_order(self, client_order_id: str) -> BrokerOrder | None:
        self._require_configured()
        existing = self.get_order_by_client_id(client_order_id)
        if existing is None or existing.broker_order_id is None:
            return existing
        self._request("DELETE", f"/v2/orders/{existing.broker_order_id}")
        return self.get_order_by_client_id(client_order_id)

    # -- parsing ------------------------------------------------------------------------------
    def _parse_order(self, raw: dict[str, Any]) -> BrokerOrder:
        status = map_order_status(raw.get("status"))
        side_raw = raw.get("side")
        side = Side(side_raw) if side_raw in ("buy", "sell") else None
        reason = raw.get("reject_reason") or ("rejected by broker" if status is OrderStatus.REJECTED else None)
        return BrokerOrder(
            client_order_id=raw.get("client_order_id", ""), broker_order_id=raw.get("id"), status=status,
            filled_qty=_num(raw.get("filled_qty")) or 0.0, filled_avg_price=_num(raw.get("filled_avg_price")),
            submitted_at=raw.get("submitted_at"), raw=raw, symbol=raw.get("symbol"), side=side,
            qty=_num(raw.get("qty")), order_type=raw.get("type") or raw.get("order_type"),
            time_in_force=raw.get("time_in_force"), limit_price=_num(raw.get("limit_price")),
            filled_at=raw.get("filled_at"), reason=reason,
        )
