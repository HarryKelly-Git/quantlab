"""AlpacaPaperBroker against a FAKE HTTP session (no network): submit, the idempotent
by_client_order_id recovery path after a submit timeout, Alpaca status mapping, an HTML 401 body,
insufficient buying power (403), and the hard live-URL refusal."""
from __future__ import annotations

import json

import pytest
import requests

from quantlab.config import load_config
from quantlab.core.types import OrderStatus, Side
from quantlab.execution.alpaca_paper import AlpacaPaperBroker, map_order_status
from quantlab.execution.broker import (
    BrokerAuthError,
    BrokerNotConfigured,
    LiveTradingForbidden,
    OrderRequest,
)

class FakeResponse:
    def __init__(self, status_code: int, body, headers: dict | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"content-type": "application/json"}
        if isinstance(body, (dict, list)):
            self.content = json.dumps(body).encode("utf-8")
        else:
            self.content = str(body).encode("utf-8")

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


class FakeSession:
    """Replays a scripted sequence of responses (or raises) per call, and records every call."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_config(overrides: dict | None = None, env=None):
    cfg = load_config(overrides={"providers": {"alpaca": {"key_id_env": "TEST_ALPACA_KEY",
                                                          "secret_env": "TEST_ALPACA_SECRET"}}, **(overrides or {})})
    return cfg


@pytest.fixture(autouse=True)
def alpaca_env(monkeypatch):
    monkeypatch.setenv("TEST_ALPACA_KEY", "key123")
    monkeypatch.setenv("TEST_ALPACA_SECRET", "secret456")
    yield


def make_broker(session: FakeSession, sleep=lambda s: None) -> AlpacaPaperBroker:
    return AlpacaPaperBroker(make_config(), session=session, max_retries=2, sleep=sleep, clock=lambda: 0.0)


ORDER_JSON = {
    "id": "broker-order-1", "client_order_id": "ql-bot-abc123", "status": "accepted",
    "filled_qty": "0", "filled_avg_price": None, "submitted_at": "2024-01-02T14:30:00Z", "symbol": "AAA",
    "side": "buy", "qty": "10", "type": "market", "time_in_force": "opg", "limit_price": None, "filled_at": None,
}


def _req(client_order_id="ql-bot-abc123") -> OrderRequest:
    return OrderRequest(client_order_id=client_order_id, symbol="AAA", side=Side.BUY, qty=10,
                        order_type="market", time_in_force="opg")


# -- live URL refusal -----------------------------------------------------------------------------
def test_construct_with_live_url_raises_live_trading_forbidden():
    cfg = make_config({"providers": {"alpaca": {"trading_base_url": "https://api.alpaca.markets"}}})
    with pytest.raises(LiveTradingForbidden):
        AlpacaPaperBroker(cfg, session=FakeSession([]))


def test_construct_with_arbitrary_non_paper_url_raises():
    cfg = make_config({"providers": {"alpaca": {"trading_base_url": "https://evil.example.com"}}})
    with pytest.raises(LiveTradingForbidden):
        AlpacaPaperBroker(cfg, session=FakeSession([]))


def test_paper_url_constructs_fine():
    broker = make_broker(FakeSession([]))
    assert broker.trading_base_url == "https://paper-api.alpaca.markets"


# -- submit_order happy path + status mapping ------------------------------------------------------
def test_submit_order_success_maps_status():
    session = FakeSession([FakeResponse(200, ORDER_JSON)])
    broker = make_broker(session)
    bo = broker.submit_order(_req())
    assert bo.status is OrderStatus.ACCEPTED
    assert bo.broker_order_id == "broker-order-1"
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"].startswith("https://paper-api.alpaca.markets")


@pytest.mark.parametrize("raw_status,expected", [
    ("new", OrderStatus.NEW), ("accepted", OrderStatus.ACCEPTED), ("pending_new", OrderStatus.ACCEPTED),
    ("partially_filled", OrderStatus.PARTIALLY_FILLED), ("filled", OrderStatus.FILLED),
    ("canceled", OrderStatus.CANCELED), ("expired", OrderStatus.EXPIRED), ("done_for_day", OrderStatus.EXPIRED),
    ("rejected", OrderStatus.REJECTED), ("replaced", OrderStatus.CANCELED),
    ("stopped", OrderStatus.UNKNOWN), ("suspended", OrderStatus.UNKNOWN), ("calculated", OrderStatus.UNKNOWN),
    ("held", OrderStatus.UNKNOWN), ("something_alpaca_invents_later", OrderStatus.UNKNOWN),
])
def test_status_mapping_never_guesses_unknown_values(raw_status, expected):
    assert map_order_status(raw_status) is expected


# -- idempotency: timeout on submit -> GET by_client_order_id, never blind resubmit ------------------
def test_submit_timeout_then_found_by_client_id_returns_that_order():
    session = FakeSession([
        requests.Timeout("boom"),  # POST /v2/orders times out
        requests.Timeout("boom"),  # retry also times out (max_retries=2 -> 3 attempts total... trimmed below)
        FakeResponse(200, ORDER_JSON),  # GET .../orders:by_client_order_id succeeds
    ])
    broker = AlpacaPaperBroker(make_config(), session=session, max_retries=1, sleep=lambda s: None, clock=lambda: 0.0)
    bo = broker.submit_order(_req())
    assert bo.status is OrderStatus.ACCEPTED
    assert bo.broker_order_id == "broker-order-1"
    methods = [c["method"] for c in session.calls]
    assert methods == ["POST", "POST", "GET"]  # 1 initial + 1 retry, then the recovery GET


def test_submit_timeout_and_broker_still_does_not_know_it_returns_unknown():
    session = FakeSession([
        requests.Timeout("boom"), requests.Timeout("boom"),
        FakeResponse(404, {"message": "order not found"}),  # by_client_order_id: still unknown
    ])
    broker = AlpacaPaperBroker(make_config(), session=session, max_retries=1, sleep=lambda s: None, clock=lambda: 0.0)
    bo = broker.submit_order(_req())
    assert bo.status is OrderStatus.UNKNOWN
    assert "unknown" in bo.reason.lower()


# -- 403 insufficient buying power ------------------------------------------------------------------
def test_submit_insufficient_buying_power_maps_to_rejected_with_reason():
    session = FakeSession([FakeResponse(403, {"code": 40310000, "message": "insufficient buying power"})])
    broker = make_broker(session)
    bo = broker.submit_order(_req())
    assert bo.status is OrderStatus.REJECTED
    assert "insufficient buying power" in bo.reason


# -- 401 with an HTML body (nginx, per docs/EXTERNAL-SERVICES.md) ------------------------------------
def test_account_401_html_body_is_not_parsed_as_json():
    html = "<html><head><title>401 Authorization Required</title></head><body>nginx</body></html>"
    session = FakeSession([FakeResponse(401, html, headers={"content-type": "text/html"})])
    broker = make_broker(session)
    with pytest.raises(BrokerAuthError) as exc_info:
        broker.account()
    assert "401 Authorization Required" in str(exc_info.value)


def test_is_available_false_on_401_and_true_on_200():
    broker = make_broker(FakeSession([FakeResponse(401, "<html><title>401</title></html>",
                                                   headers={"content-type": "text/html"})]))
    assert broker.is_available() is False

    broker2 = make_broker(FakeSession([FakeResponse(200, {"cash": "1000", "equity": "1000", "status": "ACTIVE"})]))
    assert broker2.is_available() is True


def test_not_configured_without_env_vars(monkeypatch):
    monkeypatch.delenv("TEST_ALPACA_KEY", raising=False)
    monkeypatch.delenv("TEST_ALPACA_SECRET", raising=False)
    broker = make_broker(FakeSession([]))
    assert broker.is_configured is False
    assert broker.is_available() is False
    with pytest.raises(BrokerNotConfigured):
        broker.submit_order(_req())


def test_account_and_positions_parse_alpaca_fields():
    session = FakeSession([
        FakeResponse(200, {"cash": "98765.43", "equity": "101000.00", "buying_power": "98765.43",
                          "status": "ACTIVE", "trading_blocked": False}),
        FakeResponse(200, [{"symbol": "AAA", "qty": "10", "side": "long", "avg_entry_price": "100.0",
                           "market_value": "1010.0", "current_price": "101.0"},
                          {"symbol": "BBB", "qty": "5", "side": "short", "avg_entry_price": "50.0",
                           "market_value": "-255.0", "current_price": "51.0"}]),
    ])
    broker = make_broker(session)
    account = broker.account()
    assert account.cash == pytest.approx(98765.43)
    assert account.trading_blocked is False
    positions = broker.positions()
    by_symbol = {p.symbol: p for p in positions}
    assert by_symbol["AAA"].qty == pytest.approx(10.0)
    assert by_symbol["BBB"].qty == pytest.approx(-5.0)  # short -> negative


def test_credentials_never_appear_in_a_raised_error_message():
    html = "<html><title>401</title></html>"
    session = FakeSession([FakeResponse(401, html, headers={"content-type": "text/html"})])
    broker = make_broker(session)
    with pytest.raises(BrokerAuthError) as exc_info:
        broker.account()
    assert "secret456" not in str(exc_info.value)
    assert "key123" not in str(exc_info.value)
