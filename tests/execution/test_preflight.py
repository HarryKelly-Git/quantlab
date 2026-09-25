"""Paper preflight: refuses unless the environment is EXPLICITLY paper (TRADING_MODE=PAPER,
LIVE_TRADING=false), the endpoint is the paper URL and credentials exist; records every attempt;
never stores a credential. The runner refuses to start when the preflight fails."""
from __future__ import annotations

import sqlite3

import pytest

from quantlab.config import load_config
from quantlab.db.database import open_db
from quantlab.execution.alpaca_paper import AlpacaPaperBroker
from quantlab.execution.broker import LiveTradingForbidden
from quantlab.execution.preflight import mask_account, paper_mode_checks, run_preflight
from quantlab.execution.trade_stream import TradeUpdateStream

from ..pipeline.fake_alpaca import PAPER_ENV, FakeAlpacaBroker


def _cfg(**alpaca):
    return load_config(overrides={"providers": {"alpaca": alpaca}} if alpaca else None)


@pytest.mark.parametrize("override, needle", [
    ({"TRADING_MODE": "LIVE"}, "TRADING_MODE='LIVE'"),
    ({"TRADING_MODE": "paper"}, "TRADING_MODE='paper'"),      # exact value required, no case folding
    ({"LIVE_TRADING": "true"}, "LIVE_TRADING='true'"),
    ({"LIVE_TRADING": None}, "LIVE_TRADING=None"),             # missing is a refusal, never a default
    ({"TRADING_MODE": None}, "TRADING_MODE=None"),
])
def test_refuses_unless_explicitly_paper_and_never_calls_the_broker(override, needle):
    env = {k: v for k, v in {**PAPER_ENV, **override}.items() if v is not None}
    db = open_db(":memory:")
    broker = FakeAlpacaBroker([])
    res = run_preflight(_cfg(), db, broker, env=env)
    assert res.ok is False and needle in res.reason
    assert broker.calls == 0                                 # refused before any network call
    row = db.fetchone("SELECT * FROM paper_preflights")
    assert row["ok"] == 0 and row["reason"] == res.reason


def test_missing_credentials_refused_by_name_only():
    env = {"TRADING_MODE": "PAPER", "LIVE_TRADING": "false"}
    res = run_preflight(_cfg(), open_db(":memory:"), FakeAlpacaBroker([]), env=env)
    assert not res.ok and "ALPACA_PAPER_KEY_ID" in res.reason and "ALPACA_PAPER_SECRET_KEY" in res.reason


def test_non_paper_endpoint_is_refused_everywhere():
    live = "https://api.alpaca.markets"  # noqa: forbidden-host-check (test of the refusal)
    checks = {c.name: c for c in paper_mode_checks(_cfg(trading_base_url=live), PAPER_ENV)}
    assert checks["config.endpoint"].ok is False
    res = run_preflight(_cfg(trading_base_url=live), open_db(":memory:"), FakeAlpacaBroker([]), env=PAPER_ENV)
    assert not res.ok and "trading endpoint" in res.reason
    with pytest.raises(Exception):   # config validation or the broker client refuses a live URL
        AlpacaPaperBroker(_cfg(trading_base_url=live))
    with pytest.raises(LiveTradingForbidden):
        TradeUpdateStream("wss://api.alpaca.markets/stream", dict, print, print)   # noqa: forbidden-host-check
    fake = FakeAlpacaBroker([])
    fake.trading_base_url = "https://example.com"
    with pytest.raises(LiveTradingForbidden):
        run_preflight(_cfg(), open_db(":memory:"), fake, env=PAPER_ENV)


def test_passes_and_records_account_without_secrets():
    db = open_db(":memory:")
    fake = FakeAlpacaBroker([])
    fake.pos["AAA"] = [5.0, 10.0]
    res = run_preflight(_cfg(), db, fake, env=PAPER_ENV, session_id="paper_x")
    assert res.ok, res.reason
    assert res.account_ref == "PA*******XY" and res.account_status == "ACTIVE"
    assert res.positions == [{"symbol": "AAA", "qty": 5.0, "avg_entry_price": 10.0, "market_value": 50.0}]
    dump = " ".join(str(v) for r in db.fetchall("SELECT * FROM paper_preflights") for v in r.values())
    assert PAPER_ENV["ALPACA_PAPER_SECRET_KEY"] not in dump and PAPER_ENV["ALPACA_PAPER_KEY_ID"] not in dump
    assert mask_account(None) is None


def test_broker_unreachable_is_a_refusal_not_a_crash():
    fake = FakeAlpacaBroker([])
    fake.available = False
    res = run_preflight(_cfg(), open_db(":memory:"), fake, env=PAPER_ENV)
    assert not res.ok and "not reachable" in res.reason


def test_database_rejects_a_non_paper_runner_session():
    db = open_db(":memory:")
    row = {"session_id": "s", "book": "BOT", "mode": "LIVE", "endpoint": "https://paper-api.alpaca.markets",
           "broker": "alpaca_paper", "started_at": "x", "status": "RUNNING"}
    with pytest.raises(sqlite3.IntegrityError):
        db.insert("paper_runner_sessions", row)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert("paper_runner_sessions", {**row, "mode": "PAPER", "endpoint": "https://example.com"})
