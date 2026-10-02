"""Quiver ingestion / daily refresh / CLI, offline. Without a key everything reports SKIPPED (UNKNOWN,
never zero) and no request is made; an outage is reported, never raised into the pipeline; re-running
never duplicates a disclosure."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from quantlab.context import AppContext
from quantlab.data import alt_trades as alt
from quantlab.data import catalyst_refresh as cr
from quantlab.data.providers.http import HttpClient
from quantlab.data.providers.quiver import CONGRESS_BULK, CONGRESS_LIVE, INSIDERS_LIVE, QuiverProvider

from .fakes import FakeClock, FakeResponse, FakeSession
from .quiver_fixtures import CONGRESS_V1, CONGRESS_V2, INSIDERS

KEY_ENV = "QUANTLAB_TEST_QUIVER_KEY"          # never present in a real .env
NOW = pd.Timestamp("2024-03-06T12:00:00Z")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No real .env is read and no real HTTP request can leave these tests."""
    import quantlab.secrets as secrets
    monkeypatch.setattr(secrets, "load_dotenv", lambda path: [])
    monkeypatch.delenv(KEY_ENV, raising=False)

    def no_network(*a, **k):
        raise AssertionError("a test tried to reach the network")
    monkeypatch.setattr(requests.Session, "get", no_network)


@pytest.fixture
def qctx(config):
    cfg = config.with_overrides({"providers": {"quiver": {"key_env": KEY_ENV}}})
    c = AppContext.create(cfg, init_logging=False)
    yield c
    c.close()


def fake_provider(ctx, handler, clock_now=NOW):
    clock = FakeClock()
    session = FakeSession(handler)
    http = HttpClient(session=session, max_retries=1, backoff=0.01, sleep=clock.sleep, clock=clock.time, name="quiver")
    return QuiverProvider(ctx.config, http=http, calendar=None, clock=lambda: clock_now), session


def ok_handler(url, params, headers):
    if url.endswith(CONGRESS_LIVE):
        return FakeResponse(200, json_body=CONGRESS_V1)
    if url.endswith(CONGRESS_BULK):
        return FakeResponse(200, json_body=CONGRESS_V2 if params.get("page") == 1 else [])
    if url.endswith(INSIDERS_LIVE):
        return FakeResponse(200, json_body=INSIDERS if params.get("page") == 1 else [])
    return FakeResponse(404, json_body={"detail": "Not found."})


# -- no key ---------------------------------------------------------------------------------------
def test_without_a_key_everything_is_skipped_and_nothing_is_fetched_or_written(qctx):
    res = alt.ingest_alt_trades(qctx, mode="refresh", now=NOW)
    assert res["status"] == "SKIPPED" and res["ok"] is True and res["rows"] == "SKIPPED"
    for src in ("congress", "insider"):
        assert res[src]["status"] == "SKIPPED" and KEY_ENV in res[src]["reason"] and "UNKNOWN" in res[src]["reason"]
    assert qctx.store.dataset_ids("alt_trades") == []
    out = cr.refresh_catalysts(qctx, pd.Timestamp("2024-03-05"), parts=(), now=NOW)      # the runner's daily scope
    assert out["quiver"]["status"] == "SKIPPED"
    status = alt.source_status(qctx.store, qctx.config)
    assert all("not set" in r["key"] and r["datasets"] == 0 for r in status)


def test_disabled_in_config_is_skipped(qctx, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "qv-test-token")
    cfg = qctx.config.with_overrides({"providers": {"quiver": {"enabled": False}}})
    c2 = AppContext.create(cfg, init_logging=False)
    try:
        res = alt.ingest_alt_trades(c2, mode="refresh", now=NOW)
        assert res["status"] == "SKIPPED" and "enabled is false" in res["congress"]["reason"]
    finally:
        c2.close()


# -- refresh scope ----------------------------------------------------------------------------------
def test_quiver_runs_in_the_daily_market_wide_refresh_only(qctx, monkeypatch):
    calls = []
    monkeypatch.setattr(alt, "refresh_quiver", lambda ctx, now=None, provider=None: calls.append(now) or
                        {"status": "OK", "ok": True, "rows": 0})
    cr.refresh_catalysts(qctx, pd.Timestamp("2024-03-05"), parts=(), now=NOW)                 # daily: symbols=None
    cr.refresh_catalysts(qctx, pd.Timestamp("2024-03-05"), symbols=["AAA"], parts=(), now=NOW)  # pre-open scope
    assert len(calls) == 1
    cr.refresh_catalysts(qctx, pd.Timestamp("2024-03-05"), symbols=["AAA"], parts=("quiver",), now=NOW)
    assert len(calls) == 2
    # the paper runner's daily call passes symbols=None (so Quiver runs); its pre-open call passes symbols
    from quantlab.pipeline import runner
    seen = []
    monkeypatch.setattr(cr, "refresh_catalysts", lambda ctx, session, **kw: seen.append(kw) or {})
    runner.default_catalyst_refresh(qctx, pd.Timestamp("2024-03-05").date(), "daily")
    runner.default_catalyst_refresh(qctx, pd.Timestamp("2024-03-05").date(), "preopen", symbols=["AAA"])
    assert seen[0]["symbols"] is None and seen[1]["symbols"] == ["AAA"]


# -- with a key (fake transport) -----------------------------------------------------------------------
def test_refresh_writes_dedupes_and_resyncs_the_bulk_feed_weekly(qctx, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "qv-test-token")
    prov, session = fake_provider(qctx, ok_handler)
    r1 = alt.ingest_alt_trades(qctx, mode="refresh", now=NOW, provider=prov)
    assert r1["status"] == "OK" and r1["congress"]["status"] == "OK" and r1["insider"]["status"] == "OK"
    bulk_calls = [c for c in session.calls if c["url"].endswith(CONGRESS_BULK)]
    assert bulk_calls and bulk_calls[0]["params"]["version"] == "V1"          # first refresh: bulk re-pull
    stored = qctx.store.load("alt_trades")
    n1 = len(stored)
    assert n1 == 4 + 1 + 5 and set(stored["source"]) == {"congress", "insider"}
    insider_days = [c["params"]["date"] for c in session.calls if c["url"].endswith(INSIDERS_LIVE)]
    assert len(insider_days) == len(pd.bdate_range("2024-02-28", "2024-03-06"))   # refresh_days = 7, weekdays
    # next day: no bulk re-pull yet, nothing duplicated
    prov2, s2 = fake_provider(qctx, ok_handler, NOW + pd.Timedelta(days=1))
    r2 = alt.ingest_alt_trades(qctx, mode="refresh", now=NOW + pd.Timedelta(days=1), provider=prov2)
    assert r2["status"] == "OK" and not [c for c in s2.calls if c["url"].endswith(CONGRESS_BULK)]
    assert len(qctx.store.load("alt_trades")) == n1
    # a week later: the bulk feed is re-pulled (late disclosures)
    prov3, s3 = fake_provider(qctx, ok_handler, NOW + pd.Timedelta(days=8))
    alt.ingest_alt_trades(qctx, mode="refresh", now=NOW + pd.Timedelta(days=8), provider=prov3)
    assert [c for c in s3.calls if c["url"].endswith(CONGRESS_BULK)]
    params = [json.loads(r["params_json"]) for r in qctx.db.fetchall("SELECT params_json FROM datasets WHERE kind='alt_trades'")]
    assert {p["what"] for p in params} == {"quiver_congress_live", "quiver_congress_bulk", "quiver_insider_daily"}


def test_an_outage_is_reported_never_raised(qctx, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "qv-test-token")

    def down(url, params, headers):
        return requests.ConnectionError("quiver unreachable")
    prov, _ = fake_provider(qctx, down)
    res = alt.ingest_alt_trades(qctx, mode="refresh", now=NOW, provider=prov)
    assert res["status"] == "FAILED" and res["ok"] is False and res["rows"] == "FAILED"
    assert res["congress"]["status"] == "FAILED" and res["insider"]["status"] == "FAILED"
    out = cr.refresh_catalysts(qctx, pd.Timestamp("2024-03-05"), parts=(), now=NOW, alt_provider=prov)
    assert out["quiver"]["status"] == "FAILED"                                 # reported, the refresh still returns
    assert qctx.store.dataset_ids("alt_trades") == []

    def tier1_only(url, params, headers):                                     # insiders are a Tier 2 endpoint
        if url.endswith(INSIDERS_LIVE):
            return FakeResponse(403, json_body={"detail": "You do not have permission to perform this action."})
        return ok_handler(url, params, headers)
    prov, _ = fake_provider(qctx, tier1_only)
    res = alt.ingest_alt_trades(qctx, mode="refresh", now=NOW, provider=prov)
    assert res["status"] == "PARTIAL" and res["congress"]["status"] == "OK" and "403" in res["insider"]["status"]
    assert set(qctx.store.load("alt_trades")["source"]) == {"congress"}

    def broken(*a, **k):
        raise RuntimeError("bug in the refresh")
    monkeypatch.setattr(alt, "ingest_alt_trades", broken)
    out = cr.refresh_catalysts(qctx, pd.Timestamp("2024-03-05"), parts=(), now=NOW)
    assert out["quiver"]["status"] == "FAILED" and "bug" in out["quiver"]["error"]


def test_history_ingest_and_read_helpers(qctx, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "qv-test-token")
    prov, session = fake_provider(qctx, ok_handler)
    res = alt.ingest_alt_trades(qctx, days=10, mode="history", now=NOW, provider=prov)
    assert res["status"] == "OK"
    assert not [c for c in session.calls if c["url"].endswith(CONGRESS_LIVE)]   # history uses the bulk feed
    df = alt.load_recent(qctx.store, since_days=30, now=NOW)
    assert len(df) == 1 + 5 and "raw_json" not in df.columns
    rows = alt.recent_disclosures(qctx.store, symbols=["aaa"], now=NOW, frame=df)
    assert {r["symbol"] for r in rows} == {"AAA"} and {r["source"] for r in rows} == {"insider"}
    buy = next(r for r in rows if r["side"] == "BUY")
    assert buy["amount"] == "$25,500" and buy["disclosed"] == "2024-02-28" and buy["traded"] == "2024-02-27"
    assert buy["usable_from"] == "2024-02-29 16:00 ET" and buy["usable_now"] is True
    unknown = next(r for r in alt.recent_disclosures(qctx.store, symbols=["CCC"], now=NOW, frame=df) if r["side"] == "BUY")
    assert unknown["amount"] == "UNKNOWN"


# -- CLI ------------------------------------------------------------------------------------------
def _cli_config(tmp_path):
    var = tmp_path / "clivar"
    p = tmp_path / "cli.yaml"
    p.write_text(json.dumps({"project": {"var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"),
                                         "log_dir": str(var / "logs"), "report_dir": str(var / "r"),
                                         "model_dir": str(var / "m")},
                             "providers": {"quiver": {"key_env": KEY_ENV}}}), encoding="utf-8")
    return p


def test_cli_alt_ingest_recent_status(tmp_path, capsys, monkeypatch):
    import quantlab.context
    from quantlab.cli import main
    from quantlab.config import load_config
    monkeypatch.setattr(quantlab.context, "setup_logging", lambda *a, **k: None)   # no process-wide handlers
    cfgp = _cli_config(tmp_path)
    assert main(["--config", str(cfgp), "alt", "ingest", "--days", "30"]) == 2       # no key: nothing ingested
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "SKIPPED" and out["congress"]["status"] == "SKIPPED"
    # store a real-provider disclosure the way an ingest would, then read it back through the CLI
    ctx = AppContext.create(load_config(cfgp), init_logging=False)
    try:
        prov = QuiverProvider(ctx.config, calendar=None, clock=lambda: pd.Timestamp.now(tz="UTC"))
        today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        rows = [{**INSIDERS[0], "fileDate": today, "Date": today}, {**INSIDERS[1], "fileDate": today, "Ticker": "ZZZ"}]
        ctx.store.write("alt_trades", prov.parse_insiders(rows), "quiver", params={"what": "quiver_insider_daily"})
    finally:
        ctx.close()
    assert main(["--config", str(cfgp), "alt", "recent", "--symbol", "aaa", "--data", "real"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["symbol"] == "AAA" and [r["symbol"] for r in out["rows"]] == ["AAA"]
    assert out["rows"][0]["usable_now"] is False and "never the trade date" in out["note"]
    assert main(["--config", str(cfgp), "alt", "status", "--data", "real"]) == 0
    out = json.loads(capsys.readouterr().out)
    ins = next(s for s in out["sources"] if s["source"] == "insider")
    assert ins["datasets"] == 1 and "not set" in ins["key"]
