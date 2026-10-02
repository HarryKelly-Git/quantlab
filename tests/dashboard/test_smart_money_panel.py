"""Live page: the congress / insider section renders with and without data, groups disclosures by
open positions / next-open plan / watched symbols, lists the latest market-wide disclosures, and a
failure inside it never breaks the page. Read-only; all data SYNTHETIC."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.context import AppContext
from quantlab.dashboard import app as dash
from quantlab.data.providers.quiver import QuiverProvider

from ..data.quiver_fixtures import CONGRESS_V1, INSIDERS

KEY_ENV = "QUANTLAB_TEST_QUIVER_KEY"
NOW = datetime(2024, 3, 6, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def dctx(config, monkeypatch):
    import quantlab.secrets as secrets
    monkeypatch.setattr(secrets, "load_dotenv", lambda path: [])
    monkeypatch.delenv(KEY_ENV, raising=False)
    # a fresh cache per test; a window wide enough that the 2024 fixture rows are "recent" at wall-clock time
    monkeypatch.setattr(dash, "_ALT_CACHE", dash._AltCache(since_days=5000))
    c = AppContext.create(config.with_overrides({"providers": {"quiver": {"key_env": KEY_ENV}}}), init_logging=False)
    yield c
    c.close()


def _store_disclosures(ctx):
    prov = QuiverProvider(ctx.config, calendar=None, clock=lambda: pd.Timestamp(NOW))
    df = pd.concat([prov.parse_congress(CONGRESS_V1), prov.parse_insiders(INSIDERS)], ignore_index=True)
    # no real bars in this DB, so the page reads the synthetic side: store the fixture rows there
    ctx.store.write("alt_trades", df, "synthetic", params={"what": "test"}, is_synthetic=True)
    return df


def _plan_row(sym):
    return {"symbol": sym, "selection": "SELECTED", "stage": "PLANNED", "hold": 10, "qty": 5, "ref_price": 10.0,
            "stop_price": 9.0, "setup": "momentum", "next_session": "2024-03-07", "selection_score": 0.7,
            "forecast": None}


def test_live_page_renders_without_any_disclosure_data(dctx):
    c = TestClient(dash.create_app(dctx))
    html = c.get("/live").text
    assert "Congress &amp; insider disclosures (context only)" in html and "never scored" in html
    assert "No congress/insider data stored" in html and f"{KEY_ENV} not set" in html
    sm = c.get("/api/live").json()["smart_money"]
    assert sm["market"] == [] and {r["source"] for r in sm["status"]} == {"congress", "insider"}
    assert "error" not in sm


def test_disclosures_for_held_planned_and_watched_symbols_plus_market_wide(dctx, monkeypatch):
    _store_disclosures(dctx)
    x = {"mode": "EXPLORATION", "positions": [], "plan": [_plan_row("AAA")], "watched": [{"symbol": "CCC"}, {"symbol": "ZZZ"}],
         "skipped": 0, "arms": [], "learning": {}, "plan_session": "2024-03-06", "gross": 0.0}
    sm = dash._smart_money_live(dctx, NOW, x, real=False)
    groups = {g["label"]: g for g in sm["groups"]}
    plan = groups["Next-open plan"]
    assert plan["symbols"] == ["AAA"] and {r["symbol"] for r in plan["rows"]} == {"AAA"}
    assert {r["source"] for r in plan["rows"]} == {"congress", "insider"}
    watched = groups["Watched (not traded)"]
    assert {r["symbol"] for r in watched["rows"]} == {"CCC"} and watched["quiet"] == ["ZZZ"]
    assert groups["Open positions"]["rows"] == []
    assert 0 < len(sm["market"]) <= 20
    assert sm["market"][0]["disclosed"] >= sm["market"][-1]["disclosed"]          # newest first
    exch = next(r for r in plan["rows"] if r["side"] == "OTHER")
    assert exch["amount"] == "UNKNOWN"                                            # never shown as $0
    dash._ALT_CACHE.frame = None                                                  # the page re-reads the store
    monkeypatch.setattr(dash, "_exploration_live", lambda *a, **k: x)
    html = TestClient(dash.create_app(dctx)).get("/live").text
    assert "Next-open plan: AAA" in html and "Jane Example" in html and "Dana Director" in html
    assert "No stored disclosure in the last 5000 days for ZZZ" in html


def test_a_failing_section_never_breaks_the_page(dctx, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("alt store unreadable")
    monkeypatch.setattr(dash, "_smart_money_live", boom)
    c = TestClient(dash.create_app(dctx))
    r = c.get("/live")
    assert r.status_code == 200 and "Congress/insider panel unavailable: RuntimeError: alt store unreadable" in r.text
    assert "Open positions" in r.text and "Learning" in r.text
    assert c.get("/api/live").json()["smart_money"]["error"].startswith("RuntimeError")
