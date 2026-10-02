"""/live "Congress & insider trades" section: read-only context for held / planned / watched / tracked
symbols plus the latest market-wide disclosures; fault-isolated from the rest of the page."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.context import AppContext
from quantlab.dashboard import app as dash
from quantlab.data import schemas
from quantlab.data.providers import house_ptr as hp
from quantlab.data.providers import sec_form4 as f4

from ..data.alt_fixtures import FORM4_XML, PTR_TEXT

NOW = datetime(2024, 3, 6, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def dctx(config, monkeypatch):
    import quantlab.secrets as secrets
    monkeypatch.setattr(secrets, "load_dotenv", lambda path: [])
    # a fresh cache per test; a window wide enough that the 2024 fixture rows are "recent" at wall-clock time
    monkeypatch.setattr(dash, "_ALT_CACHE", dash._AltCache(since_days=5000))
    c = AppContext.create(config, init_logging=False)
    yield c
    c.close()


def _store_disclosures(ctx):
    ts = pd.Timestamp(NOW)
    rows = f4.parse_form4_xml(FORM4_XML, accession="0000222222-24-000001", accepted_at=ts - pd.Timedelta(days=1),
                              retrieved_at=ts)
    rows += hp.ptr_rows({"doc_id": "20024001", "filing_date": "2024-03-01", "name": "Hon. Pat Example",
                         "state_dst": "CA12"}, PTR_TEXT, retrieved_at=ts)
    df = schemas.conform("alt_trades", pd.DataFrame(rows))
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
    assert "Congress &amp; insider trades (context only)" in html and "never scored" in html
    assert "No congress/insider data stored" in html and "Senate" in html
    sm = c.get("/api/live").json()["smart_money"]
    assert sm["market"] == [] and {r["source"] for r in sm["status"]} == {"congress", "insider"}
    assert "error" not in sm


def test_disclosures_for_held_planned_watched_and_tracked_symbols_plus_market_wide(dctx, monkeypatch):
    _store_disclosures(dctx)
    x = {"mode": "EXPLORATION", "positions": [{"symbol": "CCC"}], "plan": [_plan_row("AAA")],
         "watched": [{"symbol": "BBB"}, {"symbol": "ZZZ"}], "skipped": 0, "arms": [], "learning": {},
         "plan_session": "2024-03-06", "gross": 0.0}
    tracked = {"configured": ["AAA"], "rows": [{"symbol": "AAA"}]}
    sm = dash._smart_money_live(dctx, NOW, x, tracked, real=False)
    groups = {g["label"]: g for g in sm["groups"]}
    plan = groups["Next-open plan"]
    assert plan["symbols"] == ["AAA"] and {r["source"] for r in plan["rows"]} == {"congress", "insider"}
    assert {r["symbol"] for r in groups["Open positions"]["rows"]} == {"CCC"}
    assert groups["Watched (not traded)"]["quiet"] == ["ZZZ"] and groups["Tracked watchlist"]["symbols"] == ["AAA"]
    assert 0 < len(sm["market"]) <= 20 and all(r["symbol"] for r in sm["market"])
    assert sm["market"][0]["disclosed"] >= sm["market"][-1]["disclosed"]          # newest first
    other = next(r for r in plan["rows"] if r["side"] == "OTHER")
    assert other["amount"] == "UNKNOWN"                                           # never shown as $0
    dash._ALT_CACHE.frame = None                                                  # the page re-reads the store
    monkeypatch.setattr(dash, "_exploration_live", lambda *a, **k: {**x, "positions": []})   # page: no fake position rows
    html = TestClient(dash.create_app(dctx)).get("/live").text
    assert "Next-open plan: AAA" in html and "Doe Jane" in html and "Hon. Pat Example" in html
    assert "No stored disclosure in the last 5000 days for ZZZ" in html


def test_a_failing_section_never_breaks_the_page(dctx, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("alt store unreadable")
    monkeypatch.setattr(dash, "_smart_money_live", boom)
    c = TestClient(dash.create_app(dctx))
    r = c.get("/live")
    assert r.status_code == 200 and "Congress/insider section unavailable: RuntimeError: alt store unreadable" in r.text
    assert "Open positions" in r.text and "Tracked watchlist" in r.text
    assert c.get("/api/live").json()["smart_money"]["error"].startswith("RuntimeError")
