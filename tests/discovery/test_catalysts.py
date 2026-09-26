"""Catalyst discovery on a crafted SYNTHETIC world: post-earnings and material-event families,
direction never assumed, UNKNOWN vs no catalyst, evidence chain, no bypass of validation/EV/risk,
point-in-time behaviour, overnight catalyst candidates and the dashboard's catalyst panels."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.dashboard.app import create_app
from quantlab.data import schemas
from quantlab.discovery import run_discovery
from quantlab.discovery import nextsession as ns
from quantlab.discovery.catalysts import CatalystEngine, catalyst_summary, evidence_chain, setup_class
from quantlab.discovery.engine import DiscoveryEngine
from quantlab.features.base import FeatureSet
from quantlab.testing.pit import assert_truncation_invariant

from .world import BENCH, crafted_bundle

ROOT = Path(__file__).resolve().parents[2]
UTC = "UTC"
ET = "America/New_York"


def _et(d, hhmm):
    return pd.Timestamp(f"{pd.Timestamp(d).date()} {hhmm}", tz=ET).tz_convert(UTC)


def _ev(sym, etype, t, sid, payload, react=None):
    return {"symbol": sym, "event_type": etype, "event_time": t, "available_at": t, "reaction_date": react,
            "source_id": sid, "pit_status": "PIT", "provider": "synthetic", "retrieved_at": pd.Timestamp("2020-01-01", tz=UTC),
            "payload_json": json.dumps(payload)}


def _nw(sym, t, nid, head):
    return {"news_id": nid, "symbol": sym, "headline": head, "summary": "", "source": "synthetic", "url": "",
            "created_at": t, "updated_at": t, "available_at": t, "pit_status": "PIT", "provider": "synthetic",
            "retrieved_at": pd.Timestamp("2020-01-01", tz=UTC)}


def catalyst_world(extra_events=(), extra_news=()):
    cb = crafted_bundle()
    D = cb.panel.dates
    d, prev = D[-1], D[-2]
    ev = [
        # 8-K 2.02 filers (coverage): an older release each, then the event of interest
        _ev("VOLX", "earnings_release", _et(D[-70], "16:30"), "v-old", {"form": "8-K", "timing": "POST_CLOSE"}, D[-69]),
        _ev("VOLX", "earnings_release", _et(prev, "16:30"), "v-new", {"form": "8-K", "timing": "POST_CLOSE"}, d),
        _ev("DROP", "earnings_release", _et(D[-70], "07:00"), "d-old", {"form": "8-K", "timing": "PRE_MARKET"}, D[-70]),
        _ev("DROP", "earnings_release", _et(d, "07:00"), "d-new", {"form": "8-K", "timing": "PRE_MARKET"}, d),
        _ev("BRKO", "earnings_release", _et(D[-60], "16:30"), "b-old", {"form": "8-K", "timing": "POST_CLOSE"}, D[-59]),
        _ev("BRKO", "sec_8k", _et(d, "08:00"), "b-8k", {"form": "8-K", "material_items": ["1.01"],
                                                         "categories": ["contract"], "labels": ["material agreement"]}),
        *extra_events,
    ]
    news = [
        _nw("MOMO", _et(D[-30], "10:00"), "m-old", "MOMO Maintains Buy Rating"),
        _nw("N003", _et(D[-30], "10:00"), "n3-old", "N003 Launches App"),
        _nw("N003", _et(d, "10:00"), "n3-deal", "N003 Announces Agreement To Acquire Widget Co For $40M"),
        _nw("N001", _et(D[-30], "10:00"), "n-old", "N001 Unveils New Product"),
        _nw("N001", _et(d, "11:00"), "n-an", "Morgan Stanley Maintains Overweight on N001, Raises Price Target"),
        _nw("VOLX", _et(D[-30], "10:00"), "vx-old", "VOLX Launches App"),
        _nw("BRKO", _et(D[-30], "10:00"), "bx-old", "BRKO Launches App"),
        *extra_news,
    ]
    evdf = schemas.conform("events", pd.DataFrame(ev))
    ndf = schemas.conform("news", pd.DataFrame(news))
    return replace(cb, events=evdf, news=ndf), d


@pytest.fixture(scope="module")
def cfg():
    return load_config(root=ROOT, overrides={"benchmarks": BENCH})


@pytest.fixture(scope="module")
def world():
    return catalyst_world()


@pytest.fixture(scope="module")
def scanned(cfg, world):
    b, d = world
    eng = DiscoveryEngine(cfg)
    return eng, eng.scan(b, d)


def test_post_earnings_direction_is_recorded_never_assumed(scanned):
    _, scan = scanned
    v = scan.table.at["VOLX", "catalyst"]["post_earnings"]
    assert v["state"] == "FIRED" and v["reaction_direction"] == "POSITIVE" and v["volume"] == "CONFIRMED"
    assert v["sessions_since_reaction"] == 0 and v["event"]["timing"] == "POST_CLOSE"
    assert v["eps_surprise"].startswith("UNKNOWN")                                  # no consensus source, never invented
    dp = scan.table.at["DROP", "catalyst"]["post_earnings"]
    assert dp["state"] == "FIRED" and dp["reaction_direction"] == "NEGATIVE"         # an event is not bullish
    assert "post_earnings" in scan.table.at["DROP", "catalyst_fired"]


def test_material_event_needs_company_news_or_8k_and_a_market_response(scanned):
    _, scan = scanned
    t = scan.table
    br = t.at["BRKO", "catalyst"]["material_event"]
    assert br["state"] == "FIRED" and "contract" in br["categories"] and br["volume"] == "CONFIRMED"
    mo = t.at["N003", "catalyst"]["material_event"]
    assert mo["state"] == "PRESENT" and "m&a" in mo["categories"]                   # event, but no market response
    assert "material_event" not in t.at["N003", "catalyst_fired"]
    n1 = t.at["N001", "catalyst"]["material_event"]
    assert n1["state"] == "NONE"                                                    # an analyst note is not a company event


def test_unknown_coverage_is_not_no_catalyst(scanned):
    _, scan = scanned
    t = scan.table
    assert t.at["N005", "catalyst"]["coverage"]["earnings"]["known"] is False       # no 8-K 2.02 filer record
    assert t.at["N005", "catalyst"]["coverage"]["news"]["known"] is False
    assert setup_class(True, False, False) == "UNKNOWN"
    assert setup_class(True, False, True) == "TECHNICAL-ONLY"
    assert setup_class(True, True, True) == "TECHNICAL + CATALYST"
    assert setup_class(False, True, True) == "CATALYST-DRIVEN"


def _link(sid, decision, stage, ev=-50.0, reasons=""):
    return {"candidate_id": f"c-{sid}", "strategy_id": sid, "strategy_version": "1.0.0", "decision": decision,
            "reject_stage": stage, "reasons": reasons, "ev_bps": ev, "order_status": None, "refusal": None}


def test_no_catalyst_bypasses_validation_ev_or_risk(scanned):
    eng, scan = scanned
    strategies = {"breakout": {"strategy_id": "breakout", "status": "PAPER_ELIGIBLE", "evidence": None},
                  "mean_reversion": {"strategy_id": "mean_reversion", "status": "SHADOW", "evidence": None}}
    links = {"BRKO": [_link("breakout", "NO_TRADE", "EV", reasons="ev: EV -50 bps after costs <= minimum 10 bps")],
             "DROP": [_link("mean_reversion", "NO_TRADE", "STRATEGY")]}
    a = eng.assess(scan, links, strategies)
    by = {c["symbol"]: c for c in a.candidates}
    v = by["VOLX"]
    assert v["setup_class"] == "TECHNICAL + CATALYST" and v["status"] in ("WATCH", "DISCOVERED")
    assert v["chain"][-1]["stage"] == "PAPER ELIGIBILITY" and v["chain"][-1]["state"] == "FAIL"
    assert by["BRKO"]["status"] == "REJECTED"                                        # EV gate decides, not the catalyst
    ev_stage = next(s for s in by["BRKO"]["chain"] if s["stage"] == "EV")
    assert ev_stage["state"] == "FAIL"
    assert by["DROP"]["status"] == "VALIDATION_PENDING"                              # unvalidated strategy
    assert not any(c["status"] in ("PAPER_ELIGIBLE", "TRADED") for c in a.candidates)
    assert not by["DROP"].get("catalyst_watch")                                      # negative reaction: long-only
    stages = [s["stage"] for s in v["chain"]]
    assert stages == ["EVENT", "WHEN KNOWN", "PRICE RESPONSE", "VOLUME RESPONSE", "SECTOR/INDUSTRY", "FUNDAMENTALS",
                      "VALIDATION", "RISK", "EV", "PAPER ELIGIBILITY"]
    assert v["chain"][0]["provenance"]["id"] == "v-new"                              # provenance of the event
    s = catalyst_summary(v["catalyst"])
    assert s["event"].startswith("Earnings") and "positive" in s["reaction"]
    trade = eng.assess(scan, {"BRKO": [_link("breakout", "TRADE", "NONE", ev=25.0)]}, strategies)
    assert {c["symbol"]: c for c in trade.candidates}["BRKO"]["status"] == "PAPER_ELIGIBLE"   # only a TRADE decision


def test_catalyst_evaluation_is_point_in_time(cfg):
    b, _ = catalyst_world()
    D = b.panel.dates
    d = D[-15]
    fut = [_ev("N003", "earnings_release", _et(D[-10], "16:30"), "fut", {"form": "8-K"}, D[-9])]
    bf, _ = catalyst_world(extra_events=fut, extra_news=[_nw("N003", _et(D[-12], "09:00"), "fut-n", "N003 Wins Contract")])
    eng = CatalystEngine(cfg)
    syms = ["N003", "MOMO", "VOLX"]

    def rec(bb):
        view = bb.truncate(d)
        return eng.evaluate(view, FeatureSet(view), d, syms)
    a, b_ = rec(b), rec(bf)
    assert a["N003"]["families"] == b_["N003"]["families"] == []                     # future information is invisible
    assert json.dumps(a, default=str, sort_keys=True) == json.dumps(b_, default=str, sort_keys=True)

    def num(bb):
        fs = FeatureSet(bb)
        out = {}
        for dd in (D[-40], D[-15], D[-1]):
            if dd > bb.panel.dates[-1]:
                continue
            out[dd] = fs.cross_section(dd, ["reaction_ret_1d", "reaction_z_1d", "abn_ret_since_reaction",
                                            "sec_material_1d", "news_material_1d", "news_company_1d"])
        return pd.concat(out, names=["date", "symbol"])
    assert_truncation_invariant(num, bf, check_dates=[D[-40], D[-15], D[-1]], name="catalyst features")


def test_reaction_is_known_only_from_the_reaction_close(world):
    b, d = world
    fs = FeatureSet(b)
    r = fs.get("reaction_ret_1d")["VOLX"]
    ar = (b.panel.ret["VOLX"] - b.panel.ret["SPY"])
    assert r.loc[d] == pytest.approx(ar.loc[d])
    old = b.panel.dates[-69]
    assert r.loc[b.panel.dates[-2]] == pytest.approx(ar.loc[old])                    # previous event carried, not the new one


# -- overnight / next session ------------------------------------------------------------------------
@pytest.fixture
def nctx(tmp_path):
    var = tmp_path / "var"
    c = load_config(root=ROOT, overrides={"benchmarks": BENCH, "project": {
        "var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"), "log_dir": str(var / "logs"),
        "report_dir": str(var / "reports"), "model_dir": str(var / "models")}})
    ctx = AppContext.create(c, init_logging=False)
    yield ctx
    ctx.close()


def _store(ctx, b, extra_events=(), actions=()):
    d = str(b.panel.dates[-1].date())
    ctx.store.write("bars", pd.DataFrame([{"symbol": "SPY", "date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                                           "volume": 1.0, "vwap": None, "trade_count": None, "provider": "synthetic",
                                           "retrieved_at": "2024-01-01T00:00:00Z"}]), "synthetic", is_synthetic=True)
    acts = list(actions) or [{"symbol": "SPY", "ex_date": "2023-06-01", "action_type": "cash_dividend", "ratio": None,
                              "amount": 1.0, "declared_date": None, "available_at": "2023-06-01T13:30:00Z",
                              "pit_status": "PIT", "source_id": "a0", "provider": "synthetic",
                              "retrieved_at": "2024-01-01T00:00:00Z"}]
    ctx.store.write("corporate_actions", pd.DataFrame(acts), "synthetic", is_synthetic=True)
    ev = pd.concat([b.events, schemas.conform("events", pd.DataFrame(list(extra_events)))], ignore_index=True) \
        if extra_events else b.events
    ctx.store.write("events", ev, "synthetic", is_synthetic=True)
    ctx.store.write("news", b.news, "synthetic", is_synthetic=True)


def test_post_close_earnings_creates_a_watch_candidate_and_never_an_eligible_one(nctx):
    b, d = catalyst_world()
    nxt = pd.Timestamp("2024-03-25")
    later = [_ev("N005", "earnings_release", _et(d, "16:35"), "n5-post", {"form": "8-K", "timing": "POST_CLOSE"}),
             _ev("N006", "earnings_release", _et(nxt, "10:15"), "n6-after-open", {"form": "8-K"}),
             _ev("ILLQ", "earnings_release", _et(d, "17:00"), "il-post", {"form": "8-K/A"})]
    _store(nctx, b, extra_events=later)
    dr = run_discovery(nctx, b, d, links={})
    assert "N005" not in {c["symbol"] for c in dr.assessment.candidates}
    res = ns.overnight_refresh(nctx, pd.Timestamp("2024-03-23 12:00", tz=UTC))
    assert res["ok"] and res["new_candidates"] == 1
    row = nctx.db.fetchone("SELECT * FROM discovery_candidates WHERE symbol='N005'")
    assert row["created_by"] == "OVERNIGHT_REFRESH" and row["status"] == "WATCH" and row["setup_class"] == "CATALYST-DRIVEN"
    assert row["info_cutoff_at"].startswith("2024-03-23T12:00")                     # stored: when it became known
    chain = {s["stage"]: s["state"] for s in json.loads(row["evidence_chain_json"])}
    assert chain["PRICE RESPONSE"] == "PENDING" and chain["PAPER ELIGIBILITY"] == "FAIL"
    upd = nctx.db.fetchone("SELECT * FROM overnight_updates WHERE symbol='N005'")
    assert upd["effect"] == "NEW_CANDIDATE" and upd["phase"] == "POST_CLOSE"
    assert nctx.db.fetchone("SELECT 1 FROM discovery_candidates WHERE symbol='N006'") is None   # after-open: invisible
    assert nctx.db.fetchone("SELECT 1 FROM discovery_candidates WHERE symbol='ILLQ' AND created_by='OVERNIGHT_REFRESH'") is None
    split = {"symbol": "N005", "ex_date": str(nxt.date()), "action_type": "split", "ratio": 2.0, "amount": None,
             "declared_date": None, "available_at": "2024-03-23T00:00:00Z", "pit_status": "PIT", "source_id": "s-n5",
             "provider": "synthetic", "retrieved_at": "2024-01-01T00:00:00Z"}
    nctx.store.write("corporate_actions", pd.DataFrame([split]), "synthetic", is_synthetic=True)
    ns.preopen_recheck(nctx, pd.Timestamp("2024-03-25 12:00", tz=UTC))
    after = nctx.db.fetchone("SELECT status_after FROM preopen_checks WHERE symbol='N005'")
    assert after["status_after"] == "INVALIDATED"
    assert not nctx.db.fetchall("SELECT 1 FROM preopen_checks WHERE status_after='PAPER_ELIGIBLE'")
    assert nctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0


def test_terminal_shows_mode_catalysts_chain_and_coverage(nctx):
    b, d = catalyst_world()
    _store(nctx, b)
    run_discovery(nctx, b, d, links={})
    page = TestClient(create_app(nctx)).get("/", params={"at": "2024-03-25T12:00:00Z"}).text
    for s in ("PAPER MODE: STRICT", "Catalysts", "New earnings reactions today", "Evidence chain",
              "TECHNICAL + CATALYST", "Catalyst source coverage", "UNKNOWN is not", "Exploratory paper",
              "Best next-session setups", "EXPERIMENT RESULTS".title().split()[0]):
        assert s in page, s
    run = nctx.db.fetchone("SELECT source_coverage_json, catalysts_json FROM discovery_runs")
    cov = json.loads(run["source_coverage_json"])
    assert cov["sec_earnings"]["symbols_covered"] == 3 and cov["universe"]["symbols"] > 3   # counted, never assumed
    assert json.loads(run["catalysts_json"])["n_earnings_reactions_today"] == 2


@pytest.mark.slow
def test_catalyst_research_finds_a_planted_drift_and_not_a_null_one(tmp_path):
    from quantlab.data.ingest import IngestionService
    from quantlab.data.providers.synthetic import SyntheticSpec
    from quantlab.discovery.catalyst_research import GROUPS, run_catalyst_research
    out = {}
    for edge in (0.0, 0.006):
        var = tmp_path / f"v{edge}"
        c = load_config(root=ROOT, overrides={"project": {"var_dir": str(var), "db_path": str(var / "q.db"),
                        "data_dir": str(var / "data"), "log_dir": str(var / "l"), "report_dir": str(var / "r"),
                        "model_dir": str(var / "m")}})
        ctx = AppContext.create(c, init_logging=False)
        IngestionService(c, ctx.store, ctx.db).ingest_synthetic(SyntheticSpec(n_stocks=80, start="2017-01-02",
                                                                              end="2021-12-31", seed=11, pead_edge=edge))
        b = ctx.store.load_bundle(c.section("benchmarks"), synthetic=True)
        r = run_catalyst_research(ctx, b, "2018-01-02", "2021-10-29", min_obs=60, min_dates=30, report=False)
        out[edge] = {g["group"]: g for g in r["groups"]}
        ctx.close()
    a = GROUPS[4]                                                                  # all earnings reaction days
    assert out[0.0][a]["verdict"] != "PROMISING"
    h20 = out[0.006][GROUPS[0]]["horizons"][20]
    assert h20["vs_baseline"] > out[0.0][GROUPS[0]]["horizons"][20]["vs_baseline"]
    assert out[0.006][GROUPS[0]]["verdict"] == "PROMISING"
    _ = np
