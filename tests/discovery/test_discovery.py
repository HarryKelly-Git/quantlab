"""Market discovery layer: supported families, UNKNOWN propagation, ranking, reason vectors, near-miss
tracking, candidate statuses vs the unchanged decision chain, zero-discovery diagnostics, SHADOW
strategies producing WATCH/VALIDATION_PENDING without orders, dashboard funnel counts, no fabricated
data, point-in-time behaviour and forward outcomes. All data is SYNTHETIC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.dashboard.app import create_app
from quantlab.discovery import DiscoveryOutcomeTracker, run_discovery
from quantlab.discovery.engine import DiscoveryEngine, ScanResult
from quantlab.discovery.families import CONTEXT, POINTS, SCORED, SCORED_POINTS
from quantlab.pipeline.daily import DailyPipeline
from quantlab.testing.pit import assert_truncation_invariant

from ..pipeline.test_daily import world  # noqa: F401  (fixture re-export)
from .world import BENCH, crafted_bundle

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg():
    return load_config(root=ROOT, overrides={"benchmarks": BENCH})


@pytest.fixture(scope="module")
def cb():
    return crafted_bundle(news_for=("MOMO",))


@pytest.fixture(scope="module")
def scan(cfg, cb):
    return DiscoveryEngine(cfg).scan(cb, cb.panel.dates[-1])


@pytest.fixture
def dctx(tmp_path):
    var = tmp_path / "var"
    c = load_config(root=ROOT, overrides={"benchmarks": BENCH, "project": {
        "var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"), "log_dir": str(var / "logs"),
        "report_dir": str(var / "reports"), "model_dir": str(var / "models")}})
    ctx = AppContext.create(c, init_logging=False)
    yield ctx
    ctx.close()


# -- families ------------------------------------------------------------------------------------
def test_each_supported_family_fires_on_its_setup(scan):
    t = scan.table
    assert {"momentum", "relative_strength"} <= set(t.at["MOMO", "fired"])
    assert "volume_activity" in t.at["VOLX", "fired"] and t.at["VOLX", "bias"] == "BULLISH"
    assert any("relative volume" in x for x in t.at["VOLX", "reasons"]["volume_activity"])
    assert "breakout_compression" in t.at["BRKO", "fired"]
    assert any("55-day high" in x for x in t.at["BRKO", "reasons"]["breakout_compression"])
    assert "mean_reversion" in t.at["DROP", "fired"] and t.at["DROP", "bias"] == "BEARISH"
    normal = t.loc[[s for s in t.index if s.startswith("N")]]
    assert (normal["fired"].map(len) > 0).mean() < 0.5          # ordinary stocks mostly do not fire


def test_reason_vectors_points_and_provenance(scan, cb):
    r = scan.table.loc["MOMO"]
    comps = {f: r[f"c_{f}"] for f in SCORED}
    assert all(0 <= v <= POINTS for v in comps.values())
    # every scored family's points are reported, but only SCORED_POINTS feed the composite:
    # relative_strength is rank-identical to momentum (see families.SCORED_POINTS)
    known = [v for f, v in comps.items() if f in SCORED_POINTS and np.isfinite(v)]
    assert r["score"] == pytest.approx(100 * sum(known) / (POINTS * len(known)))
    assert np.isfinite(comps["relative_strength"])           # still measured and reported
    all_known = [v for v in comps.values() if np.isfinite(v)]
    assert len(known) < len(all_known), "relative_strength must be excluded from the composite"
    assert any("strong 60d return" in x for x in r["reasons"]["momentum"])
    assert any("outperforming SPY" in x for x in r["reasons"]["relative_strength"])
    d = str(cb.panel.dates[-1].date())
    for name, f in r["factors"].items():
        assert f["as_of"] == d and f["source"] and f["pit_status"]
        assert (f["value"] is None) == (f["state"] != "VALID"), name


def test_ranking_orders_by_score_and_high_rank_needs_coverage(cfg, scan):
    a = DiscoveryEngine(cfg).assess(scan, {}, {})
    scores = [c["score"] for c in a.candidates]
    assert scores == sorted(scores, reverse=True)
    assert [c["rank"] for c in a.candidates] == list(range(1, len(a.candidates) + 1))
    for c in a.candidates:
        assert c["high_quality"] == (c["score"] >= 70 and c["coverage"] >= 0.8)
    wl = [c for c in a.candidates if c["on_watchlist"]]
    assert len(wl) <= 25 and all(c["high_quality"] and not c["data_or_universe_failed"] for c in wl)


# -- UNKNOWN propagation / no fabricated data ------------------------------------------------------
def test_unknown_is_never_zero_and_invalid_is_a_data_quality_issue(cfg, scan):
    t = scan.table
    assert t.at["NEWB", "unknown"] == ["ret_120d"] and t.at["NEWB", "factors"]["ret_120d"]["value"] is None
    assert np.isfinite(t.at["NEWB", "c_momentum"])                   # 20d/60d still score momentum
    assert t.at["GAPD", "invalid"] == ["ret_120d"] and t.at["GAPD", "factors"]["ret_120d"]["state"] == "INVALID"
    mom = next(r for r in scan.coverage if r["key"] == "momentum")
    assert mom["invalid"] >= 1
    a = DiscoveryEngine(cfg).assess(scan, {}, {})
    assert any(x["code"] == "DATA_QUALITY" for x in a.diagnostics)
    # news covers MOMO only: everyone else is UNKNOWN, never a known zero
    assert t.at["MOMO", "context"]["news"]["state"] == "KNOWN"
    for s in t.index.drop("MOMO"):
        assert t.at[s, "context"]["news"]["state"] == "UNKNOWN" and "values" not in t.at[s, "context"]["news"]
    for fam in ("earnings", "fundamentals", "sector"):
        assert all(t.at[s, "context"][fam]["state"] == "UNKNOWN" for s in t.index)


def test_context_families_never_change_the_score(cfg, cb):
    eng = DiscoveryEngine(cfg)
    with_news = eng.scan(cb, cb.panel.dates[-1]).table
    without = eng.scan(crafted_bundle(news_for=()), cb.panel.dates[-1]).table
    pd.testing.assert_series_equal(with_news["score"], without["score"])
    assert with_news["fired"].equals(without["fired"])


def test_no_fabricated_inputs_reach_the_candidate_record(cfg, cb, dctx):
    dr = run_discovery(dctx, cb, cb.panel.dates[-1], links={})
    for c in dr.assessment.candidates:
        for name, f in c["factors"].items():
            if f["state"] != "VALID":
                assert f["value"] is None
        for fam in CONTEXT:
            ctx = c["context"][fam]
            assert ctx["state"] in ("KNOWN", "UNKNOWN")
            if ctx["state"] == "UNKNOWN":
                assert "values" not in ctx and ctx.get("why")
    rows = dctx.db.fetchall("SELECT discovery_score, score_coverage FROM discovery_candidates")
    assert rows and all(r["discovery_score"] is None or 0 <= r["discovery_score"] <= 100 for r in rows)


# -- validation linkage, statuses, near-misses ------------------------------------------------------
def _links():
    base = {"strategy_version": "1.0.0", "reasons": "", "ev_bps": None, "order_status": None, "refusal": None}
    return {
        "MOMO": [{**base, "candidate_id": "c1", "strategy_id": "momentum_trend", "decision": "NO_TRADE",
                  "reject_stage": "STRATEGY", "reasons": "strategy status SHADOW", "ev_bps": -20.0}],
        "VOLX": [{**base, "candidate_id": "c2", "strategy_id": "breakout", "decision": "TRADE", "reject_stage": "NONE",
                  "ev_bps": 25.0, "order_status": "accepted"}],
        "BRKO": [{**base, "candidate_id": "c3", "strategy_id": "breakout", "decision": "TRADE", "reject_stage": "NONE",
                  "ev_bps": 25.0, "refusal": "execution window missed"}],
        "DROP": [{**base, "candidate_id": "c4", "strategy_id": "mean_reversion", "decision": "NO_TRADE",
                  "reject_stage": "EV", "reasons": "EV -5 bps <= 10", "ev_bps": -5.0}],
    }


STRATS = {"momentum_trend": {"strategy_id": "momentum_trend", "status": "SHADOW", "evidence": None},
          "breakout": {"strategy_id": "breakout", "status": "PAPER_ELIGIBLE", "evidence": None},
          "mean_reversion": {"strategy_id": "mean_reversion", "status": "PAPER_ELIGIBLE", "evidence": None}}


def test_statuses_follow_the_unchanged_decision_chain(cfg, scan):
    a = DiscoveryEngine(cfg).assess(scan, _links(), STRATS)
    by = {c["symbol"]: c for c in a.candidates}
    assert by["MOMO"]["status"] == "VALIDATION_PENDING" and by["MOMO"]["block_stage"] == "STRATEGY_VALIDATION"
    assert by["VOLX"]["status"] == "TRADED" and by["VOLX"]["block_stage"] is None
    assert by["BRKO"]["status"] == "PAPER_ELIGIBLE" and by["BRKO"]["block_reason"] == "execution window missed"
    assert by["DROP"]["status"] == "REJECTED" and "EV" in by["DROP"]["block_reason"]
    assert by["ILLQ"]["status"] == "REJECTED" and by["ILLQ"]["block_reason"] == "median dollar volume < 5,000,000"
    f = a.funnel
    assert f["paper_eligible"] == 2 and f["paper_trades"] == 1 and f["in_validation"] == 4
    assert sum(f["status_counts"].values()) == f["discovered"] == len(a.candidates)
    unlinked = [c for c in a.candidates if c["symbol"] not in _links() and not c["data_or_universe_failed"]]
    assert unlinked and all(c["block_stage"] == "NO_STRATEGY_COVERAGE" for c in unlinked)
    assert all(c["status"] in ("WATCH", "DISCOVERED") for c in unlinked)


def test_shadow_strategy_setups_stay_visible_but_never_eligible(cfg, scan):
    shadow = {k: {**v, "status": "SHADOW"} for k, v in STRATS.items()}
    links = {s: [{**x, "decision": "NO_TRADE", "reject_stage": "STRATEGY", "order_status": None}
                 for x in v] for s, v in _links().items()}
    a = DiscoveryEngine(cfg).assess(scan, links, shadow)
    st = {c["symbol"]: c["status"] for c in a.candidates}
    assert st["MOMO"] == st["VOLX"] == st["BRKO"] == st["DROP"] == "VALIDATION_PENDING"
    assert a.funnel["paper_eligible"] == 0 and a.funnel["paper_trades"] == 0
    assert set(st.values()) <= {"DISCOVERED", "WATCH", "VALIDATION_PENDING", "REJECTED"}


def test_near_misses_record_passed_failed_and_unknown(cfg, scan):
    a = DiscoveryEngine(cfg).assess(scan, {}, {})
    nm = {n["symbol"]: n for n in a.near_misses}
    assert "ILLQ" in nm
    x = nm["ILLQ"]
    assert "Momentum" in x["passed"] and "Relative strength" in x["passed"]
    assert x["failed"][0] == {"stage": "UNIVERSE", "reason": "median dollar volume < 5,000,000"}
    assert any(u.startswith("News") for u in x["unknown"])
    assert all(n["block_reason"] for n in a.near_misses)


def test_blockers_explain_zero_paper_trades(cfg, scan):
    a = DiscoveryEngine(cfg).assess(scan, {}, {})
    assert a.blockers["n_discovered"] == len(a.candidates) > 0 and a.blockers["n_eligible"] == 0
    assert a.blockers["main"] == "NO_STRATEGY_COVERAGE"
    assert sum(a.blockers["by_stage"].values()) == len(a.candidates)


# -- diagnostics ----------------------------------------------------------------------------------
def test_zero_and_empty_discovery_raise_diagnostics(cfg):
    eng = DiscoveryEngine(cfg)
    flat = crafted_bundle(flat=True, specials=False)
    a = eng.assess(eng.scan(flat, flat.panel.dates[-1]), {}, {})
    assert not a.candidates
    assert any(x["code"] == "ZERO_DISCOVERIES" and x["level"] == "CRITICAL" for x in a.diagnostics)
    strict = DiscoveryEngine(load_config(root=ROOT, overrides={"benchmarks": BENCH,
                                                               "discovery": {"basic_filter": {"min_price": 1e9}}}))
    b = crafted_bundle()
    a2 = strict.assess(strict.scan(b, b.panel.dates[-1]), {}, {})
    assert any(x["code"] == "EMPTY_SCAN" and x["level"] == "CRITICAL" for x in a2.diagnostics)


def test_one_rule_blocking_everything_is_flagged(cfg, scan):
    a = DiscoveryEngine(cfg).assess(scan, None, {})          # validation never ran for this session
    assert all(c["block_stage"] in ("VALIDATION_NOT_RUN", "UNIVERSE", "DATA") for c in a.candidates)
    # realistic: every setup rejected by the SAME EV rule, each reason carrying its own EV number
    eng = DiscoveryEngine(cfg)
    base = {"strategy_version": "1.0.0", "order_status": None, "refusal": None, "strategy_id": "breakout"}
    links = {s: [{**base, "candidate_id": f"c{i}", "decision": "NO_TRADE", "reject_stage": "EV",
                  "reasons": f"risk.ev: EV {-(i + 1) * 1.7:.1f} bps after costs <= minimum 10 bps", "ev_bps": -(i + 1) * 1.7}]
             for i, s in enumerate(scan.table.index)}
    strats = {"breakout": {"strategy_id": "breakout", "status": "PAPER_ELIGIBLE", "evidence": None}}
    ev_all = eng.assess(scan, links, strats)
    ok = [c for c in ev_all.candidates if not c["data_or_universe_failed"]]
    assert len({c["block_reason"] for c in ok}) == len(ok)                 # every text differs...
    assert len({c["block_key"] for c in ok}) == 1                          # ...but it is one rule
    blocked = [c for c in ev_all.candidates if c["block_stage"]]
    if len(ok) / len(blocked) >= 0.95:
        assert any(x["code"] == "ONE_RULE_BLOCKS_ALL" for x in ev_all.diagnostics)
    dg = eng._diagnostics(ScanResult(scan.as_of, scan.table, scan.counts, scan.coverage, {}, True), ok * 2)
    assert any(x["code"] == "ONE_RULE_BLOCKS_ALL" and "risk.ev" in x["details"]["rule"] for x in dg)


# -- point in time ----------------------------------------------------------------------------------
def test_discovery_is_truncation_invariant(cfg, cb):
    eng = DiscoveryEngine(cfg)
    dates = cb.panel.dates
    check = [dates[-60], dates[-25], dates[-1]]

    def compute(b):
        rows = {d: eng.scan(b, d).table["score"] for d in check if d <= b.panel.dates[-1]}
        return pd.DataFrame(rows).T
    assert_truncation_invariant(compute, cb, check_dates=check, name="discovery score")


def test_corrupting_the_future_does_not_change_discovery(cfg, cb):
    eng = DiscoveryEngine(cfg)
    d = cb.panel.dates[-40]
    before = eng.scan(cb, d).table
    fields = {k: v.copy() for k, v in cb.panel.fields.items()}
    for k in ("open", "high", "low", "close", "aopen", "ahigh", "alow", "aclose", "volume", "dollar_volume"):
        fields[k].loc[fields[k].index > d] *= 7.0
    from dataclasses import replace
    from quantlab.data.panel import Panel
    corrupted = replace(cb, panel=Panel(fields, dict(cb.panel.meta)))
    after = eng.scan(corrupted, d).table
    pd.testing.assert_series_equal(before["score"], after["score"])
    assert before["fired"].equals(after["fired"])


# -- forward outcomes ------------------------------------------------------------------------------
def test_forward_outcomes_only_after_maturity_and_match_prices(cb, dctx):
    d = cb.panel.dates[-30]
    dr = run_discovery(dctx, cb, d, links={})
    assert dr.outcomes_written == 0
    tr = DiscoveryOutcomeTracker(dctx.db, dctx.config)
    assert tr.update(cb, d, True) == 0                                  # nothing matured on D itself
    dates = cb.panel.dates
    i = dates.get_loc(d)
    tr.update(cb, dates[i + 1], True)
    hs = {r["horizon_sessions"] for r in dctx.db.fetchall("SELECT horizon_sessions FROM discovery_outcomes")}
    assert hs == {1}
    tr.update(cb, dates[i + 5], True)
    assert tr.update(cb, dates[i + 5], True) == 0                        # idempotent
    row = dctx.db.fetchone("SELECT * FROM discovery_outcomes WHERE horizon_sessions=5 LIMIT 1")
    sym = row["discovery_id"].split(":")[-1]
    ac = cb.panel.aclose[sym]
    assert row["ret"] == pytest.approx(ac.iloc[i + 5] / ac.iloc[i] - 1)
    assert row["end_date"] == str(dates[i + 5].date())
    assert row["mfe"] >= row["mae"]


# -- pipeline + dashboard ---------------------------------------------------------------------------
@pytest.mark.slow
def test_shadow_pipeline_writes_discovery_without_orders_and_dashboard_counts_match(world):  # noqa: F811
    pipe = DailyPipeline(world, synthetic=True)
    d = pipe.full_bundle().panel.dates[-30]
    res = pipe.run(d)
    assert not res.errors and res.steps["discover"] == "succeeded"
    run = world.db.fetchone("SELECT * FROM discovery_runs WHERE run_id=?", (res.run_id,))
    assert run is not None
    cands = world.db.fetchall("SELECT status FROM discovery_candidates WHERE discovery_run_id=?",
                              (run["discovery_run_id"],))
    assert cands and {c["status"] for c in cands} <= {"DISCOVERED", "WATCH", "VALIDATION_PENDING", "REJECTED"}
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0
    # the decision tables are untouched by discovery
    n_c = world.db.fetchone("SELECT COUNT(*) AS n FROM candidates")["n"]
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM decisions")["n"] == n_c
    import json
    funnel = json.loads(run["funnel_json"])
    # the pool holds discovery setups AND strategy signals; "discovered" counts rows where a family fired
    origins = [r["origin"] for r in world.db.fetchall(
        "SELECT origin FROM discovery_candidates WHERE discovery_run_id=?", (run["discovery_run_id"],))]
    assert funnel["pool"] == len(cands) and funnel["paper_eligible"] == 0
    assert funnel["discovered"] == sum(o in ("DISCOVERY", "BOTH") for o in origins)
    assert funnel["strategy_only"] == sum(o == "STRATEGY" for o in origins)
    assert funnel["discovered"] >= funnel["high_ranked"] >= funnel["watchlist"]
    c = TestClient(create_app(world))
    page = c.get("/")
    assert page.status_code == 200 and "Top opportunities" in page.text and "PAPER ONLY" in page.text
    assert f"{funnel['discovered']} market opportunities discovered" in page.text
    assert "0 are currently paper eligible" in page.text
    api = c.get("/api/scan").json()
    by = {s["key"]: s["n"] for s in api["funnel"]}
    assert by["discovered"] == funnel["discovered"] and by["paper_trades"] == 0
    assert by["full_universe"] >= by["basic"] >= by["discovered"] >= by["watchlist"]
    assert {r["key"] for r in api["coverage"]} >= {"price_volume", *SCORED, *CONTEXT}
    # resuming the same run re-runs discovery idempotently: no duplicate rows, report keeps its section
    res2 = pipe.run(d, resume_run_id=res.run_id)
    assert not res2.errors
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM discovery_runs WHERE run_id=?", (res.run_id,))["n"] == 1
    # links are separated by data kind: a synthetic run is never linked to a real-data scan
    from quantlab.discovery import strategy_links
    assert strategy_links(world.db, d, None, synthetic=False) is None
    assert strategy_links(world.db, d, None, synthetic=True) is not None


# -- regressions from the adversarial review ---------------------------------------------------------
def test_only_placed_orders_count_as_traded_and_kill_switch_is_named(cfg, scan):
    base = {"strategy_version": "1.0.0", "reasons": "", "ev_bps": 25.0, "refusal": None, "strategy_id": "breakout"}
    links = {"VOLX": [{**base, "candidate_id": "v", "decision": "TRADE", "reject_stage": "NONE", "order_status": "rejected"}],
             "BRKO": [{**base, "candidate_id": "b", "decision": "NO_TRADE", "reject_stage": "EXECUTION", "order_status": None,
                       "reasons": "risk.system_state: system_state SYSTEM_PAUSED: new orders are blocked"}]}
    a = DiscoveryEngine(cfg).assess(scan, links, {"breakout": {"strategy_id": "breakout", "status": "PAPER_ELIGIBLE",
                                                               "evidence": None}})
    by = {c["symbol"]: c for c in a.candidates}
    assert by["VOLX"]["status"] == "PAPER_ELIGIBLE" and by["VOLX"]["block_stage"] == "EXECUTION"
    assert "rejected" in by["VOLX"]["block_reason"] and a.funnel["paper_trades"] == 0
    assert by["BRKO"]["block_stage"] == "SYSTEM_PAUSED" and a.funnel["session_orders_placed"] == 0


def test_news_coverage_is_per_symbol_and_window(cfg):
    from quantlab.discovery.engine import news_coverage
    b = crafted_bundle(news_for=("MOMO", "VOLX"))
    nw = b.news.copy()
    old = nw["symbol"] == "VOLX"
    nw.loc[old, "available_at"] = pd.to_datetime(nw.loc[old, "available_at"], utc=True) - pd.Timedelta(days=200)
    from dataclasses import replace
    b2 = replace(b, news=nw)
    t = DiscoveryEngine(cfg).scan(b2, b2.panel.dates[-1]).table
    assert t.at["VOLX", "context"]["news"]["state"] == "UNKNOWN"
    assert "not current" in t.at["VOLX", "context"]["news"]["why"]
    cov = news_coverage(nw, pd.Timestamp("2000-01-01", tz="UTC"), b2.calendar.cutoff(b2.panel.dates[-1]), 3)
    assert cov["VOLX"][0] is False


def test_universe_check_matches_the_pipeline_universe_on_full_history(cfg, cb, scan):
    from quantlab.universe import UniverseEngine
    full = UniverseEngine(cfg).explain(cb.truncate(cb.panel.dates[-1]), cb.panel.dates[-1]).set_index("symbol")["reason"]
    assert (scan.table["universe_reason"] == full.reindex(scan.table.index)).all()


def test_missing_end_bar_is_retried_not_finalised(cfg, dctx):
    from dataclasses import replace
    from quantlab.data.panel import Panel
    b = crafted_bundle()
    dates = b.panel.dates
    d = dates[-30]
    run_discovery(dctx, b, d, links={})
    sym = dctx.db.fetchone("SELECT symbol FROM discovery_candidates LIMIT 1")["symbol"]
    fields = {k: v.copy() for k, v in b.panel.fields.items()}
    i = dates.get_loc(d)
    for k in fields:
        if fields[k].dtypes.iloc[0].kind == "f":
            fields[k].loc[dates[i + 1], sym] = np.nan              # one-session data gap at the 1-day horizon
    gap = replace(b, panel=Panel(fields, dict(b.panel.meta)))
    DiscoveryOutcomeTracker(dctx.db, dctx.config).update(gap, dates[i + 3], True)
    row = dctx.db.fetchone("SELECT * FROM discovery_outcomes WHERE discovery_id LIKE ? AND horizon_sessions=1",
                           (f"%:{sym}",))
    assert row is None                                            # not finalised as DELISTED from a gap
    r3 = dctx.db.fetchone("SELECT * FROM discovery_outcomes WHERE discovery_id LIKE ? AND horizon_sessions=3",
                          (f"%:{sym}",))
    assert r3 is not None and r3["status"] == "MATURED" and r3["catalyst_persisted"] is None   # no news feed


def test_market_context_nan_is_unknown_not_a_crash(cfg):
    from dataclasses import replace
    from quantlab.data.panel import Panel
    b = crafted_bundle()
    fields = {k: v.copy() for k, v in b.panel.fields.items()}
    fields["aclose"].loc[b.panel.dates[-64], "SPY"] = np.nan
    b2 = replace(b, panel=Panel(fields, dict(b.panel.meta)))
    mc = DiscoveryEngine(cfg).scan(b2, b2.panel.dates[-1]).market_context
    assert mc["sector_etfs"] == [] and "unavailable" in mc["note"]
