"""Unified candidate pool (discovery + strategy signals), forward-outcome research, score
redundancy, the overnight / next-session mode (cutoffs, post-close inclusion, future exclusion,
pre-open recheck) and the dashboard terminal. All data is SYNTHETIC; timestamps are synthetic too."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.core.costs import CostModel
from quantlab.dashboard.app import create_app
from quantlab.data import schemas
from quantlab.discovery import DiscoveryOutcomeTracker, run_discovery
from quantlab.discovery import nextsession as ns
from quantlab.discovery import research as R
from quantlab.discovery.engine import DiscoveryEngine, _levels, next_session_info
from quantlab.monitoring.killswitch import KillSwitch

from .world import BENCH, crafted_bundle

ROOT = Path(__file__).resolve().parents[2]
UTC = "UTC"


@pytest.fixture(scope="module")
def cfg():
    return load_config(root=ROOT, overrides={"benchmarks": BENCH})


@pytest.fixture(scope="module")
def cb():
    return crafted_bundle()


@pytest.fixture
def dctx(tmp_path):
    var = tmp_path / "var"
    c = load_config(root=ROOT, overrides={"benchmarks": BENCH, "project": {
        "var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"), "log_dir": str(var / "logs"),
        "report_dir": str(var / "reports"), "model_dir": str(var / "models")}})
    ctx = AppContext.create(c, init_logging=False)
    yield ctx
    ctx.close()


def _link(sid, decision, stage, order=None, refusal=None):
    return {"candidate_id": f"c-{sid}", "strategy_id": sid, "strategy_version": "1.0.0", "decision": decision,
            "reject_stage": stage, "reasons": "", "ev_bps": 20.0, "order_status": order, "refusal": refusal}


PE = {"breakout": {"strategy_id": "breakout", "status": "PAPER_ELIGIBLE", "evidence": None},
      "mean_reversion": {"strategy_id": "mean_reversion", "status": "SHADOW", "evidence": None}}


# -- PHASE 1: master candidate pool ------------------------------------------------------------------
def test_strategy_signal_survives_a_low_discovery_score(cfg, cb):
    eng = DiscoveryEngine(cfg)
    scan = eng.scan(cb, cb.panel.dates[-1])
    low = scan.table.sort_values("score").index[0]                    # the LOWEST-scored scanned symbol
    assert not scan.table.at[low, "fired"]
    links = {low: [_link("breakout", "TRADE", "NONE", order="accepted")],
             "MOMO": [_link("mean_reversion", "NO_TRADE", "STRATEGY")],
             "ZZNOTSCANNED": [_link("breakout", "TRADE", "NONE")]}
    a = eng.assess(scan, links, PE)
    by = {c["symbol"]: c for c in a.candidates}
    assert by[low]["origin"] == "STRATEGY" and by[low]["status"] == "TRADED"   # not suppressed by its score
    assert not by[low]["high_quality"] and not by[low]["on_watchlist"]
    assert by["ZZNOTSCANNED"]["origin"] == "STRATEGY" and by["ZZNOTSCANNED"]["scanned"] is False
    assert by["ZZNOTSCANNED"]["status"] == "PAPER_ELIGIBLE" and by["ZZNOTSCANNED"]["score"] is None
    assert by["MOMO"]["origin"] == "BOTH" and by["MOMO"]["status"] == "VALIDATION_PENDING"
    assert by["VOLX"]["origin"] == "DISCOVERY"
    f = a.funnel
    assert f["strategy_only"] == f["missed_discovery_signals"] == 2 and f["both"] == 1
    assert f["pool"] == f["discovery_only"] + f["strategy_only"] + f["both"] == len(a.candidates)
    assert f["strategy_signals"] == 3 and f["paper_trades"] == 1 and f["paper_eligible"] == 2
    assert f["strategy_symbols_not_discovered_count"] == f["strategy_only"]


# -- PHASE 2/3: research -----------------------------------------------------------------------------
def test_research_replay_uses_only_past_data_and_outcomes_are_cost_adjusted(cfg):
    b = crafted_bundle(n_days=420)
    dates = b.panel.dates[310:380:7]
    obs = R.replay(cfg, b, dates)
    # corrupt everything after the last replay date: the replayed features must not change
    from dataclasses import replace
    from quantlab.data.panel import Panel
    fields = {k: v.copy() for k, v in b.panel.fields.items()}
    for k in ("open", "high", "low", "close", "aopen", "ahigh", "alow", "aclose", "volume", "dollar_volume"):
        fields[k].loc[fields[k].index > dates[-1]] *= 9.0
    obs2 = R.replay(cfg, replace(b, panel=Panel(fields, dict(b.panel.meta))), dates)
    pd.testing.assert_frame_equal(obs.reset_index(drop=True), obs2.reset_index(drop=True))
    costs = CostModel.from_config(cfg)
    out = R.attach_outcomes(obs, b, costs)
    r = out.dropna(subset=["net_5"]).iloc[0]
    i = b.panel.dates.get_loc(r["date"])
    entry, exit_ = b.panel.aopen[r["symbol"]].iloc[i + 1], b.panel.aclose[r["symbol"]].iloc[i + 5]
    assert r["gross_5"] == pytest.approx(exit_ / entry - 1)                       # NEXT open, not D's close
    ow = costs.one_way_cost_frac(r["adv20"])
    assert r["cost_5"] == pytest.approx(ow * (1 + exit_ / entry))
    assert r["net_5"] == pytest.approx(r["gross_5"] - r["cost_5"])
    stop = b.panel.dates[b.panel.dates.get_loc(dates[-1]) + 3]
    clipped = R.attach_outcomes(obs, b, costs, stop_before=stop)
    last = clipped[clipped["date"] == dates[-1]]
    assert last["net_5"].isna().all() and last["net_1"].notna().any()            # no bar at/after the stop is used


def test_unknown_stays_unknown_in_research_and_minimum_samples_apply(cfg):
    b = crafted_bundle(n_days=420)
    obs = R.replay(cfg, b, b.panel.dates[330:400:10])
    newb = obs[obs["symbol"] == "NEWB"]
    assert not newb.empty and newb["pts_momentum"].notna().all()                  # 20d/60d known
    out = R.attach_outcomes(obs, b, CostModel.from_config(cfg))
    s = R.analyse(out, min_obs=10**6, min_dates=5)
    assert all(g["verdict"] in ("INSUFFICIENT_SAMPLE", "BASELINE") for g in s["groups"])
    s2 = R.analyse(out, min_obs=5, min_dates=3)
    assert any(g["verdict"] in ("FLAT", "PROMISING", "POOR") for g in s2["groups"])


def test_score_redundancy_detects_momentum_and_relative_strength_overlap(cfg):
    b = crafted_bundle(n_days=420)
    obs = R.replay(cfg, b, b.panel.dates[330:410:5])
    red = R.redundancy(obs)
    m = np.array(red["spearman_matrix"], dtype=float)
    assert np.allclose(m, m.T, equal_nan=True) and np.allclose(np.diag(m), 1.0)
    pair = next(p for p in red["pairs"] if {p["a"], p["b"]} == {"Momentum", "Relative strength"})
    assert pair["spearman_points"] > 0.8
    assert all(p["jaccard_fired"] is None or 0 <= p["jaccard_fired"] <= 1 for p in red["pairs"])


# -- next-session mode --------------------------------------------------------------------------------
def test_market_state_and_exchange_calendar():
    fri_evening = pd.Timestamp("2024-03-22 20:30", tz="America/New_York")
    st = ns.market_state(fri_evening.tz_convert(UTC))
    assert st["status"] == "CLOSED" and st["last_completed_session"] == "2024-03-22" and st["next_session"] == "2024-03-25"
    thu = ns.market_state(pd.Timestamp("2024-03-28 17:00", tz="America/New_York").tz_convert(UTC))
    assert thu["next_session"] == "2024-04-01"                                   # Good Friday is closed
    opn = ns.market_state(pd.Timestamp("2024-03-25 10:00", tz="America/New_York").tz_convert(UTC))
    assert opn["status"] == "OPEN" and opn["current_session"] == "2024-03-25"
    info = next_session_info(pd.Timestamp("2024-03-22"))
    assert info["next_session"] == "2024-03-25"
    assert info["info_cutoff_at"] == pd.Timestamp("2024-03-22 16:00", tz="America/New_York").tz_convert(UTC).isoformat()


def test_information_phases():
    d, n = "2024-03-22", "2024-03-25"
    et = lambda s: pd.Timestamp(s, tz="America/New_York")   # noqa: E731
    assert ns.info_phase(et("2024-03-22 15:59"), d, n) == "REGULAR_SESSION"
    assert ns.info_phase(et("2024-03-22 17:00"), d, n) == "POST_CLOSE"
    assert ns.info_phase(et("2024-03-23 12:00"), d, n) == "OVERNIGHT"
    assert ns.info_phase(et("2024-03-25 06:00"), d, n) == "PRE_MARKET"
    assert ns.info_phase(et("2024-03-25 09:30"), d, n) == "NEXT_SESSION"


def _news(sym, when, nid):
    t = pd.Timestamp(when, tz="America/New_York").tz_convert(UTC)
    return {"news_id": nid, "symbol": sym, "headline": f"{sym} item {nid}", "summary": "", "source": "synthetic",
            "url": None, "created_at": t.isoformat(), "updated_at": t.isoformat(), "available_at": t, "pit_status": "PIT",
            "provider": "synthetic", "retrieved_at": t.isoformat()}


def _setup_store(ctx, cb, news_rows=(), action_rows=(), bars_end=None):
    d = bars_end or str(cb.panel.dates[-1].date())
    bars = pd.DataFrame([{"symbol": "SPY", "date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                          "vwap": None, "trade_count": None, "provider": "synthetic", "retrieved_at": "2024-01-01T00:00:00Z"}])
    ctx.store.write("bars", bars, "synthetic", is_synthetic=True)
    acts = pd.DataFrame(list(action_rows) or [{"symbol": "SPY", "ex_date": "2023-06-01", "action_type": "cash_dividend",
                                               "ratio": None, "amount": 1.0, "declared_date": None,
                                               "available_at": "2023-06-01T13:30:00Z", "pit_status": "PIT_CONSERVATIVE",
                                               "source_id": "a0", "provider": "synthetic",
                                               "retrieved_at": "2024-01-01T00:00:00Z"}])
    ctx.store.write("corporate_actions", acts, "synthetic", is_synthetic=True)
    if news_rows:
        ctx.store.write("news", pd.DataFrame(list(news_rows), columns=schemas.NEWS), "synthetic", is_synthetic=True)


def test_overnight_refresh_includes_post_close_and_excludes_the_future(dctx, cb):
    d = cb.panel.dates[-1]                                                         # 2024-03-22 (Fri)
    _setup_store(dctx, cb, news_rows=[
        _news("DROP", "2024-03-22 15:00", "reg"),       # before the cutoff: already regular-session information
        _news("DROP", "2024-03-22 17:00", "post"),      # POST_CLOSE
        _news("DROP", "2024-03-25 06:00", "pre"),       # PRE_MARKET
        _news("DROP", "2024-03-25 10:00", "next")])     # after the open: never an input
    dr = run_discovery(dctx, cb, d, links={})
    assert dr.assessment.candidates and dctx.db.fetchone("SELECT next_session FROM discovery_runs")["next_session"] == "2024-03-25"
    status_drop = next(c["status"] for c in dr.assessment.candidates if c["symbol"] == "DROP")
    assert status_drop == "DISCOVERED"
    r1 = ns.overnight_refresh(dctx, pd.Timestamp("2024-03-23 12:00", tz=UTC))
    assert r1["ok"] and r1["recorded"] == 1 and r1["promoted"] == 1
    row = dctx.db.fetchone("SELECT * FROM overnight_updates")
    assert row["phase"] == "POST_CLOSE" and row["source_id"] == "post" and row["effect"] == "PROMOTED_TO_WATCH"
    r2 = ns.overnight_refresh(dctx, pd.Timestamp("2024-03-25 12:00", tz=UTC))       # 08:00 ET, pre-market
    assert r2["recorded"] == 1
    phases = {r["source_id"]: r["phase"] for r in dctx.db.fetchall("SELECT source_id, phase FROM overnight_updates")}
    assert phases == {"post": "POST_CLOSE", "pre": "PRE_MARKET"}                   # 'reg' and 'next' never used
    r3 = ns.overnight_refresh(dctx, pd.Timestamp("2024-03-25 14:00", tz=UTC))       # after the open
    assert not r3["ok"] and "already opened" in r3["reason"]
    st = ns.next_session_state(dctx, pd.Timestamp("2024-03-25 12:30", tz=UTC))
    assert st["counts"]["promoted_overnight"] == 1
    assert dctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0          # nothing becomes a trade


def test_preopen_recheck_invalidates_rejects_and_never_uses_the_open(dctx, cb):
    d = cb.panel.dates[-1]
    split = lambda sym, avail, sid: {"symbol": sym, "ex_date": "2024-03-25", "action_type": "split", "ratio": 2.0,   # noqa: E731
                                     "amount": None, "declared_date": None, "available_at": avail,
                                     "pit_status": "PIT", "source_id": sid, "provider": "synthetic",
                                     "retrieved_at": "2024-01-01T00:00:00Z"}
    _setup_store(dctx, cb, action_rows=[split("DAC0", "2024-03-23T00:00:00Z", "s0")])
    dr = run_discovery(dctx, cb, d, links={})
    cands = {c["symbol"]: c for c in dr.assessment.candidates}
    watched = [s for s, c in cands.items() if c["status"] in ("WATCH", "DISCOVERED")]
    target, late = watched[0], watched[1]
    _setup_store(dctx, cb, action_rows=[split(target, "2024-03-23T00:00:00Z", "s1"),        # known before the open
                                        split(late, "2024-03-25T13:30:00Z", "s2")])        # only known AT the open
    res = ns.preopen_recheck(dctx, pd.Timestamp("2024-03-25 12:00", tz=UTC))
    assert res["ok"]
    got = {r["symbol"]: r for r in dctx.db.fetchall("SELECT symbol, status_after, reason FROM preopen_checks")}
    assert got[target]["status_after"] == "INVALIDATED"
    assert got[late]["status_after"] == cands[late]["status"]                    # future information excluded
    KillSwitch(dctx.db).pause("test", trigger="manual", actor="human:test")
    ns.preopen_recheck(dctx, pd.Timestamp("2024-03-25 12:10", tz=UTC))
    after = dctx.db.fetchall("SELECT status_after FROM preopen_checks WHERE checked_at LIKE '2024-03-25T12:10%'")
    assert after and {r["status_after"] for r in after} <= {"REJECTED", "INVALIDATED"}
    late_run = ns.preopen_recheck(dctx, pd.Timestamp("2024-03-25 13:30", tz=UTC))
    assert not late_run["ok"]                                                     # the open has occurred: refused
    assert dctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0


def test_preopen_transition_rules():
    t = ns.decide_transition
    kw = dict(missing_inputs=[], corporate_action=False, blocked=[], trade_decision=False, strategy_eligible=False,
              next_session="2024-03-25")
    assert t("WATCH", **kw)[0] == "WATCH"
    assert t("WATCH", **{**kw, "missing_inputs": ["stale"]})[0] == "UNKNOWN"
    assert t("WATCH", **{**kw, "corporate_action": True})[0] == "INVALIDATED"
    assert t("WATCH", **{**kw, "blocked": ["kill switch"]})[0] == "REJECTED"
    assert t("PAPER_ELIGIBLE", **{**kw, "trade_decision": True, "strategy_eligible": True})[0] == "PAPER_ELIGIBLE"
    assert t("PAPER_ELIGIBLE", **kw)[0] == "REJECTED"
    assert t("WATCH", **{**kw, "trade_decision": True, "strategy_eligible": True})[0] == "WATCH"   # cannot create eligibility


def test_stale_data_makes_preopen_unknown_and_alerts_fire(dctx, cb):
    d = cb.panel.dates[-1]
    _setup_store(dctx, cb, bars_end="2024-03-20")                                 # latest bar older than D
    run_discovery(dctx, cb, d, links={})
    ns.preopen_recheck(dctx, pd.Timestamp("2024-03-25 12:00", tz=UTC))
    assert {r["status_after"] for r in dctx.db.fetchall("SELECT status_after FROM preopen_checks")} == {"UNKNOWN"}
    # a week later, with no new end-of-day scan: stale list, missing scan, missing latest bar
    codes = {a["code"] for a in ns.next_session_alerts(dctx, pd.Timestamp("2024-03-29 01:00", tz=UTC))}
    assert {"STALE_CANDIDATE_LIST", "NO_OVERNIGHT_SCAN", "MISSING_LATEST_BAR"} <= codes


def test_conditional_setup_record_and_outcome_timestamps(dctx, cb):
    d = cb.panel.dates[-40]
    dr = run_discovery(dctx, cb, d, links={})
    c = dr.assessment.candidates[0]
    su = c["setup"]
    assert su["conditional"] is True and "Not an order" in su["condition"] and su["relevance"] == "NEXT_SESSION"
    assert su["confirm"] and su["invalidate"] and any("pre-market price: UNKNOWN" in m for m in su["missing"])
    assert not any(k in su for k in ("predicted_price", "target_price", "expected_return"))
    row = dctx.db.fetchone("SELECT * FROM discovery_candidates WHERE symbol=?", (c["symbol"],))
    assert row["relevance"] == "NEXT_SESSION" and row["info_cutoff_at"] and row["discovered_at"] and row["origin"]
    tr = DiscoveryOutcomeTracker(dctx.db, dctx.config)
    i = cb.panel.dates.get_loc(d)
    tr.update(cb, cb.panel.dates[i + 5], True)
    o = dctx.db.fetchone("SELECT * FROM discovery_outcomes WHERE discovery_id=? AND horizon_sessions=5",
                         (row["discovery_id"],))
    nxt = cb.panel.dates[i + 1]
    assert o["next_open_at"] == pd.Timestamp(f"{nxt.date()} 09:30", tz="America/New_York").tz_convert(UTC).isoformat()
    assert o["known_before_open"] == 0            # produced now (wall clock), long after that historical open
    ao = cb.panel.aopen[c["symbol"]]
    assert o["open_ret"] == pytest.approx(cb.panel.aclose[c["symbol"]].iloc[i + 5] / ao.iloc[i + 1] - 1)
    assert o["net_ret"] == pytest.approx(o["open_ret"] - o["cost_ret"])


# -- dashboard ------------------------------------------------------------------------------------------
def test_terminal_funnel_matches_the_database(dctx, cb):
    d = cb.panel.dates[-1]
    _setup_store(dctx, cb)
    dr = run_discovery(dctx, cb, d, links={"N001": [_link("breakout", "NO_TRADE", "STRATEGY")]})
    c = TestClient(create_app(dctx))
    at = "2024-03-25T12:00:00Z"
    api = c.get("/api/scan", params={"at": at}).json()
    f = dr.assessment.funnel
    by = {s["key"]: s["n"] for s in api["funnel"]}
    for k in ("full_universe", "basic", "discovered", "strategy_signals", "both", "watchlist", "validation_pending",
              "paper_eligible", "paper_trades"):
        assert by[k] == f[k], k
    assert api["market"]["next_session"] == "2024-03-25" and api["next"]["run"]["next_session"] == "2024-03-25"
    assert api["next"]["mismatch"] is None
    page = c.get("/", params={"at": at}).text
    assert "NO CURRENT PAPER TRADE" in page and "Next session setups" in page and "PAPER ONLY" in page
    assert "Missed discovery signals" in page and "What would confirm it" in page
    # a week later without a new scan the terminal must say the setups are stale, not present them as current
    page2 = c.get("/", params={"at": "2024-03-29T12:00:00Z"}).text
    assert "STALE" in page2


def test_derived_setup_inputs_and_levels_are_truncation_invariant(cfg, cb):
    """new_high_20/50, range_expansion, prev_contraction, close_location, accel, the fired masks and
    the conditional-setup levels read only rows <= D, including when the FeatureSet spans later rows."""
    from quantlab.features.base import FeatureSet
    from quantlab.testing.pit import assert_truncation_invariant
    eng = DiscoveryEngine(cfg)
    check = [cb.panel.dates[-45], cb.panel.dates[-12], cb.panel.dates[-1]]
    derived = ["new_high_20", "new_high_50", "range_expansion", "prev_contraction", "close_location", "accel"]
    benches = {cb.market_symbol, *cb.sector_etfs}

    def compute(b):
        fs = FeatureSet(b, dtype="float64")
        out = {}
        for d in check:
            if d > b.panel.dates[-1]:
                continue
            c = eng.core(b.panel, fs, d, benches)
            f = c["xs"][derived].copy()
            for fam, m in c["masks"].items():
                f[f"fired_{fam}"] = m.astype(float)
            lv = pd.DataFrame({s: _levels(c["xs"].loc[s], c["close"].get(s)) for s in c["syms"]}).T
            for col in ("atr", "ma20", "ma50", "high_55_prior"):
                f[f"lvl_{col}"] = pd.to_numeric(lv[col], errors="coerce")
            out[d] = f
        return pd.concat(out, names=["date", "symbol"])
    assert_truncation_invariant(compute, cb, check_dates=check, name="discovery setup inputs")


def test_setup_conditions_are_never_already_met_at_the_decision_close():
    """A level the close is already beyond must be something to reclaim, not a live invalidation
    (the 2026-09-24 dry run showed 'closes below the 50-day average' for names already below it)."""
    from quantlab.discovery.engine import setup_record
    import re

    def rec(fired, **lv):
        return setup_record({"fired": fired, "levels": {"atr": 1.0, **lv}, "status": "WATCH", "links": []})
    below = rec(["relative_strength", "breakout_compression"], close=66.91, ma50=68.81, high_55_prior=72.0)
    above = rec(["momentum", "breakout_compression"], close=110.0, ma50=100.0, high_55_prior=106.0)
    for r, close in ((below, 66.91), (above, 110.0)):
        for text in r["invalidate"]:
            m = re.search(r"closes (?:back )?below (?:the )?[\w\- ]*?([\d,]+\.\d+)", text)
            if m:
                assert float(m.group(1).replace(",", "")) < close, text      # not already true at D
        for text in r["confirm"]:
            m = re.search(r"holds above (?:the |its )?[\w\- ]*?([\d,]+\.\d+)", text)
            if m:
                assert float(m.group(1).replace(",", "")) <= close, text
    assert any("reclaims its 50-day average 68.81" in t for t in below["confirm"])
    assert any("through the prior 55-day high 72.00" in t for t in below["confirm"])
    assert any("closes below the 50-day average 100.00" in t for t in above["invalidate"])
    assert any("closes back below the breakout level 106.00" in t for t in above["invalidate"])
    unknown = rec(["momentum"], close=10.0)
    assert any("UNKNOWN" in t for t in unknown["confirm"])                      # missing level is UNKNOWN, not 0
