"""Regime throttle and tracked watchlist of the PAPER_EXPLORATION layer. SYNTHETIC data, stub brokers.

Regime throttle: D's regime snapshot (exact date, never later) lowers the new-entry cap when SPY is
below its 200-day average; UNKNOWN never throttles; the regime is recorded on every decision.
Tracked watchlist: recorded every session, scored by outcomes, NEVER traded, sized, counted or pending."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import quantlab.dashboard.app as dash
from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.db.database import to_json, utcnow_iso
from quantlab.discovery import run_discovery
from quantlab.exploration import ExplorationOutcomeTracker, ExplorationPolicy, experiment_results, plan_exploration, \
    preopen_submit
from quantlab.exploration.engine import _pending_symbols, held_industries, hold_for, learning_report, normalize_symbols
from quantlab.exploration.regime_throttle import RegimeThrottle, throttle_from_snapshot

from tests.discovery.world import BENCH, crafted_bundle

from .test_sanity import PRE_OPEN, SESSION, StubExec, _decision, _store

ROOT = Path(__file__).resolve().parents[2]
NO = object()


@pytest.fixture(scope="module")
def cb():
    return crafted_bundle()


def _ctx(tmp_path, name="v", mode="EXPLORATION", **exploration):
    var = tmp_path / name
    c = load_config(root=ROOT, overrides={"benchmarks": BENCH, "paper": {"mode": mode}, "exploration": exploration,
                                          "project": {"var_dir": str(var), "db_path": str(var / "q.db"),
                                                      "data_dir": str(var / "data"), "log_dir": str(var / "l"),
                                                      "report_dir": str(var / "r"), "model_dir": str(var / "m")}})
    return AppContext.create(c, init_logging=False)


def _snapshot(ctx, date, trend, label="bear_calm"):
    ctx.db.insert("regime_snapshots", {"as_of_date": date, "label": label, "run_id": None, "created_at": utcnow_iso(),
                                       "metrics_json": to_json({"market_trend_200": trend, "label": label})})


def _plan(tmp_path, cb, name, *, snaps=(), when=None, **exploration):
    ctx = _ctx(tmp_path, name, **exploration)
    _store(ctx)
    dr = run_discovery(ctx, cb, cb.panel.dates[-1] if when is None else when, links={})
    for date, trend in snaps:
        _snapshot(ctx, date, trend)
    res = plan_exploration(ctx, equity=100_000.0, now=PRE_OPEN, run_id=dr.discovery_run_id, scan=dr.scan)
    return ctx, dr, res


def _rows(ctx):
    out = {}
    for r in ctx.db.fetchall("SELECT * FROM exploration_decisions"):
        out[r["symbol"]] = {**dict(r), "pre": json.loads(r["pre_trade_json"])}
    return out


def _picks(rows):
    """SELECTED symbols in the order they were picked."""
    sel = [r for r in rows.values() if r["selection"] == "SELECTED"]
    return [r["symbol"] for r in sorted(sel, key=lambda r: r["pre"]["candidate"]["selection_order"])]


def _plain(rows):
    return {s: (r["selection"], r["qty"], r["stop_price"], r["holding_sessions"]) for s, r in rows.items()}


# -- configuration --------------------------------------------------------------------------------------
def test_repo_defaults():
    cfg = load_config(root=ROOT)
    assert RegimeThrottle.from_config(cfg) == RegimeThrottle(enabled=True, metric="market_trend_200", below=0.0,
                                                             max_new_per_session=2)
    assert ExplorationPolicy.from_config(cfg).tracked_watchlist == ()            # owner's list: config/local.yaml
    assert normalize_symbols([" aapl", "MSFT", "aapl", None, ""]) == ("AAPL", "MSFT")
    assert normalize_symbols("nvda") == ("NVDA",) and normalize_symbols(None) == ()


# -- regime throttle ------------------------------------------------------------------------------------
def test_throttle_below_threshold_caps_new_entries_and_watches_the_rest(tmp_path, cb):
    base, _, _ = _plan(tmp_path, cb, "base")
    b = _rows(base)
    picks = _picks(b)
    assert len(picks) >= 3                                                      # the cap must bind in this world
    ctx, _, res = _plan(tmp_path, cb, "thr", snaps=[(SESSION, -0.05)])
    rows = _rows(ctx)
    assert res["regime"]["state"] == "THROTTLED" and res["regime"]["effective_max_new"] == 2
    assert _picks(rows) == picks[:2]                                            # same ranking, smaller cap
    displaced = {s for s, r in rows.items() if r["pre"]["regime"]["displaced_by_throttle"]}
    assert displaced == set(picks[2:])                                          # the rest of the normal budget...
    for s in displaced:                                                         # ...is watched, naming the throttle
        assert rows[s]["selection"] == "WATCHED_NOT_TRADED" and rows[s]["qty"] == 0
        assert "regime throttle" in rows[s]["reason"] and "market_trend_200" in rows[s]["reason"]
    ordinary = lambda rs: {s for s, r in rs.items() if r["selection"] == "WATCHED_NOT_TRADED"   # noqa: E731
                           and not r["pre"]["regime"]["displaced_by_throttle"]}
    assert ordinary(rows) == ordinary(b)                                        # the ordinary comparison group is kept
    for r in rows.values():                                                     # the regime is on EVERY decision
        g = r["pre"]["regime"]
        assert (g["as_of_date"], g["label"], g["market_trend_200"], g["throttled"], g["effective_max_new"]) == \
               (SESSION, "bear_calm", -0.05, True, 2)
    planned = {e["decision_id"] for e in ctx.db.fetchall("SELECT decision_id FROM exploration_events WHERE event='PLANNED'")}
    assert planned == {r["decision_id"] for r in rows.values() if r["selection"] == "SELECTED"}
    assert all("regime throttle" in rows[s]["reason"] for s in picks[:2])
    base.close()
    ctx.close()


@pytest.mark.parametrize("snaps,extra,state", [
    ([(SESSION, 0.05)], {}, "NOT_THROTTLED"),
    ([(SESSION, 0.0)], {}, "NOT_THROTTLED"),                                    # "below" is strict
    ([(SESSION, -0.05)], {"regime_throttle": {"enabled": False}}, "DISABLED"),
    ([], {}, "UNKNOWN"),                                                        # no snapshot for D
    ([(SESSION, None)], {}, "UNKNOWN"),                                         # snapshot, metric unknown
])
def test_no_throttle_above_threshold_disabled_or_unknown(tmp_path, cb, snaps, extra, state):
    base, _, _ = _plan(tmp_path, cb, "base")
    ctx, _, res = _plan(tmp_path, cb, "x", snaps=snaps, **extra)
    rows = _rows(ctx)
    assert _plain(rows) == _plain(_rows(base))                                  # planning exactly as without it
    assert res["regime"]["state"] == state and not res["regime"]["throttled"]
    for r in rows.values():
        g = r["pre"]["regime"]
        assert g["state"] == state and g["throttled"] is False and g["effective_max_new"] == 5
        assert not g["displaced_by_throttle"]
        if not snaps:
            assert g["label"] == "UNKNOWN" and g["as_of_date"] is None and g["market_trend_200"] is None
    base.close()
    ctx.close()


def test_regime_read_has_no_look_ahead(tmp_path, cb):
    prev, nxt = str(cb.panel.dates[-2].date()), "2024-03-25"
    # only a LATER (and an earlier) session is below its 200-day average: D itself is unknown -> no throttle
    ctx, _, res = _plan(tmp_path, cb, "a", snaps=[(prev, -0.5), (nxt, -0.5)])
    assert res["regime"]["state"] == "UNKNOWN"
    assert all(r["pre"]["regime"]["as_of_date"] is None for r in _rows(ctx).values())
    ctx.close()
    ctx, _, res = _plan(tmp_path, cb, "b", snaps=[(SESSION, 0.02), (nxt, -0.5)])
    assert res["regime"]["state"] == "NOT_THROTTLED" and res["regime"]["value"] == 0.02     # D's, not D+1's
    ctx.close()


def test_throttle_decision_is_truncation_invariant(cb):
    """The throttle input (D's regime metrics) and so the cap at D never change when later data is removed."""
    from quantlab.features.base import FeatureSet
    from quantlab.regime import RegimeEngine
    from quantlab.testing.pit import assert_truncation_invariant
    cfg = load_config(root=ROOT, overrides={"benchmarks": BENCH})
    thr = RegimeThrottle.from_config(cfg)

    def caps(b):
        df = RegimeEngine(cfg).compute(FeatureSet(b))
        out = {}
        for d, row in df.iterrows():
            st = throttle_from_snapshot({"as_of_date": str(d.date()), "label": row["label"],
                                         "metrics": row.drop("label").to_dict()}, str(d.date()), 5, thr)
            out[d] = {"value": st["value"] if st["value"] is not None else float("nan"),
                      "effective_max_new": float(st["effective_max_new"]), "throttled": float(st["throttled"])}
        return pd.DataFrame.from_dict(out, orient="index")
    assert_truncation_invariant(caps, cb, min_history=230, n_dates=4, name="regime throttle")


def test_throttle_never_raises_the_normal_budget(tmp_path, cb):
    ctx, _, res = _plan(tmp_path, cb, "x", snaps=[(SESSION, -0.05)], regime_throttle={"max_new_per_session": 9})
    assert res["regime"]["state"] == "THROTTLED" and res["regime"]["effective_max_new"] == 5
    assert not any(r["pre"]["regime"]["displaced_by_throttle"] for r in _rows(ctx).values())
    ctx.close()


def test_learning_report_splits_by_throttle_and_regime(tmp_path):
    ctx = _ctx(tmp_path)
    reg = {"as_of_date": SESSION, "label": "bear_calm", "market_trend_200": -0.03, "state": "THROTTLED",
           "throttled": True, "effective_max_new": 2}
    a = _decision(ctx, symbol="AAA", pre_trade={"regime": {**reg, "displaced_by_throttle": False}})
    b = _decision(ctx, symbol="OLD")                                            # planned before regimes were recorded
    row = dict(ctx.db.fetchone("SELECT * FROM exploration_decisions WHERE decision_id=?", (a,)))
    ctx.db.insert("exploration_decisions", {**row, "decision_id": "expl_DSP", "symbol": "DSP", "selection": "WATCHED_NOT_TRADED",
                                            "qty": 0, "pre_trade_json": to_json({"regime": {**reg, "displaced_by_throttle": True}})})
    for did, net in ((a, 0.02), (b, -0.01), ("expl_DSP", 0.05)):
        ctx.db.insert("exploration_outcomes", {"decision_id": did, "horizon_sessions": 10, "entry_date": "2024-03-25",
                                               "end_date": "2024-04-08", "entry_price": 50.0, "gross_ret": net,
                                               "cost_ret": 0.0, "net_ret": net, "spy_ret": 0.01, "mfe": 0.06, "mae": -0.02,
                                               "stop_breached": 0, "thesis_valid": 1, "catalyst_persisted": None,
                                               "computed_at": utcnow_iso()})
    rep = learning_report(ctx.db, synthetic=True)
    thr = {(g["throttle"], g["selection"]): g for g in rep["by_throttle"]}
    assert thr[("THROTTLED", "SELECTED")]["n"] == 1 and thr[("NOT_RECORDED", "SELECTED")]["n"] == 1
    assert thr[("THROTTLED", "WATCHED_NOT_TRADED (throttle-displaced)")]["mean_net"] == 0.05
    regs = {(g["regime"], g["selection"]) for g in rep["by_regime"]}
    assert ("bear_calm", "SELECTED") in regs and ("NOT_RECORDED", "SELECTED") in regs
    assert rep["tracked_watchlist"] is None and rep["matured_tracked"] == 0
    json.dumps(rep)
    ctx.close()


# -- tracked watchlist ----------------------------------------------------------------------------------
def test_tracked_symbols_are_recorded_and_never_traded_sized_counted_or_pending(tmp_path, cb):
    base, dr, _ = _plan(tmp_path, cb, "base")
    b = _rows(base)
    picks = _picks(b)
    pool = {r["symbol"] for r in base.db.fetchall("SELECT symbol FROM discovery_candidates")}
    outside = next(s for s in dr.scan.table.index if s not in pool)            # scanned, no family fired
    tracked = [picks[0].lower(), outside, "SPY", "ZZZZ"]                        # a would-be pick, a quiet name,
    ctx, _, res = _plan(tmp_path, cb, "t", tracked_watchlist=tracked)           # a benchmark, an absent symbol
    rows = _rows(ctx)
    tk = {s: r for s, r in rows.items() if r["selection"] == "TRACKED"}
    assert set(tk) == {picks[0], outside, "SPY"} and set(res["tracked"]["recorded"]) == set(tk)
    assert "ZZZZ" in res["tracked"]["not_recorded"]
    # never traded / sized / planned / pending / submitted
    pol = ExplorationPolicy.from_config(ctx.config)
    for s, r in tk.items():
        assert r["qty"] == 0 and r["pre"]["sizing"]["qty"] == 0
        assert not ctx.db.fetchall("SELECT 1 FROM exploration_events WHERE decision_id=?", (r["decision_id"],))
        assert r["holding_sessions"] == hold_for(SESSION, s, pol) == r["pre"]["expected_holding_sessions"]
        assert r["pre"]["regime"]["state"] == "UNKNOWN" and r["pre"]["tracked"]["never_traded"] is True
    assert not set(tk) & _pending_symbols(ctx.db, "EXPLORATION", now=PRE_OPEN)
    # the tradeable pool without the tracked pick: budget and comparison group unchanged in size
    sel = _picks(rows)
    assert picks[0] not in sel and set(picks[1:]) <= set(sel) and len(sel) == len(picks)
    assert res["counts"]["WATCHED_NOT_TRADED"] <= pol.watched_not_traded
    assert sum(held_industries(ctx.db, "BOT").values()) == 0                    # nothing open: tracked never counts
    # features, selection score, universe percentile and upside like other candidates
    for s in (picks[0], outside):
        p = tk[s]["pre"]
        assert p["features"]["discovery"] and p["candidate"]["selection_score"] is not None
        assert 0 < p["tracked"]["universe_percentile"] <= 1
        assert p["tracked"]["universe_n"] == int(dr.scan.table["selection_score"].notna().sum())
        assert p["upside"]["state"] == "KNOWN" and p["upside"]["hold"] == tk[s]["holding_sessions"]
        assert 0 < tk[s]["stop_price"] < tk[s]["ref_price"]                     # hypothetical: outcome scoring only
    assert tk[picks[0]]["discovery_id"] and tk[outside]["origin"] == "TRACKED"
    assert tk["SPY"]["pre"]["candidate"]["selection_score"] is None and "benchmark" in tk["SPY"]["pre"]["tracked"]["source"]
    # pre-open submission never touches them
    ex = StubExec()
    sub = preopen_submit(ctx, ex, now=PRE_OPEN)
    assert {c["symbol"] for c in ex.calls} == set(sel) and sub["submitted"] == len(sel)
    for r in tk.values():
        assert not ctx.db.fetchall("SELECT 1 FROM exploration_events WHERE decision_id=?", (r["decision_id"],))
    assert plan_exploration(ctx, equity=100_000.0, now=PRE_OPEN, scan=dr.scan).get("already_planned")
    base.close()
    ctx.close()


def test_empty_or_absent_watchlist_is_a_no_op(tmp_path, cb):
    base, _, res0 = _plan(tmp_path, cb, "base")
    assert "tracked" not in res0
    for name, extra in (("empty", {"tracked_watchlist": []}), ("none", {"tracked_watchlist": None}),
                        ("absent", {"tracked_watchlist": ["ZZZZ"]})):
        ctx, _, res = _plan(tmp_path, cb, name, **extra)
        assert _plain(_rows(ctx)) == _plain(_rows(base)) and res["counts"] == res0["counts"], name
        assert not ctx.db.fetchall("SELECT 1 FROM exploration_decisions WHERE selection='TRACKED'")
        ctx.close()
    base.close()


def test_tracked_outcomes_feed_the_learning_report_and_the_dashboard(tmp_path, cb, monkeypatch):
    when = cb.panel.dates[-30]                                                  # 29 later sessions: every hold matures
    probe, dr, _ = _plan(tmp_path, cb, "probe", when=when)
    pool = {r["symbol"] for r in probe.db.fetchall("SELECT symbol FROM discovery_candidates")}
    quiet = [s for s in dr.scan.table.index if s not in pool][:2]
    probe.close()
    ctx, _, res = _plan(tmp_path, cb, "t", when=when, tracked_watchlist=quiet)
    assert sorted(res["tracked"]["recorded"]) == sorted(quiet)
    ExplorationOutcomeTracker(ctx.db, ctx.config).update(cb.panel, cb.panel.dates[-1])
    for s in quiet:
        did = ctx.db.fetchone("SELECT decision_id FROM exploration_decisions WHERE symbol=? AND selection='TRACKED'", (s,))["decision_id"]
        outs = {o["horizon_sessions"]: o for o in ctx.db.fetchall("SELECT * FROM exploration_outcomes WHERE decision_id=?",
                                                                  (did,))}
        assert {5, 10, 20} <= set(outs)
        for h in (5, 10, 20):
            o = outs[h]
            assert None not in (o["net_ret"], o["spy_ret"], o["mfe"], o["mae"], o["stop_breached"])
    rep = learning_report(ctx.db, synthetic=True)
    assert rep["matured_tracked"] == 2 and "TRACKED" in {g["selection"] for g in rep["by_selection_and_hold"]}
    tw = rep["tracked_watchlist"]
    assert {x["symbol"] for x in tw["symbols"]} == set(quiet)
    assert {"SELECTED", "TRACKED"} <= {x["selection"] for x in tw["comparison"]}
    assert not any(m["selection"] == "TRACKED" for m in rep["missed_big_winners"])
    assert experiment_results(ctx.db, horizon=5)["groups"]["Tracked watchlist (research only, never traded)"]["n"] == 2
    c = TestClient(dash.create_app(ctx))
    live = c.get("/api/live").json()
    t = {r["symbol"]: r for r in live["tracked"]["rows"]}
    assert set(t) == set(quiet) and all(t[s]["matured"] == 1 and t[s]["universe_percentile"] for s in quiet)
    page = c.get("/live").text
    assert "Tracked watchlist" in page and quiet[0] in page and "never</b> traded" in page

    def boom(*a, **k):
        raise RuntimeError("tracked panel broke")
    monkeypatch.setattr(dash, "_tracked_live", boom)                            # fault-isolated section
    r = c.get("/live")
    assert r.status_code == 200 and "Tracked-watchlist panel unavailable" in r.text and "Open positions" in r.text
    ctx.close()


def test_live_page_without_tracked_symbols(tmp_path):
    ctx = _ctx(tmp_path)
    page = TestClient(dash.create_app(ctx)).get("/live")
    assert page.status_code == 200 and "No tracked symbols" in page.text
    ctx.close()


# -- the daily pipeline: regime timing, tracked records, no orders ---------------------------------------
def test_pipeline_writes_d_regime_before_planning_and_never_orders_tracked(tmp_path):
    from quantlab.data.ingest import IngestionService
    from quantlab.data.providers.synthetic import SyntheticSpec
    from quantlab.pipeline.daily import STEPS, DailyPipeline
    assert STEPS.index("research") < STEPS.index("discover") < STEPS.index("explore")
    var = tmp_path / "p"
    cfg = load_config(root=ROOT, overrides={"paper": {"mode": "EXPLORATION"},
                                            "exploration": {"tracked_watchlist": ["SYN001", "SYN002", "SYN003"]},
                                            "project": {"var_dir": str(var), "db_path": str(var / "q.db"),
                                                        "data_dir": str(var / "data"), "log_dir": str(var / "logs"),
                                                        "report_dir": str(var / "r"), "model_dir": str(var / "m")}})
    ctx = AppContext.create(cfg, init_logging=False)
    IngestionService(cfg, ctx.store, ctx.db).ingest_synthetic(SyntheticSpec(n_stocks=40, start="2018-01-02",
                                                                            end="2020-06-30", seed=7))
    pipe = DailyPipeline(ctx, synthetic=True)
    dates = pipe.full_bundle().panel.dates
    d = dates[-40]
    r = pipe.run(d)
    assert not r.errors, r.errors
    day = str(d.date())
    snap = ctx.db.fetchone("SELECT * FROM regime_snapshots WHERE as_of_date=?", (day,))
    assert snap is not None and not ctx.db.fetchone("SELECT 1 FROM regime_snapshots WHERE as_of_date>?", (day,))
    metrics = json.loads(snap["metrics_json"])
    rows = ctx.db.fetchall("SELECT * FROM exploration_decisions WHERE session_date=?", (day,))
    assert rows and snap["created_at"] <= min(x["created_at"] for x in rows)    # written before planning
    for x in rows:
        g = json.loads(x["pre_trade_json"])["regime"]
        assert g["as_of_date"] == day and g["label"] == snap["label"]
        assert g["market_trend_200"] == pytest.approx(metrics["market_trend_200"])
        assert g["state"] == ("THROTTLED" if metrics["market_trend_200"] < 0 else "NOT_THROTTLED")
    n_sel = sum(x["selection"] == "SELECTED" for x in rows)
    assert n_sel <= (2 if metrics["market_trend_200"] < 0 else 5)
    tk = {x["symbol"]: x for x in rows if x["selection"] == "TRACKED"}
    assert set(tk) == {"SYN001", "SYN003"}                                      # SYN002 has no bar on D: not in the data
    assert all(json.loads(x["pre_trade_json"])["tracked"]["universe_percentile"] is not None for x in tk.values())
    assert not ctx.db.fetchall("SELECT 1 FROM orders WHERE symbol IN ('SYN001','SYN002','SYN003')")
    assert not any(json.loads(i["intent_json"]).get("symbol") in tk
                   for i in ctx.db.fetchall("SELECT intent_json FROM order_intents"))
    ExplorationOutcomeTracker(ctx.db, ctx.config).update(pipe.full_bundle().panel, dates[-1])
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM exploration_outcomes o JOIN exploration_decisions d "
                           "ON d.decision_id=o.decision_id WHERE d.selection='TRACKED'")["n"] == 2 * 5
    ctx.close()
