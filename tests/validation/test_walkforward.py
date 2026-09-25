"""Walk-forward runner: boundaries, no look-ahead, chronology, costs, benchmark alignment,
reproducibility, insufficient history, data-quality failure, stored records.

All data here is SYNTHETIC: these tests validate the machinery, never a strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.context import AppContext
from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import DataBundle, build_panel
from quantlab.data.providers.synthetic import SyntheticMarket, SyntheticSpec
from quantlab.experiments.registry import ExperimentRegistry
from quantlab.testing.fixtures import SYNTHETIC_BENCHMARKS, make_synthetic_bundle
from quantlab.validation.splits import walk_forward_windows
from quantlab.validation.walkforward import InsufficientHistoryError, run_walk_forward

SPEC = SyntheticSpec(n_stocks=30, start="2016-01-04", end="2020-12-31", seed=21, momentum_edge=0.001)


@pytest.fixture(scope="module")
def market():
    return SyntheticMarket(SPEC)


@pytest.fixture(scope="module")
def wf_bundle(market):
    return make_synthetic_bundle(market=market)


@pytest.fixture
def wctx(config):
    c = AppContext.create(config, init_logging=False)
    yield c
    c.close()


@pytest.fixture
def run(wctx, wf_bundle):
    return run_walk_forward(wctx, "momentum_trend", bundle=wf_bundle)


def _bundle_from_bars(market: SyntheticMarket, bars: pd.DataFrame) -> DataBundle:
    w = market.world
    cal = TradingCalendar.from_dates(sorted(bars.loc[bars["symbol"] == "SPY", "date"].unique()))
    return DataBundle(build_panel(bars, w["corporate_actions"], calendar=cal), cal, reference=w["reference"],
                      actions=w["corporate_actions"], events=w["events"], fundamentals=w["fundamentals"],
                      news=w["news"], benchmarks=SYNTHETIC_BENCHMARKS, dataset_ids=["synthetic"], is_synthetic=True)


def test_windows_are_ordered_embargoed_and_before_holdout(wf_bundle, config):
    ws = walk_forward_windows(wf_bundle.calendar, config)
    assert len(ws) >= 3
    sessions = wf_bundle.calendar.sessions
    hold = pd.Timestamp(config.get("validation.holdout.start"))
    emb = int(config.get("validation.walk_forward.embargo_sessions"))
    for a, b in zip(ws, ws[1:]):
        assert b.test_start > a.test_end and b.train_start > a.train_start
    for w in ws:
        gap = ((sessions > w.train_end) & (sessions < w.test_start)).sum()
        assert gap == emb, "exactly embargo_sessions sessions separate train and test"
        assert w.train_start < w.train_end < w.test_start <= w.test_end < hold


def test_oos_trades_stay_inside_their_window_and_chain_capital(run):
    assert run.windows and run.is_synthetic and "SYNTHETIC" in run.labels
    for prev, cur in zip(run.windows, run.windows[1:]):
        assert cur.start_capital == pytest.approx(prev.end_capital)
    total = 0
    for wr in run.windows:
        t = wr.oos.trades
        total += len(t)
        if len(t):
            assert (pd.to_datetime(t["signal_date"]) >= wr.window.test_start).all()
            assert (pd.to_datetime(t["entry_date"]) > pd.to_datetime(t["signal_date"])).all()   # next open
            assert (pd.to_datetime(t["exit_date"]) <= wr.window.test_end).all()
    assert total == len(run.result.trades) > 0
    idx = run.result.equity.index
    assert idx.is_monotonic_increasing and idx.is_unique


def test_no_lookahead_future_data_cannot_change_earlier_windows(wctx, market, wf_bundle):
    """Corrupt every price AFTER window 0's test end: window 0 must be byte-identical."""
    base = run_walk_forward(wctx, "momentum_trend", bundle=wf_bundle, persist=False)
    cut = base.windows[0].window.test_end
    bars = market.world["bars"].copy()
    rng = np.random.default_rng(0)
    late = bars["date"] > cut
    shock = rng.uniform(0.3, 3.0, late.sum())
    for c in ("open", "high", "low", "close"):
        bars.loc[late, c] = bars.loc[late, c] * shock
    bars.loc[late, "high"] = bars.loc[late, ["open", "high", "low", "close"]].max(axis=1)
    bars.loc[late, "low"] = bars.loc[late, ["open", "high", "low", "close"]].min(axis=1)
    altered = run_walk_forward(wctx, "momentum_trend", bundle=_bundle_from_bars(market, bars), persist=False)
    a, b = base.windows[0], altered.windows[0]
    pd.testing.assert_frame_equal(a.oos.trades, b.oos.trades)
    pd.testing.assert_frame_equal(a.oos.equity, b.oos.equity)
    assert a.in_sample_metrics == b.in_sample_metrics
    # later windows DO see the altered prices (sanity: the corruption was real)
    assert not base.windows[-1].oos.equity.equals(altered.windows[-1].oos.equity)


def test_costs_are_charged_and_zero_cost_equals_gross(wctx, wf_bundle, config):
    run = run_walk_forward(wctx, "momentum_trend", bundle=wf_bundle, persist=False)
    t = run.result.trades
    assert (t["cost_ret"] > 0).all() and (t["net_ret"] < t["gross_ret"]).all()
    assert run.result.metrics["total_costs"] > 0
    free = config.with_overrides({"costs": {"slippage_bps": 0.0, "half_spread_bps_tiers": [[0, 0.0]]}})
    zctx = AppContext.create(free, init_logging=False)
    try:
        z = run_walk_forward(zctx, "momentum_trend", bundle=wf_bundle, persist=False)
    finally:
        zctx.close()
    np.testing.assert_allclose(z.result.trades["gross_ret"], z.result.trades["net_ret"], atol=1e-12)
    assert z.result.metrics["total_costs"] == pytest.approx(0.0)


def test_benchmark_is_aligned_to_oos_windows(run, wf_bundle):
    b = run.benchmark_equity
    assert b is not None
    assert b.index.equals(run.result.equity.index)
    # chained SPY return equals the product of per-window SPY returns (flat across embargo gaps)
    per = np.prod([1 + w.benchmark_return for w in run.windows]) - 1
    assert b["equity"].iloc[-1] / b["equity"].iloc[0] - 1 == pytest.approx(per, rel=1e-9)
    assert run.result.metrics["benchmark"]["n_overlap"] == len(run.result.equity) - 1


def test_reproducible(wctx, wf_bundle):
    a = run_walk_forward(wctx, "momentum_trend", bundle=wf_bundle, persist=False)
    b = run_walk_forward(wctx, "momentum_trend", bundle=wf_bundle, persist=False)
    pd.testing.assert_frame_equal(a.result.trades, b.result.trades)
    assert a.inference == b.inference


def test_records_are_stored_and_feed_the_ev_statistics(run, wctx):
    exp = ExperimentRegistry(wctx.db, wctx.config).get(run.experiment_id)
    assert exp["kind"] == "walk_forward" and exp["n_variants_tested"] == 1 and exp["uses_synthetic_data"] == 1
    res = exp["results"][-1]
    assert res["status"] == "succeeded" and res["conclusion"] == run.inference["verdict"]
    assert len(res["metrics"]["windows"]) == len(run.windows)
    assert res["metrics"]["data_status"]["status"] in {"SYNTHETIC", "UNKNOWN"}
    segs = wctx.db.fetchall("SELECT segment, COUNT(*) AS n FROM backtest_trades WHERE experiment_id=? GROUP BY segment",
                            (run.experiment_id,))
    assert all(s["segment"].startswith("oos:") for s in segs)
    assert sum(s["n"] for s in segs) == len(run.result.trades)
    n_eq = wctx.db.fetchone("SELECT COUNT(*) AS n FROM backtest_equity WHERE experiment_id=? AND segment='oos'",
                            (run.experiment_id,))["n"]
    assert n_eq == len(run.result.equity)
    from quantlab.decision.stats_provider import StrategyStatsProvider
    st = StrategyStatsProvider(wctx.db, wctx.config, allow_synthetic=True).get("momentum_trend", "1.0.0")
    assert st.source.startswith("walk_forward_oos") and st.n > 0
    # default provider ignores synthetic evidence entirely
    assert StrategyStatsProvider(wctx.db, wctx.config).get("momentum_trend", "1.0.0").n == 0


def test_insufficient_history_is_recorded_as_failed(wctx):
    short = make_synthetic_bundle(SyntheticSpec(n_stocks=15, start="2016-01-04", end="2017-12-29", seed=2))
    with pytest.raises(InsufficientHistoryError):
        run_walk_forward(wctx, "momentum_trend", bundle=short)
    exps = ExperimentRegistry(wctx.db, wctx.config).list(kind="walk_forward")
    assert exps and exps[0]["status"] == "failed"


def test_data_quality_failure_blocks_evaluation(wctx, market):
    from quantlab.experiments.runner import DataValidationError
    bars = market.world["bars"]
    bars = bars[~((bars["symbol"] == "SPY") & (bars["date"].dt.year == 2019))]   # benchmark gap
    all_dates = sorted(market.world["bars"]["date"].unique())
    w = market.world
    cal = TradingCalendar.from_dates(all_dates)
    broken = DataBundle(build_panel(bars, w["corporate_actions"], calendar=cal), cal, reference=w["reference"],
                        actions=w["corporate_actions"], events=w["events"], fundamentals=w["fundamentals"],
                        news=w["news"], benchmarks=SYNTHETIC_BENCHMARKS, dataset_ids=["synthetic"], is_synthetic=True)
    with pytest.raises(DataValidationError):
        run_walk_forward(wctx, "momentum_trend", bundle=broken)
    exps = ExperimentRegistry(wctx.db, wctx.config).list(kind="walk_forward")
    assert exps[0]["status"] == "failed"


def test_window_without_signals_is_flat_not_an_error(wctx, wf_bundle, config):
    strict = config.with_overrides({"strategies": {"momentum_trend": {"params": {"min_score_pct": 1.01}}}})
    c = AppContext.create(strict, init_logging=False)
    try:
        r = run_walk_forward(c, "momentum_trend", bundle=wf_bundle, persist=False)
    finally:
        c.close()
    assert len(r.result.trades) == 0
    assert all(w.oos_return == pytest.approx(0.0) for w in r.windows)
    assert r.inference["verdict"] == "INSUFFICIENT_SAMPLE"


# --- regressions from the adversarial review ---------------------------------------------------
def test_overlapping_windows_are_refused(wf_bundle, config):
    bad = config.with_overrides({"validation": {"walk_forward": {"step_months": 3, "test_months": 6}}})
    with pytest.raises(ValueError, match="overlapping"):
        walk_forward_windows(wf_bundle.calendar, bad)


def test_aggregate_costs_sum_all_windows(wctx, wf_bundle):
    r = run_walk_forward(wctx, "momentum_trend", bundle=wf_bundle, persist=False)
    m = r.result.metrics
    assert m["total_costs"] == pytest.approx(r.result.trades["costs"].sum(), rel=1e-9)
    per_window = sum(float(w.oos.equity["cum_costs"].iloc[-1]) for w in r.windows)
    assert m["total_costs"] == pytest.approx(per_window, rel=1e-9)
    assert m["total_return_gross"] > m["total_return_net"]


def test_real_data_that_fails_the_audit_is_refused(wctx, market):
    """A non-synthetic bundle whose data fails the audit (pre-adjusted bars) must not be evaluated."""
    from quantlab.validation.walkforward import DataNotSuitableError
    w = market.world
    acts = w["corporate_actions"]
    split = acts[acts["action_type"] == "split"].iloc[0]
    bars = w["bars"].copy()
    before = (bars["symbol"] == split["symbol"]) & (bars["date"] < split["ex_date"])
    for col in ("open", "high", "low", "close"):
        bars.loc[before, col] = bars.loc[before, col] / split["ratio"]
    bars["provider"] = "fake_real"
    ids = {"bars": [wctx.store.write("bars", bars, "fake_real", params={"feed": "sip"})],
           "corporate_actions": [wctx.store.write("corporate_actions", acts.assign(provider="fake_real"), "fake_real")],
           "reference": [wctx.store.write("reference", w["reference"].assign(source="fake_real"), "fake_real")]}
    bundle = wctx.store.load_bundle(SYNTHETIC_BENCHMARKS, snapshot=ids, synthetic=False)
    assert not bundle.is_synthetic
    with pytest.raises(DataNotSuitableError):
        run_walk_forward(wctx, "momentum_trend", bundle=bundle, snapshot=ids)
    assert ExperimentRegistry(wctx.db, wctx.config).list(kind="walk_forward")[0]["status"] == "failed"
