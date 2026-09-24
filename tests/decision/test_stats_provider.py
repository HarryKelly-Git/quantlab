"""StrategyStatsProvider: OOS preference, synthetic exclusion, holdout exclusion, PIT maturity."""
from __future__ import annotations

import pytest

from quantlab.decision.stats_provider import SOURCE_IN_SAMPLE, SOURCE_NONE, SOURCE_OOS, SOURCE_OOS_SHADOW, \
    SOURCE_SHADOW, StrategyStatsProvider

from .support import seed_experiment_trades, seed_shadow_trades


def test_no_rows_gives_n_zero_source_none(db, config):
    p = StrategyStatsProvider(db, config)
    st = p.get("nope", "1.0.0")
    assert st.n == 0
    assert st.source == SOURCE_NONE
    assert not st.validated


def test_prefers_oos_over_in_sample(db, config):
    seed_experiment_trades(db, config, "s1", "1.0.0", [0.01, 0.02, -0.01], segment="full")
    seed_experiment_trades(db, config, "s1", "1.0.0", [0.03, 0.04], segment="oos:w1")
    p = StrategyStatsProvider(db, config)
    st = p.get("s1", "1.0.0")
    assert st.source == SOURCE_OOS
    assert st.n == 2
    assert st.validated


def test_in_sample_only_is_unvalidated_and_flagged(db, config):
    seed_experiment_trades(db, config, "s2", "1.0.0", [0.01, -0.02, 0.03], segment="full")
    p = StrategyStatsProvider(db, config)
    st = p.get("s2", "1.0.0")
    assert st.source == SOURCE_IN_SAMPLE
    assert st.n == 3
    assert not st.validated
    assert any("IN-SAMPLE" in n for n in st.notes)


def test_synthetic_experiments_excluded_by_default(db, config):
    seed_experiment_trades(db, config, "s3", "1.0.0", [0.05, 0.06], segment="oos:w1", uses_synthetic=True)
    p = StrategyStatsProvider(db, config)
    st = p.get("s3", "1.0.0")
    assert st.n == 0
    assert st.source == SOURCE_NONE


def test_synthetic_included_when_allowed(db, config):
    seed_experiment_trades(db, config, "s4", "1.0.0", [0.05, 0.06], segment="oos:w1", uses_synthetic=True)
    p = StrategyStatsProvider(db, config, allow_synthetic=True)
    st = p.get("s4", "1.0.0")
    assert st.n == 2
    assert st.uses_synthetic
    assert st.source == SOURCE_OOS


def test_failed_experiment_excluded_by_default(db, config):
    seed_experiment_trades(db, config, "s5", "1.0.0", [0.05], segment="oos:w1", succeeded=False)
    p = StrategyStatsProvider(db, config)
    st = p.get("s5", "1.0.0")
    assert st.n == 0


def test_holdout_trades_excluded(db, config):
    holdout_start = config.get("validation.holdout.start")
    seed_experiment_trades(db, config, "s6", "1.0.0", [0.05, 0.06], segment="oos:w1",
                          start=str(holdout_start))
    p = StrategyStatsProvider(db, config)
    st = p.get("s6", "1.0.0")
    assert st.n == 0
    assert any("holdout" in n for n in st.notes)


def test_shadow_and_oos_combine_and_dedupe(db, config):
    seed_experiment_trades(db, config, "s7", "1.0.0", [0.01, 0.02], segment="oos:w1", symbol="AAA")
    seed_shadow_trades(db, "s7", "1.0.0", [0.03, 0.04], symbol="BBB")
    p = StrategyStatsProvider(db, config)
    st = p.get("s7", "1.0.0")
    assert st.source == SOURCE_OOS_SHADOW
    assert st.n == 4
    assert st.n_backtest == 2
    assert st.n_shadow == 2


def test_shadow_only_is_validated(db, config):
    seed_shadow_trades(db, "s8", "1.0.0", [0.02, -0.01, 0.03])
    p = StrategyStatsProvider(db, config)
    st = p.get("s8", "1.0.0")
    assert st.source == SOURCE_SHADOW
    assert st.validated
    assert st.n == 3


def test_synthetic_shadow_excluded_by_default(db, config):
    seed_shadow_trades(db, "s9", "1.0.0", [0.02, 0.03], is_synthetic=True)
    p = StrategyStatsProvider(db, config)
    st = p.get("s9", "1.0.0")
    assert st.n == 0


def test_as_of_point_in_time_matches_live_pipeline(db, config):
    """A later trade that hadn't exited yet at `as_of` must not change the answer."""
    seed_experiment_trades(db, config, "s10", "1.0.0", [0.01, 0.02, 0.03, 0.04], segment="oos:w1",
                          start="2019-01-02")
    p = StrategyStatsProvider(db, config)
    all_rows = db.query_df("SELECT exit_date FROM backtest_trades WHERE strategy_id='s10' ORDER BY exit_date")
    mid_date = all_rows["exit_date"].iloc[1]
    st_asof = p.get("s10", "1.0.0", as_of=mid_date)
    st_full = p.get("s10", "1.0.0")
    assert st_asof.n == 2
    assert st_full.n == 4
    # re-querying "as of" the same cutoff again gives the same answer even though later trades exist
    p2 = StrategyStatsProvider(db, config)
    assert p2.get("s10", "1.0.0", as_of=mid_date).n == st_asof.n


def test_win_rate_and_payoffs_are_hand_computable(db, config):
    # 2 winners (+0.05 gross => +0.049 net after 0.001 cost), 1 loser (-0.03 gross => -0.031 net)
    seed_experiment_trades(db, config, "s11", "1.0.0", net_rets=[0.049, 0.049, -0.031],
                          gross_rets=[0.05, 0.05, -0.03], segment="oos:w1", cost=0.001)
    p = StrategyStatsProvider(db, config)
    st = p.get("s11", "1.0.0")
    assert st.n == 3
    assert st.win_rate == pytest.approx(2 / 3)
    assert st.avg_win == pytest.approx(0.05)
    assert st.avg_loss == pytest.approx(-0.03)
    assert st.expectancy == pytest.approx((0.049 + 0.049 - 0.031) / 3)


def test_regime_breakdown(db, config):
    seed_experiment_trades(db, config, "s12", "1.0.0", [0.02, 0.03, -0.02, -0.03], segment="oos:w1",
                          regimes=["bull", "bull", "bear", "bear"])
    p = StrategyStatsProvider(db, config)
    st = p.get("s12", "1.0.0")
    assert set(st.by_regime) == {"bull", "bear"}
    assert st.by_regime["bull"].n == 2
    assert st.by_regime["bear"].n == 2
    bear_only = p.get("s12", "1.0.0", regime="bear")
    assert bear_only.n == 2
    assert bear_only.expectancy < 0
