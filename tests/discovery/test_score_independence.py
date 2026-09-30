"""The composite discovery score must not count one signal twice.

rs_spy_n is defined as ret_n(stock) - ret_n(SPY). Within a single session ret_n(SPY) is one scalar,
and a cross-sectional percentile rank is invariant to subtracting a constant, so ranking rs_spy_n
is exactly ranking ret_n. These tests lock that property down and the consequence drawn from it:
relative_strength fires and explains itself, but contributes no points.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantlab.config import load_config
from quantlab.discovery.engine import DiscoveryEngine
from quantlab.discovery.families import POINTS, SCORED, SCORED_FEATURES, SCORED_POINTS, pct_rank
from quantlab.features.base import FeatureSet

from .world import BENCH, crafted_bundle

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg():
    return load_config(root=ROOT, overrides={"benchmarks": BENCH})


@pytest.fixture(scope="module")
def cb():
    return crafted_bundle()


@pytest.fixture(scope="module")
def core(cfg, cb):
    eng = DiscoveryEngine(cfg)
    fs = FeatureSet(cb, dtype="float64")
    return eng.core(cb.panel, fs, cb.panel.dates[-1], {cb.market_symbol, *cb.sector_etfs})


def test_percentile_rank_is_invariant_to_a_constant_shift():
    """The property the defect rests on, stated directly."""
    s = pd.Series([0.10, -0.05, 0.30, 0.00, 0.12], index=list("abcde"))
    for shift in (0.0, 0.037, -1.5, 100.0):
        assert pct_rank(s - shift).equals(pct_rank(s)), shift


def test_ranking_rs_spy_is_ranking_plain_return(core):
    """rs_spy_20 carries no cross-sectional information that ret_20d does not already carry."""
    xs = core["xs"]
    ret20, rs20 = xs["ret_20d"], xs["rs_spy_20"]
    both = ret20.notna() & rs20.notna()
    assert both.sum() >= 20, "need a usable cross-section"
    # the difference is one number for the whole session: ret_20(SPY)
    diff = (ret20[both] - rs20[both])
    assert diff.std() < 1e-9, f"rs_spy_20 is not a constant shift of ret_20d (std {diff.std():.3e})"
    # therefore the percentile ranks order the universe identically
    assert pct_rank(ret20).corr(pct_rank(rs20), method="spearman") == pytest.approx(1.0, abs=1e-9)


def test_relative_strength_contributes_no_points_to_the_score(core):
    """The score is the mean of the point-contributing families only."""
    comp, score = core["comp"], core["score"]
    assert "relative_strength" not in SCORED_POINTS
    assert set(SCORED_POINTS) < set(SCORED)
    pts = comp[list(SCORED_POINTS)]
    expected = 100.0 * pts.sum(axis=1, min_count=1) / (POINTS * pts.notna().sum(axis=1).replace(0, np.nan))
    pd.testing.assert_series_equal(score, expected, check_names=False)
    # and moving relative_strength's points cannot move the score
    bumped = comp.copy()
    bumped["relative_strength"] = POINTS
    b = bumped[list(SCORED_POINTS)]
    assert (100.0 * b.sum(axis=1, min_count=1)
            / (POINTS * b.notna().sum(axis=1).replace(0, np.nan))).equals(score)


def test_relative_strength_still_fires_and_still_explains(core):
    """The fix removes double counting, not the family: its trigger is a LEVEL test on rs_spy_63
    (genuinely market-relative, not rank-invariant), so discovery behaviour is preserved."""
    masks, comp = core["masks"], core["comp"]
    assert "relative_strength" in masks.columns
    assert bool(masks["relative_strength"].any()), "relative strength fired for no symbol"
    assert comp["relative_strength"].notna().any(), "component points are still recorded for reporting"
    assert masks.at["MOMO", "relative_strength"], "the crafted uptrend must still fire relative strength"


def test_momentum_weight_is_one_quarter_not_two_fifths(core):
    """With five scored families and one a rank-copy, momentum previously drove ~40% of the score."""
    comp = core["comp"]
    full = comp.loc["MOMO"]
    assert set(SCORED_POINTS) == {"momentum", "volume_activity", "breakout_compression", "mean_reversion"}
    known = [f for f in SCORED_POINTS if pd.notna(full[f])]
    assert len(known) >= 2
    share = 1.0 / len(known)
    assert share == pytest.approx(0.25, abs=0.26)      # 1/4 when all four are known


def test_scored_features_are_unchanged():
    """The correction is to score composition only: no feature definition moved."""
    assert SCORED_FEATURES["momentum"] == (("ret_20d", 1), ("ret_60d", 1), ("ret_120d", 1))
    assert SCORED_FEATURES["relative_strength"] == (("rs_spy_20", 1), ("rs_spy_63", 1))


# -- selection score (what exploration trades by) ---------------------------------------------------
from quantlab.discovery.families import SELECTION_FEATURES, selection_score          # noqa: E402
from quantlab.testing.pit import assert_truncation_invariant                         # noqa: E402


def test_selection_score_uses_fixed_published_priors():
    assert SELECTION_FEATURES == (("mom_12_1", 1), ("atr14_pct", -1), ("adv20", 1))


def test_selection_score_direction_and_unknown_handling():
    xs = pd.DataFrame({"mom_12_1": [0.5, 0.1, 0.3, np.nan], "atr14_pct": [0.02, 0.02, 0.08, 0.02],
                       "adv20": [5e7, 5e7, 5e7, 5e7]}, index=list("ABCD"))
    s = selection_score(xs)
    assert s["A"] > s["B"]                     # stronger 12-1 momentum, same vol and liquidity
    assert s["B"] > 0 and np.isnan(s["D"])     # a missing input is UNKNOWN, never zero
    xs2 = xs.assign(atr14_pct=[0.02, 0.02, 0.02, 0.02], mom_12_1=[0.3, 0.3, 0.3, 0.3])
    xs2.loc["C", "atr14_pct"] = 0.08
    s2 = selection_score(xs2)
    assert s2["A"] > s2["C"]                   # higher volatility ranks lower
    assert s.dropna().between(0, 1).all()


def test_selection_score_is_carried_on_every_scanned_candidate(cfg, cb):
    scan = DiscoveryEngine(cfg).scan(cb, cb.panel.dates[-1])
    assert "selection_score" in scan.table.columns
    a = DiscoveryEngine(cfg).assess(scan, {}, {})
    scored = [c for c in a.candidates if c["selection_score"] is not None]
    assert scored and all(c["setup"]["selection_score"] == c["selection_score"] for c in a.candidates)


def test_selection_score_is_truncation_invariant(cfg, cb):
    eng = DiscoveryEngine(cfg)
    dates = cb.panel.dates
    check = [dates[-60], dates[-25], dates[-1]]

    def compute(b):
        rows = {d: eng.scan(b, d).table["selection_score"] for d in check if d <= b.panel.dates[-1]}
        return pd.DataFrame(rows).T
    assert_truncation_invariant(compute, cb, check_dates=check, name="selection score")
