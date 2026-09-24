"""Strategies + arena: PIT safety, universe masking, plans, registry, holdout respect."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.types import Direction
from quantlab.features.base import FeatureSet
from quantlab.strategies.arena import HoldoutAccessError, StrategyArena, _rowwise_spearman
from quantlab.strategies.registry import STRATEGY_CLASSES, build_strategies, register_strategies
from quantlab.testing.pit import assert_truncation_invariant
from quantlab.universe import UniverseEngine


def _strategy(config, sid):
    spec = config.section("strategies")[sid]
    return STRATEGY_CLASSES[sid](sid, spec["version"], spec["params"])


@pytest.mark.parametrize("sid", sorted(STRATEGY_CLASSES))
def test_strategy_scores_are_point_in_time(sid, bundle, config):
    strat = _strategy(config, sid)
    eng = UniverseEngine(config)

    def compute(b):
        u = eng.membership(b)
        return strat.score(FeatureSet(b, universe=u), u)
    assert_truncation_invariant(compute, bundle, n_dates=4, min_history=400, name=sid)


@pytest.mark.parametrize("sid", sorted(STRATEGY_CLASSES))
def test_scores_nan_outside_universe_and_plans_sane(sid, bundle, config):
    strat = _strategy(config, sid)
    u = UniverseEngine(config).membership(bundle)
    fs = FeatureSet(bundle, universe=u)
    sc = strat.score(fs, u)
    assert sc.notna().to_numpy().sum() > 0, "strategy never fires on synthetic data"
    assert not (sc.notna() & ~u).to_numpy().any()
    d = sc.index[sc.notna().any(axis=1)][-1]
    cands = strat.candidates(fs, u, d, scores=sc)
    assert cands and all(c.strategy_id == sid for c in cands)
    for c in cands:
        assert c.plan.entry_ref_price == pytest.approx(bundle.panel.close.at[d, c.symbol])
        if c.direction is Direction.LONG and c.plan.stop_price is not None:
            assert c.plan.stop_price < c.plan.entry_ref_price
        assert c.plan.holding_sessions == strat.holding_sessions
        assert c.reasons and c.pit_status is not None


def test_registry_builds_and_registers_idempotently(config, db):
    strats = build_strategies(config)
    ids = [s.strategy_id for s in strats]
    assert "momentum_trend" in ids and "mean_reversion" in ids
    assert register_strategies(db, strats) == len(strats)
    assert register_strategies(db, strats) == 0
    row = db.fetchone("SELECT status, stage FROM strategies WHERE strategy_id='momentum_trend'")
    assert row == {"status": "SHADOW", "stage": "RESEARCH"}


def test_information_content_never_reaches_holdout(bundle, config):
    strat = _strategy(config, "momentum_trend")
    u = UniverseEngine(config).membership(bundle)
    fs = FeatureSet(bundle, universe=u)
    arena = StrategyArena([strat])
    dates = bundle.panel.dates
    hold = dates[-30]
    with pytest.raises(HoldoutAccessError):
        arena.information_content(fs, u, dates[300], hold, holdout_start=hold)
    # ending just before the holdout: horizon-20 windows must stop 21 sessions before it
    info = arena.information_content(fs, u, dates[300], dates[-31], horizons=(20,), holdout_start=hold)
    assert len(info) == 1
    sc = arena.scores(fs, u)["momentum_trend"]
    max_signals = int(sc.loc[dates[300]:dates[-30 - 21]].notna().to_numpy().sum())
    assert info.iloc[0]["n_signals"] <= max_signals


def test_rowwise_spearman_matches_pandas():
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(6, 12)))
    b = pd.DataFrame(rng.normal(size=(6, 12)))
    a.iloc[0, :3] = np.nan
    got = _rowwise_spearman(a, b, min_n=5)
    exp = [a.iloc[i].corr(b.iloc[i], method="spearman") for i in range(6)]
    np.testing.assert_allclose(got, exp, rtol=1e-10)
