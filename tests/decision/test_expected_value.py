"""EVEngine: shrinkage math, cost subtraction, and that AI can never reach this computation."""
from __future__ import annotations

import inspect
import math

import pytest

from quantlab.core.costs import CostModel
from quantlab.decision.expected_value import EVEngine
from quantlab.decision.stats_provider import SOURCE_IN_SAMPLE, SOURCE_OOS, StrategyStats

from ..shadow.support import make_candidate

COST_CONFIG = {
    "costs": {"half_spread_bps_tiers": [[1e8, 2.0], [0, 25.0]], "slippage_bps": 5.0,
             "commission_per_share": 0.0, "commission_min_per_order": 0.0, "delisting_return": -0.30},
}


def _stats(n, win_rate, avg_win, avg_loss, source=SOURCE_OOS, expectancy=None):
    if expectancy is None and n:
        expectancy = win_rate * (avg_win or 0.0) + (1 - win_rate) * (avg_loss or 0.0)
    return StrategyStats(strategy_id="s", version="1.0.0", n=n, win_rate=win_rate, avg_win=avg_win,
                         avg_loss=avg_loss, expectancy=expectancy, source=source)


def test_no_ai_parameter_anywhere_in_the_ev_path():
    """LLM confidence must have no way to reach the EV computation."""
    sig = inspect.signature(EVEngine.estimate)
    names = " ".join(sig.parameters)
    assert "ai" not in names.lower() and "confidence" not in names.lower() and "llm" not in names.lower()
    init_sig = inspect.signature(EVEngine.__init__)
    init_names = " ".join(init_sig.parameters)
    assert "ai" not in init_names.lower() and "llm" not in init_names.lower()


def test_zero_trades_gives_prior_only_ev_is_minus_cost(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    stats = StrategyStats.empty("s", "1.0.0")
    ev = eng.estimate(cand, stats, adv=1e8)
    assert ev.n_obs == 0
    assert ev.p_win == pytest.approx(eng.prior_win_rate)
    assert ev.ev == pytest.approx(-ev.cost)
    assert ev.ev < 0
    assert not eng.passes_gate(ev)


def test_stats_none_also_gives_prior_only(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    ev = eng.estimate(cand, None, adv=1e8)
    assert ev.n_obs == 0
    assert ev.ev == pytest.approx(-ev.cost)


def test_cost_uses_worst_tier_for_unknown_liquidity(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    ev_known = eng.estimate(cand, StrategyStats.empty("s", "1.0.0"), adv=1e9)
    ev_unknown = eng.estimate(cand, StrategyStats.empty("s", "1.0.0"), adv=None)
    assert ev_unknown.cost > ev_known.cost
    assert ev_unknown.cost == pytest.approx(eng.cost_model.round_trip_cost_frac(None))


def test_large_n_approaches_empirical_mean(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    stats = _stats(n=100_000, win_rate=0.6, avg_win=0.05, avg_loss=-0.03)
    ev = eng.estimate(cand, stats, adv=1e9)
    emp_gross = 0.6 * 0.05 + 0.4 * -0.03
    assert ev.ev + ev.cost == pytest.approx(emp_gross, abs=1e-4)


def test_small_n_stays_close_to_prior(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    stats = _stats(n=1, win_rate=1.0, avg_win=0.50, avg_loss=None)
    ev = eng.estimate(cand, stats, adv=1e9)
    gross = ev.ev + ev.cost
    # n_eff=1, k=prior_trades (>=1) -> shrinkage weight <= 0.5, so gross stays far below the huge
    # single-trade empirical mean and close to the zero prior.
    assert abs(gross) < 0.50 * (1.0 / (1.0 + eng.prior_trades)) + 1e-6


def test_shrinkage_identity_matches_formula(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    stats = _stats(n=40, win_rate=0.55, avg_win=0.04, avg_loss=-0.02)
    ev = eng.estimate(cand, stats, adv=1e9)
    n_eff = eng.effective_n(stats)
    k = eng.prior_trades
    emp_gross = 0.55 * 0.04 + 0.45 * -0.02
    expected_gross = n_eff / (n_eff + k) * emp_gross
    assert (ev.ev + ev.cost) == pytest.approx(expected_gross)


def test_in_sample_evidence_discounted_vs_oos(config):
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    oos = _stats(n=40, win_rate=0.55, avg_win=0.04, avg_loss=-0.02, source=SOURCE_OOS)
    ins = _stats(n=40, win_rate=0.55, avg_win=0.04, avg_loss=-0.02, source=SOURCE_IN_SAMPLE)
    ev_oos = eng.estimate(cand, oos, adv=1e9)
    ev_ins = eng.estimate(cand, ins, adv=1e9)
    assert eng.effective_n(ins) == pytest.approx(40 * eng.in_sample_weight)
    assert eng.effective_n(oos) == 40
    assert abs(ev_oos.ev) > abs(ev_ins.ev)  # OOS evidence pulls further from the zero prior


def test_passes_gate_is_strict_and_fails_closed_on_none(config):
    eng = EVEngine(config, CostModel.from_config(config))
    assert eng.passes_gate(None) is False
    from quantlab.core.types import ExpectedValue
    exact = ExpectedValue(p_win=0.5, avg_win=0.01, avg_loss=-0.01, cost=0.001,
                          ev=eng.min_ev_bps / 1e4)
    assert eng.passes_gate(exact) is False  # exactly at the threshold does not pass
    above = ExpectedValue(p_win=0.5, avg_win=0.01, avg_loss=-0.01, cost=0.001,
                          ev=(eng.min_ev_bps + 0.01) / 1e4)
    assert eng.passes_gate(above) is True


def test_unproven_strategy_never_passes_ev_gate(config):
    """n=0 -> prior only -> EV = -cost < 0 <= min_ev_bps: cannot pass regardless of threshold sign."""
    eng = EVEngine(config, CostModel.from_config(config))
    cand = make_candidate(ref=10.0, stop=9.0)
    ev = eng.estimate(cand, StrategyStats.empty("s", "1.0.0"), adv=1e9)
    assert not eng.passes_gate(ev)


def test_prior_trades_must_be_positive(config):
    with pytest.raises(ValueError):
        EVEngine(config.with_overrides({"expected_value": {"prior_trades": 0}}))
