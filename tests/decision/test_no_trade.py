"""NoTradeEngine: one behavior test per rule (pass, fail, and unknown-fails-closed where relevant)."""
from __future__ import annotations

import pytest

from quantlab.core.types import (
    AIAssessment,
    AIReview,
    CheckSeverity,
    Direction,
    MLPrediction,
    Objection,
    ObjectionSeverity,
    PitStatus,
)
from quantlab.decision.no_trade import NoTradeContext, NoTradeEngine
from quantlab.decision.stats_provider import RegimeStats, StrategyStats

from ..shadow.support import make_candidate


def _names(results, name):
    return next(r for r in results if r.name == name)


# -- liquidity -----------------------------------------------------------------------------------
def test_liquidity_unknown_fails_closed(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext(as_of=cand.as_of_date, adv20=None)
    r = _names(eng.evaluate(cand, ctx), "no_trade.liquidity")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_liquidity_below_minimum_fails(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=eng.min_adv - 1)
    assert not _names(eng.evaluate(cand, ctx), "no_trade.liquidity").passed


def test_liquidity_at_or_above_minimum_passes(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=eng.min_adv, vol_20d=0.2)
    r = _names(eng.evaluate(cand, ctx), "no_trade.liquidity")
    assert r.passed and r.severity is CheckSeverity.INFO


# -- volatility ------------------------------------------------------------------------------
def test_volatility_unknown_fails_closed(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=None)
    assert not _names(eng.evaluate(cand, ctx), "no_trade.volatility").passed


def test_volatility_above_max_fails(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=eng.max_vol + 0.01)
    assert not _names(eng.evaluate(cand, ctx), "no_trade.volatility").passed


def test_volatility_within_max_passes(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=eng.max_vol)
    assert _names(eng.evaluate(cand, ctx), "no_trade.volatility").passed


# -- earnings proximity --------------------------------------------------------------------------
def test_earnings_unknown_is_not_applicable(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2)
    r = _names(eng.evaluate(cand, ctx), "no_trade.earnings_proximity")
    assert r.passed and r.severity is CheckSeverity.INFO


def test_earnings_confirmed_inside_blackout_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, sessions_to_known_earnings=1)
    r = _names(eng.evaluate(cand, ctx), "no_trade.earnings_proximity")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_earnings_estimate_inside_blackout_is_warning_not_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, est_sessions_to_earnings=1)
    r = _names(eng.evaluate(cand, ctx), "no_trade.earnings_proximity")
    assert not r.passed and r.severity is CheckSeverity.WARNING
    assert r.details.get("estimate") is True


def test_earnings_outside_blackout_passes(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, sessions_to_known_earnings=10)
    assert _names(eng.evaluate(cand, ctx), "no_trade.earnings_proximity").passed


# -- conflicting signals --------------------------------------------------------------------------
def test_conflicting_opposite_direction_same_day_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate(direction=Direction.LONG)
    opp = make_candidate(direction=Direction.SHORT, strategy_id="other")
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, same_day_candidates=[cand, opp])
    r = _names(eng.evaluate(cand, ctx), "no_trade.conflicting_signals")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_same_direction_is_not_a_conflict(config):
    eng = NoTradeEngine(config)
    cand = make_candidate(direction=Direction.LONG)
    other = make_candidate(direction=Direction.LONG, strategy_id="other")
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, same_day_candidates=[cand, other])
    assert _names(eng.evaluate(cand, ctx), "no_trade.conflicting_signals").passed


# -- strategy history / regime -------------------------------------------------------------------
def test_strategy_history_unproven_is_warning(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, stats=StrategyStats.empty("s", "1.0.0"))
    r = _names(eng.evaluate(cand, ctx), "no_trade.strategy_history")
    assert not r.passed and r.severity is CheckSeverity.WARNING


def test_strategy_history_validated_negative_expectancy_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    stats = StrategyStats(strategy_id="s", version="1.0.0", n=eng.min_trust, win_rate=0.4,
                          expectancy=-0.01, source="walk_forward_oos", t_stat=-3.0)
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, stats=stats)
    r = _names(eng.evaluate(cand, ctx), "no_trade.strategy_history")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_strategy_history_validated_positive_expectancy_passes(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    stats = StrategyStats(strategy_id="s", version="1.0.0", n=eng.min_trust, win_rate=0.6,
                          expectancy=0.01, source="walk_forward_oos", t_stat=3.0)
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, stats=stats)
    assert _names(eng.evaluate(cand, ctx), "no_trade.strategy_history").passed


def test_regime_not_applied_when_regime_unknown(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, regime=None)
    r = _names(eng.evaluate(cand, ctx), "no_trade.regime")
    assert r.passed and r.severity is CheckSeverity.INFO


def test_regime_significantly_negative_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    stats = StrategyStats(strategy_id="s", version="1.0.0", n=100, source="walk_forward_oos",
                          by_regime={"bear": RegimeStats(n=eng.min_regime_trades, win_rate=0.3,
                                                         expectancy=-0.02, std=0.05, t_stat=-3.0)})
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, regime="bear", stats=stats)
    r = _names(eng.evaluate(cand, ctx), "no_trade.regime")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


# -- data uncertainty: pit / quarantine / missing features / signal-day bar / min price / halted --
def test_pit_unknown_blocks_when_configured(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    cand.pit_status = PitStatus.UNKNOWN
    r = _names(eng.evaluate(cand, NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2)), "no_trade.pit_status")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_pit_unknown_is_warning_when_not_blocking(config):
    cfg = config.with_overrides({"no_trade": {"block_on_unknown_pit": False}})
    eng = NoTradeEngine(cfg)
    cand = make_candidate()
    cand.pit_status = PitStatus.UNKNOWN
    r = _names(eng.evaluate(cand, NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2)), "no_trade.pit_status")
    assert not r.passed and r.severity is CheckSeverity.WARNING


def test_quarantined_symbol_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, quarantined=True, quarantine_reasons=["bad split"])
    r = _names(eng.evaluate(cand, ctx), "no_trade.quarantine")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_missing_features_is_warning(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    cand.features = {"ret_20d": float("nan")}
    r = _names(eng.evaluate(cand, NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2)), "no_trade.missing_features")
    assert not r.passed and r.severity is CheckSeverity.WARNING


def test_signal_day_bar_unknown_fails_closed(config):
    eng = NoTradeEngine(config)
    cand = make_candidate(ref=None)
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, has_bar_on_signal_day=None)
    r = _names(eng.evaluate(cand, ctx), "no_trade.signal_day_bar")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_signal_day_bar_present_passes(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, has_bar_on_signal_day=True)
    assert _names(eng.evaluate(cand, ctx), "no_trade.signal_day_bar").passed


def test_min_price_below_minimum_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate(ref=1.0)
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, last_raw_close=1.0)
    r = _names(eng.evaluate(cand, ctx), "no_trade.min_price")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_min_price_unknown_fails_closed(config):
    eng = NoTradeEngine(config)
    cand = make_candidate(ref=None)
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, last_raw_close=None)
    r = _names(eng.evaluate(cand, ctx), "no_trade.min_price")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_halted_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, halted=True)
    r = _names(eng.evaluate(cand, ctx), "no_trade.halted")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_halted_unknown_is_informational_not_blocking(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, halted=None)
    assert _names(eng.evaluate(cand, ctx), "no_trade.halted").passed


# -- model disagreement -----------------------------------------------------------------------
def test_model_disagreement_active_calibrated_spread_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    preds = [MLPrediction("m1", "1", "up", 0.8, calibrated=True, status="ACTIVE"),
            MLPrediction("m2", "1", "up", 0.2, calibrated=True, status="ACTIVE")]
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, ml_predictions=preds)
    r = _names(eng.evaluate(cand, ctx), "no_trade.model_disagreement")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_model_disagreement_uncalibrated_spread_is_warning_only(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    preds = [MLPrediction("m1", "1", "up", 0.8, calibrated=False, status="ACTIVE"),
            MLPrediction("m2", "1", "up", 0.2, calibrated=False, status="ACTIVE")]
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, ml_predictions=preds)
    r = _names(eng.evaluate(cand, ctx), "no_trade.model_disagreement")
    assert not r.passed and r.severity is CheckSeverity.WARNING


def test_model_agreement_passes(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    preds = [MLPrediction("m1", "1", "up", 0.55, calibrated=True, status="ACTIVE"),
            MLPrediction("m2", "1", "up", 0.60, calibrated=True, status="ACTIVE")]
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, ml_predictions=preds)
    assert _names(eng.evaluate(cand, ctx), "no_trade.model_disagreement").passed


# -- AI objections -----------------------------------------------------------------------------
def _assessment(role, objections):
    return AIAssessment(role=role, provider="mock", model="mock-v1", decision="REJECT", objections=objections)


def test_ai_hard_fail_objection_is_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    review = AIReview(candidate_id=cand.candidate_id, enabled=True,
                      judge=_assessment("judge", [Objection("thesis", ObjectionSeverity.HARD_FAIL, "fatal flaw")]))
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, ai_review=review)
    r = _names(eng.evaluate(cand, ctx), "no_trade.ai_hard_fail")
    assert not r.passed and r.severity is CheckSeverity.CRITICAL


def test_ai_material_concern_is_warning_not_critical(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    review = AIReview(candidate_id=cand.candidate_id, enabled=True,
                      judge=_assessment("judge", [Objection("thesis", ObjectionSeverity.MATERIAL_CONCERN, "risk")]))
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, ai_review=review)
    results = eng.evaluate(cand, ctx)
    assert _names(results, "no_trade.ai_hard_fail").passed
    r = _names(results, "no_trade.ai_material_concern")
    assert not r.passed and r.severity is CheckSeverity.WARNING


def test_ai_disabled_or_missing_is_not_blocking(config):
    eng = NoTradeEngine(config)
    cand = make_candidate()
    ctx = NoTradeContext.build(cand, adv20=1e8, vol_20d=0.2, ai_review=None)
    assert _names(eng.evaluate(cand, ctx), "no_trade.ai_hard_fail").passed


def test_panel_facts_are_truncation_invariant(config):
    from ..shadow.support import make_panel
    closes = {"AAA": [10.0 + 0.01 * i for i in range(40)]}
    panel, _ = make_panel(closes)
    from quantlab.decision.no_trade import panel_facts
    d = panel.dates[25]
    facts_full = panel_facts(panel, "AAA", d)
    truncated = panel.truncate(d)
    facts_trunc = panel_facts(truncated, "AAA", d)
    assert facts_full == facts_trunc
