"""FinalDecisionEngine: AI can never override a hard quantitative failure, and ai.unknown_policy
is honored both ways. persist_decision round-trips through the append-only `decisions` table."""
from __future__ import annotations

import pytest

from quantlab.core.types import (
    AIAssessment,
    AIDecision,
    AIReview,
    CheckResult,
    CheckSeverity,
    ExpectedValue,
    FinalDecision,
    Objection,
    ObjectionSeverity,
    RejectStage,
)
from quantlab.decision.final import FinalDecisionEngine, persist_decision
from quantlab.risk.engine import RiskResult

from ..shadow.support import make_candidate

GOOD_EV = ExpectedValue(p_win=0.55, avg_win=0.03, avg_loss=-0.02, cost=0.001, ev=0.003, n_obs=100)
CRITICAL_NT = [CheckResult(name="no_trade.liquidity", passed=False, severity=CheckSeverity.CRITICAL,
                           reason="liquidity UNKNOWN")]
CLEAN_NT = [CheckResult(name="no_trade.liquidity", passed=True, severity=CheckSeverity.INFO, reason="ok")]
PASSING_RISK = RiskResult(checks=[CheckResult("risk.ev", True, CheckSeverity.INFO, "ok")], passed=True,
                          failed_stage=RejectStage.NONE)
FAILING_RISK = RiskResult(checks=[CheckResult("risk.drawdown", False, CheckSeverity.CRITICAL, "drawdown too high")],
                          passed=False, failed_stage=RejectStage.RISK)


def _accept_review():
    return AIReview(candidate_id="c1", enabled=True, decision=AIDecision.ACCEPT,
                    judge=AIAssessment(role="judge", provider="mock", model="mock-v1", decision=AIDecision.ACCEPT))


def _reject_review():
    return AIReview(candidate_id="c1", enabled=True, decision=AIDecision.REJECT,
                    judge=AIAssessment(role="judge", provider="mock", model="mock-v1", decision=AIDecision.REJECT,
                                       objections=[Objection("thesis", ObjectionSeverity.MATERIAL_CONCERN, "weak")]))


def _watch_review():
    return AIReview(candidate_id="c1", enabled=True, decision=AIDecision.WATCH,
                    judge=AIAssessment(role="judge", provider="mock", model="mock-v1", decision=AIDecision.WATCH))


# -- hard constraints override AI -----------------------------------------------------------------
def test_ai_accept_cannot_override_critical_no_trade_failure(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), _accept_review(), CRITICAL_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.NO_TRADE
    assert out.reject_stage is RejectStage.NO_TRADE
    assert out.ai_decision is AIDecision.ACCEPT


def test_ai_accept_cannot_override_failing_risk_chain(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), _accept_review(), CLEAN_NT, GOOD_EV, FAILING_RISK)
    assert out.decision is FinalDecision.NO_TRADE
    assert out.reject_stage is RejectStage.RISK
    assert out.ai_decision is AIDecision.ACCEPT


def test_no_trade_failure_takes_precedence_over_risk_result_too(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), None, CRITICAL_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.NO_TRADE
    assert out.reject_stage is RejectStage.NO_TRADE


# -- AI decision when quant layers pass -------------------------------------------------------
def test_ai_reject_blocks_trade(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), _reject_review(), CLEAN_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.NO_TRADE
    assert out.reject_stage is RejectStage.AI
    assert out.ai_decision is AIDecision.REJECT
    assert any("weak" in r for r in out.reasons)


def test_ai_watch_gives_watch_decision(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), _watch_review(), CLEAN_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.WATCH
    assert out.reject_stage is RejectStage.AI
    assert out.ai_decision is AIDecision.WATCH


def test_ai_accept_with_clean_quant_layers_trades(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), _accept_review(), CLEAN_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.TRADE
    assert out.reject_stage is RejectStage.NONE
    assert out.ai_decision is AIDecision.ACCEPT


# -- ai.unknown_policy both ways -----------------------------------------------------------------
def test_unknown_policy_allow_continues_to_trade(config):
    eng = FinalDecisionEngine(config.with_overrides({"ai": {"unknown_policy": "allow"}}))
    out = eng.decide(make_candidate(), None, CLEAN_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.TRADE
    assert out.ai_decision is AIDecision.UNKNOWN


def test_unknown_policy_block_blocks_trade(config):
    eng = FinalDecisionEngine(config.with_overrides({"ai": {"unknown_policy": "block"}}))
    out = eng.decide(make_candidate(), None, CLEAN_NT, GOOD_EV, PASSING_RISK)
    assert out.decision is FinalDecision.NO_TRADE
    assert out.reject_stage is RejectStage.AI
    assert out.ai_decision is AIDecision.UNKNOWN


def test_invalid_unknown_policy_rejected(config):
    with pytest.raises(ValueError):
        FinalDecisionEngine(config.with_overrides({"ai": {"unknown_policy": "maybe"}}))


# -- EV fallback when no RiskResult is supplied -----------------------------------------------
def test_ev_gate_enforced_directly_when_risk_is_none(config):
    eng = FinalDecisionEngine(config)
    bad_ev = ExpectedValue(p_win=0.5, avg_win=0.01, avg_loss=-0.01, cost=0.01, ev=-0.01, n_obs=0)
    out = eng.decide(make_candidate(), _accept_review(), CLEAN_NT, bad_ev, None)
    assert out.decision is FinalDecision.NO_TRADE
    assert out.reject_stage is RejectStage.EV


def test_missing_ev_and_risk_gives_unknown_not_no_trade(config):
    eng = FinalDecisionEngine(config)
    out = eng.decide(make_candidate(), _accept_review(), CLEAN_NT, None, None)
    assert out.decision is FinalDecision.UNKNOWN
    assert out.reject_stage is RejectStage.EV


# -- persistence -----------------------------------------------------------------------------
def test_persist_decision_round_trips(db, config):
    eng = FinalDecisionEngine(config)
    cand = make_candidate()
    out = eng.decide(cand, _accept_review(), CLEAN_NT, GOOD_EV, PASSING_RISK)
    decision_id = persist_decision(db, cand.candidate_id, out, run_id="run1", ev=GOOD_EV, no_trade=CLEAN_NT)
    row = db.fetchone("SELECT * FROM decisions WHERE decision_id=?", (decision_id,))
    assert row is not None
    assert row["candidate_id"] == cand.candidate_id
    assert row["decision"] == "TRADE"
    assert row["reject_stage"] == "NONE"
    assert row["ai_decision"] == "ACCEPT"
    assert row["ev_json"] is not None
    assert row["no_trade_json"] is not None


def test_decisions_table_is_append_only(db, config):
    eng = FinalDecisionEngine(config)
    cand = make_candidate()
    out = eng.decide(cand, None, CLEAN_NT, GOOD_EV, PASSING_RISK)
    decision_id = persist_decision(db, cand.candidate_id, out)
    with pytest.raises(Exception):
        db.execute("UPDATE decisions SET decision='NO_TRADE' WHERE decision_id=?", (decision_id,))
