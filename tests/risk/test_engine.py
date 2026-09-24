"""RiskEngine: stage order DATA->SIGNAL->STRATEGY->EV->PORTFOLIO->RISK->EXECUTION, short-circuit
on the first CRITICAL failure, no-trade check categorization, and risk_checks recording."""
from __future__ import annotations

import pytest

from quantlab.core.types import Book, CheckResult, CheckSeverity, ExpectedValue, RejectStage, \
    StrategyStatus, SystemState
from quantlab.portfolio.construction import PortfolioRejection, SizedOrderIntent
from quantlab.risk.engine import STAGE_ORDER, DecisionContext, RiskEngine, record_checks

from ..shadow.support import make_candidate

GOOD_EV = ExpectedValue(p_win=0.55, avg_win=0.03, avg_loss=-0.02, cost=0.001, ev=0.003, n_obs=100)
BAD_EV = ExpectedValue(p_win=0.5, avg_win=0.01, avg_loss=-0.01, cost=0.01, ev=-0.01, n_obs=100)


def _sizing(qty=10, entry_ref_price=100.0):
    """``SizedOrderIntent.notional`` is a property (qty * entry_ref_price) — vary those, not a dict key."""
    return SizedOrderIntent(candidate_id="c1", symbol="AAA", strategy_id="s1", strategy_version="1.0.0",
                            direction=make_candidate().direction, qty=qty, entry_ref_price=entry_ref_price,
                            stop_price=entry_ref_price * 0.95, target_price=None, sector=None, sizing={})


def _clean_ctx(**overrides) -> DecisionContext:
    base = dict(candidate=make_candidate(), book=Book.BOT, no_trade_checks=[], ev=GOOD_EV,
               sizing=_sizing(), strategy_status=StrategyStatus.ACTIVE, system_state=SystemState.ACTIVE,
               broker_available=True, daily_order_count=0, book_drawdown=0.05)
    base.update(overrides)
    return DecisionContext(**base)


def test_stage_order_constant_matches_architecture():
    assert STAGE_ORDER == [RejectStage.DATA, RejectStage.SIGNAL, RejectStage.STRATEGY, RejectStage.EV,
                          RejectStage.PORTFOLIO, RejectStage.RISK, RejectStage.EXECUTION]


def test_all_clean_context_passes_every_stage(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx())
    assert result.passed
    assert result.failed_stage is RejectStage.NONE
    assert not result.blocking
    # the EV/PORTFOLIO/RISK/EXECUTION-specific checks all ran (chain was not short-circuited)
    names = {c.name for c in result.checks}
    assert {"risk.ev", "risk.portfolio", "risk.drawdown", "risk.daily_orders", "risk.order_notional",
           "risk.system_state", "risk.broker_available", "risk.price_sane", "risk.strategy_status"} <= names


def test_data_stage_critical_failure_stops_before_later_stages(config):
    eng = RiskEngine(config)
    nt = [CheckResult("no_trade.pit_status", False, CheckSeverity.CRITICAL, "PIT UNKNOWN")]
    result = eng.run(_clean_ctx(no_trade_checks=nt))
    assert not result.passed
    assert result.failed_stage is RejectStage.DATA
    # nothing from later stages (e.g. risk.ev) was ever evaluated
    assert not any(c.name == "risk.ev" for c in result.checks)


def test_signal_stage_stops_before_strategy(config):
    eng = RiskEngine(config)
    nt = [CheckResult("no_trade.liquidity", False, CheckSeverity.CRITICAL, "illiquid")]
    result = eng.run(_clean_ctx(no_trade_checks=nt))
    assert result.failed_stage is RejectStage.SIGNAL
    assert not any(c.name == "risk.strategy_status" for c in result.checks)


def test_strategy_status_not_active_blocks_bot_book(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(strategy_status=StrategyStatus.SHADOW))
    assert not result.passed
    assert result.failed_stage is RejectStage.STRATEGY


def test_strategy_status_unknown_fails_closed_for_bot(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(strategy_status=None))
    assert result.failed_stage is RejectStage.STRATEGY


def test_strategy_status_not_required_outside_bot_book(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(book=Book.SHADOW, strategy_status=None, sizing=None))
    # STRATEGY passes for non-BOT books regardless of strategy_status
    assert not any(c.name == "risk.strategy_status" and c.blocking for c in result.checks)


def test_ev_gate_uses_configured_threshold(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(ev=BAD_EV))
    assert not result.passed
    assert result.failed_stage is RejectStage.EV


def test_ev_unknown_fails_closed(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(ev=None))
    assert result.failed_stage is RejectStage.EV


def test_portfolio_rejection_blocks_at_portfolio_stage(config):
    eng = RiskEngine(config)
    rej = PortfolioRejection(candidate_id="c1", symbol="AAA", reason="no stop")
    result = eng.run(_clean_ctx(sizing=None, portfolio_rejection=rej))
    assert result.failed_stage is RejectStage.PORTFOLIO


def test_portfolio_not_applicable_outside_bot_book(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(book=Book.SHADOW, sizing=None))
    assert not any(c.name == "risk.portfolio" and c.blocking for c in result.checks)


def test_drawdown_unknown_fails_closed(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(book_drawdown=None))
    assert result.failed_stage is RejectStage.RISK


def test_drawdown_above_max_blocks(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(book_drawdown=eng.max_drawdown_pause))
    assert result.failed_stage is RejectStage.RISK


def test_daily_order_count_at_max_blocks(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(daily_order_count=eng.max_daily_orders))
    assert result.failed_stage is RejectStage.RISK


def test_order_notional_above_max_blocks(config):
    eng = RiskEngine(config)
    big = _sizing(qty=10, entry_ref_price=(eng.max_order_notional + 100) / 10)
    assert big.notional > eng.max_order_notional
    result = eng.run(_clean_ctx(sizing=big))
    assert result.failed_stage is RejectStage.RISK


def test_system_paused_blocks_at_execution(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(system_state=SystemState.PAUSED))
    assert result.failed_stage is RejectStage.EXECUTION


def test_broker_unavailable_blocks_at_execution(config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(broker_available=False))
    assert result.failed_stage is RejectStage.EXECUTION


def test_insane_price_blocks_at_execution(config):
    eng = RiskEngine(config)
    cand = make_candidate(ref=-1.0)
    result = eng.run(_clean_ctx(candidate=cand))
    assert result.failed_stage is RejectStage.EXECUTION


def test_no_trade_checks_categorized_into_correct_stage(config):
    eng = RiskEngine(config)
    nt = [
        CheckResult("no_trade.pit_status", True, CheckSeverity.INFO, "ok"),
        CheckResult("no_trade.liquidity", True, CheckSeverity.INFO, "ok"),
        CheckResult("no_trade.strategy_history", True, CheckSeverity.INFO, "ok"),
    ]
    result = eng.run(_clean_ctx(no_trade_checks=nt))
    assert result.passed
    # each no-trade check appears exactly once in the assembled chain
    names = [c.name for c in result.checks]
    assert names.count("no_trade.pit_status") == 1
    assert names.count("no_trade.liquidity") == 1
    assert names.count("no_trade.strategy_history") == 1


def test_unmapped_no_trade_check_defaults_to_signal_stage(config):
    eng = RiskEngine(config)
    nt = [CheckResult("no_trade.some_future_rule", False, CheckSeverity.CRITICAL, "new rule fired")]
    result = eng.run(_clean_ctx(no_trade_checks=nt))
    assert result.failed_stage is RejectStage.SIGNAL


def test_record_checks_writes_risk_checks_rows(db, config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx(ev=BAD_EV))
    n = record_checks(db, "cand_1", "dec_1", result.checks)
    assert n == len(result.checks)
    rows = db.fetchall("SELECT * FROM risk_checks WHERE candidate_id='cand_1'")
    assert len(rows) == len(result.checks)
    assert any(r["check_name"] == "risk.ev" and r["passed"] == 0 for r in rows)


def test_risk_checks_table_is_append_only(db, config):
    eng = RiskEngine(config)
    result = eng.run(_clean_ctx())
    record_checks(db, "cand_1", "dec_1", result.checks)
    with pytest.raises(Exception):
        db.execute("UPDATE risk_checks SET passed=0 WHERE candidate_id='cand_1'")
