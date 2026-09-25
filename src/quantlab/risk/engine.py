"""Pre-trade risk chain (ARCHITECTURE.md section 7):

    DATA -> SIGNAL -> STRATEGY -> EV -> PORTFOLIO -> RISK -> EXECUTION

Stages run IN ORDER; the first stage containing a CRITICAL failure stops the chain (later stages
are never evaluated for this candidate — ``RiskResult.failed_stage`` names it). This is the single
authoritative gate: DATA/SIGNAL/STRATEGY stages are populated by RE-CATEGORIZING the no-trade
engine's own :class:`~quantlab.core.types.CheckResult` list (so a fact is checked once, by the
no-trade engine, and simply sorted into its stage here — never recomputed), while EV/PORTFOLIO/
RISK/EXECUTION are new checks specific to this chain:

  * STRATEGY additionally requires the strategy's registry ``status`` to be ACTIVE before the BOT
    book may trade it; any other book (SHADOW, HUMAN, BACKTEST) is unaffected — an inactive
    strategy is shadow-only, not blocked outright.
  * EV requires ``expected_value.min_ev_bps_after_costs`` to be strictly exceeded (mirrors
    :meth:`quantlab.decision.expected_value.EVEngine.passes_gate`) — an unknown EV fails closed.
  * PORTFOLIO reflects whatever :class:`~quantlab.portfolio.construction.PortfolioConstructor`
    already decided for this candidate (sized -> pass, rejected -> fail); not applicable outside
    the BOT book, or when portfolio construction has not run yet for this candidate.
  * RISK checks book drawdown, today's order count and this order's notional against
    ``risk.max_drawdown_pause`` / ``risk.max_daily_orders`` / ``risk.max_order_notional``.
  * EXECUTION checks the system is ACTIVE (not SYSTEM_PAUSED), the broker is reachable, and the
    entry reference price is a sane positive finite number.

Unknown inputs fail CLOSED throughout (an unreadable drawdown, a missing strategy status, a
missing EV) — consistent with the no-trade engine's own convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from quantlab.config import Config
from quantlab.core.types import (
    Book,
    Candidate,
    CheckResult,
    CheckSeverity,
    ExpectedValue,
    RejectStage,
    StrategyStage, StrategyStatus,
    SystemState,
)
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.portfolio.construction import PortfolioRejection, SizedOrderIntent

STAGE_ORDER: list[RejectStage] = [
    RejectStage.DATA, RejectStage.SIGNAL, RejectStage.STRATEGY, RejectStage.EV,
    RejectStage.PORTFOLIO, RejectStage.RISK, RejectStage.EXECUTION,
]

# Which risk-chain stage each no-trade check belongs to. Anything not listed defaults to SIGNAL
# (a new no-trade rule is, by default, evidence about the signal, not about data availability or
# the strategy's track record) — see tests for a completeness check against NoTradeEngine.evaluate.
_NO_TRADE_STAGE: dict[str, RejectStage] = {
    "no_trade.pit_status": RejectStage.DATA,
    "no_trade.quarantine": RejectStage.DATA,
    "no_trade.missing_features": RejectStage.DATA,
    "no_trade.signal_day_bar": RejectStage.DATA,
    "no_trade.halted": RejectStage.DATA,
    "no_trade.liquidity": RejectStage.SIGNAL,
    "no_trade.volatility": RejectStage.SIGNAL,
    "no_trade.earnings_proximity": RejectStage.SIGNAL,
    "no_trade.conflicting_signals": RejectStage.SIGNAL,
    "no_trade.min_price": RejectStage.SIGNAL,
    "no_trade.model_disagreement": RejectStage.SIGNAL,
    "no_trade.ai_hard_fail": RejectStage.SIGNAL,
    "no_trade.ai_material_concern": RejectStage.SIGNAL,
    "no_trade.regime": RejectStage.STRATEGY,
    "no_trade.strategy_history": RejectStage.STRATEGY,
}


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _check(name: str, passed: bool, reason: str, severity: CheckSeverity = CheckSeverity.CRITICAL,
          **details: Any) -> CheckResult:
    return CheckResult(name=name, passed=passed, severity=CheckSeverity.INFO if passed else severity,
                       reason=reason, details=details)


@dataclass
class DecisionContext:
    """Everything :class:`RiskEngine` needs for one candidate. Fields left at their default mean
    "this layer has not run / does not apply" and are handled per-stage, never guessed."""

    candidate: Candidate
    book: Book = Book.BOT
    no_trade_checks: list[CheckResult] = field(default_factory=list)
    ev: ExpectedValue | None = None
    sizing: SizedOrderIntent | None = None
    portfolio_rejection: PortfolioRejection | None = None
    strategy_status: StrategyStatus | None = None
    strategy_stage: StrategyStage | None = None   # None = not supplied (stage not checked)
    system_state: SystemState = SystemState.ACTIVE
    broker_available: bool = True
    daily_order_count: int = 0
    book_drawdown: float | None = None
    as_of: date | None = None


@dataclass
class RiskResult:
    checks: list[CheckResult]
    passed: bool
    failed_stage: RejectStage = RejectStage.NONE

    @property
    def blocking(self) -> list[CheckResult]:
        return [c for c in self.checks if c.blocking]


class RiskEngine:
    def __init__(self, config: Config):
        self.config = config
        self.max_drawdown_pause = float(config.get("risk.max_drawdown_pause", 0.20))
        self.max_daily_orders = int(config.get("risk.max_daily_orders", 20))
        self.max_order_notional = float(config.get("risk.max_order_notional", 15000))
        self.min_ev_bps = float(config.get("expected_value.min_ev_bps_after_costs", 10))

    def run(self, ctx: DecisionContext) -> RiskResult:
        checks: list[CheckResult] = []
        failed_stage = RejectStage.NONE
        for stage in STAGE_ORDER:
            stage_checks = self._stage_checks(stage, ctx)
            checks.extend(stage_checks)
            if any(c.blocking for c in stage_checks):
                failed_stage = stage
                break
        return RiskResult(checks=checks, passed=failed_stage is RejectStage.NONE, failed_stage=failed_stage)

    # -- stage dispatch ---------------------------------------------------------------------------
    def _stage_checks(self, stage: RejectStage, ctx: DecisionContext) -> list[CheckResult]:
        if stage in (RejectStage.DATA, RejectStage.SIGNAL):
            return [c for c in ctx.no_trade_checks if _NO_TRADE_STAGE.get(c.name, RejectStage.SIGNAL) is stage]
        if stage is RejectStage.STRATEGY:
            return self._strategy(ctx)
        if stage is RejectStage.EV:
            return [self._ev(ctx)]
        if stage is RejectStage.PORTFOLIO:
            return [self._portfolio(ctx)]
        if stage is RejectStage.RISK:
            return self._risk(ctx)
        if stage is RejectStage.EXECUTION:
            return self._execution(ctx)
        return []

    # -- STRATEGY -----------------------------------------------------------------------------------
    def _strategy(self, ctx: DecisionContext) -> list[CheckResult]:
        out = [c for c in ctx.no_trade_checks if _NO_TRADE_STAGE.get(c.name) is RejectStage.STRATEGY]
        n = "risk.strategy_status"
        if ctx.book is not Book.BOT:
            out.append(_check(n, True, f"book {ctx.book.value} != BOT: ACTIVE strategy status not required "
                              "(shadow-only trading)", book=ctx.book.value))
            return out
        if ctx.strategy_status is None:
            out.append(_check(n, False, "strategy status UNKNOWN: fail closed for BOT trading"))
            return out
        ok = ctx.strategy_status is StrategyStatus.ACTIVE
        out.append(_check(n, ok, f"strategy status {ctx.strategy_status.value} "
                          + ("satisfies" if ok else "does not satisfy") + " the ACTIVE requirement for BOT trading",
                          status=ctx.strategy_status.value))
        if ok and ctx.strategy_stage is not None:
            tradable = ctx.strategy_stage in (StrategyStage.PAPER, StrategyStage.PROMOTED)
            out.append(_check("risk.strategy_stage", tradable,
                              f"strategy stage {ctx.strategy_stage.value} "
                              + ("is" if tradable else "is not") + " PAPER/PROMOTED (paper-eligible)",
                              stage=ctx.strategy_stage.value))
        return out

    # -- EV -------------------------------------------------------------------------------------
    def _ev(self, ctx: DecisionContext) -> CheckResult:
        ev = ctx.ev
        ev_bps = _num(ev.ev) * 1e4 if ev is not None else None
        if ev_bps is None:
            return _check("risk.ev", False, "EV UNKNOWN: fail closed")
        ok = ev_bps > self.min_ev_bps
        return _check("risk.ev", ok, f"EV {ev_bps:.1f} bps after costs {'>' if ok else '<='} minimum "
                      f"{self.min_ev_bps:g} bps", ev_bps=ev_bps, min_ev_bps=self.min_ev_bps)

    # -- PORTFOLIO --------------------------------------------------------------------------------
    def _portfolio(self, ctx: DecisionContext) -> CheckResult:
        n = "risk.portfolio"
        if ctx.book is not Book.BOT:
            return _check(n, True, f"book {ctx.book.value} != BOT: portfolio construction not applicable")
        if ctx.sizing is not None:
            return _check(n, True, f"sized: qty={ctx.sizing.qty} notional=${ctx.sizing.notional:,.0f}",
                          qty=ctx.sizing.qty, notional=ctx.sizing.notional)
        if ctx.portfolio_rejection is not None:
            return _check(n, False, f"portfolio construction rejected this candidate: "
                          f"{ctx.portfolio_rejection.reason}", rejection_reason=ctx.portfolio_rejection.reason)
        return _check(n, True, "portfolio construction has not run yet for this candidate")

    # -- RISK -----------------------------------------------------------------------------------
    def _risk(self, ctx: DecisionContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        if ctx.book_drawdown is None:
            out.append(_check("risk.drawdown", False, "book drawdown UNKNOWN: fail closed"))
        else:
            ok = ctx.book_drawdown < self.max_drawdown_pause
            out.append(_check("risk.drawdown", ok, f"book drawdown {ctx.book_drawdown:.1%} "
                              f"{'<' if ok else '>='} max {self.max_drawdown_pause:.1%}",
                              drawdown=ctx.book_drawdown, max=self.max_drawdown_pause))
        ok = ctx.daily_order_count < self.max_daily_orders
        out.append(_check("risk.daily_orders", ok, f"{ctx.daily_order_count} order(s) already today "
                          f"{'<' if ok else '>='} max {self.max_daily_orders}",
                          count=ctx.daily_order_count, max=self.max_daily_orders))
        notional = ctx.sizing.notional if ctx.sizing is not None else None
        if notional is None:
            out.append(_check("risk.order_notional", True, "no sized order: notional check not applicable"))
        else:
            ok = notional <= self.max_order_notional
            out.append(_check("risk.order_notional", ok, f"order notional ${notional:,.0f} "
                              f"{'<=' if ok else '>'} max ${self.max_order_notional:,.0f}",
                              notional=notional, max=self.max_order_notional))
        return out

    # -- EXECUTION --------------------------------------------------------------------------------
    def _execution(self, ctx: DecisionContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        ok = ctx.system_state is SystemState.ACTIVE
        out.append(_check("risk.system_state", ok, f"system_state {ctx.system_state.value}"
                          + ("" if ok else ": new orders are blocked"), system_state=ctx.system_state.value))
        out.append(_check("risk.broker_available", bool(ctx.broker_available),
                          "broker available" if ctx.broker_available else "broker UNAVAILABLE"))
        price = _num(ctx.candidate.plan.entry_ref_price)
        ok = price is not None and price > 0
        out.append(_check("risk.price_sane", ok,
                          f"entry_ref_price ${price:.2f} is sane" if ok else "entry_ref_price UNKNOWN/non-positive",
                          price=price))
        return out


def record_checks(db: Database, candidate_id: str, decision_id: str | None, checks: list[CheckResult]) -> int:
    """Append ``checks`` to ``risk_checks`` (001, append-only)."""
    now = utcnow_iso()
    rows = [{
        "candidate_id": candidate_id, "decision_id": decision_id, "check_name": c.name, "passed": int(c.passed),
        "severity": c.severity.value, "reason": c.reason, "details_json": to_json(c.details), "created_at": now,
    } for c in checks]
    return db.insert_many("risk_checks", rows)


__all__ = ["STAGE_ORDER", "DecisionContext", "RiskEngine", "RiskResult", "record_checks"]
