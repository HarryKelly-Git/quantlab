"""Final decision: combine the no-trade engine, the EV engine and the risk chain with the AI
layer's aggregated verdict into one :class:`DecisionOutcome`, and persist it (append-only).

Precedence (most conservative wins; AI is checked LAST and can only ever narrow a TRADE, never
widen a NO_TRADE):

  1. Any CRITICAL no-trade :class:`~quantlab.core.types.CheckResult` -> ``NO_TRADE`` at stage
     ``NO_TRADE``, regardless of AI or EV.
  2. A failed :class:`~quantlab.risk.engine.RiskResult` (which itself already encodes the EV and
     portfolio-construction outcome, among others) -> ``NO_TRADE`` at ``risk.failed_stage``.
  3. When no ``RiskResult`` is available, the EV estimate is checked directly (mirrors
     :meth:`quantlab.decision.expected_value.EVEngine.passes_gate`) -> ``NO_TRADE`` at stage
     ``EV`` if it fails, or ``UNKNOWN`` at stage ``EV`` if there is no EV estimate at all (missing
     evidence, not a decision).
  4. Only once every quantitative gate has passed does the AI layer's aggregated decision matter:
     ``REJECT`` -> ``NO_TRADE`` (stage ``AI``); ``WATCH`` -> ``WATCH`` (stage ``AI``); ``ACCEPT``
     -> falls through to TRADE; ``UNKNOWN`` (disabled, failed, or no usable verdict) follows
     ``ai.unknown_policy``: ``"allow"`` falls through to TRADE (the AI layer is an optional
     filter), ``"block"`` -> ``NO_TRADE`` (stage ``AI``).
  5. Otherwise ``TRADE`` at stage ``NONE``.

The AI's raw ``decision`` (ACCEPT/REJECT/WATCH/UNKNOWN) is always recorded in
``DecisionOutcome.ai_decision`` even when it did not determine the outcome, so the audit trail
shows what the AI said versus what actually happened.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from quantlab.config import Config
from quantlab.core.types import (
    AIDecision,
    AIReview,
    CheckResult,
    ExpectedValue,
    FinalDecision,
    RejectStage,
    new_id,
)
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.portfolio.construction import SizedOrderIntent
from quantlab.risk.engine import RiskResult

_VALID_UNKNOWN_POLICIES = frozenset({"allow", "block"})


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class DecisionOutcome:
    decision: FinalDecision
    reject_stage: RejectStage
    reasons: list[str] = field(default_factory=list)
    ai_decision: AIDecision | None = None


class FinalDecisionEngine:
    def __init__(self, config: Config):
        self.config = config
        self.min_ev_bps = float(config.get("expected_value.min_ev_bps_after_costs", 10))
        self.unknown_policy = str(config.get("ai.unknown_policy", "allow")).strip().lower()
        if self.unknown_policy not in _VALID_UNKNOWN_POLICIES:
            raise ValueError(f"ai.unknown_policy must be one of {sorted(_VALID_UNKNOWN_POLICIES)}, "
                             f"got {self.unknown_policy!r}")

    def decide(self, candidate: Any, ai_review: AIReview | None, no_trade: list[CheckResult] | None,
               ev: ExpectedValue | None, risk: RiskResult | None) -> DecisionOutcome:
        ai_decision, ai_reason = self._ai_verdict(ai_review)

        nt_blocking = [c for c in (no_trade or []) if c.blocking]
        if nt_blocking:
            return DecisionOutcome(FinalDecision.NO_TRADE, RejectStage.NO_TRADE,
                                   [f"{c.name}: {c.reason}" for c in nt_blocking], ai_decision)

        if risk is not None:
            if not risk.passed:
                blocking = risk.blocking
                reasons = [f"{c.name}: {c.reason}" for c in blocking] or ["risk chain failed"]
                return DecisionOutcome(FinalDecision.NO_TRADE, risk.failed_stage, reasons, ai_decision)
        elif ev is not None:
            ev_bps = ev.ev * 1e4 if _finite(ev.ev) else None
            if ev_bps is None:
                return DecisionOutcome(FinalDecision.UNKNOWN, RejectStage.EV, ["EV UNKNOWN"], ai_decision)
            if ev_bps <= self.min_ev_bps:
                return DecisionOutcome(FinalDecision.NO_TRADE, RejectStage.EV,
                                       [f"EV {ev_bps:.1f} bps after costs <= minimum {self.min_ev_bps:g} bps"],
                                       ai_decision)
        else:
            return DecisionOutcome(FinalDecision.UNKNOWN, RejectStage.EV,
                                   ["no expected-value estimate and no risk-chain result available"], ai_decision)

        if ai_decision is AIDecision.REJECT:
            return DecisionOutcome(FinalDecision.NO_TRADE, RejectStage.AI, [ai_reason], ai_decision)
        if ai_decision is AIDecision.WATCH:
            return DecisionOutcome(FinalDecision.WATCH, RejectStage.AI, [ai_reason], ai_decision)
        if ai_decision is AIDecision.UNKNOWN and self.unknown_policy == "block":
            return DecisionOutcome(FinalDecision.NO_TRADE, RejectStage.AI, [ai_reason], ai_decision)

        reasons = ["all quantitative gates passed"]
        if ai_decision is AIDecision.ACCEPT:
            reasons.append(ai_reason)
        elif ai_decision is AIDecision.UNKNOWN:
            reasons.append(f"{ai_reason}; ai.unknown_policy=allow: quantitative layers decide")
        return DecisionOutcome(FinalDecision.TRADE, RejectStage.NONE, reasons, ai_decision)

    @staticmethod
    def _ai_verdict(ai_review: AIReview | None) -> tuple[AIDecision, str]:
        if ai_review is None or not ai_review.enabled:
            return AIDecision.UNKNOWN, "AI review not run (layer disabled)"
        d = ai_review.decision
        if d is AIDecision.REJECT:
            texts = [f"[{a.role}] {o.category}: {o.text}" for a in
                    (ai_review.researcher, ai_review.adversary, ai_review.judge)
                    if a is not None for o in a.objections]
            return d, "AI REJECT" + (": " + "; ".join(texts) if texts else "")
        if d is AIDecision.WATCH:
            return d, "AI WATCH: recommends watching rather than trading now"
        if d is AIDecision.ACCEPT:
            return d, "AI ACCEPT"
        return AIDecision.UNKNOWN, "AI review UNKNOWN (failed call or no usable verdict)"


def persist_decision(db: Database, candidate_id: str, outcome: DecisionOutcome, run_id: str | None = None,
                     ev: ExpectedValue | None = None, no_trade: list[CheckResult] | None = None,
                     sizing: SizedOrderIntent | None = None) -> str:
    """Append one row to ``decisions`` (001, append-only). Returns the new ``decision_id``."""
    decision_id = new_id("dec")
    db.insert("decisions", {
        "decision_id": decision_id,
        "candidate_id": candidate_id,
        "run_id": run_id,
        "decision": outcome.decision.value,
        "reject_stage": outcome.reject_stage.value,
        "reasons_json": to_json(outcome.reasons),
        "ai_decision": outcome.ai_decision.value if outcome.ai_decision is not None else None,
        "ev_json": to_json(ev) if ev is not None else None,
        "no_trade_json": to_json(no_trade) if no_trade is not None else None,
        "sizing_json": to_json(sizing) if sizing is not None else None,
        "created_at": utcnow_iso(),
    })
    return decision_id


__all__ = ["DecisionOutcome", "FinalDecisionEngine", "persist_decision"]
