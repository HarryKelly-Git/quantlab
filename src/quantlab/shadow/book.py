"""Shadow book: an immutable record of EVERY serious candidate, traded or not.

Why it exists: a filter layer (AI, no-trade rules, EV, portfolio, risk, the human) can only be
judged by what happened to the opportunities it REJECTED as well as the ones it accepted. If only
traded names were kept, every filter would look good by construction (survivorship of decisions).
So the pipeline records one row per candidate with the bot's final decision and the stage that
stopped it; :mod:`quantlab.shadow.outcomes` later measures what each would have returned with the
same fill model as real paper trades.

Rows are append-only (``shadow_opportunities`` has UPDATE/DELETE triggers). Recording the same
candidate twice with the same verdict is an idempotent no-op (safe pipeline resume); recording a
DIFFERENT verdict for an already-recorded candidate is refused — history is never rewritten.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Iterable

from quantlab.core.types import (
    AIDecision,
    Candidate,
    FinalDecision,
    MLPrediction,
    Objection,
    RejectStage,
    new_id,
)
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)


class ShadowBookError(ValueError):
    """Invalid shadow record (inconsistent decision/stage, bad enum, missing field)."""


class ShadowConflictError(ShadowBookError):
    """The candidate was already recorded with a different verdict (rows are immutable)."""


def _enum(cls, value, what: str):
    try:
        return value if isinstance(value, cls) else cls(value)
    except ValueError:
        raise ShadowBookError(f"invalid {what}: {value!r}; allowed {[e.value for e in cls]}") from None


def _plain(obj: Any) -> Any:
    """Dataclass/enum-aware conversion to JSON-friendly structures."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _objections_payload(objections: Iterable[Objection | dict] | None) -> list[dict] | None:
    if objections is None:
        return None
    out = []
    for o in objections:
        d = _plain(o)
        if not isinstance(d, dict) or "text" not in d:
            raise ShadowBookError(f"objection must be an Objection or dict with 'text': {o!r}")
        out.append(d)
    return out


def _ml_payload(ml: MLPrediction | Iterable[MLPrediction | dict] | dict | None) -> Any:
    if ml is None:
        return None
    if isinstance(ml, (MLPrediction, dict)):
        return [_plain(ml)]
    return [_plain(m) for m in ml]


class ShadowBook:
    """Writer/reader for ``shadow_opportunities``."""

    def __init__(self, db: Database):
        self.db = db

    def record(
        self,
        candidate: Candidate,
        bot_decision: FinalDecision | str,
        reject_stage: RejectStage | str,
        reason: str,
        ai_decision: AIDecision | str | None = None,
        objections: Iterable[Objection | dict] | None = None,
        ml: MLPrediction | Iterable[MLPrediction | dict] | dict | None = None,
        is_synthetic: bool = False,
    ) -> str:
        """Record one candidate's final bot verdict. Returns the opportunity_id.

        Consistency rules (fail loudly rather than store an ambiguous row):
          * TRADE  <=> reject_stage NONE (a traded candidate was not rejected anywhere);
          * NO_TRADE requires a concrete reject_stage (where was it stopped?);
          * ai_decision None means "not reviewed by the AI layer" (stored as NULL), which is
            different from UNKNOWN ("reviewed, but no usable verdict").
        """
        decision = _enum(FinalDecision, bot_decision, "bot_decision")
        stage = _enum(RejectStage, reject_stage, "reject_stage")
        ai = _enum(AIDecision, ai_decision, "ai_decision") if ai_decision is not None else None
        if decision is FinalDecision.TRADE and stage is not RejectStage.NONE:
            raise ShadowBookError(f"TRADE must have reject_stage NONE, got {stage.value}")
        if decision is FinalDecision.NO_TRADE and stage is RejectStage.NONE:
            raise ShadowBookError("NO_TRADE must name the reject_stage that stopped the candidate")
        if not candidate.symbol or not candidate.strategy_id:
            raise ShadowBookError("candidate needs a symbol and strategy_id")

        existing = self.db.fetchone(
            "SELECT opportunity_id, bot_decision, reject_stage, ai_decision FROM shadow_opportunities "
            "WHERE candidate_id=? ORDER BY created_at LIMIT 1", (candidate.candidate_id,))
        if existing is not None:
            same = (existing["bot_decision"] == decision.value and existing["reject_stage"] == stage.value
                    and existing["ai_decision"] == (ai.value if ai else None))
            if not same:
                raise ShadowConflictError(
                    f"candidate {candidate.candidate_id} already recorded as {existing['bot_decision']}/"
                    f"{existing['reject_stage']} (opportunity {existing['opportunity_id']}); rows are immutable")
            return existing["opportunity_id"]

        plan = candidate.plan
        # shadow_opportunities has no direction/risk columns: keep them (and the quantitative reasons)
        # in quant_reasoning as JSON so outcome tracking never has to guess the trade direction.
        quant = {
            "reasons": list(candidate.reasons),
            "direction": candidate.direction.value,
            "features": candidate.features,
            "risk": candidate.risk,
            "pit_status": candidate.pit_status.value,
            "invalidation": plan.invalidation,
        }
        opportunity_id = new_id("opp")
        self.db.insert("shadow_opportunities", {
            "opportunity_id": opportunity_id,
            "candidate_id": candidate.candidate_id,
            "as_of_date": candidate.as_of_date.isoformat(),
            "symbol": candidate.symbol.upper(),
            "strategy_id": candidate.strategy_id,
            "strategy_version": candidate.strategy_version,
            "score": float(candidate.score) if candidate.score is not None else None,
            "quant_reasoning": to_json(quant),
            "ml_json": to_json(_ml_payload(ml)) if ml is not None else None,
            "ai_decision": ai.value if ai else None,
            "objections_json": to_json(_objections_payload(objections)) if objections is not None else None,
            "bot_decision": decision.value,
            "reject_stage": stage.value,
            "reject_reason": reason or "",
            "entry_convention": plan.entry or "next_open",
            "entry_ref_price": plan.entry_ref_price,
            "stop_price": plan.stop_price,
            "target_price": plan.target_price,
            "holding_sessions": int(plan.holding_sessions) if plan.holding_sessions else None,
            "created_at": utcnow_iso(),
            "is_synthetic": int(bool(is_synthetic)),
        })
        log_event(log, "shadow opportunity recorded", opportunity_id=opportunity_id,
                  candidate_id=candidate.candidate_id, symbol=candidate.symbol, decision=decision.value,
                  reject_stage=stage.value)
        return opportunity_id

    # -- read helpers ----------------------------------------------------------------------------
    def get(self, opportunity_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM shadow_opportunities WHERE opportunity_id=?", (opportunity_id,))
        return _decode(row) if row else None

    def for_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM shadow_opportunities WHERE candidate_id=? ORDER BY created_at LIMIT 1",
                               (candidate_id,))
        return _decode(row) if row else None

    def list(self, start: str | None = None, end: str | None = None, limit: int = 500,
             include_synthetic: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM shadow_opportunities WHERE 1=1"
        params: list[Any] = []
        if start:
            sql += " AND as_of_date >= ?"
            params.append(str(start)[:10])
        if end:
            sql += " AND as_of_date <= ?"
            params.append(str(end)[:10])
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        sql += " ORDER BY as_of_date DESC, created_at DESC LIMIT ?"
        params.append(int(limit))
        return [_decode(r) for r in self.db.fetchall(sql, params)]


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["quant"] = parse_quant_reasoning(row.get("quant_reasoning"))
    out["ml"] = _safe_json(row.get("ml_json"))
    out["objections"] = _safe_json(row.get("objections_json"))
    out["is_synthetic"] = bool(row.get("is_synthetic"))
    return out


def _safe_json(text: Any) -> Any:
    if not isinstance(text, str) or not text:
        return None
    try:
        return from_json(text)
    except ValueError:
        return None


def parse_quant_reasoning(text: Any) -> dict[str, Any]:
    """Decode ``quant_reasoning``: JSON written by :class:`ShadowBook`, or free text from other writers
    (returned as ``{"reasons": [text]}`` with no direction — callers must not guess one)."""
    parsed = _safe_json(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(text, str) and text:
        return {"reasons": [text]}
    return {}
