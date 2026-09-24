"""Persistence helpers for pipeline artefacts (all target append-only audit tables)."""
from __future__ import annotations

from typing import Iterable

from quantlab.core.types import Candidate
from quantlab.db.database import Database, to_json


def persist_candidates(db: Database, candidates: Iterable[Candidate], run_id: str | None, is_synthetic: bool) -> int:
    """Write candidates exactly once (the table is immutable; re-runs of a resumed step are no-ops)."""
    rows = []
    for c in candidates:
        rows.append({
            "candidate_id": c.candidate_id, "run_id": run_id, "as_of_date": c.as_of_date.isoformat(),
            "created_at": c.created_at.isoformat(), "symbol": c.symbol, "strategy_id": c.strategy_id,
            "strategy_version": c.strategy_version, "direction": c.direction.value, "score": float(c.score),
            "features_json": to_json(c.features), "reasons_json": to_json(c.reasons),
            "entry_convention": c.plan.entry, "entry_ref_price": c.plan.entry_ref_price,
            "stop_price": c.plan.stop_price, "target_price": c.plan.target_price,
            "holding_sessions": c.plan.holding_sessions, "invalidation": c.plan.invalidation,
            "risk_json": to_json(c.risk), "pit_status": c.pit_status.value, "is_synthetic": int(bool(is_synthetic)),
        })
    return db.insert_many("candidates", rows, or_ignore=True)


def candidates_for_run(db: Database, run_id: str) -> list[dict]:
    return db.fetchall("SELECT * FROM candidates WHERE run_id=? ORDER BY strategy_id, score DESC", (run_id,))


def load_candidates(db: Database, run_id: str) -> list[Candidate]:
    """Rebuild the Candidate objects persisted by a run (used when resuming a pipeline run, so a
    resumed run never generates a second set of candidate ids for the same session)."""
    from datetime import date, datetime

    from quantlab.core.types import Direction, PitStatus, TradePlan
    from quantlab.db.database import from_json
    out = []
    for r in candidates_for_run(db, run_id):
        out.append(Candidate(
            symbol=r["symbol"], as_of_date=date.fromisoformat(r["as_of_date"]), strategy_id=r["strategy_id"],
            strategy_version=r["strategy_version"], score=r["score"], direction=Direction(r["direction"]),
            features=from_json(r["features_json"], {}),
            plan=TradePlan(entry=r["entry_convention"], entry_ref_price=r["entry_ref_price"], stop_price=r["stop_price"],
                           target_price=r["target_price"], holding_sessions=r["holding_sessions"] or 20,
                           invalidation=r["invalidation"] or ""),
            risk=from_json(r["risk_json"], {}), pit_status=PitStatus(r["pit_status"]),
            reasons=from_json(r["reasons_json"], []), candidate_id=r["candidate_id"],
            created_at=datetime.fromisoformat(r["created_at"])))
    return out
