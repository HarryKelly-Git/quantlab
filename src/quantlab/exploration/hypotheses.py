"""Exploration -> validation research workflow (bookkeeping only; nothing here trades or promotes).

    EXPLORATION_OBSERVED -> HISTORICAL_PIT_TEST -> WALK_FORWARD -> LOCKED_HOLDOUT
      -> PROSPECTIVE_PAPER -> STRICT_ELIGIBLE        (or REJECTED from any stage)

Rules: one stage at a time; a named human actor (``human:<name>``); an evidence reference for every
move; leaving EXPLORATION_OBSERVED needs at least ``exploration.min_observations_to_propose``
exploratory outcomes. No code path advances a hypothesis automatically, and STRICT_ELIGIBLE changes
no strategy status: strategy promotion stays the separate, unchanged process.
"""
from __future__ import annotations

from typing import Any

from quantlab.core.types import new_id
from quantlab.db.database import to_json, utcnow_iso

STAGES = ("EXPLORATION_OBSERVED", "HISTORICAL_PIT_TEST", "WALK_FORWARD", "LOCKED_HOLDOUT", "PROSPECTIVE_PAPER",
          "STRICT_ELIGIBLE")


class HypothesisError(ValueError):
    pass


def _human(actor: str) -> str:
    if not str(actor or "").startswith("human:") or len(actor) <= len("human:"):
        raise HypothesisError("a hypothesis can only be created or moved by a named human actor (human:<name>)")
    return actor


def create_hypothesis(db, name: str, definition: dict[str, Any], actor: str, evidence: str) -> str:
    _human(actor)
    if not evidence:
        raise HypothesisError("evidence is required (e.g. the exploratory outcome summary it came from)")
    hid = new_id("hyp")
    now = utcnow_iso()
    with db.transaction():
        db.insert("research_hypotheses", {"hypothesis_id": hid, "name": name, "definition_json": to_json(definition),
                                          "created_by": actor, "created_at": now})
        db.insert("hypothesis_events", {"hypothesis_id": hid, "from_stage": None, "to_stage": STAGES[0], "actor": actor,
                                        "evidence": evidence, "created_at": now})
    return hid


def stage_of(db, hid: str) -> str | None:
    r = db.fetchone("SELECT to_stage FROM hypothesis_events WHERE hypothesis_id=? ORDER BY id DESC LIMIT 1", (hid,))
    return r["to_stage"] if r else None


def advance(db, hid: str, to_stage: str, actor: str, evidence: str, *, n_observations: int | None = None,
            min_observations: int = 30) -> str:
    _human(actor)
    cur = stage_of(db, hid)
    if cur is None:
        raise HypothesisError(f"unknown hypothesis {hid}")
    if cur in ("REJECTED", STAGES[-1]):
        raise HypothesisError(f"hypothesis is already {cur}")
    if not evidence:
        raise HypothesisError("every stage change needs an evidence reference")
    if to_stage != "REJECTED":
        if to_stage not in STAGES:
            raise HypothesisError(f"unknown stage {to_stage}")
        if STAGES.index(to_stage) != STAGES.index(cur) + 1:
            raise HypothesisError(f"stages cannot be skipped: {cur} -> {STAGES[STAGES.index(cur) + 1]} is next, not {to_stage}")
        if cur == STAGES[0] and (n_observations is None or n_observations < min_observations):
            raise HypothesisError(f"needs >= {min_observations} exploratory outcomes before a historical test "
                                  f"(has {n_observations or 0}): a few trades prove nothing")
    db.insert("hypothesis_events", {"hypothesis_id": hid, "from_stage": cur, "to_stage": to_stage, "actor": actor,
                                    "evidence": evidence, "created_at": utcnow_iso()})
    return to_stage


def hypotheses(db) -> list[dict[str, Any]]:
    out = []
    for h in db.fetchall("SELECT * FROM research_hypotheses ORDER BY created_at"):
        evs = db.fetchall("SELECT from_stage, to_stage, actor, evidence, created_at FROM hypothesis_events "
                          "WHERE hypothesis_id=? ORDER BY id", (h["hypothesis_id"],))
        out.append({**dict(h), "stage": evs[-1]["to_stage"] if evs else None, "history": [dict(e) for e in evs]})
    return out


__all__ = ["HypothesisError", "STAGES", "advance", "create_hypothesis", "hypotheses", "stage_of"]
