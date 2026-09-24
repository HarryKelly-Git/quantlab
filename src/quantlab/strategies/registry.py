"""Strategy registry: config -> strategy instances, and persistence of strategy definitions.

Status/stage transitions are NOT made here (that is the research promotion module's job); this
module only inserts new (strategy_id, version) rows with default status SHADOW / stage RESEARCH.
"""
from __future__ import annotations

from quantlab.config import Config
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.strategies.base import Strategy
from quantlab.strategies.breakout import Breakout
from quantlab.strategies.extreme_reversal import ExtremeReversal
from quantlab.strategies.mean_reversion import MeanReversion
from quantlab.strategies.momentum_trend import MomentumTrend
from quantlab.strategies.relative_strength import RelativeStrength
from quantlab.strategies.sector_rotation import SectorRotation

STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    "momentum_trend": MomentumTrend,
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
    "relative_strength": RelativeStrength,
    "extreme_reversal": ExtremeReversal,
    "sector_rotation": SectorRotation,
}


def build_strategies(config: Config, include_disabled: bool = False, only: list[str] | None = None) -> list[Strategy]:
    """Instantiate configured strategies that have an implementation.

    Configured-but-unimplemented strategies are skipped (reported by :func:`unimplemented`).
    """
    out: list[Strategy] = []
    for sid, spec in config.section("strategies").items():
        if only is not None and sid not in only:
            continue
        if sid not in STRATEGY_CLASSES:
            continue
        if not (spec.get("enabled", False) or include_disabled or only is not None):
            continue
        out.append(STRATEGY_CLASSES[sid](sid, str(spec.get("version", "0.0.0")), spec.get("params", {})))
    return out


def unimplemented(config: Config) -> list[str]:
    return [sid for sid in config.section("strategies") if sid not in STRATEGY_CLASSES]


def register_strategies(db: Database, strategies: list[Strategy]) -> int:
    """Insert unseen (strategy_id, version) rows; never updates or deletes existing ones."""
    now = utcnow_iso()
    n = 0
    for s in strategies:
        if db.fetchone("SELECT 1 FROM strategies WHERE strategy_id=? AND version=?", (s.strategy_id, s.version)):
            continue
        db.insert("strategies", {
            "strategy_id": s.strategy_id, "version": s.version, "family": s.family, "description": s.description,
            "params_json": to_json(s.params), "feature_deps_json": to_json(list(s.feature_deps)),
            "status": "SHADOW", "stage": "RESEARCH", "created_at": now, "updated_at": now,
        })
        n += 1
    return n
