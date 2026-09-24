"""Strategy framework. A strategy turns features into (a) a vectorized score panel used identically
by the backtester and the daily pipeline, and (b) a TradePlan for each signalled symbol.

Contract:
  * ``score(fs, universe)`` -> DataFrame (sessions x symbols). NaN = no signal. Larger = stronger.
    Must be truncation-invariant (tests enforce it) and must return NaN outside ``universe``.
  * ``plan(fs, symbol, as_of)`` -> TradePlan using only information at ``as_of`` (raw price levels).
  * Parameters come from config (``strategies.<id>.params``); never hard-code tunables.
  * ``allowed_pit``: strategies refuse features whose PIT status is outside this set.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.types import (
    HISTORICAL_RESEARCH_PIT,
    Candidate,
    Direction,
    PitStatus,
    TradePlan,
)
from quantlab.features.base import FeatureSet


class StrategyError(RuntimeError):
    pass


class Strategy(ABC):
    family: str = "abstract"
    description: str = ""
    feature_deps: tuple[str, ...] = ()
    direction: Direction = Direction.LONG
    allowed_pit: frozenset[PitStatus] = HISTORICAL_RESEARCH_PIT
    default_params: dict[str, Any] = {}

    def __init__(self, strategy_id: str, version: str, params: dict[str, Any] | None = None):
        self.strategy_id = strategy_id
        self.version = version
        self.params = {**self.default_params, **(params or {})}

    # -- required -------------------------------------------------------------------------------
    @abstractmethod
    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        ...

    # -- defaults (override when the family needs something specific) ------------------------------
    @property
    def holding_sessions(self) -> int:
        return int(self.params.get("hold_sessions", 20))

    def check_pit(self, fs: FeatureSet) -> PitStatus:
        status = fs.pit_status(self.feature_deps) if self.feature_deps else PitStatus.PIT
        if status not in self.allowed_pit:
            raise StrategyError(f"{self.strategy_id}: feature PIT status {status.value} not allowed")
        return status

    def plan(self, fs: FeatureSet, symbol: str, as_of) -> TradePlan:
        """Default: ATR-based protective stop in RAW price terms, time exit after hold_sessions."""
        d = pd.Timestamp(as_of)
        close = float(fs.panel.close.at[d, symbol])
        atr_pct = float(fs.get("atr14_pct").at[d, symbol]) if "atr14_pct" in fs.registry else np.nan
        stop = None
        k = float(self.params.get("stop_atr", 3.0))
        if np.isfinite(atr_pct) and atr_pct > 0:
            dist = k * atr_pct * close
            stop = close - dist if self.direction is Direction.LONG else close + dist
        return TradePlan(
            entry="next_open",
            entry_ref_price=close,
            stop_price=stop,
            target_price=None,
            holding_sessions=self.holding_sessions,
            invalidation=f"close beyond {k:g}x ATR(14) stop, or {self.holding_sessions} sessions elapsed",
        )

    def reasons(self, fs: FeatureSet, symbol: str, as_of) -> list[str]:
        """Human-readable quantitative reasons (FACT/MODEL_OUTPUT values of the feature deps)."""
        d = pd.Timestamp(as_of)
        out = []
        for n in self.feature_deps:
            spec = fs.registry.spec(n)
            if spec.market_level:
                v = fs.market(n).get(d, np.nan)
            else:
                v = fs.get(n).at[d, symbol]
            out.append(f"{n}={v:.4g}" if np.isfinite(v) else f"{n}=UNKNOWN")
        return out

    def candidates(self, fs: FeatureSet, universe: pd.DataFrame, as_of,
                   scores: pd.DataFrame | None = None) -> list[Candidate]:
        d = pd.Timestamp(as_of)
        pit = self.check_pit(fs)
        sc = scores if scores is not None else self.score(fs, universe)
        if d not in sc.index:
            return []
        row = sc.loc[d].dropna()
        cands = []
        for sym, s in row.sort_values(ascending=False).items():
            feats = {}
            for n in self.feature_deps:
                spec = fs.registry.spec(n)
                v = fs.market(n).get(d, np.nan) if spec.market_level else fs.get(n).at[d, sym]
                feats[n] = float(v) if np.isfinite(v) else None
            atr = fs.get("atr14_pct").at[d, sym] if "atr14_pct" in fs.registry else np.nan
            adv = fs.get("adv20").at[d, sym] if "adv20" in fs.registry else np.nan
            cands.append(Candidate(
                symbol=sym, as_of_date=d.date(), strategy_id=self.strategy_id, strategy_version=self.version,
                score=float(s), direction=self.direction, features=feats, plan=self.plan(fs, sym, d),
                risk={"atr14_pct": float(atr) if np.isfinite(atr) else None,
                      "adv20": float(adv) if np.isfinite(adv) else None},
                pit_status=pit, reasons=self.reasons(fs, sym, d),
            ))
        return cands

    def describe(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "version": self.version, "family": self.family,
                "description": self.description, "params": self.params, "feature_deps": list(self.feature_deps),
                "holding_sessions": self.holding_sessions, "direction": self.direction.value}
