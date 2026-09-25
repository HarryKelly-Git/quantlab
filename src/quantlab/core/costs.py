"""Transaction-cost model shared by the backtester, the simulated paper broker, the EV engine and
the shadow/counterfactual outcome evaluators — so every layer charges identical costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.config import Config
from quantlab.core.types import Side


@dataclass(frozen=True)
class CostModel:
    half_spread_tiers: tuple[tuple[float, float], ...]   # (min median dollar volume, half-spread bps), descending
    slippage_bps: float
    commission_per_share: float = 0.0
    commission_min_per_order: float = 0.0
    delisting_return: float = -0.30
    # A held symbol counts as DELISTED only after this many consecutive sessions without a bar while
    # the market keeps trading; the exit is booked on that session. Point-in-time: never decided by
    # looking ahead for bars that may or may not come back. Shared with the paper SimBroker.
    delisting_missing_sessions: int = 5

    @classmethod
    def from_config(cls, config: Config) -> "CostModel":
        c = config.section("costs")
        tiers = tuple(sorted(((float(a), float(b)) for a, b in c["half_spread_bps_tiers"]), key=lambda x: -x[0]))
        return cls(tiers, float(c["slippage_bps"]), float(c.get("commission_per_share", 0.0)),
                   float(c.get("commission_min_per_order", 0.0)), float(c.get("delisting_return", -0.30)),
                   int(config.get("execution.delisting_missing_sessions", 5)))

    def half_spread_bps(self, median_dollar_volume: float | None) -> float:
        """Unknown liquidity is charged the WORST tier (conservative)."""
        if median_dollar_volume is None or not np.isfinite(median_dollar_volume):
            return self.half_spread_tiers[-1][1]
        for threshold, bps in self.half_spread_tiers:
            if median_dollar_volume >= threshold:
                return bps
        return self.half_spread_tiers[-1][1]

    def one_way_cost_frac(self, median_dollar_volume: float | None) -> float:
        return (self.half_spread_bps(median_dollar_volume) + self.slippage_bps) / 1e4

    def round_trip_cost_frac(self, median_dollar_volume: float | None) -> float:
        return 2 * self.one_way_cost_frac(median_dollar_volume)

    def fill_price(self, side: Side, ref_price: float, median_dollar_volume: float | None) -> float:
        c = self.one_way_cost_frac(median_dollar_volume)
        return ref_price * (1 + c) if side is Side.BUY else ref_price * (1 - c)

    def commission(self, qty: float) -> float:
        if qty == 0:
            return 0.0
        return max(abs(qty) * self.commission_per_share, self.commission_min_per_order)
