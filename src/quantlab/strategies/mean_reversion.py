"""Short-term mean reversion (Jegadeesh 1990 / Lehmann 1990 style) inside a long-term uptrend.

Hypothesis: a sharp multi-day drop (in units of the stock's own trailing volatility) in a stock
that is still above its 200-session average is often an overreaction that partly reverses within
days. Ordinary vs extreme shocks are both recorded so research can compare them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import TradePlan
from quantlab.features.base import FeatureSet
from quantlab.strategies.base import Strategy


class MeanReversion(Strategy):
    family = "mean_reversion"
    description = ("Long after a >= |shock_z| sigma 3-session drop while price stays above its 200-session average; "
                   "target = back to the 20-session average; short hold.")
    feature_deps = ("ret_z_3d", "dist_ma200", "dist_ma20", "atr14_pct", "adv20")
    default_params = {"shock_z": -2.0, "extreme_z": -3.5, "hold_sessions": 5, "stop_atr": 2.5}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        z = fs.get("ret_z_3d")
        signal = (z <= float(self.params["shock_z"])) & (fs.get("dist_ma200") > 0) & u
        return (-z).where(signal)

    def plan(self, fs: FeatureSet, symbol: str, as_of) -> TradePlan:
        base = super().plan(fs, symbol, as_of)
        d = pd.Timestamp(as_of)
        dist20 = float(fs.get("dist_ma20").at[d, symbol])
        # dist_ma20 = close/MA20 - 1 on a scale-free series => MA20 in RAW terms at D = close / (1 + dist20)
        if np.isfinite(dist20) and dist20 < 0 and base.entry_ref_price:
            base.target_price = base.entry_ref_price / (1.0 + dist20)
        z = float(fs.get("ret_z_3d").at[d, symbol])
        kind = "extreme" if np.isfinite(z) and z <= float(self.params["extreme_z"]) else "ordinary"
        base.invalidation = f"{kind} shock; close below ATR stop, or {self.holding_sessions} sessions elapsed"
        return base
