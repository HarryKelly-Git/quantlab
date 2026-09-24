"""Relative strength vs the market, optionally requiring a strengthening sector.

Hypothesis: stocks outperforming SPY over ~3 months keep outperforming, more so when their sector
is also outperforming. Sector membership is ASSUMED_STATIC (current classification).
"""
from __future__ import annotations

import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.features.relative import xs_rank
from quantlab.strategies.base import Strategy


class RelativeStrength(Strategy):
    family = "relative_strength"
    description = "Long top cross-sectional ranks of 63-session return vs SPY; optionally sector and stock-vs-sector strength."
    feature_deps = ("rs_spy_63", "rs_sector_63", "sector_rs_spy_63", "atr14_pct", "adv20")
    default_params = {"vs_market_min_pct": 0.80, "require_sector_strength": True, "hold_sessions": 30, "stop_atr": 3.0}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        rank = xs_rank(fs.get("rs_spy_63"), u)
        signal = (rank >= float(self.params["vs_market_min_pct"])) & u
        if self.params.get("require_sector_strength", True):
            signal &= (fs.get("sector_rs_spy_63") > 0) & (fs.get("rs_sector_63") > 0)
        return rank.where(signal)
