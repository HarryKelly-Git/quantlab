"""Trend / momentum family (Jegadeesh-Titman style cross-sectional momentum with a trend filter).

Hypothesis: stocks with the strongest intermediate-term returns (skipping the most recent month,
which tends to reverse) keep outperforming for weeks. The trend filter (price above its 50-session
average) avoids buying names whose momentum is already rolling over. Parameters are deliberately
round numbers, not optimized.
"""
from __future__ import annotations

import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.features.relative import xs_rank
from quantlab.strategies.base import Strategy


class MomentumTrend(Strategy):
    family = "trend_momentum"
    description = ("Composite cross-sectional rank of 6-1 momentum and 60-session return; long the top names "
                   "that also trade above their 50-session average. Time exit + ATR stop.")
    feature_deps = ("mom_6_1", "ret_60d", "dist_ma50", "atr14_pct", "adv20")
    default_params = {"min_score_pct": 0.90, "hold_sessions": 40, "stop_atr": 3.0}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        composite = (xs_rank(fs.get("mom_6_1"), u) + xs_rank(fs.get("ret_60d"), u)) / 2.0
        signal = (composite >= float(self.params["min_score_pct"])) & (fs.get("dist_ma50") > 0) & u
        return composite.where(signal)
