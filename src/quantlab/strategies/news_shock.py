"""News shock (research-only, disabled by default): unusual news volume plus a large price move.

Hypothesis: when abnormal news flow coincides with a large move, the move continues. Uses news
COUNTS only (no invented sentiment). Direction follows the price move; shorts only if allowed.
"""
from __future__ import annotations

import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.strategies.base import Strategy


class NewsShock(Strategy):
    family = "news_shock"
    description = "Long when news count z-score and a positive one-day return z-score are both extreme (continuation)."
    feature_deps = ("news_count_z", "ret_z_1d", "atr14_pct", "adv20")
    default_params = {"news_count_z": 2.5, "min_abs_return_z": 2.0, "hold_sessions": 10, "stop_atr": 2.5}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        nz, rz = fs.get("news_count_z"), fs.get("ret_z_1d")
        signal = (nz >= float(self.params["news_count_z"])) & (rz >= float(self.params["min_abs_return_z"])) & u
        return (nz * rz).where(signal)
