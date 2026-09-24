"""Quality + momentum: momentum leaders that are also profitable, efficient and not over-levered.

Hypothesis: momentum among high-quality companies is more persistent (less crash-prone) than
momentum in general. Fundamentals come from the as-of engine (PIT_CONSERVATIVE); names without
fundamentals never signal (no fundamentals = UNKNOWN quality, not 'average').
"""
from __future__ import annotations

import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.features.relative import xs_rank
from quantlab.strategies.base import Strategy


class QualityMomentum(Strategy):
    family = "quality_momentum"
    description = ("Long names in the top momentum ranks (6-1 momentum) whose quality rank "
                   "(mean of ROE, net margin and low-leverage ranks) is also high.")
    feature_deps = ("mom_6_1", "roe", "ni_margin", "leverage", "atr14_pct", "adv20")
    default_params = {"min_momentum_pct": 0.70, "min_quality_pct": 0.70, "hold_sessions": 60, "stop_atr": 3.5}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        roe, margin, lev = fs.get("roe"), fs.get("ni_margin"), fs.get("leverage")
        has_f = roe.notna() & margin.notna() & lev.notna()
        eligible = u & has_f
        mom_rank = xs_rank(fs.get("mom_6_1"), eligible)
        quality = (xs_rank(roe, eligible) + xs_rank(margin, eligible) + xs_rank(-lev, eligible)) / 3.0
        signal = eligible & (mom_rank >= float(self.params["min_momentum_pct"])) & \
            (quality >= float(self.params["min_quality_pct"]))
        return ((mom_rank + quality) / 2.0).where(signal)
