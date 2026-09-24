"""Post-earnings-announcement drift measured by the announcement's PRICE REACTION.

Hypothesis (Chan-Jegadeesh-Lakonishok 1996; Brandt et al. 2008): stocks with a large abnormal
return around an earnings release, confirmed by heavy volume, keep drifting in the same direction
for weeks. Analyst-consensus surprises are not available point-in-time for free, so the reaction
itself is the surprise proxy. Event timing comes from SEC 8-K item 2.02 acceptance timestamps.
"""
from __future__ import annotations

import pandas as pd

from quantlab.core.types import Direction
from quantlab.features.base import FeatureSet
from quantlab.strategies.base import Strategy


class PeadEar(Strategy):
    family = "event_drift"
    description = ("Long within a few sessions after an earnings reaction of >= min_abs_ear_z abnormal-return sigmas "
                   "on >= min_rel_volume relative dollar volume (EAR is known at the close of reaction+1).")
    feature_deps = ("ear_z", "event_rel_volume", "days_since_earnings", "atr14_pct", "adv20")
    default_params = {"min_abs_ear_z": 1.5, "min_rel_volume": 1.5, "max_days_after_event": 3,
                      "hold_sessions": 40, "stop_atr": 3.0}
    direction = Direction.LONG

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        z = fs.get("ear_z")
        ds = fs.get("days_since_earnings")
        # ear_z first exists at reaction+1 (days_since == 1); allow a few sessions after that
        fresh = (ds >= 1) & (ds <= int(self.params["max_days_after_event"]))
        strong = (z >= float(self.params["min_abs_ear_z"])) & (fs.get("event_rel_volume") >= float(self.params["min_rel_volume"]))
        return z.where(fresh & strong & u)
