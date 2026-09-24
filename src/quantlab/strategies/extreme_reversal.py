"""Extreme one-day moves: do they reverse or continue?

Hypothesis (to be tested, not assumed): very large one-day drops on heavy volume overshoot and
partly reverse. Outcomes are recorded either way; the shadow book shows the continuation cases.
"""
from __future__ import annotations

import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.strategies.base import Strategy


class ExtremeReversal(Strategy):
    family = "extreme_reversal"
    description = "Long after a one-day drop beyond extreme_return_z trailing-vol units on >= min_rel_volume dollar volume."
    feature_deps = ("ret_z_1d", "rel_volume_1d", "atr14_pct", "adv20")
    default_params = {"extreme_return_z": -4.0, "min_rel_volume": 2.0, "hold_sessions": 5, "stop_atr": 3.0}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        z = fs.get("ret_z_1d")
        heavy = fs.get("rel_volume_1d") >= float(self.params["min_rel_volume"])
        signal = (z <= float(self.params["extreme_return_z"])) & heavy & u
        return (-z).where(signal)
