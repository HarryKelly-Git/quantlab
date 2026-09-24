"""Breakout after range contraction with volume expansion.

Hypothesis: a close above the prior 55-session high, after the trading range had contracted and
on expanding dollar volume, marks new demand. False breakouts are expected; they are studied from
recorded outcomes (arena / shadow book), never filtered with hindsight.
"""
from __future__ import annotations

import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.strategies.base import Strategy


class Breakout(Strategy):
    family = "breakout"
    description = ("Long a close above the prior 55-session high when the previous session's 20/60 range ratio was "
                   "contracted and today's dollar volume is expanded.")
    feature_deps = ("breakout_55", "range_contraction_20_60", "rel_volume_1d", "atr14_pct", "adv20")
    default_params = {"contraction_ratio": 0.75, "volume_mult": 1.5, "hold_sessions": 20, "stop_atr": 2.0}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        u = universe.reindex(index=fs.panel.dates, columns=fs.panel.symbols).fillna(False).astype(bool)
        brk = fs.get("breakout_55")
        contracted_before = fs.get("range_contraction_20_60").shift(1) <= float(self.params["contraction_ratio"])
        relv = fs.get("rel_volume_1d")
        signal = (brk > 0) & contracted_before & (relv >= float(self.params["volume_mult"])) & u
        return (relv * (1.0 + brk)).where(signal)
