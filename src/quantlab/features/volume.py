"""Volume-group features (ARCHITECTURE.md section 4).

All of them use ``dollar_volume`` = RAW close x RAW share volume. Dollar volume is invariant to
splits (shares double while the price halves), so neither a past nor a FUTURE split can distort
these features, and no adjusted share volume (which would need future split factors) is needed.
Windows are trailing and require a full window; division by zero yields NaN.
"""
from __future__ import annotations

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import safe_div

_SRC_DV = "panel dollar_volume (raw close x raw volume, USD)"


def _register_adv(n: int) -> None:
    @FEATURES.feature(f"adv{n}", "volume", f"median(dollar_volume, {n}) in raw USD", _SRC_DV, PitStatus.PIT,
                      lookback=n - 1)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        return fs.panel.dollar_volume.rolling(n, min_periods=n).median()


for _n in (20, 60):
    _register_adv(_n)


@FEATURES.feature("rel_volume_1d", "volume", "dv / mean(dv.shift(1),20) (dollar-based, split-invariant)",
                  _SRC_DV, PitStatus.PIT, lookback=20)
def rel_volume_1d(fs: FeatureSet) -> pd.DataFrame:
    dv = fs.panel.dollar_volume
    return safe_div(dv, dv.shift(1).rolling(20, min_periods=20).mean())


@FEATURES.feature("rel_volume_5d", "volume", "mean(dv,5)/mean(dv.shift(5),60)", _SRC_DV, PitStatus.PIT, lookback=64)
def rel_volume_5d(fs: FeatureSet) -> pd.DataFrame:
    dv = fs.panel.dollar_volume
    return safe_div(dv.rolling(5, min_periods=5).mean(), dv.shift(5).rolling(60, min_periods=60).mean())


@FEATURES.feature("volume_trend_20_60", "volume", "mean(dv,20)/mean(dv,60)", _SRC_DV, PitStatus.PIT, lookback=59)
def volume_trend_20_60(fs: FeatureSet) -> pd.DataFrame:
    dv = fs.panel.dollar_volume
    return safe_div(dv.rolling(20, min_periods=20).mean(), dv.rolling(60, min_periods=60).mean())
