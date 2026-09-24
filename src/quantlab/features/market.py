"""Market-level features (single "__market__" column), used for regime context and reporting.

All are trailing computations on the benchmark (SPY) or per-session cross-sections, so they are
point-in-time. Breadth uses ``fs.universe`` when provided (else all symbols with a value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, MARKET_COLUMN, FeatureSet
from quantlab.features.price import SQRT_TRADING_DAYS, market_nan, market_series, ret_n, rolling_std, safe_div, sma


def _m(values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({MARKET_COLUMN: values.to_numpy()}, index=values.index)


def _spy_feature(name: str, desc: str, lookback: int, fn) -> None:
    @FEATURES.feature(name, "market", desc, "panel aclose/ret of the market benchmark", PitStatus.PIT,
                      lookback=lookback, market_level=True)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        a = market_series(fs, "aclose")
        if a is None:
            return market_nan(fs)
        return _m(fn(a, market_series(fs, "ret")))


_spy_feature("market_trend_200", "SPY aclose/SMA(aclose,200)-1", 199, lambda a, r: safe_div(a, sma(a, 200)) - 1.0)
_spy_feature("market_mom_60", "SPY 60-session return", 60, lambda a, r: ret_n(a, 60))
_spy_feature("market_vol_20", "SPY std(ret,20)*sqrt(252)", 20, lambda a, r: rolling_std(r, 20) * SQRT_TRADING_DAYS)
_spy_feature("market_drawdown", "SPY aclose/cummax(aclose)-1 (expanding, trailing)", 0,
             lambda a, r: safe_div(a, a.cummax()) - 1.0)


def _breadth(fs: FeatureSet, feature: str) -> pd.DataFrame:
    dist = fs.get(feature)
    valid = dist.notna()
    if fs.universe is not None:
        valid &= fs.universe.reindex_like(dist).fillna(False).astype(bool)
    n = valid.sum(axis=1)
    above = (dist > 0).where(valid, False).sum(axis=1)
    return _m((above / n.where(n > 0)).astype("float64"))


@FEATURES.feature("breadth_50", "market", "fraction of universe members with dist_ma50 > 0", "feature dist_ma50; fs.universe",
                  PitStatus.PIT, lookback=49, market_level=True)
def breadth_50(fs: FeatureSet) -> pd.DataFrame:
    return _breadth(fs, "dist_ma50")


@FEATURES.feature("breadth_200", "market", "fraction of universe members with dist_ma200 > 0",
                  "feature dist_ma200; fs.universe", PitStatus.PIT, lookback=199, market_level=True)
def breadth_200(fs: FeatureSet) -> pd.DataFrame:
    return _breadth(fs, "dist_ma200")


@FEATURES.feature("sector_dispersion_63", "market", "cross-sectional std of sector-ETF 63-session returns",
                  "panel aclose of configured sector ETFs", PitStatus.PIT, lookback=63, market_level=True)
def sector_dispersion_63(fs: FeatureSet) -> pd.DataFrame:
    etfs = [e for e in fs.bundle.sector_etfs if e in fs.panel.symbols]
    if len(etfs) < 2:
        return market_nan(fs)
    r = ret_n(fs.panel.aclose[etfs], 63)
    enough = r.notna().sum(axis=1) >= 2
    return _m(r.std(axis=1, ddof=1).where(enough))
