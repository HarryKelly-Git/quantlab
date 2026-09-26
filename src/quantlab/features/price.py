"""Price-group features (ARCHITECTURE.md section 4) plus small helpers shared by the other groups.

Point-in-time reasoning:
  * Every computation is a trailing window, a positive shift or an expanding max over the
    (possibly truncated) panel, so the value at session D depends only on rows <= D.
  * RATIOS / RETURNS use ``ret`` or the tri-scaled ``aopen/ahigh/alow/aclose`` fields. Their scale
    is arbitrary but constant through time, so ratios are exactly what a back-adjusted-as-of-D
    series would give, without using any corporate action after D. No price LEVEL is used here.
  * Rolling windows require a FULL window (``min_periods == window``): a gap in a symbol's bars
    yields NaN (UNKNOWN) rather than a value silently computed on fewer observations.
  * Division by zero never produces +-inf; it produces NaN.

The helpers (``memo``, ``safe_div``, ``ret_n``, ``sma``, ``full_like_nan``, ``market_series``) live
here because the feature package's shared modules are foundation-owned; other group modules
import them from this module.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, MARKET_COLUMN, FeatureSet

TRADING_DAYS = 252
SQRT_TRADING_DAYS = float(np.sqrt(TRADING_DAYS))

_SRC_A = "panel aopen/ahigh/alow/aclose (tri-scaled)"
_SRC_RET = "panel ret (daily total return)"


# --------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------
def memo(fs: FeatureSet, key: str, fn: Callable[[], object]) -> object:
    """Cache an intermediate result on one FeatureSet (lives exactly as long as the FeatureSet,
    so a truncated bundle always gets its own intermediates)."""
    store = fs.__dict__.setdefault("_quantlab_memo", {})
    if key not in store:
        store[key] = fn()
    return store[key]


def safe_div(num: pd.DataFrame | pd.Series, den: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Element-wise num/den with +-inf (division by zero) mapped to NaN."""
    out = num / den
    return out.where(np.isfinite(out))


def sma(x: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """Trailing simple mean over a full window of n sessions."""
    return x.rolling(n, min_periods=n).mean()


def ret_n(aclose: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """n-session total return from tri-scaled closes (scale-free)."""
    return safe_div(aclose, aclose.shift(n)) - 1.0


def active_from(fs: FeatureSet, available_at: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` on sessions at/after the first session the SOURCE had any item available; NaN
    (UNKNOWN) before. Point in time: whether a source exists at D never depends on rows after D,
    so a count of 0 can only appear once the source is known to be delivering."""
    t = pd.to_datetime(available_at, utc=True).dropna()
    if t.empty:
        return frame.where(pd.DataFrame(False, index=frame.index, columns=frame.columns))
    first = fs.bundle.calendar.first_usable_session(t.min())
    if first is None:
        return frame.where(pd.DataFrame(False, index=frame.index, columns=frame.columns))
    return frame.where(pd.Series(frame.index >= first, index=frame.index), axis=0)


def full_like_nan(fs: FeatureSet) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=fs.panel.dates, columns=fs.panel.symbols)


def market_nan(fs: FeatureSet) -> pd.DataFrame:
    return pd.DataFrame({MARKET_COLUMN: np.nan}, index=fs.panel.dates)


def market_series(fs: FeatureSet, field: str) -> pd.Series | None:
    """A panel field for the market benchmark (SPY), or None if the benchmark is not in the panel
    (callers then return NaN = UNKNOWN; they never substitute another series)."""
    sym = fs.bundle.market_symbol
    if sym not in fs.panel.symbols:
        return None
    return fs.panel.fields[field][sym]


def rolling_std(x: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    """Trailing sample std (ddof=1) over a full window of n sessions."""
    return x.rolling(n, min_periods=n).std()


# --------------------------------------------------------------------------------------------
# returns & momentum
# --------------------------------------------------------------------------------------------
@FEATURES.feature("ret_1d", "price", "ret: daily total return (split/dividend aware, from raw data + that day's actions)",
                  _SRC_RET, PitStatus.PIT, lookback=1)
def ret_1d(fs: FeatureSet) -> pd.DataFrame:
    return fs.panel.ret


def _register_ret(n: int) -> None:
    @FEATURES.feature(f"ret_{n}d", "price", f"aclose/aclose.shift({n})-1", _SRC_A, PitStatus.PIT, lookback=n)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        return ret_n(fs.panel.aclose, n)


for _n in (5, 20, 60, 120, 252):
    _register_ret(_n)


@FEATURES.feature("mom_12_1", "price", "aclose.shift(21)/aclose.shift(252)-1 (12-1 momentum, skips the latest month)",
                  _SRC_A, PitStatus.PIT, lookback=252)
def mom_12_1(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    return safe_div(a.shift(21), a.shift(252)) - 1.0


@FEATURES.feature("mom_6_1", "price", "aclose.shift(21)/aclose.shift(126)-1 (6-1 momentum, skips the latest month)",
                  _SRC_A, PitStatus.PIT, lookback=126)
def mom_6_1(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    return safe_div(a.shift(21), a.shift(126)) - 1.0


# --------------------------------------------------------------------------------------------
# trend / moving averages
# --------------------------------------------------------------------------------------------
def _register_dist_ma(n: int) -> None:
    @FEATURES.feature(f"dist_ma{n}", "price", f"aclose/SMA(aclose,{n})-1", _SRC_A, PitStatus.PIT, lookback=n - 1)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        a = fs.panel.aclose
        return safe_div(a, sma(a, n)) - 1.0


for _n in (20, 50, 200):
    _register_dist_ma(_n)


@FEATURES.feature("ma50_over_ma200", "price", "SMA(aclose,50)/SMA(aclose,200)-1", _SRC_A, PitStatus.PIT, lookback=199)
def ma50_over_ma200(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    return safe_div(sma(a, 50), sma(a, 200)) - 1.0


def true_range(fs: FeatureSet) -> pd.DataFrame:
    """TR from tri-scaled fields: max(ahigh-alow, |ahigh-prev aclose|, |alow-prev aclose|).
    On a symbol's first bar (no previous close) TR = ahigh - alow."""
    p = fs.panel
    prev = p.aclose.shift(1)
    hl = p.ahigh - p.alow
    hc = (p.ahigh - prev).abs()
    lc = (p.alow - prev).abs()
    tr = np.fmax(np.fmax(hl, hc), lc)       # fmax ignores a NaN operand (missing previous close)
    return tr.where(hl.notna())


@FEATURES.feature("atr14_pct", "price", "SMA(TR,14)/aclose with TR from a-fields (TR uses prev aclose)",
                  _SRC_A, PitStatus.PIT, lookback=14)
def atr14_pct(fs: FeatureSet) -> pd.DataFrame:
    return safe_div(sma(true_range(fs), 14), fs.panel.aclose)


# --------------------------------------------------------------------------------------------
# volatility
# --------------------------------------------------------------------------------------------
def _register_vol(n: int) -> None:
    @FEATURES.feature(f"vol_{n}d", "price", f"std(ret,{n})*sqrt(252) (annualized, sample std)", _SRC_RET,
                      PitStatus.PIT, lookback=n)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        return rolling_std(fs.panel.ret, n) * SQRT_TRADING_DAYS


for _n in (20, 60):
    _register_vol(_n)


@FEATURES.feature("vol_ratio_20_60", "price", "vol_20d/vol_60d", _SRC_RET, PitStatus.PIT, lookback=60)
def vol_ratio_20_60(fs: FeatureSet) -> pd.DataFrame:
    return safe_div(fs.get("vol_20d"), fs.get("vol_60d"))


# --------------------------------------------------------------------------------------------
# range position / drawdown / breakout
# --------------------------------------------------------------------------------------------
@FEATURES.feature("dist_high_52w", "price", "aclose/max(ahigh,252)-1 (<= 0)", _SRC_A, PitStatus.PIT, lookback=251)
def dist_high_52w(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    return safe_div(p.aclose, p.ahigh.rolling(252, min_periods=252).max()) - 1.0


@FEATURES.feature("dist_low_52w", "price", "aclose/min(alow,252)-1 (>= 0)", _SRC_A, PitStatus.PIT, lookback=251)
def dist_low_52w(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    return safe_div(p.aclose, p.alow.rolling(252, min_periods=252).min()) - 1.0


@FEATURES.feature("drawdown_252d", "price", "aclose/max(aclose,252)-1", _SRC_A, PitStatus.PIT, lookback=251)
def drawdown_252d(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    return safe_div(a, a.rolling(252, min_periods=252).max()) - 1.0


@FEATURES.feature("breakout_55", "price", "aclose/max(ahigh.shift(1),55)-1 (> 0 = new 55-session high)",
                  _SRC_A, PitStatus.PIT, lookback=55)
def breakout_55(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    return safe_div(p.aclose, p.ahigh.shift(1).rolling(55, min_periods=55).max()) - 1.0


@FEATURES.feature("range_contraction_20_60", "price",
                  "(max(ahigh,20)-min(alow,20))/(max(ahigh,60)-min(alow,60))", _SRC_A, PitStatus.PIT, lookback=59)
def range_contraction_20_60(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    r20 = p.ahigh.rolling(20, min_periods=20).max() - p.alow.rolling(20, min_periods=20).min()
    r60 = p.ahigh.rolling(60, min_periods=60).max() - p.alow.rolling(60, min_periods=60).min()
    return safe_div(r20, r60)


# --------------------------------------------------------------------------------------------
# shocks / gaps
# --------------------------------------------------------------------------------------------
@FEATURES.feature("ret_z_1d", "price", "ret / std(ret,20).shift(1): today's move in units of trailing daily vol",
                  _SRC_RET, PitStatus.PIT, lookback=21)
def ret_z_1d(fs: FeatureSet) -> pd.DataFrame:
    r = fs.panel.ret
    return safe_div(r, rolling_std(r, 20).shift(1))


@FEATURES.feature("ret_z_3d", "price", "ret_3d / (std(ret,20).shift(3)*sqrt(3)), ret_3d = aclose/aclose.shift(3)-1",
                  _SRC_A + "; " + _SRC_RET, PitStatus.PIT, lookback=23)
def ret_z_3d(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    return safe_div(ret_n(p.aclose, 3), rolling_std(p.ret, 20).shift(3) * np.sqrt(3.0))


@FEATURES.feature("gap_1d", "price", "aopen/aclose.shift(1)-1 (overnight gap)", _SRC_A, PitStatus.PIT, lookback=1)
def gap_1d(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    return safe_div(p.aopen, p.aclose.shift(1)) - 1.0
