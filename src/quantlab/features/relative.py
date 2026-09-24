"""Relative-group features (ARCHITECTURE.md section 4): stock vs market, stock vs sector, sector vs
market, cross-sectional ranks, beta and correlation.

Point-in-time reasoning:
  * Relative returns combine trailing returns from the same session only.
  * Cross-sectional ranks are computed per session over the symbols eligible THAT session
    (``fs.universe`` when given, else every symbol with a value). The rank at D never looks at
    another date.
  * Sector membership comes from :func:`quantlab.sectors.sector_map`, a CURRENT snapshot, so the
    sector features are ASSUMED_STATIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import full_like_nan, market_series, memo, ret_n, safe_div
from quantlab.sectors import sector_map

_SRC_A = "panel aclose (tri-scaled) of stock and benchmark"


def _broadcast(series: pd.Series | None, fs: FeatureSet) -> pd.DataFrame:
    """Repeat a per-date benchmark series across all symbol columns."""
    if series is None:
        return full_like_nan(fs)
    arr = np.repeat(series.to_numpy()[:, None], len(fs.panel.symbols), axis=1)
    return pd.DataFrame(arr, index=fs.panel.dates, columns=fs.panel.symbols)


def xs_rank(values: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    """Per-session percentile rank in (0, 1] among eligible symbols; NaN when ineligible."""
    v = values if mask is None else values.where(mask.reindex_like(values).fillna(False).astype(bool))
    return v.rank(axis=1, pct=True, method="average")


def _sector_etf_of(fs: FeatureSet) -> dict[str, str | None]:
    return memo(fs, "sector_map", lambda: sector_map(fs.bundle))  # type: ignore[return-value]


def _sector_frame(fs: FeatureSet, field_fn) -> pd.DataFrame:
    """For each stock column, the value of ``field_fn(etf_symbol)`` for its sector ETF."""
    mapping = _sector_etf_of(fs)
    out = full_like_nan(fs)
    cache: dict[str, pd.Series] = {}
    for sym in fs.panel.symbols:
        etf = mapping.get(sym)
        if not etf or etf not in fs.panel.symbols or etf == sym:
            continue
        if etf not in cache:
            cache[etf] = field_fn(etf)
        out[sym] = cache[etf].to_numpy()
    return out


def _register_rs_spy(n: int) -> None:
    @FEATURES.feature(f"rs_spy_{n}", "relative", f"ret_{n}(stock) - ret_{n}(SPY)", _SRC_A, PitStatus.PIT, lookback=n)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        m = market_series(fs, "aclose")
        return ret_n(fs.panel.aclose, n) - _broadcast(None if m is None else ret_n(m, n), fs)


for _n in (20, 63, 126):
    _register_rs_spy(_n)


@FEATURES.feature("rs_sector_63", "relative", "ret_63(stock) - ret_63(sector ETF); sector map is ASSUMED_STATIC",
                  _SRC_A + "; quantlab.sectors.sector_map", PitStatus.ASSUMED_STATIC, lookback=63)
def rs_sector_63(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    return ret_n(a, 63) - _sector_frame(fs, lambda etf: ret_n(a[etf], 63))


@FEATURES.feature("sector_rs_spy_63", "relative", "ret_63(sector ETF) - ret_63(SPY), broadcast to member stocks",
                  _SRC_A + "; quantlab.sectors.sector_map", PitStatus.ASSUMED_STATIC, lookback=63)
def sector_rs_spy_63(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    m = market_series(fs, "aclose")
    if m is None:
        return full_like_nan(fs)
    spy = ret_n(m, 63)
    return _sector_frame(fs, lambda etf: ret_n(a[etf], 63) - spy)


@FEATURES.feature("xs_rank_ret_63", "relative", "per-session percentile rank of 63-session return among universe members",
                  _SRC_A + "; fs.universe", PitStatus.PIT, lookback=63)
def xs_rank_ret_63(fs: FeatureSet) -> pd.DataFrame:
    return xs_rank(ret_n(fs.panel.aclose, 63), fs.universe)


@FEATURES.feature("xs_rank_ret_126", "relative", "per-session percentile rank of 126-session return among universe members",
                  _SRC_A + "; fs.universe", PitStatus.PIT, lookback=126)
def xs_rank_ret_126(fs: FeatureSet) -> pd.DataFrame:
    return xs_rank(ret_n(fs.panel.aclose, 126), fs.universe)


@FEATURES.feature("xs_rank_mom_12_1", "relative", "per-session percentile rank of mom_12_1 among universe members",
                  "feature mom_12_1; fs.universe", PitStatus.PIT, lookback=252)
def xs_rank_mom_12_1(fs: FeatureSet) -> pd.DataFrame:
    return xs_rank(fs.get("mom_12_1"), fs.universe)


def _rolling_cov_var(fs: FeatureSet, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    m = market_series(fs, "ret")
    if m is None:
        return None
    r = fs.panel.ret
    mk = _broadcast(m, fs).where(r.notna())
    rr = r.where(mk.notna())
    mean_r = rr.rolling(n, min_periods=n).mean()
    mean_m = mk.rolling(n, min_periods=n).mean()
    cov = (rr * mk).rolling(n, min_periods=n).mean() - mean_r * mean_m
    var_m = (mk * mk).rolling(n, min_periods=n).mean() - mean_m ** 2
    var_r = (rr * rr).rolling(n, min_periods=n).mean() - mean_r ** 2
    return cov, var_m, var_r


@FEATURES.feature("beta_126", "relative", "rolling cov(ret, SPY ret)/var(SPY ret) over 126 sessions (paired observations)",
                  "panel ret of stock and SPY", PitStatus.PIT, lookback=126)
def beta_126(fs: FeatureSet) -> pd.DataFrame:
    cv = _rolling_cov_var(fs, 126)
    if cv is None:
        return full_like_nan(fs)
    cov, var_m, _ = cv
    return safe_div(cov, var_m.where(var_m > 0))


@FEATURES.feature("corr_spy_60", "relative", "rolling corr(ret, SPY ret) over 60 sessions (paired observations)",
                  "panel ret of stock and SPY", PitStatus.PIT, lookback=60)
def corr_spy_60(fs: FeatureSet) -> pd.DataFrame:
    cv = _rolling_cov_var(fs, 60)
    if cv is None:
        return full_like_nan(fs)
    cov, var_m, var_r = cv
    den = np.sqrt(var_m.where(var_m > 0) * var_r.where(var_r > 0))
    return safe_div(cov, den).clip(-1.0, 1.0)
