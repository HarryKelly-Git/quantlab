"""Momentum-breakout Stage 1: the daily setup scanner (docs/MOMENTUM-BREAKOUT-PREREG.md, section 4).

Plain rules, no fitting. Everything is computed from data up to and including the close of session D
and says whether a stock is a candidate for an opening-range breakout on session D+1.

Point-in-time:
  * Return / ratio maths uses the tri-scaled fields (``aclose``, ``ahigh``, ``alow``); level rules
    (dollar volume) use raw fields. ARCHITECTURE.md section 2.
  * Every rolling window is trailing (ends at D). Cross-sectional ranks use only session D's universe.
  * The sector map is the CURRENT snapshot (``sectors.sector_map``): ASSUMED_STATIC.
  * ``scan`` is checked with ``assert_truncation_invariant`` in tests/momentum_breakout.

Output of :func:`scan`: a long frame indexed by (date, symbol), only for universe members, with one
numeric column per metric and one boolean column per rule, so every decision can be audited.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quantlab.core.types import PitStatus
from quantlab.data.panel import DataBundle
from quantlab.sectors import sector_map

SCANNER_PIT_STATUS = PitStatus.ASSUMED_STATIC   # weakest input: today's sector map / reference data

# Add-on rules in the pre-registered ladder order. The core (regime + momentum) is always on.
FEATURE_ORDER = ("sector_rs", "stock_rs", "impulse", "consolidation", "contraction", "volume_dryup")


@dataclass(frozen=True)
class ScannerParams:
    """Pre-registered values (prereg section 4). Changing one is a new variant: amend the prereg."""

    min_median_dollar_volume: float = 20_000_000.0
    dollar_volume_window: int = 20
    regime_fast: int = 10
    regime_slow: int = 20
    momentum_lookback: int = 63
    momentum_min_pct: float = 0.95
    stock_rs_min_excess: float = 0.10
    impulse_lookback: int = 63
    impulse_min_gain: float = 0.30
    consol_min_days: int = 3
    consol_max_days: int = 40
    consol_min_frac_of_high: float = 0.85
    consol_sma: int = 20
    atr_fast: int = 5
    atr_slow: int = 20
    contraction_max_ratio: float = 0.75
    vol_fast: int = 5
    vol_slow: int = 50
    dryup_max_ratio: float = 0.80

    @classmethod
    def from_config(cls, config: Any | None) -> "ScannerParams":
        raw = (config.get("momentum_breakout.scanner", {}) or {}) if config is not None else {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown momentum_breakout.scanner keys: {sorted(unknown)}")
        return cls(**raw)


# --- building blocks (wide frames: sessions x symbols) ----------------------------------------------
def regime(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    """Market regime on the benchmark: close > SMA_fast and SMA_fast > SMA_slow."""
    sma_f = close.rolling(fast, min_periods=fast).mean()
    sma_s = close.rolling(slow, min_periods=slow).mean()
    on = (close > sma_f) & (sma_f > sma_s) & sma_s.notna()
    return pd.DataFrame({"spy_sma_fast": sma_f, "spy_sma_slow": sma_s, "regime_on": on})


def trailing_return(aclose: pd.DataFrame | pd.Series, n: int):
    return aclose / aclose.shift(n) - 1.0


def impulse_gain(ahigh: pd.DataFrame, alow: pd.DataFrame, n: int) -> pd.DataFrame:
    """Largest low-to-LATER-high gain inside the trailing n-session window ending at D.

    For each window: max_j ahigh_j / min_{i<=j} alow_i - 1. NaN if the window has any gap.
    """
    hi, lo = ahigh.to_numpy(dtype="float64"), alow.to_numpy(dtype="float64")
    out = np.full(hi.shape, np.nan)
    if hi.shape[0] >= n:
        for k in range(hi.shape[1]):
            wh = sliding_window_view(hi[:, k], n)          # (T-n+1, n), row r = sessions r..r+n-1
            wl = sliding_window_view(lo[:, k], n)
            run_min = np.minimum.accumulate(wl, axis=1)    # NaN propagates -> window rejected below
            with np.errstate(invalid="ignore", divide="ignore"):
                g = np.max(wh / run_min, axis=1) - 1.0
            g[np.isnan(wh).any(axis=1) | np.isnan(wl).any(axis=1)] = np.nan
            out[n - 1:, k] = g
    return pd.DataFrame(out, index=ahigh.index, columns=ahigh.columns)


def days_since_high(ahigh: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(n-session high, sessions since its MOST RECENT occurrence; 0 = today). NaN on any gap."""
    hi = ahigh.to_numpy(dtype="float64")
    top = np.full(hi.shape, np.nan)
    since = np.full(hi.shape, np.nan)
    if hi.shape[0] >= n:
        for k in range(hi.shape[1]):
            w = sliding_window_view(hi[:, k], n)
            bad = np.isnan(w).any(axis=1)
            m = np.where(bad[:, None], 0.0, w)
            mx = m.max(axis=1)
            last = n - 1 - np.argmax(m[:, ::-1] == mx[:, None], axis=1)
            top[n - 1:, k] = np.where(bad, np.nan, mx)
            since[n - 1:, k] = np.where(bad, np.nan, n - 1 - last)
    idx, cols = ahigh.index, ahigh.columns
    return pd.DataFrame(top, index=idx, columns=cols), pd.DataFrame(since, index=idx, columns=cols)


def atr_pct(ahigh: pd.DataFrame, alow: pd.DataFrame, aclose: pd.DataFrame, n: int) -> pd.DataFrame:
    prev = aclose.shift(1)
    tr = np.fmax(ahigh, prev) - np.fmin(alow, prev)   # true range; first bar falls back to high - low
    return (tr / aclose).rolling(n, min_periods=n).mean()


# --- the scanner --------------------------------------------------------------------------------------
def scan(bundle: DataBundle, universe: pd.DataFrame, params: ScannerParams | None = None,
         sectors: dict[str, str | None] | None = None) -> pd.DataFrame:
    """Metrics + rule flags for every (session, symbol) in the universe. See module docstring."""
    p = params or ScannerParams()
    pnl = bundle.panel
    dates, syms = pnl.dates, pnl.symbols
    aclose, ahigh, alow = pnl.aclose, pnl.ahigh, pnl.alow
    mkt = bundle.market_symbol
    if mkt not in syms:
        raise ValueError(f"market benchmark {mkt} missing from panel: regime is UNKNOWN, scanner refuses to run")

    # universe: master rules (given) + stricter liquidity floor on RAW dollar volume
    mdv = pnl.dollar_volume.rolling(p.dollar_volume_window, min_periods=p.dollar_volume_window).median()
    u = universe.reindex(index=dates, columns=syms).fillna(False).astype(bool)
    u &= (mdv >= p.min_median_dollar_volume).fillna(False)

    reg = regime(aclose[mkt], p.regime_fast, p.regime_slow)

    mom = trailing_return(aclose, p.momentum_lookback)
    mom_pct = mom.where(u).rank(axis=1, pct=True)

    sec = sectors if sectors is not None else sector_map(bundle)
    etf_of = pd.Series({s: sec.get(s) for s in syms}, dtype="object")
    etf_ret = pd.DataFrame(np.nan, index=dates, columns=syms)
    for etf in set(etf_of.dropna()):
        if etf in syms:
            cols = etf_of.index[etf_of == etf]
            r = trailing_return(aclose[etf], p.momentum_lookback)
            etf_ret[cols] = np.repeat(r.to_numpy()[:, None], len(cols), axis=1)
    spy_ret = trailing_return(aclose[mkt], p.momentum_lookback)

    imp = impulse_gain(ahigh, alow, p.impulse_lookback)
    high_n, since_high = days_since_high(ahigh, p.momentum_lookback)
    sma_c = aclose.rolling(p.consol_sma, min_periods=p.consol_sma).mean()
    frac_high = aclose / high_n

    atr_ratio = atr_pct(ahigh, alow, aclose, p.atr_fast) / atr_pct(ahigh, alow, aclose, p.atr_slow)
    # split-adjusted share volume (x an arbitrary per-symbol constant): raw dollar volume / tri close
    adj_vol = pnl.dollar_volume / aclose
    vol_ratio = (adj_vol.rolling(p.vol_fast, min_periods=p.vol_fast).mean()
                 / adj_vol.rolling(p.vol_slow, min_periods=p.vol_slow).mean())

    wide: dict[str, pd.DataFrame] = {
        "median_dollar_volume": mdv,
        "mom_ret": mom,
        "mom_pct": mom_pct,
        "sector_etf_ret": etf_ret,
        "impulse_gain": imp,
        "high_n": high_n,
        "days_since_high": since_high,
        "frac_of_high": frac_high,
        "atr_ratio": atr_ratio,
        "vol_ratio": vol_ratio,
        "momentum": (mom_pct >= p.momentum_min_pct),
        "sector_rs": etf_ret.gt(spy_ret, axis=0),
        "stock_rs": (mom - etf_ret) >= p.stock_rs_min_excess,
        "impulse": imp >= p.impulse_min_gain,
        "consolidation": ((since_high >= p.consol_min_days) & (since_high <= p.consol_max_days)
                          & (frac_high >= p.consol_min_frac_of_high) & (aclose >= sma_c)),
        "contraction": atr_ratio <= p.contraction_max_ratio,
        "volume_dryup": vol_ratio <= p.dryup_max_ratio,
    }
    stacked = {k: v.where(u).stack(future_stack=True) for k, v in wide.items()}
    out = pd.DataFrame(stacked)
    out.index.names = ["date", "symbol"]
    keep = u.stack(future_stack=True)
    out = out[keep.reindex(out.index).fillna(False).to_numpy()]
    for c in ("momentum", *FEATURE_ORDER):
        out[c] = out[c].fillna(False).astype(bool)   # NaN input (short history, unknown sector) = rule fails
    out = out.join(reg[["regime_on"]], on="date")
    out["regime_on"] = out["regime_on"].fillna(False).astype(bool)
    out["sector_etf"] = out.index.get_level_values("symbol").map(etf_of)
    return out.sort_index()


def fires(scan_out: pd.DataFrame, features: tuple[str, ...] = ()) -> pd.Series:
    """Setup fires at the close of D (trade on D+1): universe & regime & momentum & each chosen feature."""
    bad = [f for f in features if f not in FEATURE_ORDER]
    if bad:
        raise ValueError(f"unknown features {bad}; allowed {FEATURE_ORDER}")
    ok = scan_out["regime_on"] & scan_out["momentum"]
    for f in features:
        ok &= scan_out[f]
    return ok
