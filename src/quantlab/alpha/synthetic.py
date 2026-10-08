"""SYNTHETIC AlphaPanel for tests and method validation (never market evidence; ``meta['is_synthetic']``).

Market + 11 sector ETFs + stocks with market/sector loadings, idiosyncratic noise, a few delistings,
and an optional PLANTED short-term reversal (next-day return = -k x today's idiosyncratic return), so
method-validation tests can check the research stack finds a planted edge and nothing in a null world.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha.panel import SECTOR_ETFS, AlphaPanel


def make_panel(n_stocks: int = 80, n_days: int = 700, seed: int = 0, planted_reversal: float = 0.0,
               start: str = "2016-01-04") -> AlphaPanel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    etfs = list(SECTOR_ETFS)
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    syms = ["SPY", *etfs, *stocks]
    T = n_days
    mkt = rng.normal(0.0004, 0.01, T)
    sec = {e: 0.8 * mkt + rng.normal(0, 0.006, T) for e in etfs}
    rets = {"SPY": mkt, **sec}
    sect_of = {s: etfs[i % len(etfs)] for i, s in enumerate(stocks)}
    idio_prev = np.zeros(n_stocks)
    R = np.zeros((T, n_stocks))
    for t in range(T):
        idio = rng.normal(0, 0.02, n_stocks)
        R[t] = np.array([0.6 * mkt[t] + 0.6 * sec[sect_of[s]][t] for s in stocks]) + idio - planted_reversal * idio_prev
        idio_prev = idio
    for j, s in enumerate(stocks):
        rets[s] = R[:, j]
    ret = pd.DataFrame(rets, index=dates)[syms]
    # a few delistings: the last 5 stocks stop trading at different dates
    for k, s in enumerate(stocks[-5:]):
        ret.loc[dates[int(T * (0.5 + 0.08 * k)):], s] = np.nan
    adj_close = (1 + ret.fillna(0)).cumprod().where(ret.notna()) * 50
    gap = pd.DataFrame(rng.normal(0, 0.003, ret.shape), index=dates, columns=syms)
    adj_open = (adj_close.shift(1) * (1 + gap)).where(ret.notna())
    adj_open.iloc[0] = adj_close.iloc[0]
    hi = pd.concat([adj_open, adj_close]).groupby(level=0).max() * (1 + rng.uniform(0, 0.01, ret.shape))
    lo = pd.concat([adj_open, adj_close]).groupby(level=0).min() * (1 - rng.uniform(0, 0.01, ret.shape))
    vol = pd.DataFrame(rng.lognormal(14, 0.3, ret.shape), index=dates, columns=syms).where(ret.notna())
    f = {"open": adj_open, "high": hi, "low": lo, "close": adj_close, "volume": vol, "vwap": adj_close,
         "trade_count": vol / 100, "adj_open": adj_open, "adj_high": hi, "adj_low": lo, "adj_close": adj_close}
    f["dollar_volume"] = f["close"] * f["volume"]
    f["ret_cc"] = adj_close / adj_close.shift(1) - 1
    f["ret_on"] = adj_open / adj_close.shift(1) - 1
    f["ret_id"] = adj_close / adj_open - 1
    f["ret_oo"] = adj_open.shift(-1) / adj_open - 1
    f["has_bar"] = adj_close.notna().astype(float)
    f["n_hist"] = f["has_bar"].cumsum()
    master = pd.DataFrame({"symbol": syms, "sec_type": ["ETF"] * (1 + len(etfs)) + ["COMMON"] * n_stocks,
                           "status": ["active"] * (len(syms) - 5) + ["inactive"] * 5})
    meta = {"is_synthetic": True, "n_symbols": len(syms), "n_dates": T,
            "delistings": pd.DataFrame(columns=["symbol", "last_bar", "distressed", "delist_return", "renamed"])}
    return AlphaPanel(dates, pd.Index(syms), f, master, meta)


def truncate(p: AlphaPanel, d) -> AlphaPanel:
    d = pd.Timestamp(d)
    f = {k: v.loc[:d] for k, v in p.f.items()}
    return AlphaPanel(p.dates[p.dates <= d], p.symbols, f, p.master, dict(p.meta))
