"""H03 momentum matrix and H04 residual momentum (Parts 8-9). PRE-REGISTERED GRIDS (fixed before TRAIN):

H04 residual momentum (Blitz, Huij & Martens 2011 style), 16 variants:
  formation   6-1 (sessions t-126 .. t-21) | 12-1 (t-252 .. t-21)
  basis       market residual (r - beta_mkt * r_SPY) | two-factor residual (market + own statistical sector)
  scaling     sum | sum / std (residual information ratio, as in the paper)
  weighting   equal | inverse_vol
  book: long top decile, short bottom decile; hold 21 sessions (overlapping); next-open execution.

H03 momentum matrix, 6 horizons x 6 definitions = 36 + 4 conditioned variants = 40:
  horizons    5, 10, 21, 63, 126, 252 sessions (skip the last 21 sessions for 63/126/252)
  definitions absolute (raw cumulative) | market_relative (r - beta*r_SPY) | sector_relative (r - r_sector) |
              vol_adjusted (cumulative / daily vol) | consistency (share of up days in the window) |
              acceleration (second-half return minus first-half return)
  hold 5 sessions for horizons <= 10, else 21; equal weight; long top decile / short bottom decile.
  Conditioned (12-1 absolute only): within the top-3 sector ETFs by 63-day return (sector leadership) |
  only when 20d/60d volume ratio > 1 (volume rising) | only in calm markets (SPY 20d vol below its 2y
  median) | only when breadth is strong (> 60% of the universe above its 50-day average).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from quantlab.alpha.engine import quantile_weights
from quantlab.alpha.experiment import Variant, run_family
from quantlab.alpha.experiments.reversal import context


def _window_sum(x: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Sum of x over sessions t-start+1 .. t-end (end=0 -> through t)."""
    return x.rolling(start - end, min_periods=int(0.8 * (start - end))).sum().shift(end)


# ----------------------------------------------------------------------------- H04
def residual_mom_signal(d, formation: str, basis: str, scaling: str) -> pd.DataFrame:
    r = d.p["ret_cc"]
    res = (r - d.betas["beta_mkt"].mul(r["SPY"], axis=0)) if basis == "market" else d.resid
    start = 126 if formation == "6-1" else 252
    s = _window_sum(res, start, 21)
    if scaling == "ir":
        sd = res.rolling(start - 21, min_periods=int(0.8 * (start - 21))).std().shift(21)
        s = s / sd
    return s


def residual_variants(d) -> list[Variant]:
    out = []
    for f, b, sc, wt in itertools.product(("6-1", "12-1"), ("market", "two_factor"), ("sum", "ir"), ("equal", "inverse_vol")):
        spec = {"universe": "liquid survivorship-free", "signal": f"{f} {b} residual momentum ({sc})", "entry": "next open",
                "position": "decile long-short", "sizing": wt, "exit": "overlapping 21-session hold",
                "costs": "CostModel tiers; borrow 50bps/yr", "benchmark": "cash", "risk": "each side sums to 1"}
        out.append(Variant(f"RM_{f}_{b}_{sc}_{wt}", spec,
                           (lambda f=f, b=b, sc=sc, wt=wt: quantile_weights(residual_mom_signal(d, f, b, sc), d.u_liquid,
                                                                            q=0.1, weighting=wt, vol=d.vol60)), holding=21))
    return out


# ----------------------------------------------------------------------------- H03
HORIZONS = (5, 10, 21, 63, 126, 252)
DEFS = ("absolute", "market_relative", "sector_relative", "vol_adjusted", "consistency", "acceleration")


def mom_signal(d, h: int, definition: str) -> pd.DataFrame:
    r = d.p["ret_cc"]
    skip = 21 if h >= 63 else 0
    if definition == "absolute":
        base = r
    elif definition == "market_relative":
        base = r - d.betas["beta_mkt"].mul(r["SPY"], axis=0)
    elif definition == "sector_relative":
        base = r - d.sec_ret
    elif definition == "vol_adjusted":
        return _window_sum(r, h, skip) / r.rolling(h, min_periods=int(0.8 * h)).std().shift(skip)
    elif definition == "consistency":
        return (r > 0).astype(float).where(r.notna()).rolling(h - skip, min_periods=int(0.8 * (h - skip))).mean().shift(skip)
    elif definition == "acceleration":
        half = (h - skip) // 2
        recent = _window_sum(r, skip + half, skip)
        older = _window_sum(r, h, skip + half)
        return recent - older
    else:
        raise ValueError(definition)
    return _window_sum(base, h, skip)


def conditioned_weights(d, cond: str) -> pd.DataFrame:
    s = mom_signal(d, 252, "absolute")
    u = d.u_liquid.copy()
    r = d.p["ret_cc"]
    if cond == "sector_leadership":
        etf = [e for e in ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY") if e in r.columns]
        e63 = d.p["adj_close"][etf].pct_change(63)
        top3 = e63.rank(axis=1, ascending=False) <= 3
        lead = pd.DataFrame(False, index=u.index, columns=u.columns)
        for e in etf:
            lead |= (d.sectors == e) & top3[e].to_numpy()[:, None]
        u = u & lead
    elif cond == "volume_rising":
        v = d.p["volume"]
        u = u & ((v.rolling(20).mean() / v.rolling(60).mean()) > 1)
    elif cond in ("calm_market", "strong_breadth"):
        if cond == "calm_market":
            sv = r["SPY"].rolling(20).std()
            on = sv < sv.rolling(504, min_periods=126).median()
        else:
            above = d.p["adj_close"] > d.p["adj_close"].rolling(50, min_periods=40).mean()
            on = above.where(d.u_liquid).sum(axis=1) / d.u_liquid.sum(axis=1).replace(0, np.nan) > 0.6
        w = quantile_weights(s, u, q=0.1)
        return w.mul(on.astype(float), axis=0)
    return quantile_weights(s, u, q=0.1)


def matrix_variants(d) -> list[Variant]:
    out = []
    for h, df_ in itertools.product(HORIZONS, DEFS):
        hold = 5 if h <= 10 else 21
        spec = {"universe": "liquid survivorship-free", "signal": f"{df_} momentum over {h} sessions" + (" skip 21" if h >= 63 else ""),
                "entry": "next open", "position": "decile long-short (winners long)", "sizing": "equal",
                "exit": f"overlapping {hold}-session hold", "costs": "CostModel tiers; borrow 50bps/yr", "benchmark": "cash",
                "risk": "each side sums to 1"}
        out.append(Variant(f"MOM_{h}_{df_}", spec, (lambda h=h, df_=df_: quantile_weights(mom_signal(d, h, df_), d.u_liquid, q=0.1)),
                           holding=hold))
    for cond in ("sector_leadership", "volume_rising", "calm_market", "strong_breadth"):
        spec = {"universe": "liquid survivorship-free", "signal": f"12-1 absolute momentum, condition {cond}", "entry": "next open",
                "position": "decile long-short", "sizing": "equal", "exit": "overlapping 21-session hold",
                "costs": "CostModel tiers; borrow 50bps/yr", "benchmark": "cash", "risk": "flat when the condition is off"}
        out.append(Variant(f"MOM_252_cond_{cond}", spec, (lambda cond=cond: conditioned_weights(d, cond)), holding=21))
    return out


def run_residual(d) -> dict:
    return run_family(family="H04_residual_momentum", hypothesis_id="H04", variants=residual_variants(d), ctx=context(d),
                      prior_trials=1, notes="prior_trials=1: QuantLab's momentum_trend strategy")


def run_matrix(d) -> dict:
    return run_family(family="H03_momentum_matrix", hypothesis_id="H03", variants=matrix_variants(d), ctx=context(d),
                      prior_trials=3, notes="prior_trials=3: momentum_trend, relative_strength, quality_momentum strategies")
