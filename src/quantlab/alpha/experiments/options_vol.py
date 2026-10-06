"""Options volatility lab (Parts 12-20, 49). PRE-REGISTERED before any options outcome was computed.

Unit of observation: (snapshot date, underlying) using the expiry whose DTE is nearest 30 calendar days
within [20, 45] ("one-month straddle"), from alpha.options_features. Trades are ATM straddles held to
expiry: buy at the ASK (or, for the short side, sell at the BID and pay intrinsic at expiry), so the
quoted spread is always paid. One observation per underlying per calendar week (the first snapshot of
the week) so overlapping holds do not inflate the sample.

H24 IV vs RV / variance risk premium (Goyal & Saretto replication, defined-risk version)
    sort on HV - IV, HV = trailing 21-session realised vol at the snapshot. Deciles; report the
    long-straddle return per decile, the top-minus-bottom spread, and the short side as an ATM short
    straddle with a +/- 2x expected-move wing (iron butterfly, max loss known).
H33 Part 49: forecast(RV) vs IV
    model: pooled OLS of log(rv_to_exp) on [log HV5, log HV21, log HV63, log range-vol21, earnings-in-
    window flag] fitted on TRAIN snapshots only (2019-02..2021-12). Forecast quality vs IV judged out of
    sample by (a) MSE of log RV, (b) Mincer-Zarnowitz / encompassing regression log RV = a + b log IV +
    c log forecast. Trade: deciles of log(forecast / IV); same straddle / iron-butterfly legs.
H25 expected move vs realised move: em_pct vs move_to_exp by IV bucket and earnings / non-earnings.
H30 IV rank: IV percentile within the underlying's trailing 252-day IV history (deciles) and 1-week
    IV change (deciles) vs straddle returns.
H28 term structure: slope = IV(~60d) - IV(~30d) deciles vs straddle returns and vs log(RV/IV).
H29 skew: put25 IV - call25 IV deciles vs (a) straddle returns and (b) the underlying's next-21-session
    return (directional, Xing-Zhang-Zhao), market-adjusted.
H27 option momentum: past 4-week straddle return of the same underlying (previous one-month straddle
    marked at the bid) deciles vs the next straddle return, with and without controlling for the
    underlying's own 1-month return.
Every family reports deciles, the top-minus-bottom spread with a Newey-West t (weekly observations),
the split (TRAIN / VALIDATION / OOS), and earnings vs non-earnings.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantlab.alpha import registry, splits
from quantlab.alpha.options_store import options_dir
from quantlab.validation.stats import newey_west_tstat


def monthly_panel(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat[(feat["dte"] >= 20) & (feat["dte"] <= 45)].copy()
    f["dd"] = (f["dte"] - 30).abs()
    f = f.loc[f.groupby(["date", "act_symbol"])["dd"].idxmin()]
    f["week"] = pd.to_datetime(f["session"]).dt.to_period("W-FRI")
    f = f.sort_values("date").groupby(["week", "act_symbol"], observed=True).head(1)
    return f.reset_index(drop=True)


def iron_fly_return(row_spot, k, straddle_bid, em, exp_close, wing_mult: float = 2.0, wing_cost_frac: float = 0.0):
    """Short ATM straddle at the bid + long wings at k +/- wing_mult*em*spot (wing premium approximated
    as zero when not quoted: conservative for the seller is to charge it - see wing_cost_frac).
    Return per $ of max loss."""
    width = wing_mult * em * row_spot
    credit = straddle_bid * (1 - wing_cost_frac)
    loss = np.minimum(np.abs(exp_close - k), width)
    pnl = credit - loss
    max_loss = width - credit
    return np.where(max_loss > 0, pnl / max_loss, np.nan)


def decile_table(df: pd.DataFrame, sort_col: str, ret_col: str, n: int = 10) -> pd.DataFrame:
    x = df.dropna(subset=[sort_col, ret_col]).copy()
    x["dec"] = x.groupby("week", observed=True)[sort_col].transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False) if len(s) >= n else np.nan)
    return x


def long_short_series(x: pd.DataFrame, ret_col: str, n: int = 10) -> pd.Series:
    g = x.groupby(["week", "dec"], observed=True)[ret_col].mean().unstack()
    if n - 1 not in g.columns or 0 not in g.columns:
        return pd.Series(dtype=float)
    return (g[n - 1] - g[0]).dropna()


def summarize(x: pd.DataFrame, ret_col: str, n: int = 10) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sp in ("TRAIN", "VALIDATION", "OOS"):
        a, b = splits.window("options", sp)
        xs = x[(pd.to_datetime(x["session"]) >= a) & (pd.to_datetime(x["session"]) <= b)]
        if xs.empty:
            continue
        dec = xs.groupby("dec", observed=True)[ret_col].agg(["mean", "median", "count"])
        ls = long_short_series(xs, ret_col, n)
        t = newey_west_tstat(ls.to_numpy(), min_obs=20) if len(ls) else None
        out[sp] = {"decile_mean": {int(k): float(v) for k, v in dec["mean"].items()},
                   "decile_median": {int(k): float(v) for k, v in dec["median"].items()},
                   "n_obs": int(dec["count"].sum()), "top_minus_bottom_mean": float(ls.mean()) if len(ls) else None,
                   "top_minus_bottom_t_nw": t.t if t is not None else None, "n_weeks": int(len(ls))}
    return out


def load_features() -> pd.DataFrame:
    return pd.read_parquet(options_dir() / "features.parquet")
