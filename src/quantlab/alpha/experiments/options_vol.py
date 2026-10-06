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


# ------------------------------------------------------------------------------------------------------
# observation builder (features at the snapshot session; outcomes only from the features' outcome cols)
# ------------------------------------------------------------------------------------------------------
def build_obs(d, feat: pd.DataFrame | None = None, cal: pd.DataFrame | None = None) -> pd.DataFrame:
    from quantlab.alpha.experiments.earnings import clean_calendar
    feat = load_features() if feat is None else feat
    m = monthly_panel(feat)
    m = m[m["act_symbol"].isin(d.p.symbols)].copy()
    # term structure: same snapshot, expiry nearest 60 days (40..80)
    f60 = feat[(feat["dte"] >= 40) & (feat["dte"] <= 80)].copy()
    f60["dd"] = (f60["dte"] - 60).abs()
    f60 = f60.loc[f60.groupby(["date", "act_symbol"])["dd"].idxmin(), ["date", "act_symbol", "iv_atm"]].rename(columns={"iv_atm": "iv_60"})
    m = m.merge(f60, on=["date", "act_symbol"], how="left")
    # trailing realised vol and range vol at the snapshot session (adjusted prices, PIT)
    lr = np.log(d.p["adj_close"] / d.p["adj_close"].shift(1))
    hv = {k: lr.rolling(k, min_periods=int(0.8 * k)).std() * np.sqrt(252) for k in (5, 21, 63)}
    pk = (np.log(d.p["adj_high"] / d.p["adj_low"]) ** 2 / (4 * np.log(2))).rolling(21, min_periods=17).mean().pow(0.5) * np.sqrt(252)
    ci = d.p["close"].columns.get_indexer(m["act_symbol"])
    si = d.p.dates.get_indexer(pd.to_datetime(m["session"]))
    ok = (ci >= 0) & (si >= 0)
    for name, mat in (("hv5", hv[5]), ("hv21", hv[21]), ("hv63", hv[63]), ("rangevol21", pk)):
        a = mat.to_numpy()
        m[name] = np.where(ok, a[np.clip(si, 0, None), np.clip(ci, 0, None)], np.nan)
    # earnings inside the straddle's life (realised dates, PIT_ASSUMED)
    cal = clean_calendar(d) if cal is None else cal
    cal = cal[["act_symbol", "react_session"]].copy()
    cal["react_session"] = pd.to_datetime(cal["react_session"])
    mm = m[["act_symbol", "session", "expiration"]].reset_index().merge(cal, on="act_symbol", how="inner")
    hit = mm[(mm["react_session"] > pd.to_datetime(mm["session"])) & (mm["react_session"] <= pd.to_datetime(mm["expiration"]))]
    m["earn_in_window"] = m.index.isin(hit["index"].unique())
    m["cal_covered"] = pd.to_datetime(m["session"]) >= pd.Timestamp("2020-01-22")
    # IV percentile within the underlying's trailing year of one-month IVs, and 1-week IV change
    m = m.sort_values(["act_symbol", "session"])
    def _pct(s):
        return s.rolling(150, min_periods=60).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    m["iv_pctile"] = m.groupby("act_symbol", observed=True)["iv_atm"].transform(_pct)
    m["iv_chg_1w"] = m.groupby("act_symbol", observed=True)["iv_atm"].pct_change()
    # option momentum: mean hold-to-expiry return of this underlying's straddles that EXPIRED before now
    m["opt_mom_3m"] = np.nan
    m["opt_mom_12m"] = np.nan
    for sym, g in m.groupby("act_symbol", observed=True):
        done = g[["exp_session", "ret_hold_mid"]].dropna().sort_values("exp_session")
        if done.empty:
            continue
        es = pd.to_datetime(done["exp_session"]).to_numpy()
        rr = done["ret_hold_mid"].to_numpy()
        cs = np.r_[0, np.cumsum(rr)]
        sess = pd.to_datetime(g["session"]).to_numpy()
        hi = np.searchsorted(es, sess, side="left")             # expiries strictly before the snapshot session
        for col, days in (("opt_mom_3m", 91), ("opt_mom_12m", 365)):
            lo = np.searchsorted(es, sess - np.timedelta64(days, "D"), side="left")
            n = hi - lo
            val = np.where(n >= (2 if days == 91 else 6), (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
            m.loc[g.index, col] = val
    # underlying 1-month return (for the option-momentum control) and next-21-session return (skew test)
    r21 = d.p["adj_close"].pct_change(21)
    fwd21 = d.p["adj_close"].shift(-21) / d.p["adj_close"] - 1 - (d.p["adj_close"]["SPY"].shift(-21) / d.p["adj_close"]["SPY"] - 1).to_numpy()[:, None]
    ci = d.p["close"].columns.get_indexer(m["act_symbol"])
    si = d.p.dates.get_indexer(pd.to_datetime(m["session"]))
    ok = (ci >= 0) & (si >= 0)
    m["ret_1m_underlying"] = np.where(ok, r21.to_numpy()[np.clip(si, 0, None), np.clip(ci, 0, None)], np.nan)
    m["fwd21_mkt_adj"] = np.where(ok, fwd21.to_numpy()[np.clip(si, 0, None), np.clip(ci, 0, None)], np.nan)
    m["skew"] = m["iv_put25"] - m["iv_call25"]
    m["term_slope"] = m["iv_60"] - m["iv_atm"]
    m["hv_minus_iv"] = m["hv21"] - m["iv_atm"]
    m["log_rv_iv"] = np.log(m["rv_to_exp"] / m["iv_atm"])
    m["fly_ret"] = iron_fly_return(m["spot"].to_numpy(), m["k_atm"].to_numpy(), m["straddle_bid"].to_numpy(),
                                   m["em_pct"].to_numpy(), m["exp_close"].to_numpy())
    m["short_straddle_ret_per_credit"] = (m["straddle_bid"] - m["payoff"]) / m["straddle_bid"]
    return m.reset_index(drop=True)


def fit_rv_model(m: pd.DataFrame, with_iv: bool = False) -> dict:
    """Pooled OLS of log RV-to-expiry on log trailing vols (+ earnings flag) fitted on TRAIN only."""
    cols = ["hv5", "hv21", "hv63", "rangevol21"] + (["iv_atm"] if with_iv else [])
    a, b = splits.window("options", "TRAIN")
    x = m[(pd.to_datetime(m["session"]) >= a) & (pd.to_datetime(m["session"]) <= b)]
    x = x.dropna(subset=cols + ["rv_to_exp"])
    x = x[(x[cols] > 0).all(axis=1) & (x["rv_to_exp"] > 0)]
    X = np.column_stack([np.ones(len(x))] + [np.log(x[c].to_numpy()) for c in cols] + [x["earn_in_window"].astype(float).to_numpy()])
    y = np.log(x["rv_to_exp"].to_numpy())
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    return {"cols": cols, "coef": coef.tolist(), "n_train": int(len(x)), "resid_var": float(resid.var())}


def predict_rv(m: pd.DataFrame, model: dict) -> pd.Series:
    cols = model["cols"]
    X = np.column_stack([np.ones(len(m))] + [np.log(m[c].where(m[c] > 0).to_numpy()) for c in cols] + [m["earn_in_window"].astype(float).to_numpy()])
    return pd.Series(np.exp(X @ np.array(model["coef"]) + model["resid_var"] / 2), index=m.index)


def forecast_vs_iv(m: pd.DataFrame, fc: pd.Series) -> dict:
    """Out-of-sample accuracy of the forecast vs IV for log RV, and the encompassing regression."""
    out = {}
    for sp in ("VALIDATION", "OOS"):
        a, b = splits.window("options", sp)
        x = m[(pd.to_datetime(m["session"]) >= a) & (pd.to_datetime(m["session"]) <= b)].copy()
        x["fc"] = fc.reindex(x.index)
        x = x.dropna(subset=["fc", "iv_atm", "rv_to_exp"])
        x = x[(x["iv_atm"] > 0) & (x["rv_to_exp"] > 0) & (x["fc"] > 0)]
        if len(x) < 100:
            continue
        ly, li, lf = np.log(x["rv_to_exp"]), np.log(x["iv_atm"]), np.log(x["fc"])
        mse_iv = float(((ly - li) ** 2).mean())
        mse_iv_debiased = float(((ly - li - (ly - li).mean()) ** 2).mean())
        mse_fc = float(((ly - lf) ** 2).mean())
        X = np.column_stack([np.ones(len(x)), li, lf])
        coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
        res = ly - X @ coef
        # cluster-robust (by week) standard errors for the encompassing regression
        wk = pd.to_datetime(x["session"]).dt.to_period("W-FRI").astype(str).to_numpy()
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = np.zeros((3, 3))
        for w in np.unique(wk):
            idx = wk == w
            sc = X[idx].T @ res[idx]
            meat += np.outer(sc, sc)
        se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
        out[sp] = {"n": int(len(x)), "mse_log_iv": mse_iv, "mse_log_iv_debiased": mse_iv_debiased, "mse_log_forecast": mse_fc,
                   "encompassing_coef_iv": float(coef[1]), "encompassing_coef_fc": float(coef[2]),
                   "t_iv": float(coef[1] / se[1]), "t_fc": float(coef[2] / se[2]),
                   "mean_log_rv_minus_log_iv": float((ly - li).mean())}
    return out
