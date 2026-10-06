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
    IV change (deciles) vs straddle returns. Implementation: 52 weekly one-month IV observations (min 40).
    AUDIT NOTE: the first run used 150 weekly observations (~3 years) by mistake; that version is kept as
    ``iv_pctile_150w`` and reported next to the pre-registered one.
H28 term structure: slope = IV(~60d) - IV(~30d) deciles vs straddle returns and vs log(RV/IV).
H29 skew: put25 IV - call25 IV deciles vs (a) straddle returns and (b) the underlying's next-21-session
    return (directional, Xing-Zhang-Zhao), market-adjusted.
H27 option momentum: past straddle returns of the same underlying vs the next straddle return.
    AUDIT NOTE: the text first registered here ("previous one-month straddle marked at the bid") is NOT
    what the code (written before any outcome was computed) does. The code, which is what was run and
    reported: mean hold-to-expiry MID return of the underlying's one-month straddles that EXPIRED strictly
    before the snapshot, over the trailing 91 days (>= 2 expiries, ``opt_mom_3m``) and 365 days (>= 6,
    ``opt_mom_12m``; closer to Heston et al. 2023, who use 12-month histories).
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
from quantlab.validation.stats import newey_west_lags, newey_west_tstat

MIN_NW_LAGS = 5          # weekly series of ~1-month holds overlap ~4.3 weeks: never fewer than 5 HAC lags


def nw_t(x, min_obs: int = 20):
    """Newey-West t of a WEEKLY series of one-month option returns, with at least MIN_NW_LAGS lags."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return newey_west_tstat(a, lags=max(newey_west_lags(len(a)), MIN_NW_LAGS), min_obs=min_obs)


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


def iron_fly_with_model_wings(m: pd.DataFrame, wing_mult: float = 2.0, wing_spread: float = 0.10, r: float = 0.04) -> np.ndarray:
    """Short ATM straddle at the BID + long wings at k -/+ wing_mult x expected move. The chain rarely
    quotes strikes that far out, so the wings are priced by Black-Scholes at the 25-delta IV of their side
    (MODEL; ATM IV if missing) plus ``wing_spread`` of their value as cost. Return per $ of max loss."""
    from quantlab.options.pricing import bs_price
    S, k, T = m["spot"].to_numpy(), m["k_atm"].to_numpy(), np.maximum(m["dte"].to_numpy(), 1) / 365.0
    width = wing_mult * m["em_pct"].to_numpy() * S
    ivp = m["iv_put25"].fillna(m["iv_atm"]).to_numpy()
    ivc = m["iv_call25"].fillna(m["iv_atm"]).to_numpy()
    put_w = np.array([float(bs_price(s_, kk - w, t, iv, r, "put")) if np.isfinite(iv) and kk - w > 0 else np.nan
                      for s_, kk, w, t, iv in zip(S, k, width, T, ivp)])
    call_w = np.array([float(bs_price(s_, kk + w, t, iv, r, "call")) if np.isfinite(iv) else np.nan
                       for s_, kk, w, t, iv in zip(S, k, width, T, ivc)])
    credit = m["straddle_bid"].to_numpy() - (put_w + call_w) * (1 + wing_spread)
    loss = np.minimum(np.abs(m["exp_close"].to_numpy() - k), width)
    max_loss = width - credit
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(max_loss > 0, (credit - loss) / max_loss, np.nan)


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
        t = nw_t(ls.to_numpy(), min_obs=20) if len(ls) else None
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
def iv_percentile(m: pd.DataFrame, n_prev: int = 52, min_prev: int = 40, key: str = "entity") -> pd.Series:
    """Percentile of the current one-month ATM IV among the same underlying's previous ``n_prev`` weekly
    observations (at least ``min_prev``). PIT: trailing rows only. The first run used n_prev=149, min_prev=59
    (a 150-row window including the current one)."""
    def _pct(x):
        return x.rolling(n_prev + 1, min_periods=min_prev + 1).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    m = m.sort_values([key, "session"])
    return m.groupby(key, observed=True)["iv_atm"].transform(_pct).reindex(m.index)


def option_momentum(m: pd.DataFrame, key: str = "entity") -> pd.DataFrame:
    """Mean hold-to-expiry MID return of the underlying's one-month straddles whose expiry session is
    STRICTLY before the snapshot session (outcome known at the snapshot), over the trailing 91 days
    (>= 2 expiries) and 365 days (>= 6 expiries)."""
    out = pd.DataFrame(np.nan, index=m.index, columns=["opt_mom_3m", "opt_mom_12m"])
    for _, g in m.groupby(key, observed=True):
        done = g[["exp_session", "ret_hold_mid"]].dropna().sort_values("exp_session")
        if done.empty:
            continue
        es = pd.to_datetime(done["exp_session"]).to_numpy()
        cs = np.r_[0, np.cumsum(done["ret_hold_mid"].to_numpy())]
        sess = pd.to_datetime(g["session"]).to_numpy()
        hi = np.searchsorted(es, sess, side="left")             # expiries strictly before the snapshot session
        for col, days, need in (("opt_mom_3m", 91, 2), ("opt_mom_12m", 365, 6)):
            lo = np.searchsorted(es, sess - np.timedelta64(days, "D"), side="left")
            n = hi - lo
            out.loc[g.index, col] = np.where(n >= need, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return out


def earnings_in_window(m: pd.DataFrame, cal: pd.DataFrame) -> pd.Series:
    """True when the SAME company (entity) has a calendar reaction session in (snapshot, expiry]."""
    c = cal[["entity", "react_session"]].copy()
    c["react_session"] = pd.to_datetime(c["react_session"])
    mm = m[["entity", "session", "expiration"]].reset_index().merge(c, on="entity", how="inner")
    hit = mm[(mm["react_session"] > pd.to_datetime(mm["session"])) & (mm["react_session"] <= pd.to_datetime(mm["expiration"]))]
    return pd.Series(m.index.isin(hit["index"].unique()), index=m.index)


def build_obs(d, feat: pd.DataFrame | None = None, cal: pd.DataFrame | None = None) -> pd.DataFrame:
    from quantlab.alpha.entities import map_events
    from quantlab.alpha.experiments.earnings import clean_calendar
    feat = load_features() if feat is None else feat
    m = monthly_panel(feat)
    # audit fix M1: the chain's ticker on the snapshot date -> the ENTITY that used it then. The first version
    # kept only plain tickers present in the panel, which dropped every delisted 'TICKER@date' company
    # (survivorship) and looked up recycled tickers' trailing vols on the wrong company.
    from quantlab.alpha.entities import apply_aliases
    if "entity" not in m.columns:                         # features built before the fix
        m["entity"] = map_events(m, d.resolved(), "act_symbol", "session")
    else:                                                 # store entity -> the twin key the panel kept (M2)
        m["entity"] = apply_aliases(m["entity"], m["session"], d.p.meta.get("twin_aliases"))
    m = m[m["entity"].isin(d.p.symbols)].copy()
    # term structure: same snapshot, expiry nearest 60 days (40..80)
    f60 = feat[(feat["dte"] >= 40) & (feat["dte"] <= 80)].copy()
    f60["dd"] = (f60["dte"] - 60).abs()
    f60 = f60.loc[f60.groupby(["date", "act_symbol"])["dd"].idxmin(), ["date", "act_symbol", "iv_atm"]].rename(columns={"iv_atm": "iv_60"})
    m = m.merge(f60, on=["date", "act_symbol"], how="left")
    # trailing realised vol and range vol at the snapshot session (adjusted prices, PIT), by ENTITY
    lr = np.log(d.p["adj_close"] / d.p["adj_close"].shift(1))
    hv = {k: lr.rolling(k, min_periods=int(0.8 * k)).std() * np.sqrt(252) for k in (5, 21, 63)}
    pk = (np.log(d.p["adj_high"] / d.p["adj_low"]) ** 2 / (4 * np.log(2))).rolling(21, min_periods=17).mean().pow(0.5) * np.sqrt(252)
    r21 = d.p["adj_close"].pct_change(21, fill_method=None)
    spy = d.p["adj_close"]["SPY"]
    fwd21 = d.p["adj_close"].shift(-21) / d.p["adj_close"] - 1 - (spy.shift(-21) / spy - 1).to_numpy()[:, None]
    ci = d.p["close"].columns.get_indexer(m["entity"])
    si = d.p.dates.get_indexer(pd.to_datetime(m["session"]))
    ok = (ci >= 0) & (si >= 0)
    for name, mat in (("hv5", hv[5]), ("hv21", hv[21]), ("hv63", hv[63]), ("rangevol21", pk),
                      ("ret_1m_underlying", r21), ("fwd21_mkt_adj", fwd21)):
        a = mat.to_numpy()
        m[name] = np.where(ok, a[np.clip(si, 0, None), np.clip(ci, 0, None)], np.nan)
    # earnings inside the straddle's life (realised dates, PIT_ASSUMED; calendar covers 2020-01-22 on)
    cal = clean_calendar(d) if cal is None else cal
    m["earn_in_window"] = earnings_in_window(m, cal)
    m["cal_covered"] = pd.to_datetime(m["session"]) >= pd.Timestamp("2020-01-22")
    # IV percentile (pre-registered: trailing year = 52 weekly obs; as first run: 150), 1-week IV change
    m = m.sort_values(["entity", "session"])
    m["iv_pctile"] = iv_percentile(m)
    m["iv_pctile_150w"] = iv_percentile(m, n_prev=149, min_prev=59)
    m["iv_chg_1w"] = m.groupby("entity", observed=True)["iv_atm"].pct_change(fill_method=None)
    m[["opt_mom_3m", "opt_mom_12m"]] = option_momentum(m)
    m["skew"] = m["iv_put25"] - m["iv_call25"]
    m["term_slope"] = m["iv_60"] - m["iv_atm"]
    m["hv_minus_iv"] = m["hv21"] - m["iv_atm"]
    m["log_rv_iv"] = np.log(m["rv_to_exp"] / m["iv_atm"])
    m["fly_ret"] = iron_fly_with_model_wings(m)
    m["short_straddle_ret_per_credit"] = (m["straddle_bid"] - m["payoff"]) / m["straddle_bid"]
    return m.reset_index(drop=True)


def fit_rv_model(m: pd.DataFrame, with_iv: bool = False) -> dict:
    """Pooled OLS of log RV-to-expiry on log trailing vols (+ earnings-in-window flag) fitted on TRAIN only.
    Audit fix M3: only rows with earnings-calendar coverage (session >= 2020-01-22; in 2019 the flag was
    forced to 0, mislabelling ~24% of TRAIN), and an EMBARGO: a row is used only if its outcome (the expiry
    session) is inside TRAIN, so no label runs into VALIDATION."""
    cols = ["hv5", "hv21", "hv63", "rangevol21"] + (["iv_atm"] if with_iv else [])
    a, b = splits.window("options", "TRAIN")
    s = pd.to_datetime(m["session"])
    x = m[(s >= a) & (s <= b) & m["cal_covered"].astype(bool) & (pd.to_datetime(m["exp_session"]) <= b)]
    x = x.dropna(subset=cols + ["rv_to_exp"])
    x = x[(x[cols] > 0).all(axis=1) & (x["rv_to_exp"] > 0)]
    X = np.column_stack([np.ones(len(x))] + [np.log(x[c].to_numpy()) for c in cols] + [x["earn_in_window"].astype(float).to_numpy()])
    y = np.log(x["rv_to_exp"].to_numpy())
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    return {"cols": cols, "coef": coef.tolist(), "n_train": int(len(x)), "resid_var": float(resid.var()),
            "train_rows": "session in TRAIN, >= 2020-01-22 (calendar), expiry session <= TRAIN end (embargo)",
            "train_mean_log_rv_minus_log_iv": float(np.log(x["rv_to_exp"] / x["iv_atm"]).replace([np.inf, -np.inf], np.nan).mean())}


def predict_rv(m: pd.DataFrame, model: dict, kind: str = "median") -> pd.Series:
    """kind 'median': exp(E[log RV]) - the right scale for log-RV accuracy and for log(forecast / IV) sorts
    (a constant shift, so deciles are unchanged); 'mean': x exp(resid_var / 2), the lognormal mean
    (what the first run used everywhere; it inflated the forecast's log-MSE by (resid_var/2)^2)."""
    cols = model["cols"]
    X = np.column_stack([np.ones(len(m))] + [np.log(m[c].where(m[c] > 0).to_numpy()) for c in cols] + [m["earn_in_window"].astype(float).to_numpy()])
    z = X @ np.array(model["coef"]) + (model["resid_var"] / 2 if kind == "mean" else 0.0)
    return pd.Series(np.exp(z), index=m.index)


def driscoll_kraay_se(X: np.ndarray, res: np.ndarray, period: np.ndarray, lags: int) -> np.ndarray:
    """Driscoll-Kraay (1998) standard errors: per-period score sums, then a Bartlett-kernel HAC over the
    period series. Robust to cross-sectional correlation within a week AND to serial correlation across
    weeks (one-month outcomes sampled weekly overlap ~4-6 weeks; audit fix M4)."""
    order = np.unique(period)
    pos = {p: i for i, p in enumerate(order)}
    H = np.zeros((len(order), X.shape[1]))
    np.add.at(H, np.array([pos[p] for p in period]), X * res[:, None])
    S = H.T @ H
    for lag in range(1, lags + 1):
        G = H[lag:].T @ H[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))


def forecast_vs_iv(m: pd.DataFrame, fc: pd.Series, train_bias: float | None = None) -> dict:
    """Out-of-sample accuracy of the forecast vs IV for log RV, and the encompassing regression
    log RV = a + b_iv log IV + b_fc log forecast. t-stats: Driscoll-Kraay with 6 weekly lags (primary),
    12 lags (sensitivity) and the first run's week-clustered version (for comparison only).
    ``train_bias``: mean log(RV/IV) on TRAIN, used to debias IV without peeking at the evaluation split."""
    out = {}
    for sp in ("VALIDATION", "OOS"):
        a, b = splits.window("options", sp)
        x = m[(pd.to_datetime(m["session"]) >= a) & (pd.to_datetime(m["session"]) <= b)].copy()
        x["fc"] = fc.reindex(x.index)
        x = x.dropna(subset=["fc", "iv_atm", "rv_to_exp"])
        x = x[(x["iv_atm"] > 0) & (x["rv_to_exp"] > 0) & (x["fc"] > 0)]
        if len(x) < 100:
            continue
        ly, li, lf = np.log(x["rv_to_exp"]).to_numpy(), np.log(x["iv_atm"]).to_numpy(), np.log(x["fc"]).to_numpy()
        X = np.column_stack([np.ones(len(x)), li, lf])
        coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
        res = ly - X @ coef
        wk = pd.to_datetime(x["session"]).dt.to_period("W-FRI").astype(str).to_numpy()
        se6 = driscoll_kraay_se(X, res, wk, 6)
        se12 = driscoll_kraay_se(X, res, wk, 12)
        se_cl = driscoll_kraay_se(X, res, wk, 0)                       # lag 0 = clustered by week
        r = {"n": int(len(x)), "n_weeks": int(len(np.unique(wk))),
             "mse_log_iv": float(np.mean((ly - li) ** 2)),
             "mse_log_iv_debiased_oracle": float(np.mean((ly - li - (ly - li).mean()) ** 2)),
             "mse_log_forecast": float(np.mean((ly - lf) ** 2)),
             "encompassing_coef_iv": float(coef[1]), "encompassing_coef_fc": float(coef[2]),
             "t_iv": float(coef[1] / se6[1]), "t_fc": float(coef[2] / se6[2]),
             "t_iv_dk12": float(coef[1] / se12[1]), "t_fc_dk12": float(coef[2] / se12[2]),
             "t_iv_week_cluster": float(coef[1] / se_cl[1]), "t_fc_week_cluster": float(coef[2] / se_cl[2]),
             "mean_log_rv_minus_log_iv": float((ly - li).mean())}
        if train_bias is not None:
            r["mse_log_iv_debiased_train"] = float(np.mean((ly - li - train_bias) ** 2))
        out[sp] = r
    return out
