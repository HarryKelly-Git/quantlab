"""H12 machine-learning TARGET SEPARATION (Parts 27-28). PRE-REGISTERED (fixed before any fit):

Observations: every Friday close (or the last session of the week), liquid survivorship-free universe.
Features (all known at that close; PIT): ret_5, ret_21, ret_63, mom_12_1, resid_21 (two-factor residual
return), vol20, vol60, vol_ratio (vol20/vol60), rangevol21 (Parkinson), volume_ratio (20d/60d), log MDV20,
dist_52w_high, max21, ivol21, beta_mkt, sector_mom63 (own statistical-sector ETF 63d return), spy_vol20,
spy_trend (SPY / SMA200 - 1).
Targets (from the NEXT OPEN, market-adjusted with beta):
  DIRECTION  1{5-session market-adjusted return > 0}                    -> AUC
  MAGNITUDE  |5-session market-adjusted return|                         -> weekly rank IC (Spearman)
  VOLATILITY realised vol of the next 21 sessions                         -> weekly rank IC, R^2 of log
  TAIL       1{|21-session return| > 10%}                                -> AUC
  TIMING     1{+5% reached before -5% within 10 sessions, on closes}     -> AUC (only rows where one is hit)
Models: logistic (classification) / ridge (regression), random forest, histogram gradient boosting.
Fit on TRAIN (2016-2019), report VALIDATION (2020-2021), OOS (2022-2024) once. Baselines: trailing vol20
for MAGNITUDE / VOLATILITY / TAIL, 0.5 for DIRECTION / TIMING.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quantlab.alpha import splits


FEATURES = ["ret_5", "ret_21", "ret_63", "mom_12_1", "resid_21", "vol20", "vol60", "vol_ratio", "rangevol21",
            "volume_ratio", "log_mdv20", "dist_52w_high", "max21", "ivol21", "beta_mkt", "sector_mom63",
            "spy_vol20", "spy_trend"]


def build_dataset(d) -> pd.DataFrame:
    p = d.p
    r = p["ret_cc"]
    ac = p["adj_close"]
    week_last = p.dates.to_series().groupby(p.dates.to_period("W-FRI")).last()
    obs_dates = pd.DatetimeIndex(week_last.to_numpy())
    feats = {
        "ret_5": r.rolling(5).sum(), "ret_21": r.rolling(21).sum(), "ret_63": r.rolling(63).sum(),
        "mom_12_1": r.rolling(231, min_periods=180).sum().shift(21), "resid_21": d.resid.rolling(21, min_periods=17).sum(),
        "vol20": d.vol20, "vol60": d.vol60, "vol_ratio": d.vol20 / d.vol60,
        "rangevol21": (np.log(p["adj_high"] / p["adj_low"]) ** 2 / (4 * np.log(2))).rolling(21, min_periods=17).mean() ** 0.5,
        "volume_ratio": p["volume"].rolling(20).mean() / p["volume"].rolling(60).mean(),
        "log_mdv20": np.log(d.mdv20), "dist_52w_high": ac / ac.rolling(252, min_periods=200).max() - 1,
        "max21": r.rolling(21, min_periods=17).max(), "ivol21": d.resid.rolling(21, min_periods=17).std(),
        "beta_mkt": d.betas["beta_mkt"],
    }
    from quantlab.alpha.panel import SECTOR_ETFS
    sec63 = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for e in SECTOR_ETFS:
        if e in ac.columns:
            sec63 = sec63.mask(d.sectors == e, pd.DataFrame(np.broadcast_to(ac[e].pct_change(63).to_numpy()[:, None], r.shape),
                                                             index=r.index, columns=r.columns))
    feats["sector_mom63"] = sec63
    spy = ac["SPY"]
    mkt = {"spy_vol20": r["SPY"].rolling(20).std(), "spy_trend": spy / spy.rolling(200, min_periods=200).mean() - 1}
    # targets from the next open (adjusted prices), market-adjusted with beta
    ao = p["adj_open"]
    fwd5 = ac.shift(-5) / ao.shift(-1) - 1
    fwd5_m = (ac["SPY"].shift(-5) / ao["SPY"].shift(-1) - 1)
    fwd5_adj = fwd5 - d.betas["beta_mkt"].fillna(1.0).mul(fwd5_m, axis=0)
    fwd21 = ac.shift(-21) / ao.shift(-1) - 1
    lr = np.log(ac / ac.shift(1))
    fvol21 = lr[::-1].rolling(21, min_periods=17).std()[::-1].shift(-1) * np.sqrt(252)
    # timing: +5% before -5% within 10 sessions (closes vs next open)
    rows = []
    U = d.u_liquid
    for t in obs_dates:
        if t not in r.index:
            continue
        i = r.index.get_loc(t)
        u = U.loc[t]
        names = u.index[u.to_numpy()]
        if len(names) < 100:
            continue
        df = pd.DataFrame({k: v.loc[t, names] for k, v in feats.items()})
        for k, v in mkt.items():
            df[k] = float(v.loc[t])
        df["date"] = t
        df["symbol"] = names
        df["y_dir"] = (fwd5_adj.loc[t, names] > 0).astype(float).where(fwd5_adj.loc[t, names].notna())
        df["y_mag"] = fwd5_adj.loc[t, names].abs()
        df["y_vol"] = fvol21.loc[t, names]
        df["y_tail"] = (fwd21.loc[t, names].abs() > 0.10).astype(float).where(fwd21.loc[t, names].notna())
        if i + 11 < len(r.index):
            entry = ao.iloc[i + 1][names]
            path = ac.iloc[i + 1: i + 11][names] / entry - 1
            up = (path >= 0.05).to_numpy()
            dn = (path <= -0.05).to_numpy()
            first_up = np.where(up.any(axis=0), up.argmax(axis=0), 99)
            first_dn = np.where(dn.any(axis=0), dn.argmax(axis=0), 99)
            y = np.where((first_up == 99) & (first_dn == 99), np.nan, (first_up < first_dn).astype(float))
            df["y_timing"] = y
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["split"] = splits.label_series(pd.DatetimeIndex(out["date"]), "equity").to_numpy()
    return out


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(y) & np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 100 and len(np.unique(y[ok])) == 2 else None


def _weekly_ic(df: pd.DataFrame, pred: np.ndarray, y: str) -> float | None:
    x = df.assign(pred=pred)[["date", "pred", y]].dropna()
    if x.empty:
        return None
    ics = x.groupby("date").apply(lambda g: g["pred"].rank().corr(g[y].rank()) if len(g) > 30 else np.nan)
    return float(ics.mean())


def run(d, ds: pd.DataFrame | None = None) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    ds = build_dataset(d) if ds is None else ds
    out = {"n_rows": int(len(ds)), "build_seconds": round(time.time() - t0, 1)}
    targets = {"DIRECTION": ("y_dir", "clf"), "MAGNITUDE": ("y_mag", "reg"), "VOLATILITY": ("y_vol", "reg"),
               "TAIL": ("y_tail", "clf"), "TIMING": ("y_timing", "clf")}
    tr = ds[ds["split"] == "TRAIN"]
    for tname, (y, kind) in targets.items():
        trn = tr.dropna(subset=[y])
        if len(trn) > 400_000:
            trn = trn.sample(400_000, random_state=7)
        X = trn[FEATURES].to_numpy()
        Y = trn[y].to_numpy()
        if tname == "VOLATILITY":
            Y = np.log(np.maximum(Y, 1e-4))
        models = ({"linear": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=300)),
                   "random_forest": make_pipeline(SimpleImputer(), RandomForestClassifier(200, min_samples_leaf=200, n_jobs=3, random_state=7)),
                   "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=7)}
                  if kind == "clf" else
                  {"linear": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
                   "random_forest": make_pipeline(SimpleImputer(), RandomForestRegressor(200, min_samples_leaf=200, n_jobs=3, random_state=7)),
                   "gradient_boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=7)})
        res = {}
        for mname, mdl in models.items():
            mdl.fit(X, Y)
            row = {}
            for sp in ("TRAIN", "VALIDATION", "OOS"):
                te = ds[(ds["split"] == sp)].dropna(subset=[y])
                if te.empty:
                    continue
                Xt = te[FEATURES].to_numpy()
                if kind == "clf":
                    s = mdl.predict_proba(Xt)[:, 1]
                    row[sp] = {"auc": _auc(te[y].to_numpy(), s), "n": int(len(te))}
                else:
                    pr = mdl.predict(Xt)
                    row[sp] = {"weekly_rank_ic": _weekly_ic(te, pr, y), "n": int(len(te))}
            res[mname] = row
        # baselines
        base = {}
        for sp in ("TRAIN", "VALIDATION", "OOS"):
            te = ds[(ds["split"] == sp)].dropna(subset=[y])
            if te.empty:
                continue
            if kind == "clf":
                base[sp] = {"auc_vol20": _auc(te[y].to_numpy(), te["vol20"].to_numpy()), "auc_random": 0.5}
            else:
                base[sp] = {"weekly_rank_ic_vol20": _weekly_ic(te, te["vol20"].to_numpy(), y)}
        res["baseline"] = base
        out[tname] = res
        print(tname, {m: {sp: v for sp, v in r_.items()} for m, r_ in res.items()}, flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out
