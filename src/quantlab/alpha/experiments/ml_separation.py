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
    """Memory-lean: each (dates x names) matrix is computed, sampled at the weekly observation dates for
    names ever in the universe, and freed before the next one (the full set at once needs > 13 GB)."""
    import gc
    from quantlab.alpha.panel import SECTOR_ETFS
    p = d.p
    U = d.u_liquid
    cols = U.columns[U.any(axis=0).to_numpy()]
    week_last = p.dates.to_series().groupby(p.dates.to_period("W-FRI")).last()
    obs = pd.DatetimeIndex([t for t in week_last.to_numpy() if t in p.dates])
    r = p["ret_cc"][cols]
    ac = p["adj_close"]
    S: dict[str, pd.DataFrame] = {}

    def keep(name, mat):
        S[name] = mat.reindex(index=obs, columns=cols).astype("float32")

    keep("ret_5", r.rolling(5).sum()); keep("ret_21", r.rolling(21).sum()); keep("ret_63", r.rolling(63).sum())
    keep("mom_12_1", r.rolling(231, min_periods=180).sum().shift(21))
    res = d.resid[cols]
    keep("resid_21", res.rolling(21, min_periods=17).sum()); keep("ivol21", res.rolling(21, min_periods=17).std())
    del res; gc.collect()
    keep("vol20", d.vol20[cols]); keep("vol60", d.vol60[cols]); S["vol_ratio"] = S["vol20"] / S["vol60"]
    hl = np.log(p["adj_high"][cols] / p["adj_low"][cols]) ** 2 / (4 * np.log(2))
    keep("rangevol21", hl.rolling(21, min_periods=17).mean() ** 0.5); del hl; gc.collect()
    v = p["volume"][cols]
    keep("volume_ratio", v.rolling(20).mean() / v.rolling(60).mean()); del v; gc.collect()
    keep("log_mdv20", np.log(d.mdv20[cols]))
    a = ac[cols]
    keep("dist_52w_high", a / a.rolling(252, min_periods=200).max() - 1)
    keep("max21", r.rolling(21, min_periods=17).max())
    keep("beta_mkt", d.betas["beta_mkt"][cols])
    sec_at = d.sectors.reindex(index=obs, columns=cols)
    sec63 = pd.DataFrame(np.nan, index=obs, columns=cols, dtype="float32")
    for e in SECTOR_ETFS:
        if e in ac.columns:
            val = ac[e].pct_change(63).reindex(obs).to_numpy()[:, None]
            sec63 = sec63.mask(sec_at == e, pd.DataFrame(np.broadcast_to(val, sec63.shape), index=obs, columns=cols))
    S["sector_mom63"] = sec63
    spy = ac["SPY"]
    spy_vol20 = p["ret_cc"]["SPY"].rolling(20).std().reindex(obs)
    spy_trend = (spy / spy.rolling(200, min_periods=200).mean() - 1).reindex(obs)
    # targets (outcomes), from the next open, beta-adjusted with SPY
    ao = p["adj_open"]
    nxt_open = ao[cols].shift(-1)
    fwd5 = ac[cols].shift(-5) / nxt_open - 1
    fwd5_m = ac["SPY"].shift(-5) / ao["SPY"].shift(-1) - 1
    T: dict[str, pd.DataFrame] = {}
    T["fwd5_adj"] = (fwd5 - d.betas["beta_mkt"][cols].fillna(1.0).mul(fwd5_m, axis=0)).reindex(obs).astype("float32")
    del fwd5; gc.collect()
    T["fwd21"] = (ac[cols].shift(-21) / nxt_open - 1).reindex(obs).astype("float32")
    lr = np.log(a / a.shift(1))
    T["fvol21"] = (lr[::-1].rolling(21, min_periods=17).std()[::-1].shift(-1) * np.sqrt(252)).reindex(obs).astype("float32")
    del lr; gc.collect()
    rows = []
    pos = {t: i for i, t in enumerate(p.dates)}
    for t in obs:
        u = U.loc[t, cols]
        names = cols[u.to_numpy()]
        if len(names) < 100:
            continue
        df = pd.DataFrame({k: v.loc[t, names].to_numpy() for k, v in S.items()})
        df["spy_vol20"] = float(spy_vol20.loc[t]); df["spy_trend"] = float(spy_trend.loc[t])
        df["date"] = t
        df["symbol"] = names
        f5 = T["fwd5_adj"].loc[t, names].to_numpy()
        df["y_dir"] = np.where(np.isfinite(f5), (f5 > 0).astype(float), np.nan)
        df["y_mag"] = np.abs(f5)
        df["y_vol"] = T["fvol21"].loc[t, names].to_numpy()
        f21 = T["fwd21"].loc[t, names].to_numpy()
        df["y_tail"] = np.where(np.isfinite(f21), (np.abs(f21) > 0.10).astype(float), np.nan)
        i = pos[t]
        if i + 11 < len(p.dates):
            entry = ao.iloc[i + 1][names].to_numpy()
            path = ac.iloc[i + 1: i + 11][names].to_numpy() / entry - 1
            up, dn = path >= 0.05, path <= -0.05
            fu = np.where(up.any(axis=0), up.argmax(axis=0), 99)
            fd = np.where(dn.any(axis=0), dn.argmax(axis=0), 99)
            df["y_timing"] = np.where((fu == 99) & (fd == 99), np.nan, (fu < fd).astype(float))
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
