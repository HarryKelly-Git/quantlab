"""Volatility and move-magnitude forecast engine (sprint P2; docs/ALPHA-SPRINT-PREREG.md section 2). PAPER research.

This module covers MAGNITUDE and VOLATILITY (prediction problems B and C). Nothing here forecasts a
sign. Directional independence is measured, not assumed.

- **Features:** trailing only, computed at the origin close.
- **Targets:** outcomes over the next h sessions, never features.
- **Fitting:** every fitted quantity uses TRAIN rows only. That covers the HAR and QuantLab
  regressions, the GARCH (a, b), the scale-baseline calibration scalars and the Student-t degrees of
  freedom.

Forecasts are annualised volatilities. ``sigma_h x sqrt(h/252)`` is the standard deviation of the
h-session log move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

HORIZONS = (1, 3, 5, 10, 20)
FLOOR = 0.005                    # annualised RV floor for logs and QLIKE (0.5%); identical for every model
LAMBDA = 0.94
LARGE_MOVES = (0.02, 0.05, 0.10)
TAIL_Q = (0.90, 0.95, 0.99)
NU_GRID = tuple(range(3, 31))
BASELINES = ("HV21", "HV63", "EWMA", "GARCH", "HAR")
QL_FEATURES = ("l_rv1", "l_rv5", "l_rv22", "l_park21", "l_on21", "l_id21", "l_spy_hv21", "l_spy_ratio", "lmdv20",
               "ret21", "ret252", "dist_hi252", "l_maxabs21", "l_hv21_rel")
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=7)
ANN = math.sqrt(252.0)


def _lr(ret: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(ret.clip(lower=-0.99))


def _flog(x):
    return np.log(np.maximum(x, FLOOR))


# ------------------------------------------------------------------------------------------------
# recursions (numpy, one pass over dates; NaN return = no update)
# ------------------------------------------------------------------------------------------------
def ewma_var(lr2: np.ndarray, lam: float = LAMBDA, seed_n: int = 21) -> np.ndarray:
    """RiskMetrics variance known at the close of t (includes r_t). Seeded by the mean of the first
    ``seed_n`` squared returns of each series; NaN until then."""
    T, N = lr2.shape
    out = np.full((T, N), np.nan)
    s = np.full(N, np.nan)
    cnt = np.zeros(N)
    acc = np.zeros(N)
    for t in range(T):
        x = lr2[t]
        ok = np.isfinite(x)
        seeding = ok & np.isnan(s)
        acc[seeding] += x[seeding]
        cnt[seeding] += 1
        done = seeding & (cnt >= seed_n)
        s[done] = acc[done] / cnt[done]
        upd = ok & ~seeding & np.isfinite(s)
        s[upd] = lam * s[upd] + (1 - lam) * x[upd]
        out[t] = s
    return out


def garch_var(lr2: np.ndarray, target: np.ndarray, a: float, b: float) -> np.ndarray:
    """GARCH(1,1) one-step variance for t+1 known at the close of t, with variance targeting:
    s2[t+1] = (1-a-b) * V[t] + a * r2[t] + b * s2[t], V = trailing-252 variance (time-varying, PIT)."""
    T, N = lr2.shape
    out = np.full((T, N), np.nan)
    s = np.full(N, np.nan)
    w = 1.0 - a - b
    for t in range(T):
        x, v = lr2[t], target[t]
        ok = np.isfinite(x) & np.isfinite(v)
        init = ok & np.isnan(s)
        s[init] = v[init]
        upd = ok & ~init
        s[upd] = w * v[upd] + a * x[upd] + b * s[upd]
        out[t] = np.where(ok, s, np.nan)
    return out


def garch_hstep(s2_next: np.ndarray, v: np.ndarray, a: float, b: float, h: int) -> np.ndarray:
    """Mean daily variance over t+1..t+h from the GARCH(1,1) mean-reversion formula."""
    phi = a + b
    k = (1 - phi ** h) / ((1 - phi) * h) if phi < 1 else 1.0
    return v + (s2_next - v) * k


def fit_garch(lr2: np.ndarray, target: np.ndarray, seed: int = 7, n_stocks: int = 500) -> dict[str, float]:
    """Pooled Gaussian QMLE of (a, b), TRAIN rows only (caller passes TRAIN slices). Coarse grid over the whole
    stationary region (a 0.01-0.29, b 0.00-0.95), then a 0.01 grid around the coarse optimum. The first
    run used b >= 0.70 and hit that bound (TRAIN-only fit; stopped before any VAL/OOS metric); with a
    time-varying 252-session target, persistence is partly carried by the target itself."""
    rng = np.random.default_rng(seed)
    cols = np.flatnonzero(np.isfinite(lr2).sum(axis=0) > 300)
    cols = rng.choice(cols, size=min(n_stocks, len(cols)), replace=False)
    x, v = lr2[:, cols], target[:, cols]

    def nll(a: float, b: float) -> float:
        s2 = garch_var(x, v, a, b)
        pred, real = s2[:-1], x[1:]
        ok = np.isfinite(pred) & np.isfinite(real) & (pred > 0)
        return float(np.mean(np.log(pred[ok]) + real[ok] / pred[ok]))

    def search(a_grid, b_grid, best):
        for a in a_grid:
            for b in b_grid:
                if a <= 0 or b < 0 or a + b >= 0.995:
                    continue
                val = nll(float(a), float(b))
                if val < best[0]:
                    best = (val, (round(float(a), 4), round(float(b), 4)))
        return best

    best = search(np.arange(0.01, 0.30, 0.02), np.arange(0.0, 0.96, 0.05), (np.inf, None))
    a0, b0 = best[1]
    best = search(np.arange(a0 - 0.03, a0 + 0.035, 0.01), np.arange(b0 - 0.06, b0 + 0.065, 0.01), best)
    a, b = best[1]
    interior = 0.01 < a < 0.30 and 0.0 < b < 0.97 and a + b < 0.99
    return {"a": a, "b": b, "nll": best[0], "n_stocks": int(len(cols)), "interior": bool(interior)}


# ------------------------------------------------------------------------------------------------
# wide features and targets
# ------------------------------------------------------------------------------------------------
@dataclass
class Wide:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    f: dict[str, pd.DataFrame] = field(default_factory=dict)
    spy: dict[str, pd.Series] = field(default_factory=dict)


def features(p, u: pd.DataFrame, mdv20: pd.DataFrame, cols: pd.Index | None = None) -> Wide:
    """Trailing features at each close. ``p`` is an AlphaPanel (fields adj_open/high/low/close, ret_cc)."""
    cols = p.symbols if cols is None else cols
    ret = p["ret_cc"][cols]
    lr = _lr(ret)
    lr2 = lr ** 2
    ac, ao = p["adj_close"][cols], p["adj_open"][cols]
    hi, lo = p["adj_high"][cols], p["adj_low"][cols]
    W = Wide(p.dates, cols)
    f = W.f
    f["HV21"] = lr.rolling(21, min_periods=17).std() * ANN
    f["HV63"] = lr.rolling(63, min_periods=50).std() * ANN
    f["EWMA"] = pd.DataFrame(np.sqrt(ewma_var(lr2.to_numpy())) * ANN, index=p.dates, columns=cols)
    f["v252"] = lr.rolling(252, min_periods=200).var()                       # daily variance (GARCH target)
    f["lr2"] = lr2
    f["l_rv1"] = _flog(lr.abs() * ANN)
    f["l_rv5"] = _flog(np.sqrt(lr2.rolling(5, min_periods=4).mean() * 252))
    f["l_rv22"] = _flog(np.sqrt(lr2.rolling(22, min_periods=17).mean() * 252))
    pk = np.log(hi / lo) ** 2 / (4 * math.log(2))
    f["l_park21"] = _flog(np.sqrt(pk.rolling(21, min_periods=17).mean() * 252))
    on = np.log(ao / ac.shift(1))
    idr = np.log(ac / ao)
    f["l_on21"] = _flog(on.rolling(21, min_periods=17).std() * ANN)
    f["l_id21"] = _flog(idr.rolling(21, min_periods=17).std() * ANN)
    f["lmdv20"] = np.log(mdv20[cols].where(mdv20[cols] > 0))
    f["ret21"] = lr.rolling(21, min_periods=17).sum()
    f["ret252"] = lr.rolling(252, min_periods=200).sum()
    f["dist_hi252"] = np.log(ac / ac.rolling(252, min_periods=200).max())
    f["l_maxabs21"] = _flog(lr.abs().rolling(21, min_periods=17).max() * ANN)
    med = f["HV21"].where(u[cols]).median(axis=1)
    f["l_hv21_rel"] = np.log(f["HV21"].div(med, axis=0))
    s = _lr(p["ret_cc"]["SPY"])
    hv21 = s.rolling(21, min_periods=17).std() * ANN
    hv252 = s.rolling(252, min_periods=200).std() * ANN
    W.spy = {"l_spy_hv21": _flog(hv21), "l_spy_ratio": np.log(hv21 / hv252)}
    return W


def targets(ret_pnl: pd.DataFrame, cols: pd.Index, horizons=HORIZONS) -> dict[int, dict[str, pd.DataFrame]]:
    """Forward outcomes from ``ret_pnl`` (total returns with each delisting return booked on the next
    session). RV over the AVAILABLE return days in t+1..t+h (a delisting inside the window counts);
    move = compounded simple return over the same days."""
    lr = _lr(ret_pnl[cols])
    has = lr.notna()
    Q = (lr ** 2).fillna(0).cumsum()
    L = lr.fillna(0).cumsum()
    C = has.cumsum()
    out = {}
    for h in horizons:
        n = C.shift(-h) - C
        ss = Q.shift(-h) - Q
        move = np.expm1(L.shift(-h) - L)
        ok = n >= 1
        out[h] = {"rv": np.sqrt(252 * ss / n).where(ok), "move": move.where(ok), "n": n}
    return out


def weekly_origins(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period("W-FRI")).max().to_numpy())


def sample(W: Wide, pairs: pd.DataFrame) -> pd.DataFrame:
    """Long frame of features at (date, symbol) pairs."""
    di = W.dates.get_indexer(pairs["date"])
    si = W.symbols.get_indexer(pairs["symbol"])
    ok = (di >= 0) & (si >= 0)
    out = pairs.loc[ok, ["date", "symbol"]].reset_index(drop=True)
    di, si = di[ok], si[ok]
    for k, v in W.f.items():
        if k in ("lr2",):
            continue
        out[k] = v.to_numpy()[di, si].astype("float64")
    for k, v in W.spy.items():
        out[k] = v.to_numpy()[di]
    return out


# ------------------------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------------------------
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _ols_pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


@dataclass
class Fitted:
    h: int
    garch: dict[str, float]
    scal: dict[str, float]               # TRAIN median realised/forecast per scale baseline (raw -> calibrated)
    har: np.ndarray
    ql_ols: np.ndarray
    ql_ols_mu: np.ndarray
    ql_ols_sd: np.ndarray
    ql_hgb: Any
    smear: dict[str, float]              # TRAIN residual variance of log models
    nu: dict[str, int] = field(default_factory=dict)
    qlike_scalar: dict[str, float] = field(default_factory=dict)   # sensitivity: TRAIN QLIKE-optimal scale


def _har_X(df: pd.DataFrame) -> np.ndarray:
    return df[["l_rv1", "l_rv5", "l_rv22"]].to_numpy()


def _ql_X(df: pd.DataFrame) -> np.ndarray:
    return df[list(QL_FEATURES)].to_numpy()


def raw_forecasts(df: pd.DataFrame, h: int, garch: dict[str, float]) -> pd.DataFrame:
    """Scale baselines (annualised vol), uncalibrated. GARCH needs ``g_s2next`` (one-step variance)."""
    out = pd.DataFrame(index=df.index)
    out["HV21"], out["HV63"], out["EWMA"] = df["HV21"], df["HV63"], df["EWMA"]
    if "g_s2next" in df:
        out["GARCH"] = np.sqrt(garch_hstep(df["g_s2next"].to_numpy(), df["v252"].to_numpy(), garch["a"], garch["b"], h)
                               * 252)
    return out


def fit(train: pd.DataFrame, h: int, garch: dict[str, float]) -> Fitted:
    y = _flog(train[f"rv{h}"].to_numpy())
    raw = raw_forecasts(train, h, garch)
    scal = {k: float(np.nanmedian(train[f"rv{h}"].to_numpy() / raw[k].to_numpy())) for k in raw.columns}
    Xh = _har_X(train)
    ok = np.isfinite(Xh).all(axis=1) & np.isfinite(y)
    har = _ols(Xh[ok], y[ok])
    smear = {"HAR": float(np.var(y[ok] - _ols_pred(har, Xh[ok])))}
    Xq = _ql_X(train)
    okq = np.isfinite(Xq).all(axis=1) & np.isfinite(y)
    mu, sd = Xq[okq].mean(axis=0), Xq[okq].std(axis=0)
    sd[sd == 0] = 1.0
    b = _ols((Xq[okq] - mu) / sd, y[okq])
    smear["QL_OLS"] = float(np.var(y[okq] - _ols_pred(b, (Xq[okq] - mu) / sd)))
    from sklearn.ensemble import HistGradientBoostingRegressor
    okg = np.isfinite(y)
    hgb = HistGradientBoostingRegressor(**HGB_PARAMS).fit(Xq[okg], y[okg])     # native NaN handling
    smear["QL_HGB"] = float(np.var(y[okg] - hgb.predict(Xq[okg])))
    return Fitted(h, garch, scal, har, b, mu, sd, hgb, smear)


def predict(m: Fitted, df: pd.DataFrame, calibrated: bool = True) -> pd.DataFrame:
    """Annualised vol forecasts for every model. Scale baselines are multiplied by their TRAIN scalar when
    ``calibrated`` (pre-registered: information, not level bias); log models use exp(pred + s^2/2)."""
    out = raw_forecasts(df, m.h, m.garch)
    if calibrated:
        for k in list(out.columns):
            out[k] = out[k] * m.scal[k]
    Xh = _har_X(df)
    out["HAR"] = np.exp(_ols_pred(m.har, Xh) + m.smear["HAR"] / 2)
    Xq = _ql_X(df)
    out["QL_OLS"] = np.exp(_ols_pred(m.ql_ols, (Xq - m.ql_ols_mu) / m.ql_ols_sd) + m.smear["QL_OLS"] / 2)
    out["QL_HGB"] = np.exp(m.ql_hgb.predict(Xq) + m.smear["QL_HGB"] / 2)
    return out


# ------------------------------------------------------------------------------------------------
# distribution: |move_h| = sigma_h * sqrt(h/252) * |Z|, Z unit-variance Student-t
# ------------------------------------------------------------------------------------------------
def _k(nu: float) -> float:
    return math.sqrt((nu - 2.0) / nu)


def fit_nu(move: np.ndarray, sig: np.ndarray, h: int) -> int:
    s = sig * math.sqrt(h / 252.0)
    z = move / s
    z = z[np.isfinite(z)]
    best = (-np.inf, NU_GRID[-1])
    for nu in NU_GRID:
        k = _k(nu)
        ll = float(np.sum(sps.t.logpdf(z / k, nu) - math.log(k)))
        if ll > best[0]:
            best = (ll, nu)
    return int(best[1])


def p_exceed(sig: np.ndarray, h: int, nu: int, x: float) -> np.ndarray:
    s = sig * math.sqrt(h / 252.0) * _k(nu)
    return 2.0 * sps.t.sf(x / s, nu)


def abs_quantile(sig: np.ndarray, h: int, nu: int, q: float) -> np.ndarray:
    return sig * math.sqrt(h / 252.0) * _k(nu) * sps.t.ppf((1 + q) / 2, nu)


def expected_abs_move(sig: np.ndarray, h: int, nu: int) -> np.ndarray:
    """E|Z| for a unit-variance t: k * 2 sqrt(nu) Gamma((nu+1)/2) / (sqrt(pi) (nu-1) Gamma(nu/2))."""
    e = 2 * math.sqrt(nu) * math.gamma((nu + 1) / 2) / (math.sqrt(math.pi) * (nu - 1) * math.gamma(nu / 2))
    return sig * math.sqrt(h / 252.0) * _k(nu) * e


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    r2 = np.maximum(rv, FLOOR) ** 2
    f2 = np.maximum(f, FLOOR) ** 2
    return r2 / f2 - np.log(r2 / f2) - 1.0


def _auc(score: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(score) & np.isfinite(y)
    s, t = score[ok], y[ok].astype(bool)
    n1, n0 = int(t.sum()), int((~t).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = sps.rankdata(s)
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _date_ic(dates: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = pd.DataFrame({"d": dates, "a": a, "b": b}).dropna()
    d["ra"] = d.groupby("d")["a"].rank()
    d["rb"] = d.groupby("d")["b"].rank()
    ic = d.groupby("d")[["ra", "rb"]].corr().unstack().iloc[:, 1].dropna()
    return {"mean": float(ic.mean()), "share_positive": float((ic > 0).mean()), "n_dates": int(len(ic))}


def _block_ci(dates: np.ndarray, x: np.ndarray, n_boot: int = 500, seed: int = 7) -> list[float] | None:
    d = pd.DataFrame({"d": dates, "x": x}).dropna()
    g = d.groupby("d")["x"].agg(["sum", "count"])
    if len(g) < 5:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    m = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def evaluate(df: pd.DataFrame, fc: pd.DataFrame, h: int, nu: dict[str, int], ref: str = "HV21",
             regimes: dict[str, pd.Series] | None = None) -> dict[str, Any]:
    """Every pre-registered metric for one split and horizon. ``df`` holds rv{h}, move{h}, date, liq."""
    from quantlab.validation.stats import newey_west_tstat
    rv, mv = df[f"rv{h}"].to_numpy(), df[f"move{h}"].to_numpy()
    dates = df["date"].to_numpy()
    ok_t = np.isfinite(rv) & np.isfinite(mv)
    # one COMMON sample: a row counts only if every model has a forecast (fair comparison)
    ok_all = ok_t & np.all([np.isfinite(fc[k].to_numpy()) & (fc[k].to_numpy() > 0) for k in fc.columns], axis=0)
    out: dict[str, Any] = {"n": int(ok_all.sum()), "n_dates": int(pd.Series(dates[ok_all]).nunique()),
                           "rows_with_target": int(ok_t.sum()),
                           "model_coverage": {k: float((ok_t & np.isfinite(fc[k].to_numpy())).sum() / max(ok_t.sum(), 1))
                                              for k in fc.columns}, "models": {}}
    ql_by_date = {}
    for k in fc.columns:
        f = fc[k].to_numpy()
        ok = ok_all
        r, ff, mm, dd = rv[ok], f[ok], mv[ok], dates[ok]
        L = qlike(r, ff)
        ql_by_date[k] = pd.Series(L).groupby(dd).mean()
        lf, lr_ = np.log(np.maximum(ff, FLOOR)), np.log(np.maximum(r, FLOOR))
        mz = _ols(lf[:, None], lr_)
        res: dict[str, Any] = {
            "n": int(ok.sum()),
            "MAE": float(np.mean(np.abs(ff - r))), "RMSE": float(np.sqrt(np.mean((ff - r) ** 2))),
            "QLIKE": float(np.mean(L)), "bias_log": float(np.mean(lf - lr_)), "bias_level": float(np.mean(ff - r)),
            "MZ_intercept": float(mz[0]), "MZ_slope": float(mz[1]),
            "IC": _date_ic(dd, ff, r),
        }
        q = pd.qcut(pd.Series(ff).rank(method="first"), 10, labels=False)
        cal = pd.DataFrame({"q": q, "f": ff, "r": r}).groupby("q").agg(f=("f", "mean"), r=("r", "mean"), r_med=("r", "median"))
        res["calibration_deciles"] = cal.round(4).to_dict("list")
        res["calibration_monotone"] = bool(np.all(np.diff(cal["r"].to_numpy()) > 0))
        nk = nu.get(k, 5)
        probs = {}
        for x in LARGE_MOVES:
            p = p_exceed(ff, h, nk, x)
            y = (np.abs(mm) > x).astype(float)
            pq = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
            rel = pd.DataFrame({"q": pq, "p": p, "y": y}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"))
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            probs[f"gt_{int(x * 100)}pct"] = {
                "base_rate": float(y.mean()), "mean_pred": float(p.mean()), "brier": float(np.mean((p - y) ** 2)),
                "brier_climatology": float(np.mean((y.mean() - y) ** 2)),
                "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), "auc": _auc(p, y),
                "reliability": rel.round(4).to_dict("list")}
        res["large_move"] = probs
        tails = {}
        for qq in TAIL_Q:
            qa = abs_quantile(ff, h, nk, qq)
            e = (np.abs(mm) > qa).astype(float)
            tails[f"q{int(qq * 100)}"] = {"nominal_exceed": round(1 - qq, 4), "exceed_rate": float(e.mean()),
                                          "ci95_by_date": _block_ci(dd, e)}
        res["tail_coverage"] = tails
        em = expected_abs_move(ff, h, nk)
        res["expected_abs_move"] = {"mean_pred": float(em.mean()), "mean_real": float(np.abs(mm).mean()),
                                    "IC": _date_ic(dd, em, np.abs(mm))["mean"]}
        res["direction_independence"] = {"auc_sign": _auc(ff, (mm > 0).astype(float)),
                                         "ic_signed_move": _date_ic(dd, ff, mm)["mean"]}
        out["models"][k] = res
    base = ql_by_date.get(ref)
    for k, s in ql_by_date.items():
        if base is None or k == ref:
            continue
        d = (s - base).dropna()
        t = newey_west_tstat(d.to_numpy(), lags=4, min_obs=10)
        out["models"][k][f"DM_vs_{ref}"] = {"mean_diff": float(d.mean()), "t": float(t.t) if np.isfinite(t.t) else None}
    out["ql_by_date"] = ql_by_date
    return out


def dm(a: pd.Series, b: pd.Series) -> dict[str, float | None]:
    """Diebold-Mariano on weekly mean-QLIKE series (a - b < 0: a better), Newey-West 4 lags."""
    from quantlab.validation.stats import newey_west_tstat
    d = (a - b).dropna()
    t = newey_west_tstat(d.to_numpy(), lags=4, min_obs=10)
    return {"mean_diff": float(d.mean()), "t": float(t.t) if np.isfinite(t.t) else None, "n_weeks": int(len(d))}


def stability(df: pd.DataFrame, fc: pd.DataFrame, h: int, ref: str, groups: dict[str, np.ndarray]) -> dict[str, Any]:
    """QLIKE of each model relative to ``ref`` (ratio of means) within each group value."""
    rv = df[f"rv{h}"].to_numpy()
    out: dict[str, Any] = {}
    L = {k: qlike(rv, fc[k].to_numpy()) for k in fc.columns}   # NaN where a forecast is missing: excluded below
    for gname, g in groups.items():
        tab = {}
        for val in pd.unique(g[pd.notna(g)]):
            m = (g == val) & np.isfinite(rv)
            m = m & np.all([np.isfinite(L[k]) for k in fc.columns], axis=0)
            if m.sum() < 500:
                continue
            r0 = float(np.mean(L[ref][m]))
            tab[str(val)] = {k: round(float(np.mean(L[k][m])) / r0, 4) for k in fc.columns} | {"n": int(m.sum())}
        out[gname] = tab
    return out
