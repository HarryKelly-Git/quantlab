"""Portfolio construction across strategy return streams (Parts 29-30) and strategy-decay monitoring
(Part 34). PAPER research.

Weights are estimated on one period (e.g. TRAIN) and evaluated, fixed, on later periods; a walk-forward
variant re-estimates once a year on an expanding window. Every method is long-only in strategies, sums to 1:
  equal | inverse_vol | risk_parity (equal risk contribution) | max_diversification |
  min_variance (weights capped at ``cap``) | mean_variance (constrained, risk aversion ``lam``) |
  kelly_capped (fraction ``kelly_frac`` of full Kelly, projected to the simplex)
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _cov(R: pd.DataFrame) -> np.ndarray:
    return np.cov(R.to_numpy(), rowvar=False, ddof=1)


def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def _opt(obj: Callable[[np.ndarray], float], n: int, cap: float = 1.0) -> np.ndarray:
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}], options={"maxiter": 500, "ftol": 1e-12})
    return _project_simplex(res.x if res.success else x0)


def weights(R: pd.DataFrame, method: str, *, cap: float = 0.5, lam: float = 5.0, kelly_frac: float = 0.25) -> pd.Series:
    n = R.shape[1]
    S = _cov(R)
    vol = np.sqrt(np.diag(S))
    mu = R.mean().to_numpy()
    if method == "equal":
        w = np.full(n, 1.0 / n)
    elif method == "inverse_vol":
        w = _project_simplex(1.0 / np.where(vol > 0, vol, np.inf))
    elif method == "risk_parity":
        def obj(w):
            rc = w * (S @ w)
            return float(((rc - rc.mean()) ** 2).sum()) * 1e8
        w = _opt(obj, n)
    elif method == "max_diversification":
        w = _opt(lambda w: -float(w @ vol) / math.sqrt(max(float(w @ S @ w), 1e-18)), n)
    elif method == "min_variance":
        w = _opt(lambda w: float(w @ S @ w) * 1e6, n, cap=cap)
    elif method == "mean_variance":
        w = _opt(lambda w: -(float(w @ mu) - lam / 2 * float(w @ S @ w)) * 1e4, n, cap=cap)
    elif method == "kelly_capped":
        try:
            full = np.linalg.solve(S + 1e-10 * np.eye(n), mu)
        except np.linalg.LinAlgError:
            full = np.zeros(n)
        w = _project_simplex(kelly_frac * full)
    else:
        raise ValueError(method)
    return pd.Series(w, index=R.columns)


METHODS = ("equal", "inverse_vol", "risk_parity", "max_diversification", "min_variance", "mean_variance", "kelly_capped")


def walk_forward(R: pd.DataFrame, method: str, first_fit_end: str, refit: str = "Y", **kw) -> pd.Series:
    """Expanding-window weights refit at each period end (default yearly) from ``first_fit_end``;
    each fit uses only returns up to its fit date and applies to the following period."""
    R = R.dropna(how="all").fillna(0.0)
    out = pd.Series(np.nan, index=R.index)
    ends = R.loc[first_fit_end:].index.to_series().groupby(R.loc[first_fit_end:].index.to_period(refit)).last()
    fit_dates = [pd.Timestamp(first_fit_end)] + list(ends.iloc[:-1])
    for i, fd in enumerate(fit_dates):
        w = weights(R.loc[:fd], method, **kw)
        nxt = fit_dates[i + 1] if i + 1 < len(fit_dates) else R.index[-1]
        seg = R.loc[(R.index > fd) & (R.index <= nxt)]
        out.loc[seg.index] = seg.to_numpy() @ w.to_numpy()
    return out.dropna()


# --- decay monitoring --------------------------------------------------------------------------------
def rolling_health(r: pd.Series, window: int = 126) -> pd.DataFrame:
    roll = r.rolling(window, min_periods=int(0.8 * window))
    return pd.DataFrame({"rolling_sharpe": roll.mean() / roll.std() * math.sqrt(252), "rolling_hit_rate": (r > 0).astype(float).rolling(window).mean(),
                         "rolling_mean_bps": roll.mean() * 1e4})


def decay_alerts(r: pd.Series, *, turnover: pd.Series | None = None, beta: pd.Series | None = None, window: int = 126,
                 reference_sharpe: float | None = None, sharpe_floor: float = 0.0, sharpe_drop: float = 1.0,
                 turnover_change: float = 0.5, beta_drift: float = 0.3) -> list[dict]:
    """Alerts on the latest window vs the reference (default: the first window of history):
    rolling Sharpe below ``sharpe_floor``, a drop of more than ``sharpe_drop`` Sharpe units, turnover up or
    down by more than ``turnover_change`` (relative), beta moved more than ``beta_drift``."""
    h = rolling_health(r, window).dropna()
    alerts: list[dict] = []
    if h.empty:
        return [{"alert": "INSUFFICIENT_HISTORY"}]
    ref = reference_sharpe if reference_sharpe is not None else float(h["rolling_sharpe"].iloc[0])
    cur = float(h["rolling_sharpe"].iloc[-1])
    if cur < sharpe_floor:
        alerts.append({"alert": "SHARPE_BELOW_FLOOR", "current": cur, "floor": sharpe_floor})
    if ref - cur > sharpe_drop:
        alerts.append({"alert": "SHARPE_DROP", "reference": ref, "current": cur})
    hr0, hr1 = float(h["rolling_hit_rate"].iloc[0]), float(h["rolling_hit_rate"].iloc[-1])
    if hr0 - hr1 > 0.05:
        alerts.append({"alert": "HIT_RATE_DROP", "reference": hr0, "current": hr1})
    if turnover is not None and len(turnover.dropna()) > 2 * window:
        t0, t1 = float(turnover.iloc[:window].mean()), float(turnover.iloc[-window:].mean())
        if t0 > 0 and abs(t1 / t0 - 1) > turnover_change:
            alerts.append({"alert": "TURNOVER_CHANGE", "reference": t0, "current": t1})
    if beta is not None and len(beta.dropna()) > 2 * window:
        b0, b1 = float(beta.dropna().iloc[:window].mean()), float(beta.dropna().iloc[-window:].mean())
        if abs(b1 - b0) > beta_drift:
            alerts.append({"alert": "BETA_DRIFT", "reference": b0, "current": b1})
    return alerts
