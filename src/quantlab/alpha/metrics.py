"""The full metric suite for one daily return series (Part 3 of the alpha brief).

Inputs are DAILY returns of a strategy (net of costs) and optionally a benchmark over the SAME days.
Everything is computed from those two series, plus (optionally) per-name P&L contributions for the
trade-level statistics and the "remove the top x% of trades" robustness check.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as ss

from quantlab.validation.stats import newey_west_tstat

PPY = 252


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    e = equity_curve(r)
    return e / e.cummax() - 1.0


def core_metrics(r: pd.Series, bench: pd.Series | None = None, *, rf: pd.Series | None = None) -> dict[str, Any]:
    r = r.dropna()
    n = len(r)
    out: dict[str, Any] = {"n_days": n}
    if n < 20:
        out["status"] = "INSUFFICIENT_SAMPLE"
        return out
    years = n / PPY
    total = float(equity_curve(r).iloc[-1] - 1.0)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    dd = drawdown(r)
    mdd = float(dd.min())
    # drawdown duration (days below a previous peak)
    under = (dd < 0).astype(int).to_numpy()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    q05 = float(np.quantile(r, 0.05))
    q95 = float(np.quantile(r, 0.95))
    nw = newey_west_tstat(r.to_numpy())
    out.update(
        total_return=total, cagr=cagr, ann_vol=sd * math.sqrt(PPY),
        sharpe=mu / sd * math.sqrt(PPY) if sd > 0 else None,
        sortino=mu / downside * math.sqrt(PPY) if downside > 0 else None,
        calmar=cagr / abs(mdd) if mdd < 0 else None,
        max_drawdown=mdd, max_dd_days=longest,
        skew=float(ss.skew(r)), kurtosis=float(ss.kurtosis(r, fisher=False)),
        tail_loss_cvar5=float(r[r <= q05].mean()), tail_gain_cvar95=float(r[r >= q95].mean()),
        worst_day=float(r.min()), best_day=float(r.max()),
        worst_month=float(monthly.min()), best_month=float(monthly.max()),
        pct_positive_days=float((r > 0).mean()), pct_positive_months=float((monthly > 0).mean()),
        mean_daily=mu, t_mean_nw=nw.t, p_mean_nw=nw.p_two_sided,
    )
    roll = r.rolling(PPY, min_periods=PPY)
    rs = (roll.mean() / roll.std() * math.sqrt(PPY)).dropna()
    if len(rs):
        out.update(rolling_sharpe_min=float(rs.min()), rolling_sharpe_median=float(rs.median()),
                   rolling_sharpe_max=float(rs.max()), rolling_sharpe_pct_positive=float((rs > 0).mean()))
    if bench is not None:
        b = bench.reindex(r.index)
        ok = b.notna()
        if ok.sum() > 20:
            x, y = b[ok].to_numpy(), r[ok].to_numpy()
            beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
            resid = y - beta * x
            a = newey_west_tstat(resid)
            active = y - x
            te = float(np.std(active, ddof=1))
            bt = float((1 + pd.Series(x)).prod() - 1)
            out.update(beta=beta, alpha_ann=float(np.mean(resid) * PPY), alpha_t_nw=a.t, alpha_p_nw=a.p_two_sided,
                       corr_bench=float(np.corrcoef(y, x)[0, 1]),
                       information_ratio=float(np.mean(active) / te * math.sqrt(PPY)) if te > 0 else None,
                       bench_total_return=bt, excess_total_return=total - bt)
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def by_period(r: pd.Series, freq: str = "Y") -> pd.DataFrame:
    g = r.groupby(r.index.to_period(freq))
    return pd.DataFrame({
        "return": g.apply(lambda x: float((1 + x).prod() - 1)),
        "sharpe": g.apply(lambda x: float(x.mean() / x.std() * math.sqrt(PPY)) if x.std() > 0 else np.nan),
        "max_dd": g.apply(lambda x: float(drawdown(x).min())),
        "n_days": g.size(),
    })


def by_label(r: pd.Series, labels: pd.Series) -> pd.DataFrame:
    lab = labels.reindex(r.index)
    rows = {}
    for k, x in r.groupby(lab):
        if len(x) < 20:
            continue
        rows[k] = {"n_days": len(x), "mean_daily_bps": float(x.mean() * 1e4),
                   "ann_return": float(x.mean() * PPY), "sharpe": float(x.mean() / x.std() * math.sqrt(PPY)) if x.std() > 0 else None,
                   "t_nw": newey_west_tstat(x.to_numpy()).t}
    return pd.DataFrame(rows).T


# --- trade-level view from per-name contributions ---------------------------------------------------
def spells(weights: pd.DataFrame, contrib: pd.DataFrame) -> pd.DataFrame:
    """Position spells: consecutive days a name is held with the same sign. One row per spell with its
    summed gross contribution (fraction of book) and length. Vectorised per column."""
    rows = []
    W = weights.to_numpy()
    C = contrib.reindex_like(weights).fillna(0.0).to_numpy()
    for j, name in enumerate(weights.columns):
        s = np.sign(W[:, j])
        if not s.any():
            continue
        change = np.r_[True, s[1:] != s[:-1]]
        ids = np.cumsum(change)
        held = s != 0
        if not held.any():
            continue
        df = pd.DataFrame({"id": ids[held], "c": C[held, j], "side": s[held]})
        g = df.groupby("id").agg(pnl=("c", "sum"), days=("c", "size"), side=("side", "first"))
        g["name"] = name
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["pnl", "days", "side", "name"])
    return pd.concat(rows, ignore_index=True)


def trade_stats(sp: pd.DataFrame) -> dict[str, Any]:
    if sp.empty:
        return {"n_trades": 0}
    p = sp["pnl"]
    wins, losses = p[p > 0], p[p < 0]
    return {"n_trades": int(len(p)), "win_rate": float((p > 0).mean()), "expectancy": float(p.mean()),
            "avg_win": _f(wins.mean()), "avg_loss": _f(losses.mean()),
            "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
            "avg_holding_days": float(sp["days"].mean())}


def remove_top_trades(net: pd.Series, weights: pd.DataFrame, contrib: pd.DataFrame,
                      fractions=(0.01, 0.05, 0.10)) -> dict[str, Any]:
    """Recompute total return and Sharpe after deleting the best x% of position spells' contributions."""
    sp_rows = []
    W = weights.to_numpy()
    C = contrib.reindex_like(weights).fillna(0.0).to_numpy()
    T = len(weights)
    for j in range(W.shape[1]):
        s = np.sign(W[:, j])
        if not s.any():
            continue
        change = np.r_[True, s[1:] != s[:-1]]
        ids = np.cumsum(change)
        held = np.nonzero(s != 0)[0]
        if len(held) == 0:
            continue
        sid = ids[held]
        pnl = pd.Series(C[held, j]).groupby(sid).sum()
        for k, v in pnl.items():
            sp_rows.append((j, int(k), float(v)))
    if not sp_rows:
        return {}
    sp = pd.DataFrame(sp_rows, columns=["j", "sid", "pnl"]).sort_values("pnl", ascending=False)
    out = {}
    for frac in fractions:
        k = max(1, int(round(frac * len(sp))))
        top = sp.head(k)
        adj = np.zeros(T)
        for j, grp in top.groupby("j"):
            s = np.sign(W[:, j])
            ids = np.cumsum(np.r_[True, s[1:] != s[:-1]])
            mask = np.isin(ids, grp["sid"].to_numpy()) & (s != 0)
            adj += np.where(mask, C[:, j], 0.0)
        r2 = net - pd.Series(adj, index=net.index)
        m = core_metrics(r2)
        out[f"without_top_{int(frac * 100)}pct"] = {"total_return": float((1 + r2.fillna(0.0)).prod() - 1.0),
                                                     "mean_daily": float(r2.mean()), "sharpe": m.get("sharpe"),
                                                     "t_nw": m.get("t_mean_nw"), "n_removed": k}
    return out


def bucket_contrib(contrib: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    """Daily gross P&L split by a per-(date, name) bucket label (size / volatility / liquidity)."""
    b = buckets.reindex_like(contrib)
    out = {}
    for lab in pd.unique(b.to_numpy().ravel()):
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            continue
        out[lab] = contrib.where(b == lab).sum(axis=1)
    return pd.DataFrame(out)
