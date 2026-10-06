"""Cross-sectional portfolio backtester for alpha research (weights in, daily P&L out). PAPER research.

Not a duplicate of ``backtest/engine.py``: that engine simulates individual trade plans (entries,
stops, targets) for the bot. This one answers factor-style questions ("do last week's losers
outperform?") with portfolios of many names, as the academic literature does, and reports what the
same weights cost to trade.

Timing (point-in-time):
  * ``weights`` row t = target portfolio decided with information up to the CLOSE of t.
  * execution='open'  : trade at the open of t+1; held open(t+1) -> open(t+2); P&L = w_t . ret_oo[t+1].
  * execution='close' : trade at the close of t+1 (a MOC order placed on t+1); P&L = w_t . ret_cc[t+2].
  * Overlapping holding period h: the traded book is the mean of the last h target books
    (Jegadeesh-Titman), i.e. each day's cohort is held h days.
Costs: |change in weight| x (half-spread tier by 20d median dollar volume + slippage), per name per
trade (the CostModel tiers from config), x ``cost_mult`` for sensitivity; shorts pay ``borrow_bps``
a year on short gross. Optional square-root market impact for capacity at a given AUM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TIERS = ((100e6, 2.0), (20e6, 5.0), (5e6, 10.0), (0.0, 25.0))   # config/default.yaml costs
DEFAULT_SLIPPAGE_BPS = 5.0


def cost_bps_matrix(mdv: pd.DataFrame, tiers=DEFAULT_TIERS, slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> pd.DataFrame:
    """One-way cost in bps per (date, name): half-spread tier by median dollar volume + slippage.
    Unknown liquidity pays the worst tier (as core.costs.CostModel)."""
    out = pd.DataFrame(tiers[-1][1], index=mdv.index, columns=mdv.columns, dtype="float64")
    for thr, bps in sorted(tiers, key=lambda x: x[0]):          # ascending: later (higher) tiers overwrite
        out = out.mask(mdv >= thr, bps)
    return out + slippage_bps


@dataclass
class BacktestResult:
    gross: pd.Series                  # daily portfolio return before costs (indexed by P&L date)
    net: pd.Series                    # after costs (and borrow)
    costs: pd.Series
    turnover: pd.Series               # sum |dw| traded that day (one side counts once)
    gross_exposure: pd.Series
    net_exposure: pd.Series
    n_long: pd.Series
    n_short: pd.Series
    weights: pd.DataFrame             # traded book, indexed by the DECISION date t
    contrib: pd.DataFrame | None      # per-name gross P&L contribution (dates x names), optional
    meta: dict[str, Any] = field(default_factory=dict)


def run_weights(weights: pd.DataFrame, ret_oo: pd.DataFrame, cost_bps: pd.DataFrame, *, holding: int = 1,
                execution: str = "open", ret_cc: pd.DataFrame | None = None, cost_mult: float = 1.0,
                borrow_bps: float = 50.0, keep_contrib: bool = False, impact: dict | None = None) -> BacktestResult:
    if execution not in ("open", "close"):
        raise ValueError(execution)
    W = weights.fillna(0.0)
    if holding > 1:
        W = W.rolling(holding, min_periods=1).mean()
    if execution == "open":
        fwd = ret_oo.shift(-1)                  # row t -> return earned from open t+1 to open t+2
    else:
        if ret_cc is None:
            raise ValueError("execution='close' needs ret_cc")
        fwd = ret_cc.shift(-2)                  # row t -> close t+1 -> close t+2
    fwd = fwd.reindex_like(W)
    held = W != 0
    missing = held & fwd.isna()
    fwd_f = fwd.fillna(0.0)                     # a held name with no next price and no delisting record: 0, counted
    pnl_by_name = W * fwd_f
    gross = pnl_by_name.sum(axis=1)
    dW = W.diff().abs()
    dW.iloc[0] = W.iloc[0].abs()
    c = (dW * cost_bps.reindex_like(W).fillna(cost_bps.max().max()) * cost_mult / 1e4).sum(axis=1)
    if impact:
        aum = float(impact["aum"])
        k = float(impact.get("k", 0.1))
        sig = impact["daily_vol"].reindex_like(W)
        adv = impact["adv"].reindex_like(W)
        part = (dW * aum / adv).clip(lower=0)
        c = c + (dW * k * sig * np.sqrt(part)).sum(axis=1).fillna(0.0)
    short_gross = W.clip(upper=0).abs().sum(axis=1)
    borrow = short_gross * borrow_bps / 1e4 / 252.0
    net = gross - c - borrow
    # P&L is realised on the day after the decision; index by decision date (callers shift if needed)
    res = BacktestResult(
        gross=gross, net=net, costs=c + borrow, turnover=dW.sum(axis=1) / 2.0,
        gross_exposure=W.abs().sum(axis=1), net_exposure=W.sum(axis=1),
        n_long=(W > 0).sum(axis=1), n_short=(W < 0).sum(axis=1), weights=W,
        contrib=pnl_by_name if keep_contrib else None,
        meta={"holding": holding, "execution": execution, "cost_mult": cost_mult, "borrow_bps": borrow_bps,
              "missing_price_name_days": int(missing.to_numpy().sum())},
    )
    return res


# --- portfolio construction helpers ------------------------------------------------------------------
def quantile_weights(signal: pd.DataFrame, universe: pd.DataFrame, *, q: float = 0.1, long_only: bool = False,
                     weighting: str = "equal", vol: pd.DataFrame | None = None, groups: pd.DataFrame | None = None,
                     min_names: int = 20) -> pd.DataFrame:
    """Long the top ``q`` of ``signal`` (higher = better) inside ``universe``; short the bottom ``q``
    unless ``long_only``. weighting: equal | inverse_vol | rank. ``groups`` (e.g. statistical sectors)
    demeans the signal within each group first (sector-neutral ranking). Each side sums to 1 (gross 2
    long-short, 1 long-only). Dates with fewer than ``min_names`` eligible names stay flat."""
    s = signal.where(universe)
    if groups is not None:
        g = groups.reindex_like(s)
        s = s - _group_mean(s, g)
    pct = s.rank(axis=1, pct=True)
    n = s.notna().sum(axis=1)
    long_ = pct > 1 - q                     # strict: q of N names on each side (ties excepted)
    short_ = pct <= q
    if weighting == "equal":
        base = pd.DataFrame(1.0, index=s.index, columns=s.columns)
    elif weighting == "inverse_vol":
        if vol is None:
            raise ValueError("inverse_vol needs vol")
        base = 1.0 / vol.reindex_like(s).where(lambda x: x > 0)
    elif weighting == "rank":
        base = (pct - 0.5).abs()
    else:
        raise ValueError(weighting)
    wl = base.where(long_, 0.0).fillna(0.0)
    wl = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if long_only:
        w = wl
    else:
        ws = base.where(short_, 0.0).fillna(0.0)
        ws = ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        w = wl - ws
    w[n < min_names] = 0.0
    return w


def _group_mean(s: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    sv, gv = s.to_numpy(), g.to_numpy()
    for i in range(len(s)):
        row, grp = sv[i], gv[i]
        ok = np.isfinite(row) & pd.notna(grp)
        if not ok.any():
            continue
        keys = pd.Series(row[ok]).groupby(pd.Series(grp[ok]).to_numpy()).transform("mean").to_numpy()
        tmp = np.full(len(row), np.nan)
        tmp[ok] = keys
        out.iloc[i] = tmp
    return out


def beta_neutralize(w: pd.DataFrame, beta: pd.DataFrame, hedge: str = "SPY") -> pd.DataFrame:
    """Add a position in the hedge instrument so the book's market beta is ~0 (beta of the hedge = 1)."""
    b = (w * beta.reindex_like(w).fillna(1.0)).sum(axis=1)
    out = w.copy()
    if hedge not in out.columns:
        out[hedge] = 0.0
    out[hedge] = out[hedge] - b
    return out
