"""Performance metrics for backtests, baselines and walk-forward results.

``summary`` never reports total return alone: every result carries risk (vol, drawdown and its
duration), risk-adjusted return (Sharpe, Sortino, Calmar), trade statistics (win rate, payoff,
profit factor, expectancy net of costs), exposure/turnover, P&L concentration (does one lucky
trade explain everything?), a by-year table and benchmark-relative figures.

Conventions:
  * ``equity``: DataFrame indexed by session with an ``equity`` column (optionally ``cum_costs``,
    ``gross_exposure``) or a plain Series of equity values. Daily returns = pct change of equity.
  * GROSS equity = net equity + cumulative modeled costs paid (spread/slippage/commission). It is
    the additive "had we paid no costs" curve; it ignores compounding of the saved costs.
  * Sharpe/Sortino use a zero risk-free rate and sqrt(periods_per_year) annualization. That
    annualization is only exact for IID returns (see docs: Lo 2002) — treat it as descriptive; use
    validation.stats for inference.
  * Trades: rows with ``net_ret`` (fraction) and ``pnl`` (USD); optional ``mfe``, ``mae``,
    ``holding_sessions``, ``qty``/``entry_price``/``exit_qty``/``exit_price``, ``exit_date``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _equity_frame(equity: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(equity, pd.Series):
        return pd.DataFrame({"equity": equity.astype("float64")})
    if "equity" not in equity.columns:
        raise ValueError("equity frame needs an 'equity' column")
    return equity


def daily_returns(equity: pd.DataFrame | pd.Series) -> pd.Series:
    eq = _equity_frame(equity)["equity"].astype("float64")
    # equity is never missing inside a backtest; no fill is applied, so pct_change is exact
    return (eq / eq.shift(1) - 1.0).iloc[1:]


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.astype("float64")
    return eq / eq.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(max drawdown as a negative fraction, longest drawdown duration in sessions).

    Duration = sessions spent below a previous peak (peak -> recovery, or peak -> end if never
    recovered)."""
    if len(equity) == 0:
        return 0.0, 0
    dd = drawdown_series(equity)
    under = (dd < -1e-12).to_numpy()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return float(dd.min()), int(longest)


def equity_stats(equity: pd.DataFrame | pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> dict[str, Any]:
    eq = _equity_frame(equity)
    e = eq["equity"].astype("float64")
    out: dict[str, Any] = {"n_sessions": int(len(e))}
    if len(e) == 0:
        return out
    out["start"] = str(pd.Timestamp(e.index[0]).date()) if isinstance(e.index, pd.DatetimeIndex) else None
    out["end"] = str(pd.Timestamp(e.index[-1]).date()) if isinstance(e.index, pd.DatetimeIndex) else None
    e0, e1 = float(e.iloc[0]), float(e.iloc[-1])
    r = daily_returns(eq)
    n_ret = len(r)
    years = n_ret / periods_per_year if n_ret else 0.0
    total_net = e1 / e0 - 1.0
    costs = float(eq["cum_costs"].iloc[-1]) if "cum_costs" in eq.columns else 0.0
    total_gross = (e1 + costs) / e0 - 1.0
    out.update(
        total_return_net=total_net,
        total_return_gross=total_gross,
        total_costs=costs,
        cagr_net=((1 + total_net) ** (1 / years) - 1) if years > 0 and total_net > -1 else None,
        cagr_gross=((1 + total_gross) ** (1 / years) - 1) if years > 0 and total_gross > -1 else None,
    )
    sd = float(r.std(ddof=1)) if n_ret > 1 else np.nan
    mu = float(r.mean()) if n_ret else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))) if n_ret else np.nan
    out["ann_vol"] = sd * math.sqrt(periods_per_year) if np.isfinite(sd) else None
    out["sharpe"] = mu / sd * math.sqrt(periods_per_year) if np.isfinite(sd) and sd > 0 else None
    out["sortino"] = mu / downside * math.sqrt(periods_per_year) if np.isfinite(downside) and downside > 0 else None
    mdd, dur = max_drawdown(e)
    out["max_drawdown"] = mdd
    out["max_drawdown_duration"] = dur
    cagr = out["cagr_net"]
    out["calmar"] = cagr / abs(mdd) if cagr is not None and mdd < 0 else None
    if "gross_exposure" in eq.columns:
        out["avg_exposure"] = _f(eq["gross_exposure"].mean())
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def trade_stats(trades: pd.DataFrame | None, avg_equity: float | None = None, years: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"n_trades": 0}
    if trades is None or len(trades) == 0 or "net_ret" not in trades.columns:
        return out
    t = trades[trades["net_ret"].notna()]
    n = len(t)
    out["n_trades"] = n
    if n == 0:
        return out
    ret = t["net_ret"].astype("float64")
    wins, losses = ret[ret > 0], ret[ret <= 0]
    out["win_rate"] = len(wins) / n
    out["avg_win"] = _f(wins.mean()) if len(wins) else None
    out["avg_loss"] = _f(losses.mean()) if len(losses) else None
    out["payoff"] = (out["avg_win"] / abs(out["avg_loss"])
                     if out["avg_win"] is not None and out["avg_loss"] not in (None, 0.0) else None)
    out["expectancy"] = _f(ret.mean())                 # mean NET return per trade
    out["median_trade_return"] = _f(ret.median())
    out["trade_return_std"] = _f(ret.std(ddof=1)) if n > 1 else None
    if "gross_ret" in t.columns:
        out["expectancy_gross"] = _f(t["gross_ret"].mean())
    if "pnl" in t.columns:
        pnl = t["pnl"].astype("float64")
        gp, gl = float(pnl[pnl > 0].sum()), float(pnl[pnl < 0].sum())
        out["total_pnl"] = float(pnl.sum())
        out["expectancy_pnl"] = _f(pnl.mean())
        out["profit_factor"] = gp / abs(gl) if gl < 0 else None
        srt = pnl.sort_values(ascending=False)
        top5 = float(srt.iloc[:5].sum())
        top10pct = float(srt.iloc[: max(1, int(math.ceil(0.10 * n)))].sum())
        total = float(pnl.sum())
        out["pnl_share_top5"] = top5 / total if total > 0 else None
        out["pnl_share_top10pct"] = top10pct / total if total > 0 else None
        out["top5_share_of_gross_profit"] = top5 / gp if gp > 0 else None
    if "holding_sessions" in t.columns:
        out["avg_holding_sessions"] = _f(t["holding_sessions"].mean())
    for col in ("mfe", "mae"):
        if col in t.columns and t[col].notna().any():
            s = t[col].astype("float64")
            out[f"{col}_mean"] = _f(s.mean())
            out[f"{col}_median"] = _f(s.median())
            out[f"{col}_winners_mean"] = _f(s[ret > 0].mean()) if len(wins) else None
            out[f"{col}_losers_mean"] = _f(s[ret <= 0].mean()) if len(losses) else None
    if "exit_reason" in t.columns:
        out["exit_reasons"] = {str(k): int(v) for k, v in t["exit_reason"].value_counts().items()}
    need = {"qty", "entry_price", "exit_qty", "exit_price"}
    if avg_equity and years and years > 0 and need.issubset(t.columns):
        traded = (t["qty"].abs() * t["entry_price"] + t["exit_qty"].abs() * t["exit_price"]).sum()
        # one-sided annual turnover: (buys + sells) / 2 per year, relative to average equity
        out["turnover_annual"] = _f(traded / 2.0 / avg_equity / years)
    return out


def by_year(equity: pd.DataFrame | pd.Series, trades: pd.DataFrame | None = None,
            benchmark_equity: pd.DataFrame | pd.Series | None = None,
            periods_per_year: int = PERIODS_PER_YEAR) -> list[dict[str, Any]]:
    eq = _equity_frame(equity)["equity"].astype("float64")
    if len(eq) < 2 or not isinstance(eq.index, pd.DatetimeIndex):
        return []
    b = _equity_frame(benchmark_equity)["equity"].astype("float64").reindex(eq.index) if benchmark_equity is not None else None
    rows = []
    for yr, grp in eq.groupby(eq.index.year):
        # year return measured from the previous year's last equity (or the first point)
        prev = eq[eq.index < grp.index[0]]
        base = float(prev.iloc[-1]) if len(prev) else float(grp.iloc[0])
        path = pd.concat([pd.Series([base]), grp.reset_index(drop=True)])
        r = (path / path.shift(1) - 1.0).iloc[1:]
        sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
        mdd, _ = max_drawdown(path.reset_index(drop=True))
        row = {
            "year": int(yr),
            "return": float(grp.iloc[-1] / base - 1.0),
            "max_drawdown": mdd,
            "sharpe": _f(r.mean() / sd * math.sqrt(periods_per_year)) if np.isfinite(sd) and sd > 0 else None,
            "n_sessions": int(len(grp)),
        }
        if trades is not None and len(trades) and "exit_date" in trades.columns:
            ex = pd.to_datetime(trades["exit_date"])
            ty = trades[ex.dt.year == yr]
            row["n_trades"] = int(len(ty))
            row["mean_trade_return"] = _f(ty["net_ret"].mean()) if len(ty) else None
        if b is not None:
            bprev = b[b.index < grp.index[0]].dropna()
            bg = b.loc[grp.index].dropna()
            if len(bg):
                bbase = float(bprev.iloc[-1]) if len(bprev) else float(bg.iloc[0])
                row["benchmark_return"] = float(bg.iloc[-1] / bbase - 1.0)
                row["excess_return"] = row["return"] - row["benchmark_return"]
        rows.append(row)
    return rows


def benchmark_relative(equity: pd.DataFrame | pd.Series, benchmark_equity: pd.DataFrame | pd.Series,
                       periods_per_year: int = PERIODS_PER_YEAR) -> dict[str, Any]:
    r = daily_returns(equity)
    rb = daily_returns(benchmark_equity).reindex(r.index)
    both = r.notna() & rb.notna()
    r, rb = r[both], rb[both]
    out: dict[str, Any] = {"n_overlap": int(len(r))}
    if len(r) < 2:
        return out
    e = _equity_frame(equity)["equity"]
    be = _equity_frame(benchmark_equity)["equity"].reindex(e.index).dropna()
    b_total = float(be.iloc[-1] / be.iloc[0] - 1.0) if len(be) > 1 else None
    s_total = float(e.iloc[-1] / e.iloc[0] - 1.0)
    years = len(r) / periods_per_year
    out["benchmark_total_return"] = b_total
    out["excess_total_return"] = s_total - b_total if b_total is not None else None
    if b_total is not None and years > 0 and b_total > -1 and s_total > -1:
        out["benchmark_cagr"] = (1 + b_total) ** (1 / years) - 1
        out["excess_cagr"] = ((1 + s_total) ** (1 / years) - 1) - out["benchmark_cagr"]
    var_b = float(rb.var(ddof=1))
    out["beta"] = float(np.cov(r, rb, ddof=1)[0, 1] / var_b) if var_b > 0 else None
    out["correlation"] = _f(r.corr(rb))
    act = r - rb
    te = float(act.std(ddof=1))
    out["tracking_error"] = te * math.sqrt(periods_per_year) if np.isfinite(te) else None
    out["information_ratio"] = float(act.mean() / te * math.sqrt(periods_per_year)) if te > 0 else None
    if out["beta"] is not None:
        out["alpha_annual"] = float((r.mean() - out["beta"] * rb.mean()) * periods_per_year)
    return out


def summary(equity: pd.DataFrame | pd.Series, trades: pd.DataFrame | None = None,
            benchmark_equity: pd.DataFrame | pd.Series | None = None,
            periods_per_year: int = PERIODS_PER_YEAR, min_trades: int | None = None) -> dict[str, Any]:
    """Full metric set (see module docstring). JSON-friendly: non-finite values become None."""
    out = equity_stats(equity, periods_per_year)
    eq = _equity_frame(equity)
    years = max(len(eq) - 1, 0) / periods_per_year
    avg_eq = float(eq["equity"].mean()) if len(eq) else None
    out.update(trade_stats(trades, avg_eq, years))
    out["by_year"] = by_year(equity, trades, benchmark_equity, periods_per_year)
    if benchmark_equity is not None:
        out["benchmark"] = benchmark_relative(equity, benchmark_equity, periods_per_year)
    if min_trades is not None:
        out["min_trades_for_conclusion"] = int(min_trades)
        out["sample_sufficient"] = bool(out.get("n_trades", 0) >= min_trades)
    return out
