"""Replay of the paper bot's OWN strategies on the survivorship-free research store (sprint P3, P1-D, P8).
PAPER research.

Master's code runs unchanged: UniverseEngine, FeatureSet, StrategyArena and BacktestEngine, with
``config/default.yaml``. That file is identical to master's apart from the separate momentum_breakout
section. Only the INPUT changes: a DataBundle built from the research store, which includes delisted
companies (entity keys ``TICKER@last_seen``).

- **Raw OHLCV:** taken from the store. Raw high and low are the adjusted high and low scaled by that
  session's raw/adjusted close ratio.
- **Splits and cash dividends:** recovered from the raw close and the total return. The adjusted
  return ``r`` satisfies (1 + r) * prev_close = close * split + dividend, so
  ``x = (1 + r) * prev_close / close``:
  - beyond +/-15% is a split ratio (master's own SMALL_SPLIT_RATIO);
  - anything else is booked as a SIGNED cash flow of (x - 1) * close per share. Real dividends are
    positive. Rounding noise in the 4-decimal adjusted prices is either sign.

  Raw-price accounting then reproduces the total return exactly. Crediting only the positive residuals
  biased returns upward; that version was caught before any result was read.
- **Reference rows:**
  - ``security_type`` comes from the store's security master.
  - ``exchange`` is "NYSE" by construction: the store holds only SIP-reported, exchange-listed bars.
    Alpaca's CURRENT exchange is "OTC" for many delisted names and would bring survivorship bias back.
  - ``industry`` is the stock's FIRST point-in-time statistical sector, its trailing-252
    best-correlated sector ETF. It is held static afterwards, so it is stale but never from the
    future, unlike master's current-snapshot sector map.
- **Not replayed:**
  - pead_ear: needs a PIT earnings-event feed that the store lacks before 2020;
  - quality_momentum: needs fundamentals.

  Their evidence comes from the bot database only.

Sizing rules (docs/ALPHA-SPRINT-PREREG.md section 3) subclass the engine. Master's ``_decide`` picks the
candidates, slots and plans. ``SizedEngine`` then re-sizes those same entries in the same order,
against the same cash. Signals, entries, exits, stops and costs never change.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestEngine
from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import DataBundle, Panel

REPLAYED = ("momentum_trend", "mean_reversion", "breakout", "relative_strength", "extreme_reversal", "sector_rotation")
NOT_REPLAYED = {"pead_ear": "needs a point-in-time earnings-event feed (store has none before 2020)",
                "quality_momentum": "needs fundamentals (not in the research store)"}
SPLIT_LIMIT = 1.15
NOISE = 2e-4            # residuals within 2 bps are counted as rounding noise in ``meta`` (all are booked, signed)


def bundle_from_research(p, cols: pd.Index, master: pd.DataFrame, sectors: pd.DataFrame | None,
                         benchmarks: dict[str, Any], market: str = "SPY", unknown_days: str = "neutral") -> DataBundle:
    """Master DataBundle from an AlphaPanel ``p`` restricted to ``cols`` (must include the benchmarks).

    ``unknown_days`` handles the corporate-action days the sprint-c patch flagged UNKNOWN (``p.meta['ca_flags']``):
    - ``neutral`` (primary): the holder's value is unchanged that day, booked as a split ratio or cash.
      Example: UHAL's 9-for-1 distribution becomes shares x 9.87, not a -90% loss.
    - ``raw`` (sensitivity): the raw price move is booked, which ignores the value distributed."""
    close, open_ = p["close"][cols], p["open"][cols]
    k = close / p["adj_close"][cols]
    high, low = p["adj_high"][cols] * k, p["adj_low"][cols] * k
    vol = p["volume"][cols]
    ret = p["ret_cc"][cols].where(close.notna())
    prev = close.ffill().shift(1)
    fl = p.meta.get("ca_flags")
    if fl is not None and len(fl):
        mask = pd.DataFrame(False, index=p.dates, columns=cols)
        for r in fl.itertuples(index=False):
            if r.entity in mask.columns and pd.Timestamp(r.date) in mask.index:
                mask.at[pd.Timestamp(r.date), r.entity] = True
        # neutral: total return 0 (x = prev/close becomes a split ratio or a cash residual); raw: the raw move (x = 1)
        ret = ret.where(~mask, 0.0 if unknown_days == "neutral" else close / prev - 1.0)
    x = (1.0 + ret) * prev / close
    big = np.abs(np.log(x)) > math.log(SPLIT_LIMIT)
    resid = (np.abs(x - 1.0) > 1e-12) & ~big
    split = pd.DataFrame(np.where(big, x, 1.0), index=p.dates, columns=cols).fillna(1.0)
    # every non-split residual is booked as a SIGNED cash flow per share, so raw-price accounting reproduces
    # the total return exactly (no one-sided bias from crediting only the positive side)
    dividend = pd.DataFrame(np.where(resid, (x - 1.0) * close, 0.0), index=p.dates, columns=cols).fillna(0.0)
    divd = (x > 1.0 + NOISE) & ~big
    ignored = int(((x < 1.0 - NOISE) & ~big).sum().sum())
    noise = int((resid & (np.abs(x - 1.0) <= NOISE)).sum().sum())
    tri = (1.0 + ret.fillna(0.0)).cumprod().where(close.notna())
    kk = tri / close
    f = {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "ret": ret, "tri": tri,
         "aopen": open_ * kk, "ahigh": high * kk, "alow": low * kk, "aclose": tri,
         "dollar_volume": close * vol, "split_ratio": split, "dividend": dividend}
    panel = Panel(f, {"providers": ["alpaca_sip_research_store"], "negative_residual_days_over_2bp": ignored,
                      "residual_days_within_2bp": noise,
                      "n_splits": int(big.sum().sum()), "n_dividend_days": int(divd.sum().sum())})
    m = master.set_index("symbol").reindex(cols)
    first_sector = {}
    if sectors is not None:
        s = sectors.reindex(columns=cols)
        for c in cols:
            v = s[c].dropna()
            first_sector[c] = v.iloc[0] if len(v) else None
    st = m["sec_type"].fillna("UNKNOWN").astype(str)
    ref = pd.DataFrame({
        "symbol": cols, "name": cols, "exchange": "NYSE", "security_type": st.to_numpy(),
        "is_etf": (st == "ETF").to_numpy(), "is_test_issue": False, "cik": None, "sic": None, "sector": None,
        "industry": [first_sector.get(c) for c in cols], "status": m["status"].astype(str).to_numpy() if "status" in m else None,
        "source": "research_store", "retrieved_at": pd.Timestamp.now(tz="UTC"), "pit_status": "ASSUMED_STATIC_FIRST_PIT_SECTOR"})
    return DataBundle(panel=panel, calendar=TradingCalendar(p.dates), reference=ref,
                      benchmarks={"market": market, "sectors": dict(benchmarks)}, is_synthetic=bool(p.meta.get("is_synthetic")))


class SizedEngine(BacktestEngine):
    """Master's engine with a different QUANTITY rule. ``weight(i, col, strategy_id, ref)`` returns the target
    fraction of equity, or None (then the fallback weight applies and is counted). The engine runs in
    equal_weight mode, so a missing stop never removes a candidate the other rules would take."""

    def __init__(self, config, data, weight: Callable[[int, int, str], float | None], cap: float,
                 fallback: float = 0.10, **kw):
        super().__init__(config, data, **kw)
        self.sizing = "equal_weight"
        self._weight, self._cap, self._fallback = weight, float(cap), float(fallback)
        self.sizing_diag = {"fallback_weight": 0, "resized": 0}

    def _decide(self, i, S, strategies, fs, pf, pending, candidate_filter, diag, equity):
        out = super()._decide(i, S, strategies, fs, pf, pending, candidate_filter, diag, equity)
        if not out:
            return out
        avail = pf.cash
        for lot in pf.lots.values():
            if lot.pending_exit is not None:
                avail += lot.shares * lot.last_close - abs(lot.shares) * lot.last_close * lot.one_way
        avail -= sum(abs(pe.qty) * pe.entry_ref_at_signal * (1 + pe.one_way) for pe in pending)
        kept = []
        for pe in out:
            w = self._weight(i, pe.col, pe.strategy_id)
            if w is None or not np.isfinite(w) or w <= 0:
                w = self._fallback
                self.sizing_diag["fallback_weight"] += 1
            ref = pe.entry_ref_at_signal
            q = min(w, self._cap) * equity / ref
            q = min(q, max(avail, 0.0) / (ref * (1 + pe.one_way)))
            q = pf.round_qty(q)
            if not q > 0:
                diag["size_zero"] += 1
                continue
            avail -= q * ref * (1 + pe.one_way)
            pe.qty = q
            kept.append(pe)
            self.sizing_diag["resized"] += 1
        return kept


def inverse_vol_weight(vol: np.ndarray, m: float, base: float = 0.10,
                       horizon_of: Callable[[str], int] | None = None,
                       by_h: dict[int, np.ndarray] | None = None) -> Callable[[int, int, str], float | None]:
    """w = base * m / sigma at the signal session. ``by_h`` (horizon -> array) overrides ``vol`` per strategy."""
    def f(i: int, c: int, sid: str) -> float | None:
        a = by_h[horizon_of(sid)] if by_h is not None and horizon_of is not None else vol
        s = a[i, c]
        return base * m / s if np.isfinite(s) and s > 0 else None
    return f


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------
def period_metrics(equity: pd.DataFrame, trades: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    e = equity.set_index("date") if "date" in equity.columns else equity
    v = e["equity"].loc[start:end]
    if len(v) < 20:
        return {"n_days": int(len(v))}
    prev = e["equity"].loc[:start].iloc[:-1]
    base = float(prev.iloc[-1]) if len(prev) else float(v.iloc[0])
    r = pd.concat([pd.Series([base]), v.reset_index(drop=True)]).pct_change().dropna()
    yrs = len(r) / 252
    tot = float(v.iloc[-1] / base - 1)
    vol = float(r.std() * math.sqrt(252))
    dn = r[r < 0]
    curve = (1 + r).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    monthly = (1 + pd.Series(r.to_numpy(), index=v.index)).groupby(v.index.to_period("M")).prod() - 1
    gross = e["positions_value"].loc[start:end] / v
    tr = trades[(pd.to_datetime(trades["entry_date"]) >= start) & (pd.to_datetime(trades["entry_date"]) <= end)] \
        if len(trades) else trades
    cagr = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 and tot > -1 else None
    return {"n_days": int(len(r)), "total_return": tot, "cagr": cagr, "vol": vol,
            "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else None,
            "sortino": float(r.mean() / dn.std() * math.sqrt(252)) if len(dn) > 1 and dn.std() > 0 else None,
            "max_drawdown": dd, "calmar": (cagr / abs(dd)) if cagr is not None and dd < 0 else None,
            "worst_month": float(monthly.min()), "skew_daily": float(r.skew()),
            "avg_gross_exposure": float(gross.mean()),
            "return_per_unit_exposure": (cagr / float(gross.mean())) if cagr is not None and gross.mean() > 0 else None,
            "n_trades": int(len(tr)), "mean_net_ret_per_trade": float(tr["net_ret"].mean()) if len(tr) else None,
            "hit_rate": float((tr["net_ret"] > 0).mean()) if len(tr) else None,
            "turnover_trades_per_year": float(len(tr) / yrs) if yrs > 0 else None}


def sharpe_diff_ci(eq_a: pd.DataFrame, eq_b: pd.DataFrame, start: str, end: str, n_boot: int = 2000,
                   seed: int = 7, level: float = 0.90) -> dict[str, Any]:
    """Sharpe(a) - Sharpe(b) with a monthly-block bootstrap CI (same resampled months for both)."""
    def rets(eq):
        e = eq.set_index("date") if "date" in eq.columns else eq
        return e["equity"].pct_change().loc[start:end]
    a, b = rets(eq_a), rets(eq_b)
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    months = df.index.to_period("M")
    groups = [g for _, g in df.groupby(months)]
    rng = np.random.default_rng(seed)
    sh = lambda x: x.mean() / x.std() * math.sqrt(252) if x.std() > 0 else np.nan  # noqa: E731
    point = float(sh(df["a"]) - sh(df["b"]))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[j] for j in idx])
        diffs.append(sh(s["a"]) - sh(s["b"]))
    lo, hi = np.nanquantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": point, f"ci{int(level * 100)}": [float(lo), float(hi)], "n_months": len(groups)}
