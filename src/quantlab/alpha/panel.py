"""Wide point-in-time matrices for alpha research, built from the survivorship-free store.

Conventions (all trailing; anything known at the close of session t is usable for a decision taken
after that close, executed at the NEXT session's open or close):
  * Level fields (``close``, ``open``, ``volume``, ``dollar_volume``) are RAW. Level rules use only these.
  * Return fields come from ``adjustment=all`` bars, so they are total returns (splits + dividends):
      ret_cc[t] = adj_close[t] / adj_close[t-1] - 1     close-to-close
      ret_on[t] = adj_open[t]  / adj_close[t-1] - 1     overnight (close t-1 -> open t)
      ret_id[t] = adj_close[t] / adj_open[t]   - 1     intraday  (open t -> close t)
      ret_oo[t] = adj_open[t+1]/ adj_open[t]   - 1     open-to-open, the P&L of a position held from
                                                        the open of t to the open of t+1 (forward-looking:
                                                        it is an OUTCOME, never a feature)
  * Twins (audit fix M2): keys carrying the same security on the same day (identical raw close and
    volume) are de-duplicated by ``entities.dedupe_twins``; a key whose history continued under its twin
    is ``renamed``, not delisted (its last bar exits at the close, no delisting return).
  * Delistings: a symbol whose last bar is before the store's last session is delisted after that
    bar. A position still open then exits at that last close, followed by a delisting return:
    ``distressed`` (last close < $3, or adjusted close >= 50% below its 60-session adjusted high; audit
    minor fix: the first version used RAW prices, so a reverse split hid the fall) -> ``delist_distressed``
    (default -30%, Shumway 1997; sensitivity -100%); otherwise (typically an acquisition: the last price
    is near the deal price) -> ``delist_other`` (default 0%). This is an ASSUMPTION, reported with
    sensitivity, because Alpaca exposes no delisting reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.alpha.store import HOLDOUT_START, load_bars, load_master

SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")


@dataclass
class AlphaPanel:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    f: dict[str, pd.DataFrame]                      # wide fields
    master: pd.DataFrame                            # one row per symbol (sec_type, status, first/last bar)
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, k: str) -> pd.DataFrame:
        return self.f[k]

    def etf(self, sym: str, field_: str = "ret_cc") -> pd.Series:
        return self.f[field_][sym]


def _wide(df: pd.DataFrame, col: str, dates, syms) -> pd.DataFrame:
    w = df.pivot(index="date", columns="symbol", values=col)
    return w.reindex(index=dates, columns=syms).astype("float64")


def build_panel(min_price_ever: float = 3.0, min_dv_ever: float = 1e6, keep_types: tuple[str, ...] = ("COMMON",),
                extra_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "MDY", "DIA", *SECTOR_ETFS, "TLT", "IEF",
                                                  "SHY", "HYG", "LQD", "UUP", "GLD", "USO", "DBC", "VIXY", "EFA", "EEM"),
                delist_distressed: float = -0.30, delist_other: float = 0.0, dedupe: bool = True) -> AlphaPanel:
    """Load the store into wide matrices. Symbols that never reach ``min_price_ever`` raw close AND
    ``min_dv_ever`` daily dollar volume on the same day are dropped up front (they can never enter a
    research universe, which needs >= $5 and >= $1M); this is a memory filter, not a selection on outcomes."""
    bars = load_bars()
    master = load_master()
    types = master.drop_duplicates("symbol", keep="first").set_index("symbol")["sec_type"]
    bars = bars[bars["symbol"].map(types).isin(keep_types) | bars["symbol"].isin(extra_symbols)]
    if dedupe:
        from quantlab.alpha.entities import dedupe_twins
        bars, twin_aliases, renamed, twin_report = dedupe_twins(bars)
    else:
        twin_aliases, renamed, twin_report = pd.DataFrame(columns=["symbol", "date", "keeper"]), set(), pd.DataFrame()
    bars = bars.assign(dollar_volume=bars["close"] * bars["volume"])
    ok = (bars["close"] >= min_price_ever) & (bars["dollar_volume"] >= min_dv_ever)
    syms_ok = set(bars.loc[ok, "symbol"]) | set(extra_symbols)
    bars = bars[bars["symbol"].isin(syms_ok)]
    dates = pd.DatetimeIndex(sorted(bars.loc[bars["symbol"] == "SPY", "date"].unique()))
    if dates.max() >= pd.Timestamp(HOLDOUT_START):
        raise RuntimeError("holdout dates in panel: refusing")
    syms = pd.Index(sorted(bars["symbol"].unique()))
    f: dict[str, pd.DataFrame] = {}
    for c in ("open", "high", "low", "close", "volume", "vwap", "trade_count", "adj_open", "adj_high", "adj_low",
              "adj_close", "dollar_volume"):
        f[c] = _wide(bars, c, dates, syms)
    ac, ao = f["adj_close"], f["adj_open"]
    f["ret_cc"] = ac / ac.shift(1) - 1.0
    f["ret_on"] = ao / ac.shift(1) - 1.0
    f["ret_id"] = ac / ao - 1.0
    # --- open-to-open outcome with delisting handling ------------------------------------------------
    last_bar = f["close"].apply(lambda s: s.last_valid_index())
    end = dates[-1]
    roo = ao.shift(-1) / ao - 1.0
    hi60 = ac.rolling(60, min_periods=1).max()
    delisted = []
    for s in syms:
        lb = last_bar.get(s)
        if lb is None or lb >= end or s in extra_symbols:
            continue
        was_renamed = s in renamed
        distressed = (not was_renamed) and bool((f["close"].at[lb, s] < 3.0) or (ac.at[lb, s] <= 0.5 * hi60.at[lb, s]))
        dr = 0.0 if was_renamed else (delist_distressed if distressed else delist_other)
        intraday = ac.at[lb, s] / ao.at[lb, s] - 1.0 if np.isfinite(ao.at[lb, s]) else 0.0
        roo.at[lb, s] = (1.0 + intraday) * (1.0 + dr) - 1.0
        delisted.append((s, lb, distressed, dr, was_renamed))
    f["ret_oo"] = roo
    # trading-day presence and history length (PIT)
    f["has_bar"] = f["close"].notna().astype("float64")
    f["n_hist"] = f["has_bar"].cumsum()
    m = master.drop_duplicates("symbol", keep="first").set_index("symbol").reindex(syms).rename_axis("symbol")
    meta = {"n_symbols": len(syms), "n_dates": len(dates), "start": str(dates[0].date()), "end": str(end.date()),
            "delistings": pd.DataFrame(delisted, columns=["symbol", "last_bar", "distressed", "delist_return", "renamed"]),
            "delist_distressed": delist_distressed, "delist_other": delist_other,
            "twin_aliases": twin_aliases, "twin_report": twin_report, "renamed": sorted(renamed)}
    return AlphaPanel(dates, syms, f, m.reset_index(), meta)


def research_universe(p: AlphaPanel, *, min_price: float = 5.0, min_dv: float = 5e6, dv_window: int = 20,
                      min_history: int = 252, types: tuple[str, ...] = ("COMMON",)) -> pd.DataFrame:
    """Boolean (dates x symbols): eligible at the CLOSE of t. Raw price/dollar-volume, trailing only."""
    mdv = p["dollar_volume"].rolling(dv_window, min_periods=dv_window).median()
    st = p.master.set_index("symbol")["sec_type"].reindex(p.symbols)
    type_ok = pd.Series(st.isin(types).to_numpy(), index=p.symbols)
    u = (p["close"] >= min_price) & (mdv >= min_dv) & (p["n_hist"] >= min_history)
    u = u & np.broadcast_to(type_ok.to_numpy(), u.shape)
    return u.fillna(False)


def median_dollar_volume(p: AlphaPanel, window: int = 20) -> pd.DataFrame:
    return p["dollar_volume"].rolling(window, min_periods=window).median()


def statistical_sectors(p: AlphaPanel, universe: pd.DataFrame, window: int = 252,
                        etfs: tuple[str, ...] = SECTOR_ETFS) -> pd.DataFrame:
    """Each stock's 'sector' = the sector ETF its trailing ``window`` daily returns correlate with most,
    refit at each month end and applied to the FOLLOWING month (point-in-time; replaces the
    ASSUMED_STATIC current sector map, which is not available in this cloud store)."""
    r = p["ret_cc"]
    etf_r = r[[e for e in etfs if e in r.columns]]
    month_ends = r.index.to_series().groupby(r.index.to_period("M")).last()
    out = pd.DataFrame(index=r.index, columns=p.symbols, dtype=object)
    prev_assign = None
    for i, me in enumerate(month_ends):
        loc = r.index.get_loc(me)
        if loc + 1 < window:
            continue
        win = r.iloc[loc + 1 - window: loc + 1]
        ew = etf_r.iloc[loc + 1 - window: loc + 1]
        ew = ew.loc[:, ew.notna().sum() >= int(0.9 * window)]       # XLC/XLRE start later
        valid = win.notna().sum() >= int(0.8 * window)
        x = win.loc[:, valid]
        xs = (x - x.mean()) / x.std(ddof=0)
        es = (ew - ew.mean()) / ew.std(ddof=0)
        corr = (xs.fillna(0.0).T.to_numpy() @ es.fillna(0.0).to_numpy()) / x.notna().sum().to_numpy()[:, None]
        best = pd.Series(np.asarray(ew.columns)[np.nanargmax(corr, axis=1)], index=x.columns)
        nxt = month_ends.iloc[i + 1] if i + 1 < len(month_ends) else r.index[-1]
        rows = (r.index > me) & (r.index <= nxt)
        out.loc[rows, best.index] = np.broadcast_to(best.to_numpy(), (int(rows.sum()), len(best)))
        prev_assign = best
    return out


def rolling_betas(p: AlphaPanel, sectors: pd.DataFrame, window: int = 252, market: str = "SPY",
                  min_frac: float = 0.8) -> dict[str, pd.DataFrame]:
    """Betas refit at each month end on the trailing ``window`` days and applied to the FOLLOWING month
    (point-in-time). Returns:
      beta_mkt    single-factor market beta (for beta-neutral hedging)
      b1, b2      two-factor loadings on [market, own statistical-sector ETF] (for residual returns)
    Columns with complete windows are solved jointly per sector (fast); incomplete ones one by one."""
    r = p["ret_cc"]
    m = r[market]
    month_ends = r.index.to_series().groupby(r.index.to_period("M")).last()
    out = {k: np.full(r.shape, np.nan) for k in ("beta_mkt", "b1", "b2")}
    col_pos = {c: i for i, c in enumerate(r.columns)}
    R = r.to_numpy()
    M = m.to_numpy()
    for i, me in enumerate(month_ends):
        loc = r.index.get_loc(me)
        if loc + 1 < window:
            continue
        sl = slice(loc + 1 - window, loc + 1)
        nxt = month_ends.iloc[i + 1] if i + 1 < len(month_ends) else None
        rows = np.nonzero((r.index > me) & ((r.index <= nxt) if nxt is not None else True))[0]
        if len(rows) == 0:
            continue
        assign = sectors.iloc[rows[0]]
        mw = M[sl]
        Y = R[sl]
        # single-factor beta for every column
        X1 = np.column_stack([np.ones(window), mw])
        okm = np.isfinite(mw)
        complete = np.isfinite(Y).all(axis=0) & okm.all()
        if complete.any():
            coef = np.linalg.lstsq(X1, Y[:, complete], rcond=None)[0]
            out["beta_mkt"][np.ix_(rows, np.nonzero(complete)[0])] = coef[1][None, :]
        for j in np.nonzero(~complete)[0]:
            g = np.isfinite(Y[:, j]) & okm
            if g.sum() >= int(min_frac * window):
                out["beta_mkt"][rows, j] = np.linalg.lstsq(X1[g], Y[g, j], rcond=None)[0][1]
        # two-factor loadings per statistical sector
        for e in pd.unique(assign.dropna()):
            if e not in col_pos:
                continue
            ew = R[sl, col_pos[e]]
            cols = np.array([col_pos[c] for c in assign.index[assign == e]])
            if len(cols) == 0 or np.isfinite(ew).sum() < int(0.9 * window):
                continue
            X2 = np.column_stack([np.ones(window), mw, ew])
            gx = np.isfinite(X2).all(axis=1)
            Ys = Y[:, cols]
            comp = np.isfinite(Ys[gx]).all(axis=0)
            if comp.any():
                coef = np.linalg.lstsq(X2[gx], Ys[gx][:, comp], rcond=None)[0]
                out["b1"][np.ix_(rows, cols[comp])] = coef[1][None, :]
                out["b2"][np.ix_(rows, cols[comp])] = coef[2][None, :]
            for k in np.nonzero(~comp)[0]:
                y = Ys[:, k]
                g = gx & np.isfinite(y)
                if g.sum() >= int(min_frac * window):
                    cf = np.linalg.lstsq(X2[g], y[g], rcond=None)[0]
                    out["b1"][rows, cols[k]] = cf[1]
                    out["b2"][rows, cols[k]] = cf[2]
    return {k: pd.DataFrame(v, index=r.index, columns=r.columns) for k, v in out.items()}


def sector_return(p: AlphaPanel, sectors: pd.DataFrame, field_: str = "ret_cc") -> pd.DataFrame:
    """Each stock's own statistical-sector ETF return on each day (NaN when unassigned)."""
    r = p[field_]
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for e in SECTOR_ETFS:
        if e not in r.columns:
            continue
        mask = (sectors == e).to_numpy()
        vals = np.broadcast_to(r[e].to_numpy()[:, None], r.shape)
        out = out.mask(mask, pd.DataFrame(vals, index=r.index, columns=r.columns))
    return out


def residual_returns(p: AlphaPanel, sectors: pd.DataFrame, betas: dict[str, pd.DataFrame],
                     market: str = "SPY", field_: str = "ret_cc") -> pd.DataFrame:
    """r_i - b1_i * r_mkt - b2_i * r_sector(i), with loadings fitted on data BEFORE the current month."""
    r = p[field_]
    m = r[market]
    s = sector_return(p, sectors, field_)
    return r - betas["b1"].mul(m, axis=0) - betas["b2"] * s
