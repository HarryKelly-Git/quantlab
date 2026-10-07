"""Build-once, cache-on-disk research matrices shared by every equity experiment.

Everything here is trailing / point-in-time except the ``ret_*`` P&L outcomes the engine consumes.
Cached under var/alpha/cache (git-ignored). ``get(refresh=True)`` rebuilds from the store.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.alpha.engine import cost_bps_matrix
from quantlab.alpha.panel import (SECTOR_ETFS, AlphaPanel, build_panel, median_dollar_volume, research_universe,
                                  residual_returns, rolling_betas, sector_return, statistical_sectors)
from quantlab.alpha.store import store_dir

PANEL_VERSION = ("2026-10-audit-b: zero-volume filler bars dropped; twins de-duplicated (M2) on real trading days with "
                 ">= 50% identical common days; distress flag on adjusted prices; spy_vol fixed bins")
CACHE_FIELDS = ("open", "close", "volume", "dollar_volume", "adj_open", "adj_high", "adj_low", "adj_close",
                "ret_cc", "ret_on", "ret_id", "ret_oo", "n_hist")


@dataclass
class EquityData:
    p: AlphaPanel
    u_liquid: pd.DataFrame          # >= $5, MDV20 >= $5M, >= 252 sessions, COMMON
    u_large: pd.DataFrame           # u_liquid & top 500 by MDV60 (large-cap proxy)
    u_survivor: pd.DataFrame        # u_liquid restricted to names still listed at the store end (biased on purpose)
    mdv20: pd.DataFrame
    cost_bps: pd.DataFrame
    sectors: pd.DataFrame
    betas: dict[str, pd.DataFrame]
    resid: pd.DataFrame             # daily residual returns (market + own statistical sector)
    sec_ret: pd.DataFrame
    vol20: pd.DataFrame
    vol60: pd.DataFrame
    regimes: dict[str, pd.Series]
    buckets: dict[str, pd.DataFrame]
    alt_ret_oo: dict[str, pd.DataFrame]
    manifest: dict[str, Any]

    def resolved(self) -> pd.DataFrame:
        """(ticker, date) -> entity (alpha.entities.resolve), cached on the object. Resolved against EVERY
        store entity (so a filtered-out company can never be replaced by another one using its ticker),
        then twin keys removed from the panel are replaced by the key that was kept (panel aliases).
        Synthetic / store-less panels resolve against the panel itself."""
        r = getattr(self, "_resolved", None)
        if r is None:
            from quantlab.alpha.entities import apply_aliases, resolve
            src, renames = None, None
            if not (self.manifest.get("synthetic") or self.p.meta.get("is_synthetic")):
                try:
                    from quantlab.alpha.store import load_bars, load_name_changes
                    src = load_bars(columns=["symbol", "date", "volume"])
                    src = src.loc[(src["date"] <= self.p.dates[-1]) & (src["volume"] > 0), ["symbol", "date"]]
                    renames = load_name_changes()
                except FileNotFoundError:
                    src = None
            if src is None:
                # pandas 3 stack keeps NaN: drop them, an entity only "has a bar" where it has a close
                long = self.p["close"].stack(future_stack=True).dropna().rename("close").reset_index()
                long.columns = ["date", "symbol", "close"]
                src = long[["symbol", "date"]]
            r = resolve(src, renames)
            r["entity"] = apply_aliases(r["entity"], r["date"], self.p.meta.get("twin_aliases"))
            object.__setattr__(self, "_resolved", r)
        return r

    @property
    def ret_cc_pnl(self) -> pd.DataFrame:
        """``ret_cc`` for P&L under next-CLOSE execution, with each delisting return booked on the session
        after the last bar (audit minor fix: the robustness check 'next-close execution' used ret_cc, which
        has no delisting return). ``ret_cc`` itself stays a pure price return for signals."""
        r = getattr(self, "_ret_cc_pnl", None)
        if r is None:
            r = self.p["ret_cc"].copy()
            dates = self.p.dates
            for row in self.p.meta["delistings"].itertuples(index=False):
                dr = float(row.delist_return)
                i = dates.get_loc(row.last_bar)
                if dr != 0.0 and i + 1 < len(dates) and row.symbol in r.columns:
                    r.iat[i + 1, r.columns.get_loc(row.symbol)] = dr
            object.__setattr__(self, "_ret_cc_pnl", r)
        return r

    @property
    def bench_oo(self) -> pd.Series:
        """SPY P&L aligned to decision dates under next-open execution."""
        return self.p["ret_oo"]["SPY"].shift(-1)


def _cache() -> Path:
    d = store_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(df: pd.DataFrame, name: str) -> None:
    df.astype("float32").to_parquet(_cache() / f"{name}.parquet") if df.dtypes.iloc[0] != object else \
        df.to_parquet(_cache() / f"{name}.parquet")


def _load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(_cache() / f"{name}.parquet")
    return df.astype("float64") if df.dtypes.iloc[0] != object else df


def _delist_variant(p: AlphaPanel, distressed: float, other: float) -> pd.DataFrame:
    roo = p["ret_oo"].copy()
    dl = p.meta["delistings"]
    ac, ao = p["adj_close"], p["adj_open"]
    for row in dl.itertuples(index=False):
        s, lb = row.symbol, row.last_bar
        dr = 0.0 if bool(getattr(row, "renamed", False)) else (distressed if row.distressed else other)
        intraday = ac.at[lb, s] / ao.at[lb, s] - 1.0 if np.isfinite(ao.at[lb, s]) else 0.0
        roo.at[lb, s] = (1 + intraday) * (1 + dr) - 1
    return roo


def _terciles(x: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    pct = x.where(u).rank(axis=1, pct=True)
    lab = pd.DataFrame(np.where(pct <= 1 / 3, "low", np.where(pct <= 2 / 3, "mid", "high")), index=x.index, columns=x.columns)
    return lab.where(pct.notna())


def get(refresh: bool = False) -> EquityData:
    """Load the cached research data (pickle) or build it."""
    import pickle
    path = _cache() / "equity_data.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            d = pickle.load(fh)
        if d.manifest.get("panel_version") != PANEL_VERSION:
            raise RuntimeError(f"cached research data is panel version {d.manifest.get('panel_version')!r}, code expects "
                               f"{PANEL_VERSION!r}: rebuild with run_batch.py build")
        return d
    d = build()
    with path.open("wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def build(refresh: bool = False) -> EquityData:
    return from_panel(build_panel())


def from_panel(p: AlphaPanel, manifest: dict | None = None) -> EquityData:
    """Every derived research matrix from an AlphaPanel (used on truncated panels by the PIT tests)."""
    t0 = time.time()
    u_liquid = research_universe(p)
    mdv20 = median_dollar_volume(p, 20)
    mdv60 = median_dollar_volume(p, 60)
    rank60 = mdv60.where(u_liquid).rank(axis=1, ascending=False)
    u_large = u_liquid & (rank60 <= 500)
    active_end = set(p.master.loc[(p.master["status"] == "active"), "symbol"])
    last = p["close"].apply(lambda s: s.last_valid_index())
    survivors = [s for s in p.symbols if s in active_end and last.get(s) == p.dates[-1]]
    u_survivor = u_liquid & pd.DataFrame(np.broadcast_to(p.symbols.isin(survivors), u_liquid.shape),
                                         index=u_liquid.index, columns=u_liquid.columns)
    cost = cost_bps_matrix(mdv20)
    print(f"panel {p.meta['n_symbols']} symbols x {p.meta['n_dates']} days; built in {time.time() - t0:.0f}s", flush=True)
    sectors = statistical_sectors(p, u_liquid)
    print(f"sectors {time.time() - t0:.0f}s", flush=True)
    betas = rolling_betas(p, sectors)
    print(f"betas {time.time() - t0:.0f}s", flush=True)
    resid = residual_returns(p, sectors, betas)
    sec_ret = sector_return(p, sectors)
    r = p["ret_cc"]
    vol20 = r.rolling(20, min_periods=15).std()
    vol60 = r.rolling(60, min_periods=45).std()
    spy = p["adj_close"]["SPY"]
    ac = p["adj_close"]
    regimes: dict[str, pd.Series] = {
        "spy_trend": pd.Series(np.where(spy > spy.rolling(200, min_periods=200).mean(), "above_200d", "below_200d"),
                               index=p.dates).where(spy.rolling(200, min_periods=200).mean().notna()),
        # audit minor fix: fixed bins on the TRAILING percentile (qcut took full-sample breakpoints)
        "spy_vol": pd.Series(pd.cut(r["SPY"].rolling(20).std().rolling(504, min_periods=126).rank(pct=True),
                                    [0, 1 / 3, 2 / 3, 1.0], labels=["calm", "normal", "stressed"],
                                    include_lowest=True).astype(object), index=p.dates),
    }
    if {"HYG", "LQD"} <= set(ac.columns):
        regimes["credit"] = pd.Series(np.where((ac["HYG"] / ac["LQD"]).pct_change(63) > 0, "risk_on", "risk_off"), index=p.dates)
    if "TLT" in ac.columns:
        regimes["rates"] = pd.Series(np.where(ac["TLT"].pct_change(63) > 0, "yields_falling", "yields_rising"), index=p.dates)
    if "UUP" in ac.columns:
        regimes["dollar"] = pd.Series(np.where(ac["UUP"].pct_change(63) > 0, "usd_up", "usd_down"), index=p.dates)
    buckets = {"liquidity": _terciles(mdv20, u_liquid), "volatility": _terciles(vol60, u_liquid)}
    alt = {"distressed_minus100": _delist_variant(p, -1.0, 0.0), "all_zero": _delist_variant(p, 0.0, 0.0)}
    dl = p.meta["delistings"]
    if manifest is None:
        mp = store_dir() / "manifest.json"
        manifest = json.loads(mp.read_text()) if mp.exists() else {}
    ren = dl["renamed"].astype(bool) if "renamed" in dl else pd.Series(False, index=dl.index)
    tw = p.meta.get("twin_report")
    manifest.update({"panel_version": PANEL_VERSION,
                     "n_twin_keys_deduplicated": int(len(tw)) if tw is not None else 0,
                     "n_twin_rows_removed": int(len(p.meta.get("twin_aliases", []))),
                     "n_renamed_not_delisted": int(ren.sum()),
                     "n_zero_volume_bars_dropped": int(p.meta.get("n_zero_volume_bars_dropped", 0)),
                     "n_delisted_in_panel": int((~ren).sum()), "n_delisted_distressed": int(dl["distressed"].astype(bool).sum()),
                     "n_survivor_symbols": len(survivors), "panel_symbols": int(p.meta["n_symbols"]),
                     "universe_liquid_avg_names": float(u_liquid.sum(axis=1).mean()),
                     "universe_large_avg_names": float(u_large.sum(axis=1).mean())})
    print(f"research data ready in {time.time() - t0:.0f}s: {manifest['universe_liquid_avg_names']:.0f} names/day", flush=True)
    return EquityData(p, u_liquid, u_large, u_survivor, mdv20, cost, sectors, betas, resid, sec_ret, vol20, vol60,
                      regimes, buckets, alt, manifest)
