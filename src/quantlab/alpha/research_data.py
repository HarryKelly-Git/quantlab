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
    for s, lb, dis, _ in dl.itertuples(index=False):
        dr = distressed if dis else other
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
            return pickle.load(fh)
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
        "spy_vol": pd.Series(pd.qcut(r["SPY"].rolling(20).std().rolling(504, min_periods=126).rank(pct=True),
                                     [0, 1 / 3, 2 / 3, 1.0], labels=["calm", "normal", "stressed"]).astype(object),
                             index=p.dates),
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
    manifest.update({"n_delisted_in_panel": int(len(dl)), "n_delisted_distressed": int(dl["distressed"].sum()),
                     "n_survivor_symbols": len(survivors), "panel_symbols": int(p.meta["n_symbols"]),
                     "universe_liquid_avg_names": float(u_liquid.sum(axis=1).mean()),
                     "universe_large_avg_names": float(u_large.sum(axis=1).mean())})
    print(f"research data ready in {time.time() - t0:.0f}s: {manifest['universe_liquid_avg_names']:.0f} names/day", flush=True)
    return EquityData(p, u_liquid, u_large, u_survivor, mdv20, cost, sectors, betas, resid, sec_ret, vol20, vol60,
                      regimes, buckets, alt, manifest)
