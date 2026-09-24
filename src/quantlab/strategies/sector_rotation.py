"""Sector rotation: MARKET -> SECTOR -> STOCK.

Hypothesis: equally strong stocks behave better inside strengthening sectors. Sectors are ranked by
63-session return vs SPY (sector-ETF prices, PIT); within the top sectors, stocks are ranked by
strength vs their own sector. Sector membership is ASSUMED_STATIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.base import FeatureSet
from quantlab.features.price import market_series, ret_n
from quantlab.features.relative import _sector_etf_of, xs_rank
from quantlab.strategies.base import Strategy


class SectorRotation(Strategy):
    family = "sector_rotation"
    description = "Long the strongest stocks (vs own sector) inside the top-N sectors by 63-session return vs SPY."
    feature_deps = ("rs_sector_63", "sector_rs_spy_63", "atr14_pct", "adv20")
    default_params = {"sector_lookback": 63, "top_sectors": 3, "stock_rank_pct": 0.80, "hold_sessions": 40, "stop_atr": 3.0}

    def score(self, fs: FeatureSet, universe: pd.DataFrame) -> pd.DataFrame:
        p = fs.panel
        u = universe.reindex(index=p.dates, columns=p.symbols).fillna(False).astype(bool)
        etfs = [e for e in fs.bundle.sector_etfs if e in p.symbols]
        mkt = market_series(fs, "aclose")
        if not etfs or mkt is None:
            return pd.DataFrame(np.nan, index=p.dates, columns=p.symbols)
        n = int(self.params["sector_lookback"])
        etf_rs = ret_n(p.aclose[etfs], n).sub(ret_n(mkt, n), axis=0)
        top = etf_rs.rank(axis=1, ascending=False, method="first") <= int(self.params["top_sectors"])
        mapping = _sector_etf_of(fs)
        in_top = pd.DataFrame(False, index=p.dates, columns=p.symbols)
        for sym in p.symbols:
            etf = mapping.get(sym)
            if etf in top.columns and etf != sym:
                in_top[sym] = top[etf].to_numpy()
        eligible = u & in_top
        stock_rank = xs_rank(fs.get("rs_sector_63"), eligible)
        signal = eligible & (stock_rank >= float(self.params["stock_rank_pct"]))
        return stock_rank.where(signal)
