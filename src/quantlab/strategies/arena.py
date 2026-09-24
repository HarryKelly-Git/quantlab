"""Strategy arena: every strategy generates candidates independently; the arena measures what
information each strategy's signals actually carry. It never counts votes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import Candidate
from quantlab.features.base import FeatureSet
from quantlab.strategies.base import Strategy


class HoldoutAccessError(RuntimeError):
    pass


def _rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 5) -> np.ndarray:
    """Per-row Spearman correlation over columns where both are present (vectorized)."""
    both = a.notna() & b.reindex_like(a).notna()
    ra = a.where(both).rank(axis=1)
    rb = b.reindex_like(a).where(both).rank(axis=1)
    n = both.sum(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    cov = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    rho = (cov / den.where(den > 0)).where(n >= min_n)
    return rho.dropna().to_numpy()


class StrategyArena:
    def __init__(self, strategies: list[Strategy]):
        self.strategies = {s.strategy_id: s for s in strategies}
        self._scores: dict[tuple[int, str], pd.DataFrame] = {}

    def scores(self, fs: FeatureSet, universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
        out = {}
        for sid, s in self.strategies.items():
            key = (id(fs), sid)
            if key not in self._scores:
                s.check_pit(fs)
                self._scores[key] = s.score(fs, universe)
            out[sid] = self._scores[key]
        return out

    def candidates(self, fs: FeatureSet, universe: pd.DataFrame, as_of) -> list[Candidate]:
        """All strategies' candidates for one session (a symbol may appear under several strategies)."""
        sc = self.scores(fs, universe)
        out: list[Candidate] = []
        for sid, s in self.strategies.items():
            out.extend(s.candidates(fs, universe, as_of, scores=sc[sid]))
        return out

    def information_content(self, fs: FeatureSet, universe: pd.DataFrame, start, end,
                            horizons=(5, 10, 20), holdout_start=None) -> pd.DataFrame:
        """MODEL_OUTPUT: forward excess return of signalled names (next-open entry, vs benchmark).

        Uses future prices by design (it is an evaluation), so it refuses any window reaching the
        locked holdout.
        """
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        p = fs.panel
        hold_i = len(p.dates)
        if holdout_start is not None:
            if e >= pd.Timestamp(holdout_start):
                raise HoldoutAccessError(f"information_content end {e.date()} reaches the locked holdout")
            hold_i = int(p.dates.searchsorted(pd.Timestamp(holdout_start), side="left"))
        mkt = fs.bundle.market_symbol
        rows = []
        for sid, sc in self.scores(fs, universe).items():
            for h in horizons:
                # a signal at index d needs prices up to d+h: never let that reach the holdout
                last_ok = hold_i - h - 1
                e_h = min(e, p.dates[last_ok]) if last_ok >= 0 else s - pd.Timedelta(days=1)
                win = sc.loc[s:e_h]
                fwd = p.forward_returns(h, "next_open").loc[s:e_h]
                bench = fwd[mkt] if mkt in fwd.columns else pd.Series(0.0, index=fwd.index)
                excess = fwd.sub(bench, axis=0)
                sig = win.notna()
                vals = excess.where(sig).stack().dropna()
                per_date = excess.where(sig).mean(axis=1).dropna()
                ics = _rowwise_spearman(win, excess, min_n=5)
                n_dates = len(per_date)
                t = per_date.mean() / (per_date.std(ddof=1) / np.sqrt(n_dates)) if n_dates > 2 and per_date.std() > 0 else np.nan
                rows.append({"strategy_id": sid, "horizon": h, "n_signals": int(len(vals)), "n_dates": n_dates,
                             "mean_excess": float(vals.mean()) if len(vals) else np.nan,
                             "hit_rate": float((vals > 0).mean()) if len(vals) else np.nan,
                             "mean_ic": float(np.nanmean(ics)) if len(ics) else np.nan,
                             "t_stat_dates": float(t) if np.isfinite(t) else np.nan,
                             "info_kind": "MODEL_OUTPUT"})
        return pd.DataFrame(rows)

    def overlap(self, fs: FeatureSet, universe: pd.DataFrame, start, end) -> pd.DataFrame:
        """Jaccard overlap of signalled (date, symbol) sets between strategies."""
        sc = {k: v.loc[pd.Timestamp(start):pd.Timestamp(end)].notna() for k, v in self.scores(fs, universe).items()}
        ids = list(sc)
        m = pd.DataFrame(np.nan, index=ids, columns=ids)
        for a in ids:
            for b in ids:
                inter = (sc[a] & sc[b]).to_numpy().sum()
                union = (sc[a] | sc[b]).to_numpy().sum()
                m.at[a, b] = inter / union if union else np.nan
        return m
