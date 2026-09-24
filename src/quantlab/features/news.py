"""News-volume features. Counts only: an item exists from its ``available_at`` (creation time).

Headline/summary text may be revised after creation (PIT_CONSERVATIVE rows), so no text-derived
sentiment is computed here — sentiment would be an AI_OPINION requiring forward validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import full_like_nan, memo, safe_div

_SRC_N = "bundle.news (count of items by first usable session of available_at)"


def _daily_counts(fs: FeatureSet) -> pd.DataFrame:
    """Items attributed to the first session whose cutoff is >= available_at (0 where none)."""
    def build() -> pd.DataFrame:
        p = fs.panel
        out = pd.DataFrame(0.0, index=p.dates, columns=p.symbols)
        n = fs.bundle.news
        if n.empty:
            return out
        n = n[n["symbol"].isin(p.symbols)]
        if n.empty:
            return out
        sess = fs.bundle.calendar.first_usable_sessions(n["available_at"])
        counts = pd.DataFrame({"s": sess.to_numpy(), "sym": n["symbol"].to_numpy()}).dropna()
        counts = counts.groupby(["s", "sym"]).size().unstack(fill_value=0)
        counts = counts.reindex(index=p.dates, columns=p.symbols, fill_value=0).astype("float64")
        return out.add(counts, fill_value=0.0)
    return memo(fs, "news_daily_counts", build)  # type: ignore[return-value]


@FEATURES.feature("news_count_1d", "news", "items usable in (cutoff(D-1), cutoff(D)]", _SRC_N, PitStatus.PIT_CONSERVATIVE)
def news_count_1d(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)          # no news source => UNKNOWN, not zero
    return _daily_counts(fs)


@FEATURES.feature("news_count_5d", "news", "items usable in the last 5 sessions", _SRC_N, PitStatus.PIT_CONSERVATIVE, lookback=4)
def news_count_5d(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)
    return _daily_counts(fs).rolling(5, min_periods=5).sum()


@FEATURES.feature("news_count_z", "news", "(news_count_1d - mean of previous 60 sessions) / their std",
                  _SRC_N, PitStatus.PIT_CONSERVATIVE, lookback=60)
def news_count_z(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)
    c = _daily_counts(fs)
    prev = c.shift(1).rolling(60, min_periods=60)
    sd = prev.std()
    return safe_div(c - prev.mean(), sd.where(sd > 0, np.nan))
