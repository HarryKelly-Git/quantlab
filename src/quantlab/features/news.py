"""News-volume features. Counts only: an item exists from its ``available_at`` (creation time).

Headline/summary text may be revised after creation (PIT_CONSERVATIVE rows), so no text-derived
sentiment is computed here — sentiment would be an AI_OPINION requiring forward validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import active_from, full_like_nan, memo, safe_div

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
    return active_from(fs, fs.bundle.news["available_at"], _daily_counts(fs))


@FEATURES.feature("news_count_5d", "news", "items usable in the last 5 sessions", _SRC_N, PitStatus.PIT_CONSERVATIVE, lookback=4)
def news_count_5d(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)
    return active_from(fs, fs.bundle.news["available_at"], _daily_counts(fs).rolling(5, min_periods=5).sum())


@FEATURES.feature("news_count_z", "news", "(news_count_1d - mean of previous 60 sessions) / their std",
                  _SRC_N, PitStatus.PIT_CONSERVATIVE, lookback=60)
def news_count_z(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)
    c = active_from(fs, fs.bundle.news["available_at"], _daily_counts(fs))
    prev = c.shift(1).rolling(60, min_periods=60)
    sd = prev.std()
    return safe_div(c - prev.mean(), sd.where(sd > 0, np.nan))


def _classified_counts(fs: FeatureSet, col: str) -> pd.DataFrame:
    """Counts of items with ``col`` True (company-specific / material), by first usable session.
    Tags are counted over ALL stored rows of an article before restricting to panel symbols."""
    def build() -> dict[str, pd.DataFrame]:
        from quantlab.data.news_classify import classify_frame
        p = fs.panel
        base = pd.DataFrame(0.0, index=p.dates, columns=p.symbols)
        n = classify_frame(fs.bundle.news)
        n = n[n["symbol"].isin(p.symbols)]
        out = {}
        for c in ("company_specific", "material"):
            sub = n[n[c]]
            if sub.empty:
                out[c] = base.copy()
                continue
            sess = fs.bundle.calendar.first_usable_sessions(sub["available_at"])
            k = pd.DataFrame({"s": sess.to_numpy(), "sym": sub["symbol"].to_numpy()}).dropna()
            k = k.groupby(["s", "sym"]).size().unstack(fill_value=0).reindex(index=p.dates, columns=p.symbols,
                                                                              fill_value=0)
            out[c] = base.add(k.astype("float64"), fill_value=0.0)
        return out
    return memo(fs, "news_classified_counts", build)[col]  # type: ignore[index]


_SRC_NC = _SRC_N + "; headline rules in quantlab.data.news_classify; tags counted over all rows of an article"


@FEATURES.feature("news_company_1d", "news", "company-specific items (article tagged with <= 2 symbols) usable at D",
                  _SRC_NC, PitStatus.PIT_CONSERVATIVE)
def news_company_1d(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)
    return active_from(fs, fs.bundle.news["available_at"], _classified_counts(fs, "company_specific"))


@FEATURES.feature("news_material_1d", "news",
                  "company-specific items whose headline category is a company event (earnings, guidance, m&a, "
                  "financing, contract, management, regulatory, legal, product, capital_return) usable at D",
                  _SRC_NC, PitStatus.PIT_CONSERVATIVE)
def news_material_1d(fs: FeatureSet) -> pd.DataFrame:
    if fs.bundle.news.empty:
        return full_like_nan(fs)
    return active_from(fs, fs.bundle.news["available_at"], _classified_counts(fs, "material"))
