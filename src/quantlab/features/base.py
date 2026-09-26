"""Feature framework: specs, a global registry, and a lazy per-bundle FeatureSet.

A feature is a function ``fn(fs: FeatureSet) -> DataFrame`` returning a WIDE frame
(index = bundle sessions, columns = bundle symbols; market-level features return a single-column
frame named "__market__"). It may read ``fs.bundle`` and other features via ``fs.get(name)``.

Rules (enforced by tests/features via the truncation-invariance check):
  * Only backward-looking operations (rolling windows, shift(+n), expanding, per-date cross
    sections). Never shift(-n), centered windows, bfill, or full-sample statistics.
  * Price LEVELS -> raw fields (``close``, ``dollar_volume``). RATIOS/RETURNS -> tri-scaled fields
    (``aclose`` etc.) or ``ret``. Never mix (e.g. raw close / adjusted MA).
  * Declare the honest ``pit_status``. Features built on ASSUMED_STATIC data (e.g. current sector
    membership) must say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.data.panel import DataBundle

MARKET_COLUMN = "__market__"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str                    # price | volume | relative | event | fundamental | news | market
    description: str              # exact calculation in words
    source: str                   # which bundle fields / datasets it reads
    pit_status: PitStatus
    lookback: int                 # sessions of history needed before the first valid value
    fn: Callable[["FeatureSet"], pd.DataFrame] = field(compare=False, repr=False)
    market_level: bool = False    # True => single "__market__" column


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._specs:
            raise ValueError(f"duplicate feature name {spec.name!r}")
        self._specs[spec.name] = spec
        return spec

    def feature(self, name: str, group: str, description: str, source: str,
                pit_status: PitStatus = PitStatus.PIT, lookback: int = 0, market_level: bool = False):
        def deco(fn: Callable[["FeatureSet"], pd.DataFrame]):
            self.register(FeatureSpec(name, group, description, source, pit_status, lookback, fn, market_level))
            return fn
        return deco

    def spec(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown feature {name!r}; registered: {sorted(self._specs)}") from None

    def names(self, group: str | None = None) -> list[str]:
        return sorted(n for n, s in self._specs.items() if group is None or s.group == group)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def catalog(self) -> pd.DataFrame:
        return pd.DataFrame([{k: getattr(s, k) for k in ("name", "group", "description", "source", "pit_status", "lookback")}
                             for s in self._specs.values()]).sort_values(["group", "name"]).reset_index(drop=True)


# The default registry. Feature modules register into it at import time; ``load_all_features``
# imports them all.
FEATURES = FeatureRegistry()


def load_all_features() -> FeatureRegistry:
    import importlib
    for mod in ("price", "volume", "relative", "event", "fundamental", "news", "market", "industry"):
        try:
            importlib.import_module(f"quantlab.features.{mod}")
        except ModuleNotFoundError as exc:  # a group may not exist yet
            if exc.name != f"quantlab.features.{mod}":
                raise
    return FEATURES


class FeatureSet:
    """Lazy, cached feature computation over one DataBundle (full-history or truncated)."""

    def __init__(self, bundle: DataBundle, registry: FeatureRegistry | None = None,
                 universe: pd.DataFrame | None = None, dtype: str = "float64"):
        self.bundle = bundle
        self.registry = registry or load_all_features()
        self.universe = universe          # optional dates x symbols bool mask (for cross-sectional ranks)
        self.dtype = dtype
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def panel(self):
        return self.bundle.panel

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            spec = self.registry.spec(name)
            out = spec.fn(self)
            if not isinstance(out, pd.DataFrame):
                raise TypeError(f"feature {name} must return a DataFrame")
            if spec.market_level:
                out = out.reindex(index=self.panel.dates)
            else:
                out = out.reindex(index=self.panel.dates, columns=self.panel.symbols)
            self._cache[name] = out.astype(self.dtype)
        return self._cache[name]

    def market(self, name: str) -> pd.Series:
        """Market-level feature as a Series indexed by date."""
        df = self.get(name)
        return df[MARKET_COLUMN] if MARKET_COLUMN in df.columns else df.iloc[:, 0]

    def many(self, names: Iterable[str]) -> dict[str, pd.DataFrame]:
        return {n: self.get(n) for n in names}

    def cross_section(self, date, names: Iterable[str]) -> pd.DataFrame:
        """symbols x features on one session (market-level features broadcast)."""
        d = pd.Timestamp(date)
        cols = {}
        for n in names:
            df = self.get(n)
            if self.registry.spec(n).market_level:
                cols[n] = pd.Series(df.loc[d].iloc[0], index=self.panel.symbols)
            else:
                cols[n] = df.loc[d]
        return pd.DataFrame(cols)

    def stack(self, names: Iterable[str], mask: pd.DataFrame | None = None) -> pd.DataFrame:
        """Long (date, symbol) x features frame for ML. ``mask`` restricts rows (e.g. universe)."""
        names = list(names)
        parts = {}
        for n in names:
            df = self.get(n)
            if self.registry.spec(n).market_level:
                df = pd.DataFrame(np.repeat(df.iloc[:, [0]].to_numpy(), len(self.panel.symbols), axis=1),
                                  index=df.index, columns=self.panel.symbols)
            if mask is not None:
                df = df.where(mask.reindex_like(df).fillna(False).astype(bool))
            parts[n] = df.stack(future_stack=True)
        out = pd.DataFrame(parts)
        out.index.names = ["date", "symbol"]
        return out.dropna(how="all")

    def pit_status(self, names: Iterable[str]) -> PitStatus:
        return PitStatus.weakest([self.registry.spec(n).pit_status for n in names])
