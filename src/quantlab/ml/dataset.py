"""Point-in-time ML datasets.

A dataset row is a (signal session D, symbol) pair. It has two halves with opposite time rules:

* **Features X** must be computable at cutoff(D). They come ONLY from a
  :class:`~quantlab.features.base.FeatureSet`, whose features are individually truncation-invariant.
  :func:`build_features` adds nothing but per-row selection (universe mask, date range, dropping
  all-NaN rows), each of which depends only on information at D. So X at D is identical whether
  it is built from the full bundle or from ``bundle.truncate(D)``; tests/ml prove this.
* **Labels y** use the future by definition. The trade convention is the house one (signal at the
  close of D, entry at the open of D+1), so the label is the return from open D+1 to close
  D+h minus the benchmark's return over the same interval. It is only *known* at the close of
  D+h, which is recorded per row as ``label_end``. Walk-forward purging (``ml.walkforward``) uses
  ``label_end`` to drop training rows whose label overlaps a test period.

Survivorship in labels: a symbol that stops trading inside the horizon would otherwise have a NaN
label and silently drop out, biasing labels upward. Following ARCHITECTURE.md section 2.10, its
label is the last close x (1 + ``costs.delisting_return``) relative to the entry.

Guards (fail loudly rather than train on leaked data): feature names that look like labels
('fwd', 'forward', 'future') are refused, as are features whose values track the label almost
perfectly, and features whose weakest PIT status is not allowed for historical research.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from quantlab.core.types import HISTORICAL_RESEARCH_PIT, PitStatus
from quantlab.data.panel import DataBundle, Panel
from quantlab.features.base import FeatureSet
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

TARGETS = ("positive_excess_return", "forward_excess_return", "hit_target")
CLASSIFICATION_TARGETS = frozenset({"positive_excess_return", "hit_target"})
FORBIDDEN_NAME_TOKENS = ("fwd", "forward", "future")
FORBIDDEN_SOURCE_TOKENS = ("forward_returns", "shift(-")
DEFAULT_DELISTING_RETURN = -0.30      # mirrors costs.delisting_return; used only when no config is given
DEFAULT_HIT_TARGET_RETURN = 0.05
DEFAULT_LEAK_CORR = 0.95
FROM_CONFIG: Any = object()           # sentinel: "read this parameter from config"


class DatasetError(ValueError):
    """The dataset cannot be built honestly (bad inputs, missing benchmark, empty result)."""


class LabelLeakError(DatasetError):
    """A feature looks like (or is built from) the label."""


class PitStatusError(DatasetError):
    """A feature's PIT status is not allowed for this use (e.g. UNKNOWN in historical research)."""


def task_for_target(target: str) -> str:
    if target not in TARGETS:
        raise DatasetError(f"unknown target {target!r}; expected one of {TARGETS}")
    return "classification" if target in CLASSIFICATION_TARGETS else "regression"


@dataclass
class MLDataset:
    """Aligned rows indexed by (date, symbol).

    ``X`` features (PIT), ``y`` target, ``label_end`` session at whose close the label is fully
    known, ``fwd_excess`` the forward excess return (kept for trading-relevance evaluation even
    when ``y`` is binary). ``fwd_excess`` and ``y`` are LABEL information: never feed them to a
    model as inputs.
    """

    X: pd.DataFrame
    y: pd.Series
    label_end: pd.Series
    fwd_excess: pd.Series
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("y", "label_end", "fwd_excess"):
            s = getattr(self, name)
            if not s.index.equals(self.X.index):
                raise DatasetError(f"MLDataset.{name} index is not aligned with X")
        if list(self.X.index.names) != ["date", "symbol"]:
            raise DatasetError("MLDataset.X must be indexed by (date, symbol)")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def target(self) -> str:
        return str(self.meta.get("target", "positive_excess_return"))

    @property
    def task(self) -> str:
        return task_for_target(self.target)

    @property
    def horizon(self) -> int:
        return int(self.meta.get("horizon", 0))

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    @property
    def row_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.X.index.get_level_values("date"))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.row_dates.unique()).sort_values()

    def subset(self, mask: np.ndarray | pd.Series, **meta_updates: Any) -> "MLDataset":
        m = np.asarray(mask, dtype=bool)
        meta = {**self.meta, **meta_updates}
        out = MLDataset(self.X[m], self.y[m], self.label_end[m], self.fwd_excess[m], meta)
        out.meta["n_rows"] = len(out)
        return out

    def before(self, cutoff) -> "MLDataset":
        """Rows whose signal date AND label end are strictly before ``cutoff``.

        Use this to keep research clear of the locked holdout: a row dated before the holdout
        whose label ends inside it still reads holdout prices.
        """
        c = pd.Timestamp(cutoff)
        keep = (self.row_dates < c) & (pd.DatetimeIndex(self.label_end) < c)
        return self.subset(keep, truncated_before=c.date().isoformat())

    def between(self, start=None, end=None) -> "MLDataset":
        d = self.row_dates
        keep = np.ones(len(d), dtype=bool)
        if start is not None:
            keep &= d >= pd.Timestamp(start)
        if end is not None:
            keep &= d <= pd.Timestamp(end)
        return self.subset(keep)


# ------------------------------------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------------------------------------
def check_feature_names(names: Sequence[str]) -> None:
    """Refuse label-like feature names. Cheap, catches the most common accidental leak."""
    if len(set(names)) != len(names):
        raise DatasetError(f"duplicate feature names: {list(names)}")
    for n in names:
        low = n.lower()
        for tok in FORBIDDEN_NAME_TOKENS:
            if tok in low:
                raise LabelLeakError(f"feature {n!r} contains {tok!r}: label-like names are refused as ML inputs")


def _check_feature_sources(fs: FeatureSet, names: Sequence[str]) -> None:
    for n in names:
        spec = fs.registry.spec(n)
        src = f"{spec.source} {spec.description}".lower().replace(" ", "")
        for tok in FORBIDDEN_SOURCE_TOKENS:
            if tok in src:
                raise LabelLeakError(f"feature {n!r} declares a future-looking source ({tok!r})")


def check_label_leak(X: pd.DataFrame, labels: dict[str, pd.Series], threshold: float = DEFAULT_LEAK_CORR,
                     min_rows: int = 30) -> None:
    """Raise if any X column is (nearly) a copy of a label column.

    Legitimate predictive features correlate with forward returns at |r| ~ 0.01-0.1. |r| above
    ``threshold`` means the column was built from the label (or the label from it).
    """
    for col in X.columns:
        x = X[col].to_numpy(dtype="float64")
        for lname, lab in labels.items():
            yv = lab.to_numpy(dtype="float64")
            ok = np.isfinite(x) & np.isfinite(yv)
            if ok.sum() < min_rows:
                continue
            xs, ys = x[ok], yv[ok]
            if np.std(xs) == 0 or np.std(ys) == 0:
                continue
            r = float(np.corrcoef(xs, ys)[0, 1])
            if abs(r) >= threshold:
                raise LabelLeakError(
                    f"feature {col!r} has |corr|={abs(r):.3f} with label {lname!r}: X appears to be built from labels")


def check_pit(fs: FeatureSet, names: Sequence[str], allowed: Iterable[PitStatus]) -> PitStatus:
    status = fs.pit_status(names) if names else PitStatus.UNKNOWN
    allowed = frozenset(PitStatus(a) for a in allowed)
    if status not in allowed:
        raise PitStatusError(f"weakest feature PIT status {status.value} not in allowed "
                             f"{sorted(a.value for a in allowed)}")
    return status


# ------------------------------------------------------------------------------------------------
# Features (PIT half)
# ------------------------------------------------------------------------------------------------
def _benchmark_symbols(bundle: DataBundle) -> set[str]:
    out = {bundle.market_symbol}
    out.update(bundle.sector_etfs.keys())
    return out


def universe_mask(fs: FeatureSet, universe: pd.DataFrame | None) -> pd.DataFrame:
    """Boolean (dates x symbols) row mask aligned to the FeatureSet panel.

    ``None`` means ``fs.universe`` if set, else every listed symbol except the benchmark/sector
    ETFs. Missing entries count as NOT in the universe (fail-safe).
    """
    panel = fs.panel
    if universe is None:
        universe = fs.universe
    if universe is None:
        m = panel.listed.copy()
        drop = [c for c in m.columns if c in _benchmark_symbols(fs.bundle)]
        if drop:
            m[drop] = False
        return m
    u = universe.reindex(index=panel.dates, columns=panel.symbols)
    return u.eq(True)


def build_features(
    fs: FeatureSet,
    feature_names: Sequence[str],
    universe: pd.DataFrame | None = None,
    start=None,
    end=None,
    allowed_pit: Iterable[PitStatus] = HISTORICAL_RESEARCH_PIT,
) -> tuple[pd.DataFrame, PitStatus]:
    """Long (date, symbol) x features frame built ONLY from the FeatureSet (PIT).

    Rows: universe members on sessions in [start, end] with at least one finite feature.
    Non-finite values (inf from divisions) become NaN; models impute on training data only.
    """
    names = list(feature_names)
    if not names:
        raise DatasetError("at least one feature is required")
    check_feature_names(names)
    _check_feature_sources(fs, names)
    status = check_pit(fs, names, allowed_pit)

    panel = fs.panel
    dates, symbols = panel.dates, panel.symbols
    mask = universe_mask(fs, universe).to_numpy(dtype=bool)
    if start is not None:
        mask[np.asarray(dates < pd.Timestamp(start))] = False
    if end is not None:
        mask[np.asarray(dates > pd.Timestamp(end))] = False
    ri, ci = np.nonzero(mask)

    cols: dict[str, np.ndarray] = {}
    for n in names:
        arr = fs.get(n).to_numpy(dtype="float64")
        if fs.registry.spec(n).market_level:
            v = arr[ri, 0]
        else:
            v = arr[ri, ci]
        cols[n] = np.where(np.isfinite(v), v, np.nan)
    index = pd.MultiIndex.from_arrays([dates[ri], symbols[ci]], names=["date", "symbol"])
    X = pd.DataFrame(cols, index=index, columns=names)
    X = X[X.notna().any(axis=1)]
    return X, status


def features_for_candidates(fs: FeatureSet, feature_names: Sequence[str], candidates: Sequence[Any]
                            ) -> tuple[pd.DataFrame, list[str | None]]:
    """X rows for pipeline candidates (objects with ``symbol``, ``as_of_date``, ``candidate_id``).

    One row per candidate in input order (several strategies may flag the same symbol), so the
    returned ``candidate_ids`` align with X for :meth:`ModelRegistry.predict_candidates`.
    Unknown symbols/dates give all-NaN rows, which the registry turns into UNKNOWN predictions.
    """
    names = list(feature_names)
    check_feature_names(names)
    rows, keys, ids = [], [], []
    frames = {n: fs.get(n) for n in names}
    for c in candidates:
        d = pd.Timestamp(c.as_of_date)
        sym = c.symbol
        vals = []
        for n in names:
            df = frames[n]
            if d not in df.index:
                vals.append(np.nan)
            elif fs.registry.spec(n).market_level:
                vals.append(float(df.loc[d].iloc[0]))
            elif sym in df.columns:
                vals.append(float(df.at[d, sym]))
            else:
                vals.append(np.nan)
        rows.append([v if np.isfinite(v) else np.nan for v in vals])
        keys.append((d, sym))
        ids.append(getattr(c, "candidate_id", None))
    index = pd.MultiIndex.from_tuples(keys, names=["date", "symbol"]) if keys else \
        pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), pd.Index([], dtype=object)], names=["date", "symbol"])
    return pd.DataFrame(rows, index=index, columns=names, dtype="float64"), ids


# ------------------------------------------------------------------------------------------------
# Labels (future half, by design)
# ------------------------------------------------------------------------------------------------
@dataclass
class Labels:
    """Wide (dates x symbols) label frames plus the per-date label end session."""

    fwd_excess: pd.DataFrame
    y: pd.DataFrame
    label_end: pd.Series          # indexed by signal date; NaT where the horizon runs past the data
    n_delisting_filled: int = 0


def _shift_up(a: np.ndarray, k: int) -> np.ndarray:
    """out[t] = a[t+k] (NaN past the end)."""
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[: len(a) - k] = a[k:]
    return out


def build_labels(
    panel: Panel,
    horizon: int,
    target: str = "positive_excess_return",
    benchmark: str = "SPY",
    delisting_return: float | None = DEFAULT_DELISTING_RETURN,
    hit_target_return: float = DEFAULT_HIT_TARGET_RETURN,
) -> Labels:
    """Forward labels with next-open entry. Uses FUTURE data by definition: labels only.

    * forward return = aclose[D+h] / aopen[D+1] - 1 (``panel.forward_returns(h, 'next_open')``)
    * excess = stock forward return - benchmark forward return over the same interval
    * delisted inside the horizon: last aclose x (1 + delisting_return) / entry - 1
      (``delisting_return=None`` disables this and such rows get no label)
    * hit_target: 1 if any close in D+1..D+h is >= entry x (1 + hit_target_return), else 0
      (absolute, not excess; defined where the excess label is defined)
    """
    task_for_target(target)
    if horizon < 1:
        raise DatasetError("horizon must be >= 1 session")
    if benchmark not in panel.symbols:
        raise DatasetError(f"benchmark {benchmark!r} not in panel: cannot compute excess returns")
    dates = panel.dates
    fwd = panel.forward_returns(horizon, "next_open")
    fwd_np = fwd.to_numpy(dtype="float64").copy()
    n_filled = 0

    ac = panel.aclose.to_numpy(dtype="float64")
    entry = _shift_up(panel.aopen.to_numpy(dtype="float64"), 1)
    if delisting_return is not None:
        # Last bar per symbol in the (label-side, full) panel. Rows whose horizon ends after that
        # bar but inside the data window are delistings (or halts lasting to the end of data).
        has_bar = np.isfinite(ac)
        T = len(dates)
        last_idx = np.where(has_bar.any(axis=0), T - 1 - np.argmax(has_bar[::-1], axis=0), -1)
        t_idx = np.arange(T)[:, None]
        delisted = (
            np.isnan(fwd_np) & np.isfinite(entry)
            & (last_idx[None, :] >= t_idx + 1) & (last_idx[None, :] < t_idx + horizon)
            & (t_idx + horizon <= T - 1)
        )
        if delisted.any():
            last_close = ac[last_idx.clip(min=0), np.arange(ac.shape[1])]
            dl_ret = last_close[None, :] * (1.0 + delisting_return) / entry - 1.0
            fwd_np = np.where(delisted, dl_ret, fwd_np)
            n_filled = int(delisted.sum())

    bench_j = list(panel.symbols).index(benchmark)
    bench = fwd_np[:, bench_j]
    excess = fwd_np - bench[:, None]
    fwd_excess = pd.DataFrame(excess, index=dates, columns=panel.symbols)

    if target == "positive_excess_return":
        yv = np.where(np.isfinite(excess), (excess > 0).astype("float64"), np.nan)
    elif target == "forward_excess_return":
        yv = excess.copy()
    else:  # hit_target
        path_max = np.full_like(ac, -np.inf)
        for k in range(1, horizon + 1):
            path_max = np.fmax(path_max, _shift_up(ac, k))
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = (path_max / entry - 1.0) >= hit_target_return
        yv = np.where(np.isfinite(excess), hit.astype("float64"), np.nan)
    y = pd.DataFrame(yv, index=dates, columns=panel.symbols)
    label_end = pd.Series(pd.DatetimeIndex(dates).to_series().shift(-horizon).to_numpy(), index=dates)
    return Labels(fwd_excess, y, label_end, n_filled)


# ------------------------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------------------------
def dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, X.columns)).encode())
    h.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(y, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def _stack_at(wide: pd.DataFrame, index: pd.MultiIndex) -> np.ndarray:
    ri = wide.index.get_indexer(index.get_level_values("date"))
    ci = wide.columns.get_indexer(index.get_level_values("symbol"))
    if (ri < 0).any() or (ci < 0).any():
        raise DatasetError("label frame does not cover every feature row")
    return wide.to_numpy()[ri, ci]


def build_dataset(
    fs: FeatureSet,
    feature_names: Sequence[str],
    universe: pd.DataFrame | None,
    horizon: int,
    target: str = "positive_excess_return",
    start=None,
    end=None,
    benchmark: str | None = None,
    *,
    config: Any = None,
    allowed_pit: Iterable[PitStatus] = HISTORICAL_RESEARCH_PIT,
    delisting_return: Any = FROM_CONFIG,
    hit_target_return: float | None = None,
) -> MLDataset:
    """PIT feature matrix + forward labels for rows in the universe with an available label.

    ``benchmark`` defaults to the bundle's market symbol (SPY). ``config`` (optional) supplies
    ``costs.delisting_return``, ``ml.hit_target_return`` and ``ml.leak_corr_threshold``.
    ``delisting_return=None`` explicitly disables the delisting label (rows then drop out).
    """
    task = task_for_target(target)
    bench = benchmark or fs.bundle.market_symbol
    if delisting_return is FROM_CONFIG:
        delisting_return = float(config.get("costs.delisting_return", DEFAULT_DELISTING_RETURN)) if config \
            else DEFAULT_DELISTING_RETURN
    if hit_target_return is None:
        hit_target_return = float(config.get("ml.hit_target_return", DEFAULT_HIT_TARGET_RETURN)) if config \
            else DEFAULT_HIT_TARGET_RETURN
    leak_thr = float(config.get("ml.leak_corr_threshold", DEFAULT_LEAK_CORR)) if config else DEFAULT_LEAK_CORR

    X, status = build_features(fs, feature_names, universe, start, end, allowed_pit)
    labels = build_labels(fs.panel, horizon, target, bench, delisting_return, hit_target_return)

    fwd = _stack_at(labels.fwd_excess, X.index)
    yv = _stack_at(labels.y, X.index)
    keep = np.isfinite(fwd) & np.isfinite(yv)
    X = X[keep]
    idx = X.index
    y = pd.Series(yv[keep], index=idx, name=target)
    if task == "classification":
        y = y.astype("int64")
    fwd_s = pd.Series(fwd[keep], index=idx, name="fwd_excess")
    le = labels.label_end.reindex(idx.get_level_values("date")).to_numpy()
    label_end = pd.Series(pd.DatetimeIndex(le), index=idx, name="label_end")
    if len(X) == 0:
        raise DatasetError("dataset is empty after applying universe, date range and label availability")
    if label_end.isna().any():
        raise DatasetError("internal: a labelled row has no label_end")

    check_label_leak(X, {"fwd_excess": fwd_s, target: y.astype("float64")}, threshold=leak_thr)

    row_dates = idx.get_level_values("date")
    meta: dict[str, Any] = {
        "features": list(X.columns),
        "pit_status": status.value,
        "horizon": int(horizon),
        "target": target,
        "task": task,
        "benchmark": bench,
        "entry": "next_open",
        "n_rows": int(len(X)),
        "n_dates": int(row_dates.nunique()),
        "n_symbols": int(idx.get_level_values("symbol").nunique()),
        "start": row_dates.min().date().isoformat(),
        "end": row_dates.max().date().isoformat(),
        "label_end_max": pd.Timestamp(label_end.max()).date().isoformat(),
        "dataset_ids": list(fs.bundle.dataset_ids),
        "is_synthetic": bool(fs.bundle.is_synthetic),
        "delisting_return": delisting_return,
        "n_delisting_labels_in_panel": int(labels.n_delisting_filled),
        "dataset_hash": dataset_hash(X, y),
    }
    if target == "hit_target":
        meta["hit_target_return"] = hit_target_return
    if task == "classification":
        meta["positive_rate"] = float(y.mean())
    log_event(log, "ml dataset built", **{k: meta[k] for k in ("target", "horizon", "n_rows", "n_dates",
                                                              "pit_status", "dataset_hash")})
    return MLDataset(X, y, label_end, fwd_s, meta)
