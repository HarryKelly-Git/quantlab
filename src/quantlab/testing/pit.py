"""Point-in-time leakage detector: the TRUNCATION-INVARIANCE test.

If a computation only uses information available at each date, then computing it on the full
history and reading the value at date D must give exactly the same answer as computing it on the
data truncated at D (``bundle.truncate(D)``). Any centered window, negative shift, backfill,
full-sample normalization, future-dependent universe, or late-arriving event breaks this.

Every feature, signal, universe rule, regime metric and ML feature matrix in QuantLab is run
through this check in the test-suite.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from quantlab.data.panel import DataBundle


class LookaheadError(AssertionError):
    pass


def _row_at(obj: Any, d: pd.Timestamp) -> pd.Series:
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex):
            lvl = obj.index.get_level_values(0)
            s = obj[lvl == d]
            s.index = s.index.droplevel(0)
            return s.sort_index()
        return pd.Series({"value": obj.get(d, np.nan)})
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.MultiIndex):
            lvl = obj.index.get_level_values(0)
            sub = obj[lvl == d]
            sub.index = sub.index.droplevel(0)
            return sub.stack(future_stack=True).sort_index()
        if d not in obj.index:
            return pd.Series(dtype="float64")
        return obj.loc[d].sort_index()
    raise TypeError(f"unsupported output type {type(obj)!r}")


def _compare(a: pd.Series, b: pd.Series, atol: float, rtol: float) -> list[str]:
    idx = a.index.union(b.index)
    a, b = a.reindex(idx), b.reindex(idx)
    bad: list[str] = []
    for k in idx:
        x, y = a.get(k), b.get(k)
        xn = x is None or (isinstance(x, float) and np.isnan(x)) or (x is pd.NA)
        yn = y is None or (isinstance(y, float) and np.isnan(y)) or (y is pd.NA)
        if xn and yn:
            continue
        if xn != yn:
            bad.append(f"{k}: full={x!r} truncated={y!r}")
            continue
        try:
            if not np.isclose(float(x), float(y), atol=atol, rtol=rtol):
                bad.append(f"{k}: full={x!r} truncated={y!r}")
        except (TypeError, ValueError):
            if x != y:
                bad.append(f"{k}: full={x!r} truncated={y!r}")
    return bad


def sample_dates(bundle: DataBundle, n: int = 5, min_history: int = 60) -> list[pd.Timestamp]:
    dates = bundle.panel.dates[min_history:]
    if len(dates) == 0:
        raise ValueError("not enough history to sample check dates")
    picks = np.linspace(0, len(dates) - 1, num=min(n, len(dates))).round().astype(int)
    return [dates[i] for i in sorted(set(picks))]


def assert_truncation_invariant(
    compute: Callable[[DataBundle], Any],
    bundle: DataBundle,
    check_dates: Iterable[pd.Timestamp] | None = None,
    n_dates: int = 5,
    min_history: int = 60,
    atol: float = 1e-9,
    rtol: float = 1e-7,
    name: str = "computation",
) -> None:
    """Raise LookaheadError if ``compute`` gives a different value at D on truncated data."""
    full = compute(bundle)
    dates = list(check_dates) if check_dates is not None else sample_dates(bundle, n_dates, min_history)
    problems: list[str] = []
    for d in dates:
        d = pd.Timestamp(d)
        trunc = compute(bundle.truncate(d))
        bad = _compare(_row_at(full, d), _row_at(trunc, d), atol, rtol)
        if bad:
            problems.append(f"{d.date()}: {len(bad)} mismatches, e.g. {bad[:3]}")
    if problems:
        raise LookaheadError(f"{name} is NOT point-in-time safe:\n  " + "\n  ".join(problems))
