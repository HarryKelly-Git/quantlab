"""Move distributions for the stock-vs-option comparator.

A :class:`MoveDistribution` is the comparator's ONLY view of the future: an array of underlying
returns at the horizon (``end_returns`` = S_h / spot - 1) with optional weights, and optionally the
path: ``max_favourable`` / ``max_adverse`` (both >= 0) are the largest move in / against the thesis
direction at any point within the horizon.

:func:`empirical_distribution` builds the default used by the CLI: every overlapping
``horizon``-session window of the underlying's own recent history. It is UNCONDITIONAL (it assumes
no edge: the thesis adds nothing to history), overlapping windows are not independent, and the
bars are split/dividend adjusted because they only describe returns already known today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class DistributionError(ValueError):
    pass


@dataclass(frozen=True)
class MoveDistribution:
    end_returns: np.ndarray
    weights: np.ndarray | None = None
    max_favourable: np.ndarray | None = None
    max_adverse: np.ndarray | None = None
    label: str = "user-supplied"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        r = np.asarray(self.end_returns, dtype=float).ravel()
        if r.size == 0 or not np.all(np.isfinite(r)) or np.any(r <= -1.0):
            raise DistributionError("end_returns must be a non-empty array of finite returns > -100%")
        object.__setattr__(self, "end_returns", r)
        if self.weights is None:
            w = np.full(r.size, 1.0 / r.size)
        else:
            w = np.asarray(self.weights, dtype=float).ravel()
            if w.size != r.size or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise DistributionError("weights must be finite, >= 0, one per return, with a positive sum")
            w = w / w.sum()
        object.__setattr__(self, "weights", w)
        for name in ("max_favourable", "max_adverse"):
            v = getattr(self, name)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if v.size != r.size or not np.all(np.isfinite(v)) or np.any(v < 0):
                raise DistributionError(f"{name} must be finite, >= 0 and one per return")
            object.__setattr__(self, name, v)

    @property
    def path_based(self) -> bool:
        return self.max_favourable is not None

    @property
    def n(self) -> int:
        return int(self.end_returns.size)

    def mean(self) -> float:
        return float(np.dot(self.weights, self.end_returns))

    def std(self) -> float:
        m = self.mean()
        return float(np.sqrt(np.dot(self.weights, (self.end_returns - m) ** 2)))

    def to_dict(self, max_points: int = 5000) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "n": self.n, "path_based": self.path_based,
                             "has_max_adverse": self.max_adverse is not None, "mean": self.mean(), "std": self.std(),
                             "meta": self.meta}
        if self.n <= max_points:
            d["end_returns"] = [round(float(x), 6) for x in self.end_returns]
            d["weights"] = [round(float(x), 8) for x in self.weights]
            if self.max_favourable is not None:
                d["max_favourable"] = [round(float(x), 6) for x in self.max_favourable]
            if self.max_adverse is not None:
                d["max_adverse"] = [round(float(x), 6) for x in self.max_adverse]
        else:
            d["truncated"] = True
        return d


def empirical_distribution(bars: pd.DataFrame, horizon: int, direction: str, lookback: int) -> MoveDistribution:
    """Overlapping ``horizon``-session windows over the last ``lookback`` sessions of ``bars``
    (columns date/high/low/close, adjusted). Entry at a session's close; path from the next
    session's high/low through the horizon session."""
    if horizon < 1:
        raise DistributionError("horizon must be >= 1 session")
    b = bars.dropna(subset=["close", "high", "low"]).sort_values("date").tail(lookback + horizon).reset_index(drop=True)
    n = len(b) - horizon
    if n < 20:
        raise DistributionError(f"only {max(n, 0)} complete {horizon}-session windows in the bars (need >= 20)")
    close, high, low = (b[c].to_numpy(dtype=float) for c in ("close", "high", "low"))
    end = np.array([close[i + horizon] / close[i] - 1.0 for i in range(n)])
    hi = np.array([high[i + 1:i + horizon + 1].max() / close[i] - 1.0 for i in range(n)])
    lo = np.array([1.0 - low[i + 1:i + horizon + 1].min() / close[i] for i in range(n)])
    up, down = np.maximum(hi, 0.0), np.maximum(lo, 0.0)
    fav, adv = (up, down) if direction == "LONG" else (down, up)
    return MoveDistribution(end_returns=end, max_favourable=fav, max_adverse=adv,
                            label="EMPIRICAL_UNCONDITIONAL",
                            meta={"windows": n, "horizon": horizon, "first": str(pd.Timestamp(b["date"].iloc[0]).date()),
                                  "last": str(pd.Timestamp(b["date"].iloc[-1]).date()), "overlapping": True,
                                  "note": "unconditional history of the underlying: assumes NO edge; overlapping "
                                          "windows are not independent"})


def atr(bars: pd.DataFrame, n: int = 14) -> float | None:
    b = bars.dropna(subset=["close", "high", "low"]).sort_values("date")
    if len(b) < n + 1:
        return None
    prev = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - prev).abs(), (b["low"] - prev).abs()], axis=1).max(axis=1)
    v = float(tr.tail(n).mean())
    return v if np.isfinite(v) and v > 0 else None


def as_distribution(x: Any, label: str = "ndarray") -> MoveDistribution:
    """Accept a MoveDistribution or a plain array of end returns (e.g. the 21 equal-weight quantiles
    from ``exploration.upside.move_distribution``). NOTE: those quantiles are the STOCK trade's net
    returns (stop + time exit, after costs); used here as underlying end returns they already carry
    the stop truncation and costs, so the stock leg is charged twice and option payoffs never see
    moves beyond the stop. Labelled as such; prefer raw underlying returns when available."""
    if isinstance(x, MoveDistribution):
        return x
    return MoveDistribution(end_returns=np.asarray(x, dtype=float), label=label)
