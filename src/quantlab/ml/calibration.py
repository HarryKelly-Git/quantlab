"""Probability calibration (isotonic / Platt) and calibration metrics.

Why calibrate at all: downstream layers (EV, no-trade rules) may read ``MLPrediction.probability``
as a probability. Tree ensembles and regularized logits are systematically mis-scaled, so a raw
score is only a ranking. A calibrated probability is only claimed (``calibrated=True``) when a
calibrator was actually fitted on enough data.

Point-in-time rule: the calibrator is fitted on the MOST RECENT time-ordered slice of the TRAIN
window only (see :func:`calibration_split` and ``models.fit_model``), never on test data, and the
model itself is fitted on the earlier part with rows whose labels reach into the calibration
slice purged. Otherwise the calibrator would learn from labels the model has effectively seen and
come out overconfident.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from quantlab.core.types import MLPrediction

METHODS = ("isotonic", "platt", "none")
_EPS = 1e-6


def _arrays(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype="float64").ravel()
    y = np.asarray(y, dtype="float64").ravel()
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have the same length")
    ok = np.isfinite(p) & np.isfinite(y)
    return p[ok], y[ok]


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def brier_score(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p, y, eps: float = 1e-15) -> float:
    p, y = _arrays(p, y)
    if not len(p):
        return float("nan")
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _bin_ids(p: np.ndarray, n_bins: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")
    ids = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return ids, edges


def reliability_table(p, y, n_bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    """Per-bin mean predicted probability vs observed frequency (empty bins omitted)."""
    p, y = _arrays(p, y)
    cols = ["bin", "lo", "hi", "count", "p_mean", "y_rate", "gap"]
    if not len(p):
        return pd.DataFrame(columns=cols)
    ids, edges = _bin_ids(p, n_bins, strategy)
    rows = []
    for b in np.unique(ids):
        m = ids == b
        pm, yr = float(p[m].mean()), float(y[m].mean())
        rows.append((int(b), float(edges[b]), float(edges[b + 1]), int(m.sum()), pm, yr, yr - pm))
    return pd.DataFrame(rows, columns=cols)


def expected_calibration_error(p, y, n_bins: int = 10, strategy: str = "uniform") -> float:
    """ECE = sum over bins of (n_b / N) * |observed rate - mean predicted probability|."""
    t = reliability_table(p, y, n_bins, strategy)
    if t.empty:
        return float("nan")
    return float((t["count"] * t["gap"].abs()).sum() / t["count"].sum())


def calibration_report(p, y, n_bins: int = 10) -> dict[str, Any]:
    pa, ya = _arrays(p, y)
    base = float(ya.mean()) if len(ya) else float("nan")
    return {
        "n": int(len(pa)),
        "base_rate": base,
        "brier": brier_score(pa, ya),
        "brier_climatology": base * (1 - base) if len(ya) else float("nan"),
        "log_loss": log_loss(pa, ya),
        "ece": expected_calibration_error(pa, ya, n_bins),
        "reliability": reliability_table(pa, ya, n_bins).to_dict(orient="records"),
    }


# ------------------------------------------------------------------------------------------------
# Calibrator
# ------------------------------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    """Maps raw model probabilities to calibrated ones. Identity (and ``calibrated=False``) when
    it could not be fitted honestly; ``reason`` says why."""

    method: str = "isotonic"
    fitted: bool = False
    n_rows: int = 0
    reason: str = ""
    fit_report: dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method!r}; expected {METHODS}")

    @property
    def calibrated(self) -> bool:
        return self.fitted and self.method != "none"

    def fit(self, p_raw, y, min_rows: int = 200) -> "Calibrator":
        p, yv = _arrays(p_raw, y)
        self.n_rows = int(len(p))
        if self.method == "none":
            self.fitted, self.reason = False, "calibration disabled (method=none)"
            return self
        if len(p) < min_rows:
            self.fitted, self.reason = False, f"too few calibration rows ({len(p)} < {min_rows})"
            return self
        if len(np.unique(yv)) < 2:
            self.fitted, self.reason = False, "calibration slice has a single class"
            return self
        if self.method == "isotonic":
            m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            m.fit(p, yv)
        else:  # platt: logistic regression on the raw logit (nearly unregularized)
            m = LogisticRegression(C=1e6, max_iter=1000)
            m.fit(_logit(p).reshape(-1, 1), yv.astype(int))
        self._model = m
        self.fitted, self.reason = True, ""
        after = self.transform(p)
        self.fit_report = {"ece_before": expected_calibration_error(p, yv),
                           "ece_after_in_slice": expected_calibration_error(after, yv),
                           "brier_before": brier_score(p, yv), "brier_after_in_slice": brier_score(after, yv)}
        return self

    def transform(self, p_raw) -> np.ndarray:
        p = np.asarray(p_raw, dtype="float64").ravel()
        if not self.calibrated:
            return p.copy()
        out = np.full_like(p, np.nan)
        ok = np.isfinite(p)
        if ok.any():
            if self.method == "isotonic":
                out[ok] = self._model.predict(p[ok])
            else:
                out[ok] = self._model.predict_proba(_logit(p[ok]).reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    def describe(self) -> dict[str, Any]:
        return {"method": self.method, "calibrated": self.calibrated, "n_rows": self.n_rows,
                "reason": self.reason, **self.fit_report}


def fit_calibrator(method: str, p_raw, y, min_rows: int = 200) -> Calibrator:
    return Calibrator(method=method).fit(p_raw, y, min_rows=min_rows)


def calibration_split(row_dates: pd.DatetimeIndex | np.ndarray, fraction: float) -> pd.Timestamp | None:
    """First session of the most recent ``fraction`` of the unique TRAIN sessions.

    Returns None when the fraction leaves no calibration or no fitting sessions.
    """
    if not 0.0 < fraction < 1.0:
        return None
    u = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(row_dates))).sort_values()
    n_cal = int(np.floor(len(u) * fraction))
    if n_cal < 1 or n_cal >= len(u):
        return None
    return u[len(u) - n_cal]


def to_ml_prediction(model_id: str, model_version: str, target: str, probability: float | None,
                     calibrated: bool, status: str = "ACTIVE") -> MLPrediction:
    """Wrap one probability into the shared ``MLPrediction`` record (None => UNKNOWN)."""
    if probability is not None and not np.isfinite(probability):
        probability = None
    return MLPrediction(model_id=model_id, model_version=model_version, target=target,
                        probability=None if probability is None else float(probability),
                        calibrated=bool(calibrated and probability is not None), status=status)
