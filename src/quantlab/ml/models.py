"""Model factory and the fitted-model container.

Kinds (config ``ml.models``): 'logistic' (median imputation + standardization + L2 logistic
regression), 'random_forest' (median imputation + shallow forest with large leaves),
'hist_gradient_boosting' (shallow, regularized; handles NaN natively). 'prior' predicts the
training base rate and is the no-skill baseline every model must beat.

Choices that matter for honesty:
  * Fixed seeds from ``ml.random_seed``, so a model version is reproducible from its inputs.
  * Imputation/scaling live INSIDE the pipeline, so their statistics come from training rows only.
  * HistGradientBoosting early stopping is refused: it holds out a RANDOM (not time-ordered)
    validation split, which leaks future rows into model selection.
  * :func:`fit_model` fits the estimator on the older part of the train window and the calibrator
    on the most recent slice, purging fit rows whose labels end inside the calibration slice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantlab.core.types import utcnow
from quantlab.logging_setup import get_logger, log_event
from quantlab.ml.calibration import Calibrator, calibration_split
from quantlab.ml.monitor import DistributionProfile

log = get_logger(__name__)

MODEL_KINDS = ("logistic", "random_forest", "hist_gradient_boosting", "prior")
DEFAULT_SEED = 11

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic": {"C": 0.5, "max_iter": 2000, "ridge_alpha": 10.0},
    "random_forest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 100, "max_features": "sqrt"},
    "hist_gradient_boosting": {"max_depth": 3, "learning_rate": 0.05, "max_iter": 150, "min_samples_leaf": 200,
                               "l2_regularization": 1.0},
    "prior": {},
}


class ModelError(RuntimeError):
    pass


class InsufficientDataError(ModelError):
    """Not enough (or single-class) training data to fit honestly."""


class MissingFeaturesError(ModelError):
    """Prediction input lacks feature columns the model was trained on."""


def _cfg(config: Any, key: str, default: Any) -> Any:
    return config.get(key, default) if config is not None else default


def model_params(kind: str, config: Any = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if kind not in MODEL_KINDS:
        raise ModelError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    params = dict(DEFAULT_PARAMS[kind])
    params.update(_cfg(config, f"ml.model_params.{kind}", {}) or {})
    params.update(overrides or {})
    if kind == "hist_gradient_boosting" and params.get("early_stopping") not in (None, False):
        raise ModelError("early_stopping uses a random (not time-ordered) validation split: refused")
    return params


def build_model(kind: str, config: Any = None, task: str = "classification", seed: int | None = None,
                params: dict[str, Any] | None = None) -> Any:
    """Unfitted sklearn estimator for ``kind``/``task`` with the configured params and seed."""
    if task not in ("classification", "regression"):
        raise ModelError(f"unknown task {task!r}")
    p = model_params(kind, config, params)
    seed = int(seed if seed is not None else _cfg(config, "ml.random_seed", DEFAULT_SEED))
    n_jobs = int(_cfg(config, "ml.n_jobs", 1))
    clf = task == "classification"
    if kind == "logistic":
        est = LogisticRegression(C=float(p["C"]), max_iter=int(p["max_iter"])) if clf \
            else Ridge(alpha=float(p["ridge_alpha"]))
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()), ("model", est)])
    if kind == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        est = cls(n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
                  min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
                  random_state=seed, n_jobs=n_jobs)
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", est)])
    if kind == "hist_gradient_boosting":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_depth=p["max_depth"], learning_rate=float(p["learning_rate"]), max_iter=int(p["max_iter"]),
                   min_samples_leaf=int(p["min_samples_leaf"]), l2_regularization=float(p["l2_regularization"]),
                   early_stopping=False, random_state=seed)
    return DummyClassifier(strategy="prior") if clf else DummyRegressor(strategy="mean")


def build_models(config: Any, task: str = "classification") -> dict[str, Any]:
    """All configured kinds (``ml.models``) as unfitted estimators."""
    kinds = list(_cfg(config, "ml.models", ["logistic"]))
    return {k: build_model(k, config, task) for k in kinds}


# ------------------------------------------------------------------------------------------------
# Fitted model
# ------------------------------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """A fitted estimator plus everything needed to reproduce, audit and monitor it."""

    kind: str
    task: str
    target: str
    horizon: int
    features: list[str]
    estimator: Any
    calibrator: Calibrator | None
    params: dict[str, Any]
    seed: int
    train_start: str
    train_end: str                  # last signal date used anywhere (fit or calibration)
    fit_end: str                    # last signal date used to fit the estimator
    calib_start: str | None
    n_fit: int
    n_calib: int
    label_end_max: str              # last session whose prices the labels used
    feature_profiles: dict[str, DistributionProfile] = field(default_factory=dict)
    prediction_profile: DistributionProfile | None = None
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utcnow().isoformat())

    @property
    def calibrated(self) -> bool:
        return self.calibrator is not None and self.calibrator.calibrated

    def matrix(self, X: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise MissingFeaturesError(f"missing feature columns {missing}")
        return X[self.features].to_numpy(dtype="float64")

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(y=1) for classification, predicted value for regression."""
        m = self.matrix(X)
        if len(m) == 0:
            return np.array([], dtype="float64")
        if self.task == "classification":
            proba = self.estimator.predict_proba(m)
            classes = list(getattr(self.estimator, "classes_", [0, 1]))
            return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(m))
        return np.asarray(self.estimator.predict(m), dtype="float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability when a calibrator was fitted, else the raw output."""
        raw = self.predict_raw(X)
        if self.task == "classification" and self.calibrated:
            return self.calibrator.transform(raw)
        return raw

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "task": self.task, "target": self.target, "horizon": self.horizon,
            "features": list(self.features), "params": self.params, "seed": self.seed,
            "train_start": self.train_start, "train_end": self.train_end, "fit_end": self.fit_end,
            "calib_start": self.calib_start, "n_fit": self.n_fit, "n_calib": self.n_calib,
            "label_end_max": self.label_end_max, "calibrated": self.calibrated,
            "calibration": self.calibrator.describe() if self.calibrator else None,
            "notes": list(self.notes), "created_at": self.created_at,
        }


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).date().isoformat()


def fit_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    row_dates: pd.DatetimeIndex | np.ndarray,
    label_end: pd.Series | np.ndarray,
    *,
    kind: str,
    config: Any = None,
    task: str = "classification",
    target: str = "positive_excess_return",
    horizon: int = 0,
    calibration: str | None = None,
    calibration_fraction: float | None = None,
    calibration_min_rows: int | None = None,
    min_rows: int | None = None,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> TrainedModel:
    """Fit ``kind`` on a TRAIN window (already purged against any test period by the caller).

    Classification with calibration: the most recent ``calibration_fraction`` of train sessions
    is reserved for the calibrator; the estimator is fitted on earlier rows whose ``label_end`` is
    before the calibration slice starts (purge). If that leaves too little data, the model is
    fitted on everything and marked uncalibrated (never silently "calibrated").
    """
    features = list(X.columns)
    yv = np.asarray(y, dtype="float64")
    dates = pd.DatetimeIndex(row_dates)
    lend = pd.DatetimeIndex(label_end)
    if not (len(X) == len(yv) == len(dates) == len(lend)):
        raise ModelError("X, y, row_dates and label_end must have the same length")
    ok = np.isfinite(yv)
    X, yv, dates, lend = X[ok], yv[ok], dates[ok], lend[ok]
    min_rows = int(min_rows if min_rows is not None else _cfg(config, "ml.min_train_rows", 5000))
    if len(X) < min_rows:
        raise InsufficientDataError(f"{len(X)} training rows < min_train_rows={min_rows}")
    if task == "classification":
        yv = yv.astype(int)
        if len(np.unique(yv)) < 2:
            raise InsufficientDataError("training labels contain a single class")

    method = calibration if calibration is not None else str(_cfg(config, "ml.calibration", "isotonic"))
    frac = float(calibration_fraction if calibration_fraction is not None
                 else _cfg(config, "ml.calibration_fraction", 0.2))
    cal_min = int(calibration_min_rows if calibration_min_rows is not None
                  else _cfg(config, "ml.calibration_min_rows", 200))
    seed = int(seed if seed is not None else _cfg(config, "ml.random_seed", DEFAULT_SEED))
    notes: list[str] = []

    fit_mask = np.ones(len(X), dtype=bool)
    cal_mask = np.zeros(len(X), dtype=bool)
    calib_start = None
    if task == "classification" and method != "none":
        calib_start = calibration_split(dates, frac)
        if calib_start is None:
            notes.append("calibration skipped: train window too short to reserve a slice")
        else:
            cal_mask = np.asarray(dates >= calib_start)
            fit_mask = np.asarray((dates < calib_start) & (lend < calib_start))
            enough = fit_mask.sum() >= max(50, min_rows // 2) and len(np.unique(yv[fit_mask])) == 2
            if not enough:
                notes.append("calibration skipped: purged fit part too small; fitted on the full train window")
                fit_mask, cal_mask, calib_start = np.ones(len(X), dtype=bool), np.zeros(len(X), dtype=bool), None

    est = build_model(kind, config, task, seed, params)
    Xf = X[fit_mask]
    est.fit(Xf.to_numpy(dtype="float64"), yv[fit_mask])
    model = TrainedModel(
        kind=kind, task=task, target=target, horizon=int(horizon), features=features, estimator=est,
        calibrator=None, params=model_params(kind, config, params), seed=seed,
        train_start=_iso(dates.min()), train_end=_iso(dates.max()), fit_end=_iso(dates[fit_mask].max()),
        calib_start=None if calib_start is None else _iso(calib_start),
        n_fit=int(fit_mask.sum()), n_calib=int(cal_mask.sum()), label_end_max=_iso(lend.max()), notes=notes,
    )
    if task == "classification":
        if cal_mask.any():
            model.calibrator = Calibrator(method=method).fit(model.predict_raw(X[cal_mask]), yv[cal_mask], cal_min)
            if not model.calibrator.calibrated:
                notes.append(f"uncalibrated: {model.calibrator.reason}")
        elif method != "none":
            notes.append("uncalibrated: no calibration slice")
    n_bins = int(_cfg(config, "ml.monitor.n_bins", 10))
    model.feature_profiles = {f: DistributionProfile.from_sample(Xf[f].to_numpy(), n_bins) for f in features}
    ref_rows = X[cal_mask] if cal_mask.any() else Xf
    model.prediction_profile = DistributionProfile.from_sample(model.predict(ref_rows), n_bins)
    log_event(log, "ml model fitted", kind=kind, target=target, n_fit=model.n_fit, n_calib=model.n_calib,
              calibrated=model.calibrated, train_start=model.train_start, train_end=model.train_end)
    return model
