"""Purged + embargoed walk-forward evaluation (Lopez de Prado, AFML ch. 7) for ML datasets.

Each window trains only on rows dated before its test window and predicts the test rows. Only
out-of-sample predictions are returned, each stamped with the window's model version.

Leakage controls (all on SESSION dates):
  * **Purge**: a training row whose label interval [date, label_end] overlaps the test period
    is dropped. For rows before the test window that means ``label_end >= test_start``; rows
    after the test window (possible in non-walk-forward splits) are dropped while their date is
    <= the last test label end. ``purge_sessions`` adds a buffer: rows dated in the last
    ``purge_sessions`` sessions before the test window are dropped too (belt and braces against a
    mis-specified horizon). Rows with an unknown label end are dropped (fail-safe).
  * **Embargo**: rows dated within ``embargo_sessions`` sessions AFTER the test end are dropped
    from training, because serial correlation leaks test information into them. In pure
    walk-forward no training row lies after the test window; the rule matters for the general
    splitter (:func:`purge_embargo_masks`) and is tested there.
  * **Locked holdout**: if any row's date or label end reaches ``validation.holdout.start`` the
    run is refused unless a caller-supplied ``holdout_guard(period_start, period_end, reason)``
    returns True. The guard (e.g. the validation package's HoldoutGuard) owns unlock logging;
    it is passed in as a callable so this module does not depend on that package.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from quantlab.logging_setup import get_logger, log_event
from quantlab.ml.dataset import MLDataset
from quantlab.ml.evaluate import EvaluationReport, evaluate_predictions
from quantlab.ml.models import InsufficientDataError, TrainedModel, fit_model

log = get_logger(__name__)

HoldoutGuard = Callable[[pd.Timestamp, pd.Timestamp, str], bool]


class HoldoutLockedError(PermissionError):
    """The run would read data at/after the locked holdout start and no guard granted access."""


class WalkForwardError(RuntimeError):
    pass


@dataclass(frozen=True)
class WalkForwardSpec:
    scheme: str = "expanding"        # expanding | rolling
    train_sessions: int = 756        # rolling: window length; expanding: history before the first test
    test_sessions: int = 126
    step_sessions: int = 126
    purge_sessions: int = 10
    embargo_sessions: int = 10
    min_train_rows: int = 5000

    def __post_init__(self) -> None:
        if self.scheme not in ("expanding", "rolling"):
            raise WalkForwardError(f"unknown walk-forward scheme {self.scheme!r}")
        if min(self.train_sessions, self.test_sessions, self.step_sessions) < 1:
            raise WalkForwardError("train/test/step sessions must be >= 1")
        if self.step_sessions < self.test_sessions:
            raise WalkForwardError("step_sessions < test_sessions would overlap test windows (double-counted OOS rows)")
        if self.purge_sessions < 0 or self.embargo_sessions < 0:
            raise WalkForwardError("purge/embargo sessions must be >= 0")

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> "WalkForwardSpec":
        """``ml.walk_forward.*`` first, falling back to ``validation.walk_forward`` (years/months
        converted at 252 sessions/year, 21 sessions/month) and ``ml.purge/embargo_sessions``."""
        g = config.get
        vw = g("validation.walk_forward", {}) or {}
        vals = {
            "scheme": g("ml.walk_forward.scheme", "expanding"),
            "train_sessions": g("ml.walk_forward.train_sessions", None) or int(round(float(vw.get("train_years", 3)) * 252)),
            "test_sessions": g("ml.walk_forward.test_sessions", None) or int(round(float(vw.get("test_months", 6)) * 21)),
            "step_sessions": g("ml.walk_forward.step_sessions", None) or int(round(float(vw.get("step_months", 6)) * 21)),
            "purge_sessions": int(g("ml.purge_sessions", 10)),
            "embargo_sessions": int(g("ml.embargo_sessions", vw.get("embargo_sessions", 10))),
            "min_train_rows": int(g("ml.min_train_rows", 5000)),
        }
        vals.update(overrides)
        return cls(**{k: (v if k == "scheme" else int(v)) for k, v in vals.items()})


@dataclass(frozen=True)
class Window:
    index: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(sessions: pd.DatetimeIndex, spec: WalkForwardSpec) -> list[Window]:
    """Consecutive, non-overlapping test windows; training always strictly before each test."""
    s = pd.DatetimeIndex(sessions).unique().sort_values()
    out: list[Window] = []
    i, k = spec.train_sessions, 0
    while i < len(s):
        j = min(i + spec.test_sessions, len(s)) - 1
        tr0 = s[0] if spec.scheme == "expanding" else s[max(0, i - spec.train_sessions)]
        out.append(Window(k, tr0, s[i], s[j]))
        i += spec.step_sessions
        k += 1
    return out


def _session_at(sessions: pd.DatetimeIndex, anchor: pd.Timestamp, offset: int) -> pd.Timestamp:
    i = int(sessions.searchsorted(anchor, side="left"))
    return sessions[int(np.clip(i + offset, 0, len(sessions) - 1))]


def purge_embargo_masks(
    row_dates: Any,
    label_end: Any,
    test_start: Any,
    test_end: Any,
    sessions: pd.DatetimeIndex,
    purge_sessions: int = 0,
    embargo_sessions: int = 0,
) -> dict[str, np.ndarray]:
    """Boolean masks over rows: ``test``, ``purged``, ``embargoed`` and ``train`` (= everything
    else, i.e. usable for training when the test window is [test_start, test_end])."""
    d = pd.DatetimeIndex(row_dates)
    le = pd.DatetimeIndex(label_end)
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    sessions = pd.DatetimeIndex(sessions).unique().sort_values()
    test = np.asarray((d >= ts) & (d <= te))
    known = np.asarray(le.notna())
    test_le = le[test & known]
    test_label_end = test_le.max() if len(test_le) else te
    overlap_before = np.asarray(d < ts) & known & np.asarray(le >= ts)
    overlap_after = np.asarray((d > te) & (d <= test_label_end))
    buffer = np.zeros(len(d), dtype=bool)
    if purge_sessions > 0:
        buffer = np.asarray((d >= _session_at(sessions, ts, -purge_sessions)) & (d < ts))
    purged = ~test & (overlap_before | overlap_after | buffer | ~known)
    embargoed = np.zeros(len(d), dtype=bool)
    if embargo_sessions > 0:
        emb_end = _session_at(sessions, te, embargo_sessions)
        embargoed = np.asarray((d > te) & (d <= emb_end)) & ~purged & ~test
    train = ~test & ~purged & ~embargoed
    return {"test": test, "purged": purged, "embargoed": embargoed, "train": train}


def check_holdout(row_dates: Any, label_end: Any, holdout_start: Any, guard: HoldoutGuard | None,
                  reason: str = "") -> bool:
    """Refuse (raise) if any row date or label end is >= holdout_start, unless ``guard`` grants it.

    Returns True when the holdout is touched with permission, False when it is not touched."""
    hs = pd.Timestamp(holdout_start)
    d = pd.DatetimeIndex(row_dates)
    le = pd.DatetimeIndex(label_end)
    latest = max(d.max(), le.max()) if len(d) else None
    if latest is None or pd.isna(latest) or latest < hs:
        return False
    if guard is None:
        raise HoldoutLockedError(
            f"data through {latest.date()} touches the locked holdout (start {hs.date()}); use "
            f"MLDataset.before(holdout_start) or pass an explicit holdout_guard with a logged reason")
    granted = guard(hs, pd.Timestamp(latest), reason)
    if granted is not True:
        raise HoldoutLockedError(f"holdout guard refused access to {hs.date()}..{latest.date()}")
    log_event(log, "ml walk-forward reading LOCKED HOLDOUT with guard permission",
              holdout_start=str(hs.date()), period_end=str(latest.date()), reason=reason)
    return True


@dataclass
class Split:
    window: Window
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_purged: int
    n_embargoed: int


def iter_splits(ds: MLDataset, spec: WalkForwardSpec) -> Iterator[Split]:
    """Row positions for each window: train (before test, purged/embargoed) and test."""
    dates = ds.row_dates
    sessions = ds.dates
    le = pd.DatetimeIndex(ds.label_end)
    for w in make_windows(sessions, spec):
        m = purge_embargo_masks(dates, le, w.test_start, w.test_end, sessions,
                                spec.purge_sessions, spec.embargo_sessions)
        in_train_range = np.asarray((dates >= w.train_start) & (dates < w.test_start))
        train = m["train"] & in_train_range
        purged = m["purged"] & in_train_range
        yield Split(w, np.flatnonzero(train), np.flatnonzero(m["test"]), int(purged.sum()),
                    int((m["embargoed"] & in_train_range).sum()))


def _stamp(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    windows: pd.DataFrame
    meta: dict[str, Any]
    models: dict[int, TrainedModel] = field(default_factory=dict)

    def evaluate(self, config: Any = None, **kwargs: Any) -> EvaluationReport:
        task = self.meta.get("task", "classification")
        return evaluate_predictions(self.predictions, "score", "y", "fwd_excess", task=task,
                                    horizon=self.meta.get("horizon"), config=config, **kwargs)


def run_walk_forward(
    ds: MLDataset,
    model_kind: str,
    config: Any = None,
    spec: WalkForwardSpec | None = None,
    *,
    holdout_guard: HoldoutGuard | None = None,
    holdout_reason: str = "",
    holdout_start: Any = None,
    params: dict[str, Any] | None = None,
    calibration: str | None = None,
    model_id: str | None = None,
    keep_models: bool = False,
) -> WalkForwardResult:
    """Walk-forward OOS predictions for ``model_kind`` on ``ds``.

    Windows with too few training rows (or one class) are skipped and listed in ``windows`` with
    the reason; their test rows get no prediction (never a made-up one).
    """
    if spec is None:
        if config is None:
            raise WalkForwardError("pass a WalkForwardSpec or a config")
        spec = WalkForwardSpec.from_config(config)
    hs = holdout_start if holdout_start is not None else (config.get("validation.holdout.start") if config else None)
    if hs is None:
        raise WalkForwardError("holdout start unknown: pass config or holdout_start (refusing to run unguarded)")
    touched = check_holdout(ds.row_dates, ds.label_end, hs, holdout_guard, holdout_reason)

    task, target, horizon = ds.task, ds.target, ds.horizon
    mid = model_id or f"wf.{target}.{model_kind}.h{horizon}"
    cfg_hash = getattr(config, "hash", None)
    X, y = ds.X, ds.y.to_numpy(dtype="float64")
    dates = ds.row_dates
    le = pd.DatetimeIndex(ds.label_end)

    parts: list[pd.DataFrame] = []
    info: list[dict[str, Any]] = []
    models: dict[int, TrainedModel] = {}
    for sp in iter_splits(ds, spec):
        w = sp.window
        row = {"window": w.index, "train_start": w.train_start.date().isoformat(),
               "test_start": w.test_start.date().isoformat(), "test_end": w.test_end.date().isoformat(),
               "n_train": int(len(sp.train_idx)), "n_purged": sp.n_purged, "n_embargoed": sp.n_embargoed,
               "n_test": int(len(sp.test_idx)), "model_version": None, "calibrated": False,
               "n_fit": 0, "n_calib": 0, "train_label_end_max": None, "skipped": ""}
        if len(sp.test_idx) == 0:
            row["skipped"] = "empty test window"
            info.append(row)
            continue
        tr = sp.train_idx
        try:
            model = fit_model(X.iloc[tr], y[tr], dates[tr], le[tr], kind=model_kind, config=config, task=task,
                              target=target, horizon=horizon, calibration=calibration,
                              min_rows=spec.min_train_rows, params=params)
        except InsufficientDataError as exc:
            row["skipped"] = str(exc)
            info.append(row)
            log_event(log, "ml walk-forward window skipped", window=w.index, reason=str(exc))
            continue
        version = f"wf{w.index:02d}-" + _stamp({
            "kind": model_kind, "target": target, "horizon": horizon, "features": ds.features,
            "train": [model.train_start, model.train_end], "test": [row["test_start"], row["test_end"]],
            "n_train": row["n_train"], "dataset": ds.meta.get("dataset_hash"), "config": cfg_hash,
            "params": model.params, "seed": model.seed, "calibration": calibration})
        if pd.Timestamp(model.label_end_max) >= w.test_start:
            raise WalkForwardError(f"window {w.index}: training labels reach into the test window (purge failed)")
        te = sp.test_idx
        Xt = X.iloc[te]
        raw = model.predict_raw(Xt)
        prob = model.predict(Xt) if task == "classification" else np.full(len(te), np.nan)
        part = pd.DataFrame({
            "window": w.index, "model_id": mid, "model_version": version, "calibrated": model.calibrated,
            "raw": raw, "prob": prob, "score": prob if task == "classification" else raw,
            "y": y[te], "fwd_excess": ds.fwd_excess.to_numpy()[te], "label_end": le[te],
        }, index=Xt.index)
        parts.append(part)
        row.update(model_version=version, calibrated=model.calibrated, n_fit=model.n_fit, n_calib=model.n_calib,
                   train_label_end_max=model.label_end_max)
        info.append(row)
        if keep_models:
            models[w.index] = model
        log_event(log, "ml walk-forward window", window=w.index, version=version, n_train=row["n_train"],
                  n_test=row["n_test"], n_purged=sp.n_purged, calibrated=model.calibrated)

    windows = pd.DataFrame(info)
    preds = pd.concat(parts) if parts else pd.DataFrame(
        columns=["window", "model_id", "model_version", "calibrated", "raw", "prob", "score", "y",
                 "fwd_excess", "label_end"])
    meta = {
        "model_id": mid, "model_kind": model_kind, "target": target, "task": task, "horizon": horizon,
        "features": ds.features, "spec": asdict(spec), "config_hash": cfg_hash,
        "dataset_hash": ds.meta.get("dataset_hash"), "dataset_ids": ds.meta.get("dataset_ids", []),
        "pit_status": ds.meta.get("pit_status"), "is_synthetic": ds.meta.get("is_synthetic"),
        "holdout_start": str(pd.Timestamp(hs).date()), "touches_holdout": touched,
        "n_windows": int(len(windows)), "n_skipped_windows": int((windows["skipped"] != "").sum()) if len(windows) else 0,
        "n_oos_rows": int(len(preds)), "calibration": calibration,
    }
    if not len(windows):
        meta["note"] = "no walk-forward window fits: dataset shorter than train_sessions"
    return WalkForwardResult(preds, windows, meta, models)
