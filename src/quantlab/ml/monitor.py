"""Model monitoring: input/output drift (PSI), rolling calibration on MATURED predictions, error
rate, and a conservative status recommendation.

Design rules:
  * Only matured predictions (label_end <= the latest session with data) are scored. Scoring a
    prediction before its label is known would use a partial outcome.
  * The drift reference is stored with the model at fit time (``DistributionProfile``: decile
    edges + bin fractions + missing rate), so monitoring never needs the training data again.
  * The monitor can only keep or DEMOTE a status: ACTIVE -> MONITORED -> PAUSED. It never
    promotes (promotion is a research decision with its own evidence), a PAUSED model resumes only
    by a human, and RETIRED is terminal. Missing evidence is a warning, never a pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.core.types import ModelStatus, new_id, utcnow
from quantlab.logging_setup import get_logger, log_event
from quantlab.ml.calibration import brier_score, expected_calibration_error
from quantlab.ml.dataset import DEFAULT_DELISTING_RETURN, build_labels

log = get_logger(__name__)


# ------------------------------------------------------------------------------------------------
# Distribution profiles and PSI
# ------------------------------------------------------------------------------------------------
@dataclass
class DistributionProfile:
    """Reference distribution: quantile cut points of the finite values, the fraction of ALL
    values in each bin, and the missing fraction (a missing-rate shift is drift too)."""

    edges: list[float] = field(default_factory=list)
    fractions: list[float] = field(default_factory=list)
    missing_fraction: float = 0.0
    n: int = 0

    @classmethod
    def from_sample(cls, values: Any, n_bins: int = 10) -> "DistributionProfile":
        v = np.asarray(values, dtype="float64").ravel()
        n = len(v)
        if n == 0:
            return cls()
        fin = v[np.isfinite(v)]
        edges = np.unique(np.quantile(fin, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(fin) else np.array([])
        prof = cls(edges=[float(e) for e in edges], n=n)
        frac, miss = prof.bin_fractions(v)
        prof.fractions, prof.missing_fraction = [float(f) for f in frac], float(miss)
        return prof

    def bin_fractions(self, values: Any) -> tuple[np.ndarray, float]:
        v = np.asarray(values, dtype="float64").ravel()
        if len(v) == 0:
            return np.zeros(len(self.edges) + 1), float("nan")
        fin = v[np.isfinite(v)]
        ids = np.searchsorted(np.asarray(self.edges, dtype="float64"), fin, side="right")
        counts = np.bincount(ids, minlength=len(self.edges) + 1)
        return counts / len(v), (len(v) - len(fin)) / len(v)

    def psi(self, values: Any, eps: float = 1e-4) -> float:
        """Population stability index of ``values`` against this reference (NaN if either is empty)."""
        v = np.asarray(values, dtype="float64").ravel()
        if self.n == 0 or len(v) == 0:
            return float("nan")
        a_frac, a_miss = self.bin_fractions(v)
        e = np.clip(np.r_[np.asarray(self.fractions, dtype="float64"), self.missing_fraction], eps, None)
        a = np.clip(np.r_[a_frac, a_miss], eps, None)
        return float(np.sum((a - e) * np.log(a / e)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DistributionProfile":
        return cls(list(d.get("edges", [])), list(d.get("fractions", [])), float(d.get("missing_fraction", 0.0)),
                   int(d.get("n", 0)))


def psi(expected: Any, actual: Any, n_bins: int = 10) -> float:
    """PSI of ``actual`` vs ``expected`` using decile bins of ``expected``.
    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift."""
    return DistributionProfile.from_sample(expected, n_bins).psi(actual)


# ------------------------------------------------------------------------------------------------
# Thresholds / report
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MonitorThresholds:
    psi_warn: float = 0.10
    psi_pause: float = 0.25
    feature_psi_warn: float = 0.10
    feature_psi_pause: float = 0.25
    feature_frac_pause: float = 0.30      # fraction of features with PSI > feature_psi_pause
    ece_warn: float = 0.05
    ece_pause: float = 0.10
    brier_tolerance: float = 0.01         # pause if Brier > base-rate (climatology) Brier + tolerance
    error_rate_warn: float = 0.01
    error_rate_pause: float = 0.05
    min_matured: int = 200
    min_recent: int = 50
    rolling_window_sessions: int = 63
    n_bins: int = 10

    @classmethod
    def from_config(cls, config: Any) -> "MonitorThresholds":
        if config is None:
            return cls()
        vals = {}
        for k, default in asdict(cls()).items():
            v = config.get(f"ml.monitor.{k}", default)
            vals[k] = type(default)(v)
        return cls(**vals)


@dataclass
class MonitorReport:
    model_id: str
    version: str
    as_of: str | None
    current_status: str
    recommended_status: str
    reasons: list[str]
    metrics: dict[str, Any]
    feature_psi: dict[str, float] = field(default_factory=dict)
    rolling: pd.DataFrame | None = None
    report_id: str = field(default_factory=lambda: new_id("mmon"))
    created_at: str = field(default_factory=lambda: utcnow().isoformat())

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in ("report_id", "model_id", "version", "as_of", "current_status",
                                           "recommended_status", "reasons", "metrics", "feature_psi", "created_at")}
        d["rolling"] = None if self.rolling is None else self.rolling.assign(
            date=self.rolling["date"].astype(str)).to_dict(orient="records")
        return d


def _finite(x: Any) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------
# Outcomes
# ------------------------------------------------------------------------------------------------
def attach_realized_outcomes(predictions: pd.DataFrame, panel: Any, horizon: int,
                             target: str = "positive_excess_return", benchmark: str = "SPY",
                             delisting_return: float | None = DEFAULT_DELISTING_RETURN) -> pd.DataFrame:
    """Add ``y``, ``fwd_excess``, ``label_end`` and ``matured`` to prediction rows.

    ``predictions`` needs ``as_of_date`` and ``symbol`` columns. A row is matured only when its
    full label horizon lies inside ``panel`` (label_end <= last session) and the label is defined.
    """
    out = predictions.copy()
    labels = build_labels(panel, horizon, target, benchmark, delisting_return)
    d = pd.to_datetime(out["as_of_date"]).dt.normalize()
    ri = labels.y.index.get_indexer(d)
    ci = labels.y.columns.get_indexer(out["symbol"])
    ok = (ri >= 0) & (ci >= 0)
    yv = np.full(len(out), np.nan)
    fx = np.full(len(out), np.nan)
    yv[ok] = labels.y.to_numpy()[ri[ok], ci[ok]]
    fx[ok] = labels.fwd_excess.to_numpy()[ri[ok], ci[ok]]
    le = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    le[ok] = labels.label_end.to_numpy().astype("datetime64[ns]")[ri[ok]]
    out["y"], out["fwd_excess"], out["label_end"] = yv, fx, pd.DatetimeIndex(le)
    out["matured"] = np.isfinite(yv) & out["label_end"].notna()
    return out


def load_predictions(db: Any, model_id: str, version: str, start: str | None = None,
                     end: str | None = None) -> pd.DataFrame:
    sql = ("SELECT candidate_id, model_id, model_version, as_of_date, symbol, target, probability, prediction, "
           "created_at FROM ml_predictions WHERE model_id=? AND model_version=?")
    params: list[Any] = [model_id, version]
    if start:
        sql += " AND as_of_date >= ?"
        params.append(str(start))
    if end:
        sql += " AND as_of_date <= ?"
        params.append(str(end))
    return db.query_df(sql + " ORDER BY as_of_date, id", params)


# ------------------------------------------------------------------------------------------------
# Monitor
# ------------------------------------------------------------------------------------------------
class ModelMonitor:
    def __init__(self, config: Any = None, thresholds: MonitorThresholds | None = None):
        self.config = config
        self.t = thresholds or MonitorThresholds.from_config(config)

    # -- components -------------------------------------------------------------------------------
    def prediction_drift(self, reference: DistributionProfile | Any, recent: Any) -> float:
        prof = reference if isinstance(reference, DistributionProfile) else \
            DistributionProfile.from_sample(reference, self.t.n_bins)
        return prof.psi(recent)

    def feature_drift(self, reference: Mapping[str, DistributionProfile] | pd.DataFrame,
                      recent: pd.DataFrame) -> dict[str, float]:
        """PSI per feature. A feature missing from ``recent`` is maximal drift (inf)."""
        if isinstance(reference, pd.DataFrame):
            reference = {c: DistributionProfile.from_sample(reference[c].to_numpy(), self.t.n_bins)
                         for c in reference.columns}
        out: dict[str, float] = {}
        for name, prof in reference.items():
            out[name] = prof.psi(recent[name].to_numpy()) if name in recent.columns else float("inf")
        return out

    @staticmethod
    def error_rate(predictions: Sequence[Any] | np.ndarray | pd.Series) -> float:
        """Fraction of predictions that are missing/non-finite (failed or abstained)."""
        v = pd.to_numeric(pd.Series(list(predictions), dtype="object"), errors="coerce").to_numpy(dtype="float64")
        return float(np.mean(~np.isfinite(v))) if len(v) else float("nan")

    def calibration_metrics(self, prob: Any, y: Any) -> dict[str, float]:
        p = np.asarray(prob, dtype="float64")
        yy = np.asarray(y, dtype="float64")
        ok = np.isfinite(p) & np.isfinite(yy)
        p, yy = p[ok], yy[ok]
        if not len(p):
            return {"n": 0, "brier": float("nan"), "ece": float("nan"), "base_rate": float("nan"),
                    "brier_climatology": float("nan")}
        base = float(yy.mean())
        return {"n": int(len(p)), "brier": brier_score(p, yy), "ece": expected_calibration_error(p, yy, self.t.n_bins),
                "base_rate": base, "brier_climatology": base * (1 - base)}

    def rolling_metrics(self, matured: pd.DataFrame, window_sessions: int | None = None) -> pd.DataFrame:
        """Brier / ECE over trailing windows of ``window_sessions`` distinct prediction dates.

        ``matured`` needs columns ``as_of_date``, ``probability``, ``y`` (matured rows only)."""
        w = int(window_sessions or self.t.rolling_window_sessions)
        cols = ["date", "n", "brier", "ece", "base_rate"]
        if matured.empty:
            return pd.DataFrame(columns=cols)
        df = matured.assign(_d=pd.to_datetime(matured["as_of_date"]).dt.normalize())
        df = df[np.isfinite(df["probability"].astype("float64")) & np.isfinite(df["y"].astype("float64"))]
        dates = pd.DatetimeIndex(df["_d"].unique()).sort_values()
        rows = []
        for i, d in enumerate(dates):
            lo = dates[max(0, i - w + 1)]
            sub = df[(df["_d"] >= lo) & (df["_d"] <= d)]
            m = self.calibration_metrics(sub["probability"], sub["y"])
            rows.append((d, m["n"], m["brier"], m["ece"], m["base_rate"]))
        return pd.DataFrame(rows, columns=cols)

    # -- decision ---------------------------------------------------------------------------------
    def recommend_status(self, current_status: ModelStatus | str, metrics: Mapping[str, Any]
                         ) -> tuple[ModelStatus, list[str]]:
        cur = ModelStatus(current_status)
        t = self.t
        if cur is ModelStatus.RETIRED:
            return ModelStatus.RETIRED, ["RETIRED is terminal"]
        pause: list[str] = []
        warn: list[str] = []

        er = metrics.get("error_rate")
        if not _finite(er):
            warn.append("error rate UNKNOWN (no recent predictions)")
        elif er > t.error_rate_pause:
            pause.append(f"error rate {er:.3f} > {t.error_rate_pause}")
        elif er > t.error_rate_warn:
            warn.append(f"error rate {er:.3f} > {t.error_rate_warn}")

        p = metrics.get("prediction_psi")
        if not _finite(p):
            warn.append("prediction drift UNKNOWN (not measured)")
        elif p > t.psi_pause:
            pause.append(f"prediction PSI {p:.3f} > {t.psi_pause}")
        elif p > t.psi_warn:
            warn.append(f"prediction PSI {p:.3f} > {t.psi_warn}")

        fpsi = metrics.get("feature_psi") or {}
        if not fpsi:
            warn.append("feature drift UNKNOWN (not measured)")
        else:
            vals = {k: float(v) for k, v in fpsi.items()}
            major = [k for k, v in vals.items() if not np.isfinite(v) or v > t.feature_psi_pause]
            moderate = [k for k, v in vals.items() if np.isfinite(v) and t.feature_psi_warn < v <= t.feature_psi_pause]
            if len(major) / len(vals) >= t.feature_frac_pause:
                pause.append(f"{len(major)}/{len(vals)} features drifted (PSI > {t.feature_psi_pause}): {sorted(major)}")
            elif major or moderate:
                warn.append(f"feature drift: major={sorted(major)} moderate={sorted(moderate)}")

        n_mat = int(metrics.get("n_matured") or 0)
        if n_mat < t.min_matured:
            warn.append(f"only {n_mat} matured predictions (< {t.min_matured}): calibration not yet verified")
        else:
            ece = metrics.get("recent_ece")
            brier, clim = metrics.get("recent_brier"), metrics.get("brier_climatology")
            if _finite(ece) and ece > t.ece_pause:
                pause.append(f"recent ECE {ece:.3f} > {t.ece_pause}")
            elif _finite(ece) and ece > t.ece_warn:
                warn.append(f"recent ECE {ece:.3f} > {t.ece_warn}")
            if _finite(brier) and _finite(clim) and brier > clim + t.brier_tolerance:
                pause.append(f"recent Brier {brier:.4f} worse than base-rate predictor {clim:.4f} + {t.brier_tolerance}")

        if pause:
            return ModelStatus.PAUSED, pause + warn
        if cur is ModelStatus.PAUSED:
            return ModelStatus.PAUSED, ["paused models resume only by a human decision", *warn]
        if warn:
            return ModelStatus.MONITORED, warn
        return cur, ["all monitored metrics within thresholds"]

    def assess(
        self,
        *,
        model_id: str,
        version: str,
        current_status: ModelStatus | str,
        recent_predictions: Any = None,
        prediction_reference: DistributionProfile | Any = None,
        X_recent: pd.DataFrame | None = None,
        feature_reference: Mapping[str, DistributionProfile] | pd.DataFrame | None = None,
        matured: pd.DataFrame | None = None,
        as_of: Any = None,
        model: Any = None,
    ) -> MonitorReport:
        """Compute all monitor metrics and a recommended status.

        ``model`` (a ``TrainedModel``) supplies the stored prediction/feature references when the
        explicit references are not given. ``matured`` needs ``as_of_date``, ``probability``, ``y``.
        """
        if model is not None:
            prediction_reference = prediction_reference if prediction_reference is not None else model.prediction_profile
            feature_reference = feature_reference if feature_reference is not None else model.feature_profiles
        metrics: dict[str, Any] = {}
        recent = np.asarray([] if recent_predictions is None else pd.to_numeric(
            pd.Series(list(recent_predictions), dtype="object"), errors="coerce"), dtype="float64")
        metrics["n_recent"] = int(len(recent))
        metrics["error_rate"] = self.error_rate(recent) if len(recent) else float("nan")
        fin = recent[np.isfinite(recent)]
        metrics["prediction_psi"] = (self.prediction_drift(prediction_reference, fin)
                                     if prediction_reference is not None and len(fin) >= self.t.min_recent
                                     else float("nan"))
        fpsi: dict[str, float] = {}
        if feature_reference is not None and X_recent is not None and len(X_recent) >= self.t.min_recent:
            fpsi = self.feature_drift(feature_reference, X_recent)
        metrics["feature_psi"] = fpsi
        metrics["max_feature_psi"] = max(fpsi.values()) if fpsi else float("nan")
        rolling = None
        if matured is not None and len(matured):
            m = self.calibration_metrics(matured["probability"], matured["y"])
            metrics.update(n_matured=m["n"], recent_brier=m["brier"], recent_ece=m["ece"],
                           base_rate=m["base_rate"], brier_climatology=m["brier_climatology"])
            rolling = self.rolling_metrics(matured)
        else:
            metrics["n_matured"] = 0
        status, reasons = self.recommend_status(current_status, metrics)
        report = MonitorReport(model_id=model_id, version=version,
                               as_of=None if as_of is None else pd.Timestamp(as_of).date().isoformat(),
                               current_status=ModelStatus(current_status).value, recommended_status=status.value,
                               reasons=reasons, metrics={k: v for k, v in metrics.items() if k != "feature_psi"},
                               feature_psi=fpsi, rolling=rolling)
        log_event(log, "ml monitor assessed", model_id=model_id, version=version,
                  current=report.current_status, recommended=report.recommended_status, n_reasons=len(reasons))
        return report

    @staticmethod
    def record(db: Any, report: MonitorReport) -> str:
        """Append the report to ``ml_monitor_reports`` (append-only). Returns the report id."""
        m = report.metrics
        db.insert("ml_monitor_reports", {
            "report_id": report.report_id, "model_id": report.model_id, "version": report.version,
            "as_of_date": report.as_of, "created_at": report.created_at,
            "prediction_psi": _num(m.get("prediction_psi")), "max_feature_psi": _num(m.get("max_feature_psi")),
            "recent_brier": _num(m.get("recent_brier")), "recent_ece": _num(m.get("recent_ece")),
            "error_rate": _num(m.get("error_rate")), "n_matured": int(m.get("n_matured") or 0),
            "current_status": report.current_status, "recommended_status": report.recommended_status,
            "reasons_json": report.reasons, "report_json": _jsonable(report.as_dict()),
        })
        return report.report_id


def _num(x: Any) -> float | None:
    return float(x) if _finite(x) else None


def _jsonable(obj: Any) -> Any:
    """Replace non-finite floats (not valid JSON) with None, recursively."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj
