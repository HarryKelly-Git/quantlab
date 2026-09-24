"""Out-of-sample evaluation with honest uncertainty.

Statistical classification metrics (AUC, Brier, ECE, precision@top-decile) and TRADING
relevance: the mean forward excess return of the top-decile predictions vs the bottom decile and
vs the unfiltered universe, per signal date.

Why the bootstrap resamples DATES (clusters) with a stationary block scheme (Politis & Romano):
  * rows on the same date share market/sector shocks, so rows are not independent;
  * labels overlap in time (a 10-session horizon makes consecutive dates' labels share 9
    sessions), so consecutive dates are serially correlated.
Resampling individual rows would make confidence intervals far too narrow and "find" edges in a
null world. Block length defaults to max(validation.bootstrap.block_length, horizon).

Everything here is self-contained (no dependency on the validation/backtest packages).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.ml.calibration import brier_score, expected_calibration_error, log_loss


# ------------------------------------------------------------------------------------------------
# Bootstrap machinery
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapSpec:
    n_resamples: int = 1000
    block_length: int = 20
    seed: int = 7
    alpha: float = 0.05

    @classmethod
    def from_config(cls, config: Any = None, horizon: int | None = None) -> "BootstrapSpec":
        if config is None:
            base = cls()
        else:
            base = cls(
                n_resamples=int(config.get("ml.evaluate.n_resamples",
                                           config.get("validation.bootstrap.n_resamples", 1000))),
                block_length=int(config.get("ml.evaluate.block_length",
                                            config.get("validation.bootstrap.block_length", 20))),
                seed=int(config.get("ml.evaluate.seed", config.get("validation.bootstrap.seed", 7))),
                alpha=float(config.get("ml.evaluate.alpha", 0.05)),
            )
        if horizon:
            base = cls(base.n_resamples, max(base.block_length, int(horizon)), base.seed, base.alpha)
        return base


def stationary_bootstrap_indices(n: int, n_resamples: int, block_length: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """(n_resamples, n) indices: blocks of geometric length (mean ``block_length``) starting at
    uniform positions, wrapping around the end."""
    if n <= 0:
        return np.zeros((n_resamples, 0), dtype=np.int64)
    p = 1.0 / max(float(block_length), 1.0)
    new = rng.random((n_resamples, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, (n_resamples, n))
    pos = np.arange(n)
    start_pos = np.maximum.accumulate(np.where(new, pos[None, :], 0), axis=1)
    start_val = np.take_along_axis(starts, start_pos, axis=1)
    return (start_val + (pos[None, :] - start_pos)) % n


class DateBootstrap:
    """Stationary block bootstrap over the sorted unique dates of a row set.

    ``date_weights[k, d]`` = how many times date d appears in resample k. Any statistic that can
    be written with per-row or per-date weights is bootstrapped by re-weighting, which keeps
    paired comparisons (model A vs model B on the same resampled dates) exact and fast.
    """

    def __init__(self, row_dates: Any, spec: BootstrapSpec = BootstrapSpec()):
        codes, uniques = pd.factorize(pd.DatetimeIndex(row_dates), sort=True)
        self.codes = codes.astype(np.int64)
        self.dates = pd.DatetimeIndex(uniques)
        self.n_dates = len(self.dates)
        self.spec = spec
        rng = np.random.default_rng(spec.seed)
        idx = stationary_bootstrap_indices(self.n_dates, spec.n_resamples, spec.block_length, rng)
        flat = (idx + (np.arange(spec.n_resamples, dtype=np.int64) * self.n_dates)[:, None]).ravel()
        self.date_weights = np.bincount(flat, minlength=spec.n_resamples * self.n_dates).reshape(
            spec.n_resamples, self.n_dates).astype("float64")

    def row_weights(self, k: int) -> np.ndarray:
        return self.date_weights[k][self.codes]

    def interval(self, samples: np.ndarray) -> tuple[float, float]:
        s = np.asarray(samples, dtype="float64")
        s = s[np.isfinite(s)]
        if not len(s):
            return float("nan"), float("nan")
        a = self.spec.alpha / 2
        return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))

    def daily_mean(self, values_by_date: pd.Series) -> "Estimate":
        """Mean over dates of a per-date statistic (NaN dates ignored), with a CI."""
        v = values_by_date.reindex(self.dates).to_numpy(dtype="float64")
        ok = np.isfinite(v)
        if not ok.any():
            return Estimate()
        vv = np.where(ok, v, 0.0)
        num = self.date_weights @ vv
        den = self.date_weights @ ok.astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = num / den
        lo, hi = self.interval(samples)
        return Estimate(float(v[ok].mean()), lo, hi, int(ok.sum()))


@dataclass
class Estimate:
    value: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n: int = 0

    def covers(self, x: float) -> bool:
        return bool(np.isfinite(self.lo) and np.isfinite(self.hi) and self.lo <= x <= self.hi)

    @property
    def significantly_positive(self) -> bool:
        return bool(np.isfinite(self.lo) and self.lo > 0)


# ------------------------------------------------------------------------------------------------
# AUC (weighted, tie-aware, one sort for all resamples)
# ------------------------------------------------------------------------------------------------
class _SortedAuc:
    def __init__(self, y: np.ndarray, s: np.ndarray):
        order = np.argsort(s, kind="mergesort")
        self.order = order
        ss = s[order]
        self.y = y[order].astype("float64")
        _, self.group = np.unique(ss, return_inverse=True)
        self.n_groups = int(self.group.max()) + 1 if len(ss) else 0

    def auc(self, w: np.ndarray | None = None) -> float:
        if self.n_groups == 0:
            return float("nan")
        ww = np.ones_like(self.y) if w is None else w[self.order]
        pos = np.bincount(self.group, ww * self.y, minlength=self.n_groups)
        neg = np.bincount(self.group, ww * (1 - self.y), minlength=self.n_groups)
        P, N = pos.sum(), neg.sum()
        if P <= 0 or N <= 0:
            return float("nan")
        neg_below = np.cumsum(neg) - neg
        return float(np.sum(pos * (neg_below + 0.5 * neg)) / (P * N))


def _finite_pair(y: Any, s: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype="float64")
    s = np.asarray(s, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(s)
    return y, s, ok


def auc_score(y: Any, score: Any, weights: Any = None) -> float:
    """ROC AUC (ties count 1/2); NaN when only one class is present."""
    y, s, ok = _finite_pair(y, score)
    w = None if weights is None else np.asarray(weights, dtype="float64")[ok]
    return _SortedAuc(y[ok], s[ok]).auc(w)


def auc_with_ci(y: Any, score: Any, boot: DateBootstrap) -> Estimate:
    y, s, ok = _finite_pair(y, score)
    sorter = _SortedAuc(y[ok], s[ok])
    point = sorter.auc()
    samples = np.array([sorter.auc(boot.row_weights(k)[ok]) for k in range(boot.spec.n_resamples)])
    lo, hi = boot.interval(samples)
    return Estimate(point, lo, hi, int(ok.sum()))


def paired_auc_difference(y: Any, score_a: Any, score_b: Any, boot: DateBootstrap) -> Estimate:
    """AUC(a) - AUC(b) on the same rows and the same resampled dates."""
    y = np.asarray(y, dtype="float64")
    a = np.asarray(score_a, dtype="float64")
    b = np.asarray(score_b, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    sa, sb = _SortedAuc(y[ok], a[ok]), _SortedAuc(y[ok], b[ok])
    point = sa.auc() - sb.auc()
    samples = np.empty(boot.spec.n_resamples)
    for k in range(boot.spec.n_resamples):
        w = boot.row_weights(k)[ok]
        samples[k] = sa.auc(w) - sb.auc(w)
    lo, hi = boot.interval(samples)
    return Estimate(point, lo, hi, int(ok.sum()))


def daily_auc(y: Any, score: Any, row_dates: Any, min_names: int = 10) -> pd.Series:
    """Cross-sectional AUC per date (dates with < min_names rows or one class are NaN)."""
    df = pd.DataFrame({"d": pd.DatetimeIndex(row_dates), "y": np.asarray(y, dtype="float64"),
                       "s": np.asarray(score, dtype="float64")}).dropna()
    out = {}
    for d, g in df.groupby("d", sort=True):
        out[d] = auc_score(g["y"].to_numpy(), g["s"].to_numpy()) if len(g) >= min_names else np.nan
    return pd.Series(out, dtype="float64")


# ------------------------------------------------------------------------------------------------
# Deciles
# ------------------------------------------------------------------------------------------------
def _tie_break(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def rank_within_dates(score: Any, row_dates: Any, seed: int = 0) -> pd.DataFrame:
    """Per-row descending rank within its date (0 = best) and the date's row count.

    Ties are broken by a seeded random key so equal scores are not ordered by symbol name."""
    s = np.asarray(score, dtype="float64")
    df = pd.DataFrame({"d": pd.DatetimeIndex(row_dates), "s": s, "tb": _tie_break(len(s), seed)})
    df["pos"] = np.arange(len(df))
    df = df[np.isfinite(df["s"])]
    df = df.sort_values(["d", "s", "tb"], ascending=[True, False, False])
    df["rank"] = df.groupby("d").cumcount()
    df["n"] = df.groupby("d")["s"].transform("size")
    return df.set_index("pos")[["d", "rank", "n"]]


def decile_returns_by_date(score: Any, fwd: Any, row_dates: Any, frac: float = 0.1, min_names: int = 10,
                           seed: int = 0) -> pd.DataFrame:
    """Per date: mean fwd excess of the top ``frac``, bottom ``frac`` and all rows (dates with at
    least ``min_names`` scored rows only)."""
    r = rank_within_dates(score, row_dates, seed)
    f = np.asarray(fwd, dtype="float64")[r.index.to_numpy()]
    r = r.assign(f=f)
    r = r[np.isfinite(r["f"]) & (r["n"] >= min_names)]
    if r.empty:
        return pd.DataFrame(columns=["n", "n_sel", "top", "bottom", "universe"])
    k = np.maximum(1, np.floor(r["n"].to_numpy() * frac)).astype(int)
    r = r.assign(top=r["rank"].to_numpy() < k, bottom=r["rank"].to_numpy() >= r["n"].to_numpy() - k, k=k)
    g = r.groupby("d")
    out = pd.DataFrame({
        "n": g.size(),
        "n_sel": g["k"].first(),
        "top": r[r["top"]].groupby("d")["f"].mean(),
        "bottom": r[r["bottom"]].groupby("d")["f"].mean(),
        "universe": g["f"].mean(),
    })
    out.index.name = "date"
    return out


def precision_top_by_date(y: Any, score: Any, row_dates: Any, frac: float = 0.1, min_names: int = 10,
                          seed: int = 0) -> pd.DataFrame:
    """Per date: hits and count of positives among the top ``frac`` rows (min_names rows needed)."""
    r = rank_within_dates(score, row_dates, seed)
    yy = np.asarray(y, dtype="float64")[r.index.to_numpy()]
    r = r.assign(y=yy)
    r = r[np.isfinite(r["y"]) & (r["n"] >= min_names)]
    if r.empty:
        return pd.DataFrame(columns=["hits", "count"])
    k = np.maximum(1, np.floor(r["n"].to_numpy() * frac)).astype(int)
    top = r[r["rank"].to_numpy() < k]
    return pd.DataFrame({"hits": top.groupby("d")["y"].sum(), "count": top.groupby("d")["y"].size()})


def pooled_top_mask(score: Any, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Rows in the top ``frac`` of ALL finite scores (random tie-break). For sparse candidate
    sets where most dates have too few rows for a per-date decile."""
    s = np.asarray(score, dtype="float64")
    ok = np.flatnonzero(np.isfinite(s))
    mask = np.zeros(len(s), dtype=bool)
    if not len(ok):
        return mask
    k = max(1, int(np.floor(len(ok) * frac)))
    order = np.lexsort((-_tie_break(len(ok), seed), -s[ok]))
    mask[ok[order[:k]]] = True
    return mask


def weighted_mean_with_ci(values: Any, mask: np.ndarray, boot: DateBootstrap) -> Estimate:
    """Mean of ``values`` over ``mask`` rows; CI by date-cluster re-weighting."""
    v = np.asarray(values, dtype="float64")
    m = np.asarray(mask, dtype=bool) & np.isfinite(v)
    if not m.any():
        return Estimate()
    vv = np.where(m, v, 0.0)
    num = np.bincount(boot.codes, vv, minlength=boot.n_dates)
    den = np.bincount(boot.codes, m.astype("float64"), minlength=boot.n_dates)
    with np.errstate(invalid="ignore", divide="ignore"):
        samples = (boot.date_weights @ num) / (boot.date_weights @ den)
    lo, hi = boot.interval(samples)
    return Estimate(float(v[m].mean()), lo, hi, int(m.sum()))


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    n_rows: int
    n_dates: int
    task: str
    score_col: str
    base_rate: float = float("nan")
    auc: Estimate = field(default_factory=Estimate)
    mean_daily_auc: Estimate = field(default_factory=Estimate)
    brier: float = float("nan")
    brier_climatology: float = float("nan")
    log_loss: float = float("nan")
    ece: float = float("nan")
    precision_top: Estimate = field(default_factory=Estimate)
    top_excess: Estimate = field(default_factory=Estimate)
    bottom_excess: Estimate = field(default_factory=Estimate)
    universe_excess: Estimate = field(default_factory=Estimate)
    top_minus_bottom: Estimate = field(default_factory=Estimate)
    top_minus_universe: Estimate = field(default_factory=Estimate)
    n_decile_dates: int = 0
    top_fraction: float = 0.1
    bootstrap: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def significant_auc(self) -> bool:
        return bool(np.isfinite(self.auc.lo) and self.auc.lo > 0.5)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["significant_auc"] = self.significant_auc
        return d


def evaluate_predictions(
    pred: pd.DataFrame,
    score_col: str = "prob",
    y_col: str = "y",
    fwd_col: str = "fwd_excess",
    *,
    task: str = "classification",
    horizon: int | None = None,
    config: Any = None,
    bootstrap: BootstrapSpec | None = None,
    top_fraction: float | None = None,
    min_names: int | None = None,
    n_bins: int = 10,
) -> EvaluationReport:
    """Evaluate OOS predictions indexed by (date, symbol) with score, label and fwd excess columns.

    For regression targets the AUC is of the score against ``fwd > 0`` (a ranking check) and the
    probability metrics (Brier/ECE/log-loss) are not computed.
    """
    frac = float(top_fraction if top_fraction is not None else
                 (config.get("ml.evaluate.top_fraction", 0.1) if config is not None else 0.1))
    mn = int(min_names if min_names is not None else
             (config.get("ml.evaluate.min_names_per_date", 10) if config is not None else 10))
    spec = bootstrap or BootstrapSpec.from_config(config, horizon)
    df = pred[np.isfinite(pred[score_col].astype("float64")) & np.isfinite(pred[fwd_col].astype("float64"))]
    rep = EvaluationReport(n_rows=int(len(df)), n_dates=0, task=task, score_col=score_col, top_fraction=frac,
                           bootstrap=asdict(spec))
    if df.empty:
        rep.notes.append("no scored rows: nothing to evaluate")
        return rep
    dates = pd.DatetimeIndex(df.index.get_level_values("date"))
    rep.n_dates = int(dates.nunique())
    s = df[score_col].to_numpy(dtype="float64")
    fwd = df[fwd_col].to_numpy(dtype="float64")
    y = df[y_col].to_numpy(dtype="float64") if task == "classification" else (fwd > 0).astype("float64")
    boot = DateBootstrap(dates, spec)

    rep.base_rate = float(np.nanmean(y))
    rep.auc = auc_with_ci(y, s, boot)
    rep.mean_daily_auc = boot.daily_mean(daily_auc(y, s, dates, mn))
    if task == "classification" and np.nanmin(s) >= 0 and np.nanmax(s) <= 1:
        rep.brier = brier_score(s, y)
        rep.brier_climatology = rep.base_rate * (1 - rep.base_rate)
        rep.log_loss = log_loss(s, y)
        rep.ece = expected_calibration_error(s, y, n_bins)

    prec = precision_top_by_date(y, s, dates, frac, mn, spec.seed)
    if not prec.empty:
        est = boot.daily_mean(prec["hits"] / prec["count"])
        rep.precision_top = est
    dec = decile_returns_by_date(s, fwd, dates, frac, mn, spec.seed)
    rep.n_decile_dates = int(len(dec))
    if len(dec):
        rep.top_excess = boot.daily_mean(dec["top"])
        rep.bottom_excess = boot.daily_mean(dec["bottom"])
        rep.universe_excess = boot.daily_mean(dec["universe"])
        rep.top_minus_bottom = boot.daily_mean(dec["top"] - dec["bottom"])
        rep.top_minus_universe = boot.daily_mean(dec["top"] - dec["universe"])
    else:
        rep.notes.append(f"no date has >= {mn} scored rows: decile statistics not computed")
    rep.notes.append("returns are gross forward excess returns (no costs); trading costs apply downstream")
    return rep
