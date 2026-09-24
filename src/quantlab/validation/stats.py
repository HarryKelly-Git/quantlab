"""Research statistics: block-bootstrap CIs, PSR / DSR, multiple-testing corrections, Newey-West t.

Formulas follow docs/EXTERNAL-SERVICES.md "Research methods & formulas" exactly:
  * PSR(SR*) = Phi((SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2)) with PER-OBSERVATION
    Sharpe ratios and RAW kurtosis (Normal = 3). scipy's default kurtosis is EXCESS kurtosis — we
    always call it with ``fisher=False``.
  * DSR = PSR(SR0), SR0 = sqrt(V[SR_n]) * ((1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))).
  * Holm step-down and Benjamini-Hochberg (c(M)=1) / Benjamini-Yekutieli (c(M)=sum 1/j).
  * Stationary bootstrap (Politis-Romano): geometric block lengths with mean ``block_length``,
    wrapping circularly, so serial dependence in daily returns is preserved.

WHY the minimum-sample guards: with a handful of observations every one of these statistics is
noise dressed up as a number. Below the configured minimum, functions return status
``INSUFFICIENT_SAMPLE`` (and NaN values) instead of a misleadingly precise estimate.

All randomness uses ``numpy.random.default_rng(seed)`` so results are reproducible.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as ss

OK = "OK"
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
UNDEFINED = "UNDEFINED"          # formula not defined for the inputs (e.g. zero variance)

EULER_MASCHERONI = 0.5772156649015329
DEFAULT_MIN_OBS = 30


def _clean(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype="float64")
    if a.ndim == 1:
        return a[np.isfinite(a)]
    if a.ndim == 2:
        return a[np.isfinite(a).all(axis=1)]
    raise ValueError("expected a 1-D or 2-D array")


def _nan_to_none(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------
# Stationary block bootstrap
# ------------------------------------------------------------------------------------------------
def stationary_bootstrap_indices(n: int, block_length: float, n_resamples: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """(n_resamples, n) index matrix for the Politis-Romano stationary bootstrap.

    Each position starts a new block with probability p = 1/block_length (so block lengths are
    geometric with mean block_length); otherwise it continues the previous block at index+1,
    wrapping circularly. block_length <= 1 gives the ordinary iid bootstrap.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = 1.0 if block_length <= 1 else 1.0 / float(block_length)
    starts = rng.integers(0, n, size=(n_resamples, n))
    new_block = rng.random((n_resamples, n)) < p
    new_block[:, 0] = True
    pos = np.arange(n)
    # position (column) at which the current block began, carried forward
    block_begin = np.where(new_block, pos[None, :], 0)
    block_begin = np.maximum.accumulate(block_begin, axis=1)
    rows = np.arange(n_resamples)[:, None]
    begin_idx = starts[rows, block_begin]
    return (begin_idx + (pos[None, :] - block_begin)) % n


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    p_value: float                 # one-sided, H0: statistic <= null_value
    n: int
    n_resamples: int
    block_length: float
    alpha: float
    null_value: float = 0.0
    status: str = OK
    method: str = "stationary_bootstrap_percentile"

    @property
    def significant_positive(self) -> bool:
        return self.status == OK and np.isfinite(self.ci_low) and self.ci_low > self.null_value

    @property
    def significant_negative(self) -> bool:
        return self.status == OK and np.isfinite(self.ci_high) and self.ci_high < self.null_value

    def to_dict(self) -> dict[str, Any]:
        return _nan_to_none(asdict(self))


def _resolve_bootstrap_params(config, n_resamples, block_length, seed):
    cfg = config.section("validation").get("bootstrap", {}) if config is not None else {}
    return (int(n_resamples if n_resamples is not None else cfg.get("n_resamples", 2000)),
            float(block_length if block_length is not None else cfg.get("block_length", 20)),
            int(seed if seed is not None else cfg.get("seed", 7)))


def bootstrap_distribution(data: Any, statistic: Callable[[np.ndarray], float], *,
                           n_resamples: int = 2000, block_length: float = 20.0,
                           seed: int = 7, chunk: int = 250) -> np.ndarray:
    """Bootstrap distribution of ``statistic`` (rows of ``data`` resampled jointly, so a 2-D
    array keeps paired observations together)."""
    x = _clean(data)
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        idx = stationary_bootstrap_indices(n, block_length, b, rng)
        for k in range(b):
            out[done + k] = statistic(x[idx[k]])
        done += b
    return out


def bootstrap_ci(data: Any, statistic: Callable[[np.ndarray], float] = np.mean, *,
                 config=None, n_resamples: int | None = None, block_length: float | None = None,
                 seed: int | None = None, alpha: float = 0.05, null_value: float = 0.0,
                 min_n: int = DEFAULT_MIN_OBS) -> BootstrapResult:
    """Percentile CI + one-sided p-value for any statistic via the stationary block bootstrap.

    Bootstrap parameters default to config ``validation.bootstrap`` (n_resamples, block_length,
    seed). The p-value tests H0: statistic <= null_value by centring the bootstrap distribution
    on the null: p = (1 + #{theta*_b - theta_hat >= theta_hat - null}) / (B + 1).
    """
    B, L, sd = _resolve_bootstrap_params(config, n_resamples, block_length, seed)
    x = _clean(data)
    n = len(x)
    if n < max(2, min_n):
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan, np.nan, n, B, L, alpha, null_value,
                               INSUFFICIENT_SAMPLE)
    est = float(statistic(x))
    if not np.isfinite(est):
        return BootstrapResult(est, np.nan, np.nan, np.nan, np.nan, n, B, L, alpha, null_value, UNDEFINED)
    L_eff = min(L, n)  # a mean block longer than the sample degenerates to a circular shift
    dist = bootstrap_distribution(x, statistic, n_resamples=B, block_length=L_eff, seed=sd)
    dist = dist[np.isfinite(dist)]
    if len(dist) < max(10, B // 2):
        return BootstrapResult(est, np.nan, np.nan, np.nan, np.nan, n, B, L_eff, alpha, null_value, UNDEFINED)
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    p = (1 + np.sum(dist - est >= est - null_value)) / (len(dist) + 1)
    return BootstrapResult(est, float(lo), float(hi), float(np.std(dist, ddof=1)), float(p), n, B, L_eff,
                           alpha, null_value, OK)


# -- ready-made statistics ------------------------------------------------------------------------
def stat_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def stat_sharpe(x: np.ndarray) -> float:
    """Per-observation Sharpe (mean / sample std). NaN when the std is zero."""
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd) if sd > 0 else np.nan


def stat_paired_mean_diff(x: np.ndarray) -> float:
    """For a 2-column array [a, b]: mean(b - a)."""
    return float(np.mean(x[:, 1] - x[:, 0]))


def stat_paired_sharpe_diff(x: np.ndarray) -> float:
    """For a 2-column array [a, b]: Sharpe(b) - Sharpe(a) (per observation)."""
    return stat_sharpe(x[:, 1]) - stat_sharpe(x[:, 0])


def bootstrap_unpaired_diff(a: Any, b: Any, statistic: Callable[[np.ndarray], float] = np.mean, *,
                            config=None, n_resamples: int | None = None, block_length: float | None = None,
                            seed: int | None = None, alpha: float = 0.05,
                            min_n: int = DEFAULT_MIN_OBS) -> BootstrapResult:
    """CI for statistic(b) - statistic(a) with independent resampling of the two samples."""
    B, L, sd = _resolve_bootstrap_params(config, n_resamples, block_length, seed)
    xa, xb = _clean(a), _clean(b)
    n = min(len(xa), len(xb))
    if n < max(2, min_n):
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan, np.nan, n, B, L, alpha, 0.0, INSUFFICIENT_SAMPLE,
                               "unpaired_bootstrap_percentile")
    est = float(statistic(xb) - statistic(xa))
    da = bootstrap_distribution(xa, statistic, n_resamples=B, block_length=min(L, len(xa)), seed=sd)
    db = bootstrap_distribution(xb, statistic, n_resamples=B, block_length=min(L, len(xb)), seed=sd + 1)
    dist = db - da
    dist = dist[np.isfinite(dist)]
    if not np.isfinite(est) or len(dist) < max(10, B // 2):
        return BootstrapResult(est, np.nan, np.nan, np.nan, np.nan, n, B, L, alpha, 0.0, UNDEFINED,
                               "unpaired_bootstrap_percentile")
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    p = (1 + np.sum(dist - est >= est)) / (len(dist) + 1)
    return BootstrapResult(est, float(lo), float(hi), float(np.std(dist, ddof=1)), float(p), n, B, L, alpha,
                           0.0, OK, "unpaired_bootstrap_percentile")


# ------------------------------------------------------------------------------------------------
# Sharpe-ratio inference (PSR, DSR, MinTRL)
# ------------------------------------------------------------------------------------------------
def moments(returns: Any) -> tuple[float, int, float, float]:
    """(per-observation Sharpe, n, skewness, RAW kurtosis) of a return series."""
    x = _clean(returns)
    n = len(x)
    if n < 3:
        return np.nan, n, np.nan, np.nan
    sd = np.std(x, ddof=1)
    if not sd > 0:
        return np.nan, n, np.nan, np.nan
    sr = float(np.mean(x) / sd)
    skew = float(ss.skew(x, bias=False)) if n > 3 else 0.0
    kurt = float(ss.kurtosis(x, fisher=False, bias=False)) if n > 3 else 3.0
    return sr, n, skew, kurt


def psr(sr: float, n_obs: float, skew: float = 0.0, kurtosis: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio. ALL Sharpe inputs per observation; ``kurtosis`` is RAW (Normal=3)."""
    if not (np.isfinite(sr) and np.isfinite(n_obs) and n_obs > 1):
        return np.nan
    var_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
    if not var_term > 0:
        return np.nan
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(var_term)
    return float(ss.norm.cdf(z))


def min_track_record_length(sr: float, skew: float = 0.0, kurtosis: float = 3.0, sr_benchmark: float = 0.0,
                            prob: float = 0.95) -> float:
    """MinTRL in OBSERVATIONS (not years). inf when sr <= sr_benchmark."""
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return math.inf
    var_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
    return float(1.0 + var_term * (ss.norm.ppf(prob) / (sr - sr_benchmark)) ** 2)


def expected_max_sharpe(n_trials: int, var_trials: float, mean_trials: float = 0.0) -> float:
    """E[max SR] over N independent trials (Bailey & Lopez de Prado 2014): the DSR threshold SR0.

    N <= 1 means no selection happened, so no deflation: SR0 = mean_trials.
    """
    if n_trials is None or n_trials <= 1:
        return float(mean_trials)
    if not (np.isfinite(var_trials) and var_trials >= 0):
        return np.nan
    g = EULER_MASCHERONI
    max_z = (1 - g) * ss.norm.ppf(1 - 1.0 / n_trials) + g * ss.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return float(mean_trials + math.sqrt(var_trials) * max_z)


def null_sharpe_variance(n_obs: int) -> float:
    """Sampling variance of a per-observation Sharpe estimate under the null SR = 0: 1/(T-1).

    Used as V[{SR_n}] when the individual trial Sharpes were not recorded (only the COUNT of
    variants is known). It is what independent zero-edge trials would scatter by.
    """
    return 1.0 / (n_obs - 1.0) if n_obs and n_obs > 1 else np.nan


@dataclass
class SharpeInference:
    sr: float                      # per observation
    sr_annualized: float
    n: int
    skew: float
    kurtosis: float                # raw
    sr_benchmark: float
    psr: float
    status: str = OK

    def to_dict(self) -> dict[str, Any]:
        return _nan_to_none(asdict(self))


@dataclass
class DeflatedSharpe:
    sr: float
    sr_annualized: float
    n: int
    skew: float
    kurtosis: float
    n_trials: int
    var_trials: float
    var_source: str                # "trials" | "null_asymptotic" | "given"
    sr0: float                     # expected max Sharpe under the null (per observation)
    dsr: float
    significant: bool
    confidence: float
    status: str = OK
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _nan_to_none(asdict(self))


def sharpe_inference(returns: Any, sr_benchmark: float = 0.0, periods_per_year: int = 252,
                     min_obs: int = DEFAULT_MIN_OBS) -> SharpeInference:
    sr, n, skew, kurt = moments(returns)
    if n < max(3, min_obs):
        return SharpeInference(np.nan, np.nan, n, np.nan, np.nan, sr_benchmark, np.nan, INSUFFICIENT_SAMPLE)
    if not np.isfinite(sr):
        return SharpeInference(np.nan, np.nan, n, np.nan, np.nan, sr_benchmark, np.nan, UNDEFINED)
    p = psr(sr, n, skew, kurt, sr_benchmark)
    return SharpeInference(sr, sr * math.sqrt(periods_per_year), n, skew, kurt, sr_benchmark, p,
                           OK if np.isfinite(p) else UNDEFINED)


def deflated_sharpe(returns: Any, n_trials: int, *, trial_sharpes: Sequence[float] | None = None,
                    var_trials: float | None = None, periods_per_year: int = 252, confidence: float = 0.95,
                    min_obs: int = DEFAULT_MIN_OBS) -> DeflatedSharpe:
    """Deflated Sharpe Ratio of ``returns`` after ``n_trials`` variants were tried.

    V[{SR_n}] comes from (in order of preference) ``var_trials``, the variance of
    ``trial_sharpes`` (per-observation), or the null asymptotic 1/(T-1). ``n_trials`` must count
    EVERY configuration tried (not just the ones kept) — undercounting overstates the DSR.
    """
    sr, n, skew, kurt = moments(returns)
    n_trials = max(1, int(n_trials or 1))
    notes: list[str] = []
    if n < max(3, min_obs):
        return DeflatedSharpe(np.nan, np.nan, n, np.nan, np.nan, n_trials, np.nan, "none", np.nan, np.nan, False,
                              confidence, INSUFFICIENT_SAMPLE, [f"n={n} < min_obs={min_obs}"])
    if var_trials is not None:
        v, src = float(var_trials), "given"
    elif trial_sharpes is not None and len(_clean(trial_sharpes)) >= 2:
        v, src = float(np.var(_clean(trial_sharpes), ddof=1)), "trials"
    else:
        v, src = null_sharpe_variance(n), "null_asymptotic"
        if n_trials > 1:
            notes.append("trial Sharpes not supplied; V[SR_n] = 1/(T-1) (null asymptotic)")
    sr0 = expected_max_sharpe(n_trials, v)
    if not np.isfinite(sr):
        return DeflatedSharpe(np.nan, np.nan, n, skew, kurt, n_trials, v, src, sr0, np.nan, False, confidence,
                              UNDEFINED, notes + ["zero-variance returns"])
    d = psr(sr, n, skew, kurt, sr0)
    status = OK if np.isfinite(d) else UNDEFINED
    return DeflatedSharpe(sr, sr * math.sqrt(periods_per_year), n, skew, kurt, n_trials, v, src, sr0, d,
                          bool(status == OK and d >= confidence), confidence, status, notes)


# ------------------------------------------------------------------------------------------------
# Multiple testing
# ------------------------------------------------------------------------------------------------
def _pvals(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype="float64").copy()
    p[~np.isfinite(p)] = 1.0            # an unknown p-value is never evidence (fail safe)
    return np.clip(p, 0.0, 1.0)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down (FWER under any dependence). Returns (reject mask, adjusted p) in input order."""
    p = _pvals(pvalues)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= alpha, adj


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05,
                       dependent: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control; ``dependent=True`` gives Benjamini-Yekutieli (c(M)=sum 1/j),
    which is valid under ANY dependence — prefer it for strategies tested on the same universe."""
    p = _pvals(pvalues)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if dependent else 1.0
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, m + 1)
    raw = p[order] * m * c / ranks
    adj_sorted = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= alpha, adj


def benjamini_yekutieli(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    return benjamini_hochberg(pvalues, alpha, dependent=True)


# ------------------------------------------------------------------------------------------------
# t-statistic with Newey-West (HAC) standard errors
# ------------------------------------------------------------------------------------------------
def newey_west_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 * (n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0


@dataclass
class TStat:
    mean: float
    se: float
    t: float
    lags: int
    n: int
    p_two_sided: float
    p_one_sided: float               # H0: mean <= null
    null_value: float = 0.0
    status: str = OK

    def to_dict(self) -> dict[str, Any]:
        return _nan_to_none(asdict(self))


def newey_west_tstat(x: Any, lags: int | None = None, null_value: float = 0.0,
                     min_obs: int = DEFAULT_MIN_OBS) -> TStat:
    """t-statistic of the mean with a Bartlett-kernel HAC standard error:
    S = gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k,  se = sqrt(S / n)."""
    a = _clean(x)
    n = len(a)
    if n < max(3, min_obs):
        return TStat(np.nan, np.nan, np.nan, 0, n, np.nan, np.nan, null_value, INSUFFICIENT_SAMPLE)
    L = newey_west_lags(n) if lags is None else int(lags)
    L = max(0, min(L, n - 1))
    mu = float(np.mean(a))
    e = a - mu
    s = float(np.dot(e, e) / n)
    for k in range(1, L + 1):
        gk = float(np.dot(e[k:], e[:-k]) / n)
        s += 2.0 * (1.0 - k / (L + 1.0)) * gk
    if not s > 0:
        return TStat(mu, np.nan, np.nan, L, n, np.nan, np.nan, null_value, UNDEFINED)
    se = math.sqrt(s / n)
    t = (mu - null_value) / se
    return TStat(mu, se, t, L, n, float(2 * ss.t.sf(abs(t), n - 1)), float(ss.t.sf(t, n - 1)), null_value, OK)
