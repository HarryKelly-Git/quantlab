"""Multiple-hypothesis statistics for alpha research (Part 4 of the alpha brief).

Reuses ``validation.stats`` (stationary bootstrap, PSR/DSR, Holm, BH, BY, Newey-West) and adds:
  * White (2000) Reality Check for data snooping: is the BEST of K strategies better than 0 once
    the search over K is accounted for?
  * Hansen (2005) Superior Predictive Ability, consistent version (studentised, recentred so poor
    strategies do not dilute power).
  * Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2017).
  * Effective number of independent trials from the trial-return correlation matrix.
  * A cross-sectional permutation test helper (shuffle the signal across names within each date).

Every function takes a T x K matrix of DAILY strategy returns (excess over the benchmark when a
benchmark applies; long-short books are already excess over cash).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.validation.stats import stationary_bootstrap_indices


def _clean_matrix(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    a = np.asarray(X, dtype="float64")
    if a.ndim == 1:
        a = a[:, None]
    return a[np.isfinite(a).all(axis=1)]


@dataclass
class SnoopingTest:
    test: str
    statistic: float
    p_value: float
    best_index: int
    n_strategies: int
    n_obs: int
    n_resamples: int
    block_length: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def white_reality_check(X, *, n_resamples: int = 2000, block_length: float = 10.0, seed: int = 11) -> SnoopingTest:
    a = _clean_matrix(X)
    T, K = a.shape
    mean = a.mean(axis=0)
    stat = math.sqrt(T) * mean.max()
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < n_resamples:
        b = min(200, n_resamples - done)
        idx = stationary_bootstrap_indices(T, block_length, b, rng)
        for k in range(b):
            mb = a[idx[k]].mean(axis=0)
            if math.sqrt(T) * (mb - mean).max() >= stat:
                exceed += 1
        done += b
    return SnoopingTest("white_reality_check", float(stat), (exceed + 1) / (n_resamples + 1), int(mean.argmax()),
                        K, T, n_resamples, block_length)


def hansen_spa(X, *, n_resamples: int = 2000, block_length: float = 10.0, seed: int = 13) -> SnoopingTest:
    """SPA_c: studentised max statistic; strategies with clearly negative means are not recentred."""
    a = _clean_matrix(X)
    T, K = a.shape
    mean = a.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx_all = stationary_bootstrap_indices(T, block_length, n_resamples, rng)
    boot_means = np.empty((n_resamples, K))
    for k in range(n_resamples):
        boot_means[k] = a[idx_all[k]].mean(axis=0)
    omega = np.sqrt(T) * boot_means.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    tstat = math.sqrt(T) * mean / omega
    stat = max(0.0, float(np.nanmax(tstat)))
    thresh = -math.sqrt(2 * math.log(math.log(T))) if T > 15 else -np.inf
    # Hansen (2005) g_c: strategies not clearly worse than the benchmark are centred at 0 (null mu = 0);
    # clearly bad ones (t below -sqrt(2 log log T)) keep their negative mean so they cannot dilute power.
    g = np.where(tstat >= thresh, mean, 0.0)
    tb = math.sqrt(T) * (boot_means - g[None, :]) / omega[None, :]
    tb = np.maximum(np.nanmax(tb, axis=1), 0.0)
    p = (np.sum(tb >= stat) + 1) / (n_resamples + 1)
    return SnoopingTest("hansen_spa_consistent", stat, float(p), int(np.nanargmax(tstat)), K, T, n_resamples, block_length)


def _sharpe_cols(a: np.ndarray) -> np.ndarray:
    sd = a.std(axis=0, ddof=1)
    return np.where(sd > 0, a.mean(axis=0) / sd, -np.inf)


def pbo_cscv(X, *, n_blocks: int = 16, max_combinations: int = 2000, seed: int = 17) -> dict[str, Any]:
    """Probability that the in-sample-best configuration ranks below the median out of sample.

    The T days are cut into ``n_blocks`` contiguous blocks; every split of half the blocks into IS and
    the rest into OOS (sampled if there are too many) picks the best IS Sharpe and records its OOS
    relative rank w; PBO = share of splits with logit(w) <= 0. Also returns the mean OOS Sharpe of the
    IS winner and the IS->OOS degradation slope."""
    a = _clean_matrix(X)
    T, K = a.shape
    if K < 2:
        return {"status": "NEEDS_2+_CONFIGS"}
    blocks = np.array_split(np.arange(T), n_blocks)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combinations:
        sel = rng.choice(len(combos), max_combinations, replace=False)
        combos = [combos[i] for i in sel]
    logits, is_best_sr, oos_sr_of_best = [], [], []
    for c in combos:
        is_idx = np.concatenate([blocks[i] for i in c])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in c])
        sr_is = _sharpe_cols(a[is_idx])
        sr_oos = _sharpe_cols(a[oos_idx])
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1)
        w = (rank + 1) / (K + 1)
        logits.append(math.log(w / (1 - w)))
        is_best_sr.append(sr_is[best])
        oos_sr_of_best.append(sr_oos[best])
    logits = np.array(logits)
    slope = float(np.polyfit(is_best_sr, oos_sr_of_best, 1)[0]) if len(set(np.round(is_best_sr, 12))) > 1 else None
    return {"status": "OK", "pbo": float((logits <= 0).mean()), "n_splits": len(combos), "n_configs": K,
            "mean_oos_sharpe_of_is_best_ann": float(np.mean(oos_sr_of_best) * math.sqrt(252)),
            "degradation_slope": slope}


def effective_trials(X) -> dict[str, float]:
    """Independent-trial count from the eigenvalues of the trial correlation matrix:
    participation ratio (sum l)^2 / sum l^2 (1 = all identical, K = all independent)."""
    a = _clean_matrix(X)
    if a.shape[1] < 2:
        return {"n_trials": float(a.shape[1]), "effective_trials": float(a.shape[1])}
    c = np.corrcoef(a, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(c), 0, None)
    return {"n_trials": float(a.shape[1]), "effective_trials": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "mean_abs_corr": float(np.abs(c[np.triu_indices_from(c, 1)]).mean())}


def permutation_pvalue(observed: float, statistic_under_permutation: Callable[[np.random.Generator], float], *,
                       n_permutations: int = 200, seed: int = 19) -> dict[str, Any]:
    """One-sided p-value of ``observed`` against a permutation null (larger = better)."""
    rng = np.random.default_rng(seed)
    null = np.array([statistic_under_permutation(rng) for _ in range(n_permutations)])
    return {"observed": float(observed), "p_value": float((np.sum(null >= observed) + 1) / (n_permutations + 1)),
            "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)), "n_permutations": n_permutations}


def shuffle_within_dates(signal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute the non-missing signal values across names within each date (keeps the cross-sectional
    distribution and the universe; destroys any link between signal and name)."""
    a = signal.to_numpy().copy()
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i])
        if ok.sum() > 1:
            a[i, ok] = rng.permutation(a[i, ok])
    return pd.DataFrame(a, index=signal.index, columns=signal.columns)
