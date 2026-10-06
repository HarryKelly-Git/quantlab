"""Cross-sectional anomaly families. PRE-REGISTERED GRIDS (fixed before TRAIN):

H16 MAX / lottery (Bali, Cakici & Whitelaw 2011), 4 variants:
  signal -MAX1 (largest daily return, last 21 sessions) | -MAX5 (mean of the 5 largest); weighting equal |
  inverse_vol; decile long-short (low MAX long); hold 21; next open.
H17 idiosyncratic volatility (Ang, Hodrick, Xing & Zhang 2006), 8 variants:
  signal -std(residual returns) over 21 | 63 sessions; residual market | two_factor; weighting equal |
  inverse_vol; decile long-short (low IVOL long); hold 21; next open.
H07 intraday gap reversal, 4 variants (Part 23/24: separate overnight from intraday):
  at the open of t, gap = open_t / close_{t-1} - 1 (known at the open); long the most negative gap
  decile, short the most positive, exit at the close of t: P&L = intraday return of t, minus costs
  plus an extra 10 bps per side for trading right after the open. universe liquid | large; weighting
  equal | inverse_vol.
H06 decomposition (descriptive, no trial): overnight vs intraday P&L of the selected H01/H03/H04
  variants (a book held close-to-close earns ret_on + ret_id).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from quantlab.alpha.engine import quantile_weights
from quantlab.alpha.experiment import Context, Variant, run_family
from quantlab.alpha.experiments.reversal import context


def max_signal(d, k: int) -> pd.DataFrame:
    r = d.p["ret_cc"]
    if k == 1:
        return -r.rolling(21, min_periods=17).max()
    # mean of the k largest daily returns in the window
    arr = r.to_numpy()
    out = np.full(arr.shape, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(arr, 21, axis=0)            # (T-20, N, 21)
    srt = np.sort(np.where(np.isfinite(W), W, -np.inf), axis=2)[:, :, -k:]
    valid = np.isfinite(W).sum(axis=2) >= 17
    m = np.where(valid, np.mean(np.where(np.isfinite(srt), srt, np.nan), axis=2), np.nan)
    out[20:] = m
    return -pd.DataFrame(out, index=r.index, columns=r.columns)


def ivol_signal(d, window: int, basis: str) -> pd.DataFrame:
    r = d.p["ret_cc"]
    res = (r - d.betas["beta_mkt"].mul(r["SPY"], axis=0)) if basis == "market" else d.resid
    return -res.rolling(window, min_periods=int(0.8 * window)).std()


def max_variants(d) -> list[Variant]:
    out = []
    for k, wt in itertools.product((1, 5), ("equal", "inverse_vol")):
        spec = {"universe": "liquid survivorship-free", "signal": f"-MAX{k} over 21 sessions", "entry": "next open",
                "position": "decile long-short", "sizing": wt, "exit": "overlapping 21-session hold",
                "costs": "CostModel tiers; borrow 50bps/yr", "benchmark": "cash", "risk": "each side sums to 1"}
        out.append(Variant(f"MAX{k}_{wt}", spec, (lambda k=k, wt=wt: quantile_weights(max_signal(d, k), d.u_liquid, q=0.1,
                                                                                         weighting=wt, vol=d.vol60)), holding=21))
    return out


def ivol_variants(d) -> list[Variant]:
    out = []
    for w_, b, wt in itertools.product((21, 63), ("market", "two_factor"), ("equal", "inverse_vol")):
        spec = {"universe": "liquid survivorship-free", "signal": f"-IVOL {w_}d {b}", "entry": "next open",
                "position": "decile long-short", "sizing": wt, "exit": "overlapping 21-session hold",
                "costs": "CostModel tiers; borrow 50bps/yr", "benchmark": "cash", "risk": "each side sums to 1"}
        out.append(Variant(f"IVOL{w_}_{b}_{wt}", spec, (lambda w_=w_, b=b, wt=wt: quantile_weights(ivol_signal(d, w_, b), d.u_liquid,
                                                                                                    q=0.1, weighting=wt, vol=d.vol60)),
                           holding=21))
    return out


def gap_context(d) -> Context:
    """P&L of a book entered at the open of t and exited at the close of t, indexed by t: the engine's
    'open' execution reads ret_oo.shift(-1) at the decision row, so we feed it ret_id shifted back one row
    and decide on row t-1 with the gap of row t (known at the open of t)."""
    c = context(d)
    c.ret_oo = d.p["ret_id"]                      # decision row t-1 -> ret_oo.shift(-1) = ret_id[t]
    c.cost_bps = d.cost_bps + 10.0                # extra 10 bps per side for trading just after the open
    c.alt_ret_oo = {}
    c.intraday_only = True                        # every position opened at the open and closed at the close
    return c


def gap_weights(d, universe, wt) -> pd.DataFrame:
    gap = d.p["ret_on"]                           # row t: overnight gap into the open of t
    sig = -gap.shift(-1)                          # place it on row t-1 (the decision row for P&L day t)
    u = universe                                  # row t-1 universe = known at the close of t-1: PIT
    return quantile_weights(sig, u, q=0.1, weighting=wt, vol=d.vol60)


def gap_variants(d) -> list[Variant]:
    out = []
    for un, wt in itertools.product(("liquid", "large"), ("equal", "inverse_vol")):
        uni = d.u_liquid if un == "liquid" else d.u_large
        spec = {"universe": f"{un} survivorship-free", "signal": "-overnight gap at the open (known at the open)",
                "entry": "just after the open (+10bps)", "position": "decile long-short", "sizing": wt,
                "exit": "same-day close", "costs": "CostModel tiers + 10bps; borrow 50bps/yr", "benchmark": "cash",
                "risk": "intraday only, flat overnight"}
        out.append(Variant(f"GAP_{un}_{wt}", spec, (lambda uni=uni, wt=wt: gap_weights(d, uni, wt)), holding=1))
    return out


def run_max(d) -> dict:
    return run_family(family="H16_max_lottery", hypothesis_id="H16", variants=max_variants(d), ctx=context(d))


def run_ivol(d) -> dict:
    return run_family(family="H17_idiosyncratic_vol", hypothesis_id="H17", variants=ivol_variants(d), ctx=context(d),
                      prior_trials=1, notes="prior_trials=1: the selection score's low-volatility term")


def run_gap(d) -> dict:
    return run_family(family="H07_gap_reversal_intraday", hypothesis_id="H07", variants=gap_variants(d), ctx=gap_context(d))


def decompose(d, w: pd.DataFrame, holding: int = 1) -> pd.DataFrame:
    """Overnight vs intraday P&L of a book decided at close t and held close-to-close from close t+1
    (so both legs of day t+2 are earned): returns daily overnight and intraday P&L series."""
    W = w.fillna(0.0).rolling(holding, min_periods=1).mean() if holding > 1 else w.fillna(0.0)
    on = (W * d.p["ret_on"].shift(-2).reindex_like(W).fillna(0.0)).sum(axis=1)
    idd = (W * d.p["ret_id"].shift(-2).reindex_like(W).fillna(0.0)).sum(axis=1)
    return pd.DataFrame({"overnight": on, "intraday": idd})
