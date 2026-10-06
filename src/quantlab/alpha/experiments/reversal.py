"""H01 short-term reversal (Part 7). PRE-REGISTERED GRID (fixed before TRAIN was run):

  lookback L      1, 2, 3, 5, 10 sessions (sum of daily returns up to the close of t)
  ranking basis   raw | market (r - beta_mkt * r_SPY) | sector (r - r_own_statistical_sector_ETF) |
                  residual (two-factor residual, loadings from data before the current month)
  weighting       equal | inverse_vol (1 / 60-day vol)
  book            ls: long bottom decile (losers), short top decile (winners), each side sums to 1
                  lo_hedged: long bottom decile, short SPY x the book's market beta
  holding         1 | 5 sessions (overlapping cohorts)
  -> 5 x 4 x 2 x 2 x 2 = 160 variants. Universe: liquid (>= $5, MDV20 >= $5M, >= 252 sessions, common
  stock), survivorship-free. Execution: next open. Costs: CostModel tiers; borrow 50 bps/yr on shorts.
Signal: s = -cumulative basis return (higher = bigger recent loser). Selection: TRAIN net Sharpe.
Extra skeptic checks for the selected variant: large-cap universe, survivors-only universe (the
survivorship-bias measurement, H02), and correlation with the Ken French short-term-reversal factor.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from quantlab.alpha.engine import beta_neutralize, quantile_weights
from quantlab.alpha.experiment import Context, Variant, run_family

LOOKBACKS = (1, 2, 3, 5, 10)
BASES = ("raw", "market", "sector", "residual")
WEIGHTINGS = ("equal", "inverse_vol")
BOOKS = ("ls", "lo_hedged")
HOLDINGS = (1, 5)
Q = 0.10


def basis_returns(d, basis: str) -> pd.DataFrame:
    r = d.p["ret_cc"]
    if basis == "raw":
        return r
    if basis == "market":
        return r - d.betas["beta_mkt"].mul(r["SPY"], axis=0)
    if basis == "sector":
        return r - d.sec_ret
    if basis == "residual":
        return d.resid
    raise ValueError(basis)


def signal(d, L: int, basis: str) -> pd.DataFrame:
    return -basis_returns(d, basis).rolling(L, min_periods=L).sum()


def weights(d, L, basis, weighting, book, universe=None) -> pd.DataFrame:
    u = d.u_liquid if universe is None else universe
    s = signal(d, L, basis)
    w = quantile_weights(s, u, q=Q, long_only=(book == "lo_hedged"), weighting=weighting, vol=d.vol60)
    if book == "lo_hedged":
        w = beta_neutralize(w, d.betas["beta_mkt"])
    return w


def variants(d) -> list[Variant]:
    out = []
    for L, b, wt, bk, h in itertools.product(LOOKBACKS, BASES, WEIGHTINGS, BOOKS, HOLDINGS):
        spec = {"universe": "liquid: >=$5, MDV20>=$5M, >=252 sessions, COMMON, survivorship-free",
                "signal": f"-sum of {b} returns over {L} sessions", "entry": "next open", "position": f"decile {bk}",
                "sizing": wt, "exit": f"overlapping {h}-session holding", "costs": "CostModel tiers + 5bps; borrow 50bps/yr",
                "benchmark": "cash (zero-investment book)", "risk": "each side sums to 1; beta-hedged if lo_hedged"}
        out.append(Variant(f"L{L}_{b}_{wt}_{bk}_h{h}", spec, (lambda L=L, b=b, wt=wt, bk=bk: weights(d, L, b, wt, bk)), holding=h))
    return out


def context(d) -> Context:
    return Context(ret_oo=d.p["ret_oo"], ret_cc=d.ret_cc_pnl, cost_bps=d.cost_bps, bench=d.bench_oo,
                   regimes=d.regimes, buckets=d.buckets, alt_ret_oo=d.alt_ret_oo, data_manifest=d.manifest)


def run(d) -> dict:
    return run_family(family="H01_short_term_reversal", hypothesis_id="H01", variants=variants(d), ctx=context(d),
                      prior_trials=2, notes="prior_trials=2: QuantLab's mean_reversion and extreme_reversal strategies")
