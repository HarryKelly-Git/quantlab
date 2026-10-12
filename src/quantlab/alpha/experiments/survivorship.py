"""H02 survivorship-bias measurement (Part 6). PRE-REGISTERED books, each run on the survivorship-free
liquid universe and on the SAME universe restricted to names still listed at the store end (what a
current-listings-only store, like QuantLab's old one, sees). Difference = the bias.

Books (long-only, equal weight, next-open execution, CostModel tiers):
  universe_ew        every eligible name (the "market" of liquid stocks), rebalanced monthly (21-day hold)
  momentum_12_1      top decile of 12-1 momentum, 21-day hold
  reversal_losers_5d bottom decile of 5-day return (biggest recent losers), 5-day hold
  low_price_vol      top decile of 60-day volatility (distressed / speculative names), 21-day hold
Delisting-return sensitivity for the full universe: distressed -30% (default), -100%, and 0%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha import splits
from quantlab.alpha.engine import quantile_weights, run_weights
from quantlab.alpha.metrics import core_metrics
from quantlab.alpha.experiments.momentum import mom_signal


def _books(d, u):
    r = d.p["ret_cc"]
    ew = u.astype(float)
    ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {
        "universe_ew": (ew, 21),
        "momentum_12_1": (quantile_weights(mom_signal(d, 252, "absolute"), u, q=0.1, long_only=True), 21),
        "reversal_losers_5d": (quantile_weights(-r.rolling(5, min_periods=5).sum(), u, q=0.1, long_only=True), 5),
        "high_vol": (quantile_weights(d.vol60, u, q=0.1, long_only=True), 21),
    }


def run(d) -> dict:
    out = {"n_names_full": float(d.u_liquid.sum(axis=1).mean()), "n_names_survivors": float(d.u_survivor.sum(axis=1).mean())}
    full = _books(d, d.u_liquid)
    surv = _books(d, d.u_survivor)
    for name in full:
        wf, h = full[name]
        ws, _ = surv[name]
        rf = run_weights(wf, d.p["ret_oo"], d.cost_bps, holding=h).net
        rs = run_weights(ws, d.p["ret_oo"], d.cost_bps, holding=h).net
        rows = {}
        for lab, x in (("full", rf), ("survivors_only", rs)):
            m = core_metrics(x.loc["2016-01-01":"2024-12-31"])
            rows[lab] = {k: m.get(k) for k in ("cagr", "sharpe", "max_drawdown", "total_return")}
        rows["bias_cagr_pp"] = (rows["survivors_only"]["cagr"] - rows["full"]["cagr"]) * 100
        for nm, alt in d.alt_ret_oo.items():
            ra = run_weights(wf, alt, d.cost_bps, holding=h).net
            rows[f"full_delist_{nm}_cagr"] = core_metrics(ra.loc["2016-01-01":"2024-12-31"]).get("cagr")
        out[name] = rows
    return out
