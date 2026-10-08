"""Point-in-time safety of every alpha-discovery equity signal: the value at date D must be identical
whether computed on the full synthetic history or on data truncated at D (CLAUDE.md PIT rule), plus a
method-validation check that the reversal research finds a PLANTED reversal and not a null one."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import research_data
from quantlab.alpha.engine import quantile_weights, run_weights
from quantlab.alpha.experiments import anomalies, momentum, reversal
from quantlab.alpha.synthetic import make_panel, truncate


def _rd(p):
    rd = research_data.from_panel(p, manifest={"synthetic": True})
    # synthetic stocks are cheap and small: open the universe up so signals are populated
    rd.u_liquid = (p["close"].notna() & (p["n_hist"] >= 60)) & pd.DataFrame(
        np.broadcast_to(p.master.set_index("symbol")["sec_type"].reindex(p.symbols).eq("COMMON").to_numpy(), p["close"].shape),
        index=p.dates, columns=p.symbols)
    return rd


@pytest.fixture(scope="module")
def full():
    p = make_panel(n_stocks=40, n_days=600, seed=3)
    return p, _rd(p)


SIGNALS = {
    "rev_raw_5": lambda d: reversal.signal(d, 5, "raw"),
    "rev_market_3": lambda d: reversal.signal(d, 3, "market"),
    "rev_sector_2": lambda d: reversal.signal(d, 2, "sector"),
    "rev_residual_5": lambda d: reversal.signal(d, 5, "residual"),
    "resmom_6_1_ir": lambda d: momentum.residual_mom_signal(d, "6-1", "two_factor", "ir"),
    "mom_63_abs": lambda d: momentum.mom_signal(d, 63, "absolute"),
    "mom_126_voladj": lambda d: momentum.mom_signal(d, 126, "vol_adjusted"),
    "mom_21_accel": lambda d: momentum.mom_signal(d, 21, "acceleration"),
    "mom_63_consistency": lambda d: momentum.mom_signal(d, 63, "consistency"),
    "max5": lambda d: anomalies.max_signal(d, 5),
    "ivol21_two_factor": lambda d: anomalies.ivol_signal(d, 21, "two_factor"),
}


@pytest.mark.parametrize("name", list(SIGNALS))
def test_signal_truncation_invariant(full, name):
    p, d = full
    sig_full = SIGNALS[name](d)
    checked = 0
    for D in p.dates[[300, 420, 599]]:
        dt = _rd(truncate(p, D))
        sig_t = SIGNALS[name](dt)
        a = sig_full.loc[D]
        b = sig_t.loc[D].reindex(a.index)
        both = a.notna() | b.notna()
        assert (a.notna() == b.notna())[both].all(), f"{name}: NaN pattern differs at {D.date()}"
        np.testing.assert_allclose(a[a.notna()].to_numpy(), b[a.notna()].to_numpy(), rtol=1e-9, atol=1e-12,
                                   err_msg=f"{name} not point-in-time at {D.date()}")
        checked += int(a.notna().sum())
    assert checked > 0, f"{name}: vacuous test (no values)"


def test_planted_reversal_found_and_null_not():
    def edge(planted: float, seed: int = 11) -> float:
        p = make_panel(n_stocks=60, n_days=500, seed=seed, planted_reversal=planted)
        d = _rd(p)
        w = quantile_weights(reversal.signal(d, 1, "raw"), d.u_liquid, q=0.2, min_names=10)
        cost = pd.DataFrame(0.0, index=p.dates, columns=p.symbols)
        r = run_weights(w, p["ret_oo"], cost)          # decide at close t, trade next open
        x = r.gross.dropna()
        return x.mean() / x.std() * np.sqrt(len(x))
    assert edge(0.3) > 3.0          # planted: next-day reversal of 30% of today's idiosyncratic move
    # null world: one seed can land in a 1-in-300 tail (seed 11 gives t = -2.96), so judge the AVERAGE
    # t over six worlds: its standard error is 1/sqrt(6) = 0.41, so +/-1.25 is a 3-sigma band
    null_ts = [edge(0.0, seed) for seed in range(6)]
    assert abs(float(np.mean(null_ts))) < 1.25, null_ts
