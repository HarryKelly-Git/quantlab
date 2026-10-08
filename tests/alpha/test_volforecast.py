"""Sprint P2: the volatility-forecast features are point-in-time, and the method recovers known truths on
SYNTHETIC worlds: the distribution maths gives nominal tail coverage when sigma is right, and the pooled
GARCH fit recovers planted (a, b)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import research_data
from quantlab.alpha import volforecast as vf
from quantlab.alpha.synthetic import make_panel, truncate


def _rd(p):
    rd = research_data.from_panel(p, manifest={"synthetic": True})
    rd.u_liquid = p["close"].notna() & (p["n_hist"] >= 60)
    return rd


@pytest.fixture(scope="module")
def full():
    p = make_panel(n_stocks=30, n_days=520, seed=5)
    return p, _rd(p)


def test_vol_features_truncation_invariant(full):
    p, d = full
    W = vf.features(p, d.u_liquid, d.mdv20)
    g = vf.garch_var(W.f["lr2"].to_numpy(), W.f["v252"].to_numpy(), 0.08, 0.90)
    checked = 0
    for D in p.dates[[300, 410, 519]]:
        pt = truncate(p, D)
        dt = _rd(pt)
        Wt = vf.features(pt, dt.u_liquid, dt.mdv20)
        for k in W.f:
            a = W.f[k].loc[D]
            b = Wt.f[k].loc[D].reindex(a.index)
            assert (a.notna() == b.notna()).all(), f"{k}: NaN pattern differs at {D.date()}"
            np.testing.assert_allclose(a[a.notna()], b[a.notna()], rtol=1e-9, atol=1e-12, err_msg=f"{k} not PIT")
            checked += int(a.notna().sum())
        for k in W.spy:
            assert np.isclose(W.spy[k].loc[D], Wt.spy[k].loc[D], equal_nan=True), k
        gt = vf.garch_var(Wt.f["lr2"].to_numpy(), Wt.f["v252"].to_numpy(), 0.08, 0.90)
        np.testing.assert_allclose(g[p.dates.get_loc(D)], gt[-1], rtol=1e-9, equal_nan=True)
    assert checked > 1000


def test_targets_are_forward_only(full):
    p, d = full
    T = vf.targets(d.ret_cc_pnl, p.symbols)
    D = p.dates[300]
    lr = np.log1p(d.ret_cc_pnl.loc[p.dates[301:306], "S000"])
    assert math.isclose(T[5]["rv"].at[D, "S000"], math.sqrt(252 * (lr ** 2).mean()), rel_tol=1e-12)
    assert math.isclose(T[5]["move"].at[D, "S000"], float(np.expm1(lr.sum())), rel_tol=1e-12)


def test_distribution_calibrated_when_sigma_is_right():
    rng = np.random.default_rng(1)
    nu, h, n = 5, 5, 200_000
    sig = rng.uniform(0.15, 0.9, n)
    z = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    move = sig * math.sqrt(h / 252) * z
    assert vf.fit_nu(move, sig, h) in (4, 5, 6)
    for q in vf.TAIL_Q:
        e = (np.abs(move) > vf.abs_quantile(sig, h, nu, q)).mean()
        assert abs(e - (1 - q)) < 0.004, (q, e)
    p = vf.p_exceed(sig, h, nu, 0.05)
    assert abs(p.mean() - (np.abs(move) > 0.05).mean()) < 0.004
    assert abs(vf.expected_abs_move(sig, h, nu).mean() - np.abs(move).mean()) / np.abs(move).mean() < 0.01


def test_garch_fit_recovers_planted_parameters():
    rng = np.random.default_rng(3)
    T, N, a, b, v = 1500, 60, 0.07, 0.90, 0.0004
    r2 = np.empty((T, N))
    s2 = np.full(N, v)
    for t in range(T):
        r = np.sqrt(s2) * rng.standard_normal(N)
        r2[t] = r ** 2
        s2 = (1 - a - b) * v + a * r ** 2 + b * s2
    fit = vf.fit_garch(r2, np.full((T, N), v), n_stocks=60)
    assert abs(fit["a"] - a) <= 0.02 and abs(fit["b"] - b) <= 0.03, fit


def test_qlike_prefers_the_true_sigma():
    rng = np.random.default_rng(2)
    sig = rng.uniform(0.1, 0.8, 50_000)
    rv = sig * np.sqrt(rng.chisquare(20, sig.size) / 20)
    assert vf.qlike(rv, sig).mean() < vf.qlike(rv, np.full_like(sig, sig.mean())).mean()
