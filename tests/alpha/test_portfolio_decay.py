"""Portfolio construction and decay monitoring on synthetic return streams."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import portfolio as pf


def _streams(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    a = rng.normal(0.0004, 0.01, n)
    b = rng.normal(0.0002, 0.02, n)
    c = -0.5 * a + rng.normal(0.0003, 0.008, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}, index=idx)


@pytest.mark.parametrize("method", pf.METHODS)
def test_weights_valid(method):
    w = pf.weights(_streams(), method)
    assert w.min() >= -1e-9 and w.sum() == pytest.approx(1.0, abs=1e-6)


def test_inverse_vol_and_risk_parity_shape():
    R = _streams()
    iv = pf.weights(R, "inverse_vol")
    assert iv["b"] < iv["a"]                          # the noisiest stream gets the least weight
    rp = pf.weights(R, "risk_parity")
    S = np.cov(R.to_numpy(), rowvar=False)
    rc = rp.to_numpy() * (S @ rp.to_numpy())
    assert rc.max() / rc.min() < 1.1                   # equal risk contributions


def test_min_variance_beats_equal_in_sample():
    R = _streams()
    S = np.cov(R.to_numpy(), rowvar=False)
    v = lambda w: float(w @ S @ w)
    assert v(pf.weights(R, "min_variance").to_numpy()) <= v(pf.weights(R, "equal").to_numpy()) + 1e-12


def test_walk_forward_uses_only_past():
    R = _streams(n=1300)
    out = pf.walk_forward(R, "inverse_vol", "2017-12-29")
    assert out.index.min() > pd.Timestamp("2017-12-29")
    R2 = R.copy()
    R2.loc["2019-06-01":] *= 5.0                        # change the future: past outputs must not move
    out2 = pf.walk_forward(R2, "inverse_vol", "2017-12-29")
    pd.testing.assert_series_equal(out.loc[:"2018-12-31"], out2.loc[:"2018-12-31"])


def test_decay_alerts():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2016-01-01", periods=600)
    good = rng.normal(0.002, 0.01, 300)
    bad = rng.normal(-0.002, 0.01, 300)
    r = pd.Series(np.r_[good, bad], index=idx)
    kinds = {a["alert"] for a in pf.decay_alerts(r)}
    assert "SHARPE_DROP" in kinds and "SHARPE_BELOW_FLOOR" in kinds
    steady = pd.Series(rng.normal(0.002, 0.01, 600), index=idx)
    assert not {"SHARPE_DROP", "SHARPE_BELOW_FLOOR"} & {a["alert"] for a in pf.decay_alerts(steady)}
