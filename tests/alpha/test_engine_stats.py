"""Alpha research engine: hand-checked P&L/cost maths, timing, metrics, and method validation of the
multiple-testing statistics (null world must not pass; a planted edge must be found)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import mht, registry, splits
from quantlab.alpha.engine import cost_bps_matrix, quantile_weights, run_weights
from quantlab.alpha.metrics import core_metrics, remove_top_trades, spells, trade_stats


def _idx(n):
    return pd.bdate_range("2018-01-01", periods=n)


def test_run_weights_hand_computed():
    d = _idx(4)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.0], "B": [-0.5, -0.5, 0.0, 0.0]}, index=d)
    roo = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.0], "B": [0.00, 0.01, 0.03, 0.0]}, index=d)
    cost = pd.DataFrame(10.0, index=d, columns=["A", "B"])         # 10 bps one way
    r = run_weights(w, roo, cost, borrow_bps=0.0)
    # decision t earns ret_oo[t+1]: day0 -> 0.5*0.02 - 0.5*0.01 = 0.005 ; day1 -> 0.5*-0.01 - 0.5*0.03 = -0.02
    assert r.gross.iloc[0] == pytest.approx(0.005)
    assert r.gross.iloc[1] == pytest.approx(-0.02)
    # costs: day0 trade |0.5|+|0.5| = 1.0 at 10bps = 0.001; day1 no change; day2 close out 1.0 -> 0.001
    assert r.costs.iloc[0] == pytest.approx(0.001)
    assert r.costs.iloc[1] == pytest.approx(0.0)
    assert r.costs.iloc[2] == pytest.approx(0.001)
    assert r.net.iloc[0] == pytest.approx(0.004)
    assert r.turnover.iloc[0] == pytest.approx(0.5)


def test_holding_period_averages_cohorts_and_borrow():
    d = _idx(5)
    w = pd.DataFrame({"A": [1.0, 0, 0, 0, 0]}, index=d)
    roo = pd.DataFrame({"A": [0.0] * 5}, index=d)
    r = run_weights(w, roo, pd.DataFrame(0.0, index=d, columns=["A"]), holding=2)
    assert list(r.weights["A"]) == [1.0, 0.5, 0.0, 0.0, 0.0]
    ws = pd.DataFrame({"A": [-1.0] * 5}, index=d)
    rb = run_weights(ws, roo, pd.DataFrame(0.0, index=d, columns=["A"]), borrow_bps=252.0)
    assert rb.costs.iloc[2] == pytest.approx(1e-4)                    # 252 bps / 252 days on gross short 1


def test_lookahead_signal_is_rewarded_and_lagged_signal_is_not():
    """Sanity of the timing convention: a signal equal to the FUTURE P&L return makes money, the same
    values one day stale do not (iid returns)."""
    rng = np.random.default_rng(0)
    d = _idx(600)
    cols = [f"S{i}" for i in range(60)]
    roo = pd.DataFrame(rng.normal(0, 0.02, (600, 60)), index=d, columns=cols)
    u = pd.DataFrame(True, index=d, columns=cols)
    cost = pd.DataFrame(0.0, index=d, columns=cols)
    cheat = roo.shift(-1)                      # the return the book will earn: forbidden information
    good = run_weights(quantile_weights(cheat, u, q=0.2, min_names=10), roo, cost)
    stale = run_weights(quantile_weights(roo, u, q=0.2, min_names=10), roo, cost)
    assert good.gross.mean() > 0.01
    assert abs(stale.gross.mean()) < 0.002


def test_cost_tiers():
    mdv = pd.DataFrame({"A": [2e8], "B": [3e7], "C": [6e6], "D": [1e6], "E": [np.nan]})
    c = cost_bps_matrix(mdv)
    assert list(c.iloc[0]) == [7.0, 10.0, 15.0, 30.0, 30.0]


def test_quantile_weights_sides_and_min_names():
    d = _idx(2)
    sig = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 2, index=d, columns=list("ABCDEFGHIJ"), dtype=float)
    u = pd.DataFrame(True, index=d, columns=sig.columns)
    w = quantile_weights(sig, u, q=0.2, min_names=5)
    assert w.iloc[0][["I", "J"]].tolist() == [0.5, 0.5]
    assert w.iloc[0][["A", "B"]].tolist() == [-0.5, -0.5]
    assert w.iloc[0].sum() == pytest.approx(0.0)
    assert (quantile_weights(sig, u, q=0.2, min_names=20) == 0).all().all()


def test_core_metrics_known_values():
    rng = np.random.default_rng(9)
    d = _idx(1000)
    bench = pd.Series(rng.normal(0.0004, 0.01, 1000), index=d)
    r = 0.5 * bench + pd.Series(rng.normal(0.0002, 0.005, 1000), index=d)
    m = core_metrics(r, bench=bench)
    assert m["n_days"] == 1000
    assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert m["max_drawdown"] <= 0
    assert m["beta"] == pytest.approx(0.5, abs=0.05)
    assert m["alpha_ann"] == pytest.approx((r - m["beta"] * bench).mean() * 252)
    eq = (1 + r).cumprod()
    assert m["max_drawdown"] == pytest.approx(float((eq / eq.cummax() - 1).min()))


def test_spells_and_remove_top_trades():
    d = _idx(6)
    w = pd.DataFrame({"A": [1, 1, 0, 1, 0, 0], "B": [0, 0, 1, 1, 1, 0]}, index=d, dtype=float)
    c = pd.DataFrame({"A": [0.05, 0.05, 0, -0.01, 0, 0], "B": [0, 0, 0.001, 0.001, 0.001, 0]}, index=d)
    sp = spells(w, c)
    assert len(sp) == 3
    ts = trade_stats(sp)
    assert ts["n_trades"] == 3 and ts["win_rate"] == pytest.approx(2 / 3)
    net = c.sum(axis=1)
    rt = remove_top_trades(net, w, c, fractions=(0.34,))
    assert rt["without_top_34pct"]["total_return"] < (1 + net).prod() - 1


# --- multiple testing: method validation ---------------------------------------------------------
def test_null_world_does_not_pass_and_planted_edge_does():
    rng = np.random.default_rng(42)
    T, K = 1500, 40
    null = rng.normal(0, 0.01, (T, K))
    assert mht.white_reality_check(null, n_resamples=500).p_value > 0.05
    assert mht.hansen_spa(null, n_resamples=500).p_value > 0.05
    planted = null.copy()
    planted[:, 7] += 0.0012                                         # ~1.9 annual Sharpe on 1% daily vol
    spa = mht.hansen_spa(planted, n_resamples=500)
    assert spa.p_value < 0.05 and spa.best_index == 7
    assert mht.white_reality_check(planted, n_resamples=500).p_value < 0.05


def test_pbo_null_vs_planted():
    rng = np.random.default_rng(3)
    null = rng.normal(0, 0.01, (1200, 30))
    p0 = mht.pbo_cscv(null, n_blocks=10)["pbo"]
    assert 0.25 < p0 < 0.85                                          # no skill: around one half
    planted = null.copy()
    planted[:, 4] += 0.0015
    assert mht.pbo_cscv(planted, n_blocks=10)["pbo"] < 0.1


def test_effective_trials():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.01, (800, 1))
    same = np.hstack([base + rng.normal(0, 1e-4, (800, 1)) for _ in range(10)])
    indep = rng.normal(0, 0.01, (800, 10))
    assert mht.effective_trials(same)["effective_trials"] < 1.5
    assert mht.effective_trials(indep)["effective_trials"] > 8


def test_permutation_shuffle_keeps_rows():
    rng = np.random.default_rng(1)
    s = pd.DataFrame(rng.normal(size=(5, 8)))
    s.iloc[0, 3] = np.nan
    p = mht.shuffle_within_dates(s, rng)
    assert np.isnan(p.iloc[0, 3])
    for i in range(5):
        assert sorted(s.iloc[i].dropna()) == pytest.approx(sorted(p.iloc[i].dropna()))


# --- splits / registry ------------------------------------------------------------------------------
def test_splits_are_pinned_and_holdout_is_unreachable():
    assert splits.SPLITS["equity"]["OOS"] == ("2022-01-01", "2024-12-31")
    assert splits.SPLITS["equity"]["HOLDOUT"][0] == "2025-01-01"
    s = pd.Series(1.0, index=pd.bdate_range("2015-06-01", "2024-12-31"))
    assert splits.slice_split(s, "equity", "TRAIN").index.min() >= pd.Timestamp("2016-01-01")
    with pytest.raises(splits.HoldoutAccessError):
        splits.slice_split(s, "equity", "HOLDOUT")


def test_registry_oos_once_and_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "DIR", tmp_path)
    monkeypatch.setattr(registry, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(registry, "QUEUE", tmp_path / "queue.json")
    registry.append_run(hypothesis_id="H1", family="f", spec={"q": 0.1}, split="OOS", metrics={"t": 1})
    registry.append_run(hypothesis_id="H1", family="f", spec={"q": 0.1}, split="OOS", metrics={"t": 1})  # same spec ok
    with pytest.raises(registry.OOSReuseError):
        registry.append_run(hypothesis_id="H1", family="f", spec={"q": 0.2}, split="OOS", metrics={})
    assert registry.trial_counts("f")["configs"] == 1
    E = registry.Evidence
    assert registry.classify(E(False, None, None, None, None, None, None, None, None, None, None))[0] == "F"
    assert registry.classify(E(True, 0.5, 2.5, None, None, None, None, None, None, None, None))[0] == "D"
    assert registry.classify(E(True, 2.5, 3.0, 0.3, 0.0001, 0.33, 0.97, 0.01, 0.2, True, True))[0] == "G"
    assert registry.classify(E(True, 2.5, 3.0, 2.4, 0.0005, 1.0, 0.97, 0.01, 0.2, True, True))[0] == "A"
    assert registry.classify(E(True, 2.5, 3.0, 2.4, 0.0005, 1.0, 0.97, 0.01, 0.2, False, True))[0] == "B"


def test_intraday_roundtrip_costs():
    d = _idx(3)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=d)
    roo = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=d)
    cost = pd.DataFrame(10.0, index=d, columns=["A"])
    r = run_weights(w, roo, cost, roundtrip_each_period=True, borrow_bps=0.0)
    assert list(r.costs) == pytest.approx([0.002, 0.002, 0.002])        # in at the open, out at the close, daily
