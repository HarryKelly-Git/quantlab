"""Sprint P3 harness: master's own strategies and engine run on a DataBundle built from an AlphaPanel, and
the re-sizing subclass changes ONLY quantities. At a constant 10% it reproduces master's native equal_weight
run exactly (same entries, exits and equity). SYNTHETIC data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import master_replay as mr
from quantlab.alpha.synthetic import make_panel
from quantlab.backtest.engine import BacktestEngine
from quantlab.config import load_config
from quantlab.features.base import FeatureSet
from quantlab.strategies.arena import StrategyArena
from quantlab.strategies.registry import build_strategies
from quantlab.universe.engine import UniverseEngine


@pytest.fixture(scope="module")
def setup():
    p = make_panel(n_stocks=40, n_days=700, seed=4)
    cfg = load_config(overrides={"backtest": {"sizing": "equal_weight"}})
    sectors = pd.DataFrame(np.broadcast_to(np.array(["XLK"] * len(p.symbols), dtype=object), p["close"].shape),
                           index=p.dates, columns=p.symbols)
    b = mr.bundle_from_research(p, p.symbols, p.master, sectors, cfg.section("benchmarks").get("sectors", {}))
    uni = UniverseEngine(cfg).membership(b)
    fs = FeatureSet(b, universe=uni)
    strats = build_strategies(cfg, only=["momentum_trend", "mean_reversion"])
    sig = StrategyArena(strats).scores(fs, uni)
    return p, cfg, b, uni, fs, strats, sig


def test_bundle_reproduces_total_returns(setup):
    p, cfg, b, *_ = setup
    tri = b.panel.aclose["S001"].dropna()
    ac = p["adj_close"]["S001"].dropna()
    np.testing.assert_allclose((tri / tri.iloc[0]).to_numpy(), (ac / ac.iloc[0]).to_numpy(), rtol=1e-9)
    assert (b.reference.set_index("symbol").loc["SPY", "is_etf"]) and b.reference["exchange"].eq("NYSE").all()


def test_constant_weight_resize_equals_native_equal_weight(setup):
    p, cfg, b, uni, fs, strats, sig = setup
    S = {s.strategy_id: s for s in strats}
    native = BacktestEngine(cfg, b).run(sig, S, uni, fs=fs)
    sized_eng = mr.SizedEngine(cfg, b, weight=lambda i, c, sid: 0.10, cap=0.10)
    sized = sized_eng.run(sig, S, uni, fs=fs)
    assert len(native.trades) > 10
    cols = ["symbol", "strategy_id", "entry_date", "exit_date", "qty"]
    pd.testing.assert_frame_equal(native.trades[cols].reset_index(drop=True), sized.trades[cols].reset_index(drop=True))
    np.testing.assert_allclose(native.equity["equity"].to_numpy(), sized.equity["equity"].to_numpy(), rtol=1e-12)
    assert sized_eng.sizing_diag["fallback_weight"] == 0


def test_inverse_vol_resize_changes_only_quantities(setup):
    p, cfg, b, uni, fs, strats, sig = setup
    S = {s.strategy_id: s for s in strats}
    vol = b.panel.ret.rolling(21, min_periods=17).std().to_numpy() * np.sqrt(252)
    w = mr.inverse_vol_weight(vol, m=float(np.nanmedian(vol)))
    a = BacktestEngine(cfg, b).run(sig, S, uni, fs=fs).trades
    c = mr.SizedEngine(cfg, b, weight=w, cap=0.10).run(sig, S, uni, fs=fs).trades
    key = ["symbol", "strategy_id", "entry_date"]
    shared = a[key].merge(c[key], on=key)
    assert len(shared) >= 0.8 * len(a)            # same signals and slots; cash limits may shift a few entries
    assert not np.allclose(a["qty"].to_numpy()[: len(shared)], c["qty"].to_numpy()[: len(shared)])


def test_period_metrics_and_sharpe_ci(setup):
    p, cfg, b, uni, fs, strats, sig = setup
    S = {s.strategy_id: s for s in strats}
    r = BacktestEngine(cfg, b).run(sig, S, uni, fs=fs)
    d = p.dates
    m = mr.period_metrics(r.equity, r.trades, str(d[300].date()), str(d[-1].date()))
    assert m["n_days"] > 300 and np.isfinite(m["vol"])
    ci = mr.sharpe_diff_ci(r.equity, r.equity, str(d[300].date()), str(d[-1].date()), n_boot=50)
    assert abs(ci["diff"]) < 1e-12
