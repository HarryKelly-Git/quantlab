"""Backtester: consistency with the shared trade semantics, accounting identities, holdout lock."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import BacktestEngine, save_to_db
from quantlab.core.calendar import TradingCalendar
from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import TradePlan
from quantlab.data.panel import DataBundle, build_panel
from quantlab.experiments.registry import ExperimentRegistry
from quantlab.strategies.base import Strategy
from quantlab.validation.holdout import HoldoutGuard, HoldoutLockedError


class FixedPlan(Strategy):
    """Test strategy: signals supplied externally, fixed plan."""
    family = "test"

    def __init__(self, plan: TradePlan):
        super().__init__("fixed", "0.0.1", {})
        self._plan = plan

    def score(self, fs, universe):  # pragma: no cover - signals are passed directly
        raise NotImplementedError

    def plan(self, fs, symbol, as_of):
        close = float(fs.panel.close.at[pd.Timestamp(as_of), symbol])
        return TradePlan(entry_ref_price=close, stop_price=self._plan.stop_price,
                         target_price=self._plan.target_price, holding_sessions=self._plan.holding_sessions)


def _bundle(closes, opens=None, actions=None):
    dates = pd.bdate_range("2020-01-01", periods=len(next(iter(closes.values()))))
    rows = []
    for sym, cl in closes.items():
        op = (opens or {}).get(sym, cl)
        for d, c, o in zip(dates, cl, op):
            if c is None:
                continue
            rows.append({"symbol": sym, "date": d, "open": o, "high": max(o, c) * 1.01, "low": min(o, c) * 0.99,
                         "close": c, "volume": 1e6, "vwap": c, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    cal = TradingCalendar.from_dates(dates)
    return DataBundle(build_panel(pd.DataFrame(rows), actions, calendar=cal), cal,
                      benchmarks={"market": "SPY", "sectors": {}})


def _one_signal(bundle, sym, i):
    sig = pd.DataFrame(np.nan, index=bundle.panel.dates, columns=bundle.panel.symbols)
    sig.iloc[i, list(bundle.panel.symbols).index(sym)] = 1.0
    return sig


@pytest.fixture
def bt_config(config):
    return config.with_overrides({"backtest": {"sizing": "equal_weight", "max_position_weight": 0.5,
                                               "max_positions": 2, "fractional": True}})


@pytest.mark.parametrize("case", ["time", "stop", "split", "delisted"])
def test_single_trade_matches_simulate_plan(case, bt_config):
    n = 14
    spy = [100.0] * n
    if case == "time":
        cl = [10, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13, 13, 13, 13, 13, 13]
        plan = TradePlan(stop_price=None, holding_sessions=4)
        acts = None
    elif case == "stop":
        cl = [10, 10, 9.5, 9, 7.5, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        plan = TradePlan(stop_price=8.0, holding_sessions=10)
        acts = None
    elif case == "split":
        cl = [10, 10, 10.2, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6, 6, 6]
        plan = TradePlan(stop_price=9.0, holding_sessions=5)
        acts = "split"
    else:
        cl = [10, 10, 10, 9, 8] + [None] * (n - 5)
        plan = TradePlan(stop_price=None, holding_sessions=10)
        acts = None
    closes = {"AAA": cl, "SPY": spy}
    b0 = _bundle(closes)
    actions = None
    if acts == "split":
        actions = pd.DataFrame([{"symbol": "AAA", "ex_date": b0.panel.dates[3], "action_type": "split", "ratio": 2.0,
                                 "amount": np.nan, "declared_date": b0.panel.dates[1],
                                 "available_at": pd.Timestamp("2020-01-02 21:00", tz="UTC"), "pit_status": "PIT",
                                 "source_id": "s", "provider": "test", "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")}])
    b = _bundle(closes, actions=actions)
    strat = FixedPlan(plan)
    eng = BacktestEngine(bt_config, b)
    res = eng.run({"fixed": _one_signal(b, "AAA", 1)}, {"fixed": strat}, start=b.panel.dates[0], end=b.panel.dates[-1])
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    ref_plan = strat.plan(type("FS", (), {"panel": b.panel})(), "AAA", b.panel.dates[1])
    out = simulate_plan(b.panel, "AAA", b.panel.dates[1], ref_plan, CostModel.from_config(bt_config),
                        force_close_at_end=True)
    assert t["exit_reason"] == out.exit_reason
    assert pd.Timestamp(t["exit_date"]) == out.exit_date
    assert t["net_ret"] == pytest.approx(out.net_ret, abs=1e-9)


def test_equity_identity_and_costs(bundle, config):
    from quantlab.features.base import FeatureSet
    from quantlab.strategies.registry import build_strategies
    from quantlab.universe import UniverseEngine
    u = UniverseEngine(config).membership(bundle)
    fs = FeatureSet(bundle, universe=u)
    strat = build_strategies(config)[0]
    res = BacktestEngine(config, bundle).run({strat.strategy_id: strat.score(fs, u)}, {strat.strategy_id: strat}, u,
                                             bundle.panel.dates[300], bundle.panel.dates[-1], fs=fs)
    eq = res.equity
    np.testing.assert_allclose(eq["equity"], eq["cash"] + eq["positions_value"], rtol=1e-12)
    assert len(res.trades) > 20
    assert (res.trades["cost_ret"] > 0).all()
    # final equity = initial + sum of trade P&L (every position closed at the end)
    assert eq["equity"].iloc[-1] == pytest.approx(config.get("backtest.initial_capital") + res.trades["pnl"].sum(), rel=1e-9)
    assert eq["positions"].max() <= config.get("backtest.max_positions")


def test_holdout_lock_blocks_and_token_is_single_use(bundle, config, db):
    cfg = config.with_overrides({"validation": {"holdout": {"start": str(bundle.panel.dates[-50].date())}}})
    strat = FixedPlan(TradePlan(holding_sessions=3))
    sig = _one_signal(bundle, "SYN001", len(bundle.panel.dates) - 20)
    eng = BacktestEngine(cfg, bundle, db)
    with pytest.raises(HoldoutLockedError):
        eng.run({"fixed": sig}, {"fixed": strat})
    token = HoldoutGuard(cfg, db).unlock("verifying the final locked holdout once", "human:test",
                                          bundle.panel.dates[0], bundle.panel.dates[-1])
    res = eng.run({"fixed": sig}, {"fixed": strat}, holdout_token=token)
    assert "HOLDOUT" in res.labels
    with pytest.raises(HoldoutLockedError):
        eng.run({"fixed": sig}, {"fixed": strat}, holdout_token=token)
    assert HoldoutGuard(cfg, db).access_count() == 1


def test_save_to_db_append_only(bundle, bt_config, db):
    config = bt_config
    strat = FixedPlan(TradePlan(holding_sessions=3))
    res = BacktestEngine(config, bundle).run({"fixed": _one_signal(bundle, "SYN002", 300)}, {"fixed": strat})
    exp = ExperimentRegistry(db, config).start("t", "backtest", uses_synthetic=True)
    assert save_to_db(db, exp, res)["trades"] == 1
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM backtest_trades")
