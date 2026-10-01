"""The share-based backtester and simulate_plan agree trade-for-trade under BOTH stop models."""
from __future__ import annotations

import pandas as pd
import pytest

from quantlab.backtest.engine import BacktestEngine
from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import TradePlan

from .test_engine import FixedPlan, _bundle, _one_signal, bt_config  # noqa: F401  (fixture re-export)

CASES = {
    #          closes                                         opens                                          stop
    "touch": ([10, 10, 9.6, 9.0, 8.8, 8.8, 8.8, 8.8], [10, 10, 9.7, 9.4, 8.9, 8.8, 8.8, 8.8], 9.2),
    "gap": ([10, 10, 9.6, 8.4, 8.4, 8.4, 8.4, 8.4], [10, 10, 9.7, 8.5, 8.4, 8.4, 8.4, 8.4], 9.2),
    "entry_day": ([10, 10, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5], [10, 10, 10.4, 10.5, 10.5, 10.5, 10.5, 10.5], 9.95),
}


@pytest.mark.parametrize("model,distance", [("intraday", 1.0), ("intraday", 1.75), ("close", 1.0)])
@pytest.mark.parametrize("case", sorted(CASES))
def test_backtester_matches_simulate_plan(case, model, distance, bt_config):  # noqa: F811
    cfg = bt_config.with_overrides({"execution": {"stop_model": model, "protective_stop": {"distance": distance}}})
    closes, opens, stop = CASES[case]
    b = _bundle({"AAA": closes, "SPY": [100.0] * len(closes)}, opens={"AAA": opens, "SPY": [100.0] * len(closes)})
    strat = FixedPlan(TradePlan(stop_price=stop, holding_sessions=5))
    res = BacktestEngine(cfg, b).run({"fixed": _one_signal(b, "AAA", 0)}, {"fixed": strat},
                                     start=b.panel.dates[0], end=b.panel.dates[-1])
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    costs = CostModel.from_config(cfg)
    assert costs.stop_model == model and costs.broker_stop_distance == distance
    ref_plan = strat.plan(type("FS", (), {"panel": b.panel})(), "AAA", b.panel.dates[0])
    out = simulate_plan(b.panel, "AAA", b.panel.dates[0], ref_plan, costs, force_close_at_end=True)
    # the entry-day dip never CLOSES below the stop: only the intraday model sees it
    expected = "TIME" if (case == "entry_day" and model == "close") else "STOP"
    assert t["exit_reason"] == out.exit_reason == expected
    assert pd.Timestamp(t["exit_date"]) == out.exit_date
    assert t["net_ret"] == pytest.approx(out.net_ret, abs=1e-9)
