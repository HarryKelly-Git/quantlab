"""Trade simulation semantics (shared by backtest, shadow book, counterfactuals)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import TradePlan
from quantlab.data.panel import build_panel

ZERO_COST = CostModel(((0.0, 0.0),), 0.0, delisting_return=-0.3)


def _bars(closes: dict[str, list[float]], opens: dict[str, list[float]] | None = None, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = []
    for sym, cl in closes.items():
        op = (opens or {}).get(sym, cl)
        for d, c, o in zip(dates, cl, op):
            if c is None:
                continue
            rows.append({"symbol": sym, "date": d, "open": o, "high": max(o, c) * 1.01, "low": min(o, c) * 0.99,
                         "close": c, "volume": 1e6, "vwap": c, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    return pd.DataFrame(rows), TradingCalendar.from_dates(dates)


def test_time_exit_next_open_after_hold():
    bars, cal = _bars({"AAA": [10, 10, 11, 12, 13, 14, 15], "SPY": [100] * 7},
                      opens={"AAA": [10, 10, 10.5, 11.5, 12.5, 13.5, 14.5], "SPY": [100] * 7})
    p = build_panel(bars, calendar=cal)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(holding_sessions=3), ZERO_COST)
    # entry at open of session 1 (10), held sessions 1,2,3 -> exit at open of session 4 (12.5)
    assert out.status == "complete" and out.exit_reason == "TIME"
    assert out.entry_date == cal.sessions[1] and out.exit_date == cal.sessions[4]
    assert out.gross_ret == pytest.approx(12.5 / 10 - 1)
    assert out.holding_sessions == 3


def test_stop_on_close_exits_next_open_with_gap():
    bars, cal = _bars({"AAA": [10, 10, 9, 7, 7.5], "SPY": [100] * 5},
                      opens={"AAA": [10, 10, 9.8, 8.8, 6.5], "SPY": [100] * 5})
    p = build_panel(bars, calendar=cal)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(stop_price=8.0, holding_sessions=10), ZERO_COST)
    # close 7 < stop 8 on session 3 -> exit at session 4 open 6.5 (gap paid in full)
    assert out.exit_reason == "STOP" and out.exit_date == cal.sessions[4]
    assert out.gross_ret == pytest.approx(6.5 / 10 - 1)


def test_split_during_hold_is_neutral():
    # 2-for-1 split effective session 3: raw price halves, economic value unchanged
    bars, cal = _bars({"AAA": [10, 10, 10, 5, 5, 5], "SPY": [100] * 6})
    acts = pd.DataFrame([{"symbol": "AAA", "ex_date": cal.sessions[3], "action_type": "split", "ratio": 2.0,
                          "amount": np.nan, "declared_date": cal.sessions[1],
                          "available_at": pd.Timestamp("2024-01-02 21:00", tz="UTC"), "pit_status": "PIT",
                          "source_id": "x", "provider": "test", "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")}])
    p = build_panel(bars, acts, calendar=cal)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(stop_price=9.0, holding_sessions=4), ZERO_COST)
    assert out.exit_reason == "TIME"          # the raw drop to 5 must NOT trigger the 9.0 stop
    assert out.gross_ret == pytest.approx(0.0, abs=1e-12)


def test_delisting_applies_haircut_after_missing_sessions():
    # last bar at session 2; the rule fires after 5 consecutive missing sessions (session 7)
    bars, cal = _bars({"AAA": [10, 10, 10] + [None] * 7, "SPY": [100] * 10})
    p = build_panel(bars, calendar=cal)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(holding_sessions=20), ZERO_COST)
    assert out.status == "delisted" and out.exit_reason == "DELISTED"
    assert out.gross_ret == pytest.approx(-0.3)
    assert out.exit_date == cal.sessions[7]          # booked when it became knowable, not at the last bar


def test_short_gap_is_not_a_delisting():
    # 3 missing sessions then the symbol trades again: no delisting, the position is still held
    bars, cal = _bars({"AAA": [10, 10, 10, None, None, None, 11, 12], "SPY": [100] * 8})
    p = build_panel(bars, calendar=cal)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(holding_sessions=20), ZERO_COST)
    assert out.status == "open" and out.exit_reason is None
    gap_at_end, cal2 = _bars({"AAA": [10, 10, 10, None, None], "SPY": [100] * 5})
    out2 = simulate_plan(build_panel(gap_at_end, calendar=cal2), "AAA", cal2.sessions[0], TradePlan(holding_sessions=20),
                         ZERO_COST, force_close_at_end=True)
    assert out2.exit_reason == "END_OF_TEST" and out2.gross_ret == pytest.approx(0.0)   # no haircut from a data gap


def test_open_when_data_ends():
    bars, cal = _bars({"AAA": [10, 10, 11], "SPY": [100] * 3})
    p = build_panel(bars, calendar=cal)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(holding_sessions=10), ZERO_COST)
    assert out.status == "open" and out.net_ret is None


def test_costs_are_charged_round_trip():
    bars, cal = _bars({"AAA": [10] * 6, "SPY": [100] * 6})
    p = build_panel(bars, calendar=cal)
    cm = CostModel(((0.0, 10.0),), 5.0)
    out = simulate_plan(p, "AAA", cal.sessions[0], TradePlan(holding_sessions=2), cm)
    assert out.cost_ret == pytest.approx(2 * 15 / 1e4)
    assert out.net_ret == pytest.approx(-2 * 15 / 1e4)
