"""execution.stop_model = intraday: a broker-held stop-market order, as the paper runner places it.
Gap through -> that session's open; intraday touch -> the stop price, same session; on the entry
session only the low counts and only when the entry was above the stop. ``close`` keeps the old rule."""
from __future__ import annotations

from dataclasses import replace

import pytest

from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import TradePlan
from quantlab.data.panel import build_panel

from .test_tradesim import ZERO_COST, _bars

INTRADAY = replace(ZERO_COST, stop_model="intraday")


def _run(closes, opens, stop, costs, hold=10):
    bars, cal = _bars({"AAA": closes, "SPY": [100.0] * len(closes)},
                      opens={"AAA": opens, "SPY": [100.0] * len(closes)})
    p = build_panel(bars, calendar=cal)
    return simulate_plan(p, "AAA", cal.sessions[0], TradePlan(stop_price=stop, holding_sessions=hold), costs), cal


def test_intraday_touch_exits_at_the_stop_on_that_session():
    # session 3 opens 9.4 (above the 9.2 stop), trades down to 8.91, closes 9.0
    closes, opens = [10, 10, 9.6, 9.0, 8.8, 8.8], [10, 10, 9.7, 9.4, 8.9, 8.8]
    out, cal = _run(closes, opens, 9.2, INTRADAY)
    assert out.exit_reason == "STOP" and out.exit_date == cal.sessions[3]
    assert out.exit_price_raw == pytest.approx(9.2) and out.gross_ret == pytest.approx(9.2 / 10 - 1)
    old, _ = _run(closes, opens, 9.2, ZERO_COST)              # close rule: next open, 8.9
    assert old.exit_date == cal.sessions[4] and old.gross_ret == pytest.approx(8.9 / 10 - 1)


def test_gap_through_the_stop_fills_at_that_open():
    closes, opens = [10, 10, 9.6, 8.4, 8.4], [10, 10, 9.7, 8.5, 8.4]
    out, cal = _run(closes, opens, 9.2, INTRADAY)
    assert out.exit_reason == "STOP" and out.exit_date == cal.sessions[3]
    assert out.gross_ret == pytest.approx(8.5 / 10 - 1)       # the gap is paid, not the stop price


def test_entry_session_counts_only_when_the_entry_is_above_the_stop():
    # entry at 10, the entry session's low is 9.9: a 9.95 stop resting after the fill is hit that day
    out, cal = _run([10, 10, 10.5, 10.5], [10, 10, 10.4, 10.5], 9.95, INTRADAY)
    assert out.exit_date == cal.sessions[1] and out.exit_price_raw == pytest.approx(9.95)
    # entry gapped BELOW the stop (9.0 < 9.5): no broker stop that session, the close (9.6) is fine;
    # from the next session the stop rests and the 9.41 low (9.5 x 0.99) touches it
    out, cal = _run([10, 9.6, 9.5, 9.5], [10, 9.0, 9.6, 9.5], 9.5, INTRADAY)
    assert out.exit_date == cal.sessions[2] and out.exit_price_raw == pytest.approx(9.5)


def test_time_exits_are_unchanged_by_the_intraday_model():
    closes, opens = [10, 10, 11, 12, 13, 14, 15], [10, 10, 10.5, 11.5, 12.5, 13.5, 14.5]
    a, _ = _run(closes, opens, 5.0, INTRADAY, hold=3)
    b, _ = _run(closes, opens, 5.0, ZERO_COST, hold=3)
    assert (a.exit_reason, a.exit_date, a.gross_ret) == (b.exit_reason, b.exit_date, b.gross_ret) and a.exit_reason == "TIME"


def test_stop_model_is_validated():
    with pytest.raises(ValueError, match="stop_model"):
        replace(ZERO_COST, stop_model="sometimes")
    assert CostModel(((0.0, 0.0),), 0.0).stop_model == "close"       # direct constructions keep the old rule
