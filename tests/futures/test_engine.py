"""Synthetic proofs for the futures backtester (brief section 20): timing, no look-ahead, stops, targets,
conservative same-bar resolution, costs, slippage, sessions/DST, sizing, daily loss limit, drawdown,
determinism. Every expected number is worked out by hand in the comments."""
from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from quantlab.futures.engine import ES, MES, CostModel, Order, run
from quantlab.futures.sessions import label_bars

TZ = "America/Chicago"


def session(day: str, rows, start: str = "08:30") -> pd.DataFrame:
    """1-minute bars from ``start`` CT on ``day``; rows = (o, h, l, c)."""
    t0 = pd.Timestamp(f"{day} {start}", tz=TZ)
    idx = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=k) for k in range(len(rows))]).tz_convert("UTC")
    a = np.array(rows, float)
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
                         "volume": 100.0}, index=idx)


FLAT = (100.0, 100.0, 100.0, 100.0)


class Script:
    """Returns pre-set orders after given bar indices; records what it was shown."""

    def __init__(self, plan: dict[int, list[Order]]):
        self.plan, self.seen = plan, []

    def on_bar(self, ctx):
        self.seen.append((ctx.i, len(ctx.close), float(ctx.close[-1])))
        return self.plan.get(ctx.i)


NOCOST = CostModel(0, 0, 0)


# ---------------------------------------------------------------- sessions / DST
def test_globex_evening_bar_belongs_to_next_trading_date():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-10-04 17:00", tz=TZ)]).tz_convert("UTC")   # Sunday 17:00 CT
    lab = label_bars(idx)
    assert lab["session_date"].iloc[0] == date(2026, 10, 5) and not lab["is_rth"].iloc[0]


def test_rth_open_follows_dst_not_a_fixed_utc_offset():
    # US DST starts 2026-03-08: 08:30 CT is 14:30 UTC on Mar 6 but 13:30 UTC on Mar 9
    idx = pd.DatetimeIndex(["2026-03-06 14:30", "2026-03-09 13:30", "2026-03-09 14:30"], tz="UTC")
    lab = label_bars(idx)
    assert list(lab["minute_of_rth"]) == [0, 0, 60]


# ---------------------------------------------------------------- timing / look-ahead
def test_market_entry_fills_at_next_bar_open_with_slippage():
    bars = session("2026-10-05", [FLAT, (101, 102, 100, 101.5), (101.5, 102, 101, 102)])
    s = Script({0: [Order("market", +1)]})
    r = run(bars, s, ES, CostModel(0, 0, 1))
    t = r.trades[0]
    assert t.entry_time == bars.index[1]                 # decided after bar 0 closed -> bar 1 open
    assert t.entry_price == 101.25                       # open 101 + 1 tick
    assert t.exit_reason == "session_end" and t.exit_price == 101.75   # last close 102 - 1 tick


def test_strategy_only_ever_sees_closed_bars():
    bars = session("2026-10-05", [(100 + k, 100 + k, 100 + k, 100 + k) for k in range(10)])
    s = Script({})
    run(bars, s, ES, NOCOST)
    assert all(n == i + 1 and c == 100 + i for i, n, c in s.seen)


def test_truncation_invariance_of_decisions():
    """Removing future bars must not change any decision made before them."""
    rng = np.random.default_rng(3)
    p = 100 + np.cumsum(rng.normal(0, 0.5, 120))
    rows = [(x, x + 0.5, x - 0.5, x + rng.normal(0, 0.2)) for x in p]
    full = session("2026-10-05", rows)

    class Cross:
        def __init__(self):
            self.decisions = []

        def on_bar(self, ctx):
            if ctx.i < 20 or ctx.position:
                return None
            fast, slow = ctx.close[-5:].mean(), ctx.close[-20:].mean()
            self.decisions.append((ctx.i, fast > slow))
            return [Order("market", +1 if fast > slow else -1, stop_loss=None)]

    a, b = Cross(), Cross()
    run(full, a, ES, NOCOST)
    run(full.iloc[:60], b, ES, NOCOST)
    assert b.decisions == [d for d in a.decisions if d[0] < 60]


# ---------------------------------------------------------------- stops / targets
def test_stop_loss_fills_at_stop_plus_slippage():
    bars = session("2026-10-05", [FLAT, FLAT, (100, 100.5, 98.5, 99)])
    s = Script({0: [Order("market", +1, stop_loss=99.0)]})
    t = run(bars, s, ES, CostModel(0, 0, 1)).trades[0]
    assert t.entry_price == 100.25 and t.exit_price == 98.75 and t.exit_reason == "stop"


def test_gap_through_stop_fills_at_the_open():
    bars = session("2026-10-05", [FLAT, FLAT, (97, 97.5, 96.5, 97)])
    t = run(bars, Script({0: [Order("market", +1, stop_loss=99.0)]}), ES, NOCOST).trades[0]
    assert t.exit_price == 97.0                          # opened below the stop: no fill at 99


def test_target_needs_a_trade_through_not_a_touch():
    touch = session("2026-10-05", [FLAT, FLAT, (100, 102.0, 100, 101), FLAT])
    t = run(touch, Script({0: [Order("market", +1, take_profit=102.0)]}), ES, NOCOST).trades[0]
    assert t.exit_reason == "session_end"                # high == target: not filled
    through = session("2026-10-05", [FLAT, FLAT, (100, 102.25, 100, 101), FLAT])
    t = run(through, Script({0: [Order("market", +1, take_profit=102.0)]}), ES, NOCOST).trades[0]
    assert t.exit_reason == "target" and t.exit_price == 102.0


def test_same_bar_stop_and_target_assumes_the_stop():
    bars = session("2026-10-05", [FLAT, FLAT, (100, 103, 98, 101)])
    t = run(bars, Script({0: [Order("market", +1, stop_loss=99, take_profit=102)]}), ES, NOCOST).trades[0]
    assert t.exit_reason == "stop" and t.exit_price == 99.0


def test_entry_bar_can_stop_but_never_target():
    bars = session("2026-10-05", [FLAT, (100, 103, 98, 101), FLAT])
    t = run(bars, Script({0: [Order("market", +1, stop_loss=99, take_profit=102)]}), ES, NOCOST).trades[0]
    assert t.exit_reason == "stop"
    bars = session("2026-10-05", [FLAT, (100, 103, 99.5, 101), FLAT])
    t = run(bars, Script({0: [Order("market", +1, stop_loss=99, take_profit=102)]}), ES, NOCOST).trades[0]
    assert t.exit_reason == "session_end"                # target on the entry bar is not credited


def test_limit_entry_needs_trade_through():
    bars = session("2026-10-05", [FLAT, (100, 100, 99.0, 99.5), (99.5, 99.5, 98.75, 99), FLAT])
    t = run(bars, Script({0: [Order("limit", +1, price=99.0)]}), ES, NOCOST).trades[0]
    assert t.entry_time == bars.index[2] and t.entry_price == 99.0   # bar 1 only touched 99.0


# ---------------------------------------------------------------- costs / slippage / sizing
def test_commission_fees_and_slippage_are_charged_both_sides():
    bars = session("2026-10-05", [FLAT, FLAT, (101, 101, 101, 101)])
    plan = {0: [Order("market", +1)]}
    gross = run(bars, Script(plan), ES, NOCOST).trades[0].pnl                         # (101-100)*50 = 50
    assert gross == 50.0
    net = run(bars, Script(plan), ES, CostModel(2.25, 1.40, 0)).trades[0].pnl
    assert net == pytest.approx(50 - 2 * 3.65)
    for k in (1, 2, 3):                                                                 # slippage sensitivity
        p = run(bars, Script(plan), ES, CostModel(0, 0, k)).trades[0].pnl
        assert p == pytest.approx(50 - 2 * k * ES.tick_value)


def test_position_size_scales_pnl():
    bars = session("2026-10-05", [FLAT, FLAT, (101, 101, 101, 101)])
    p1 = run(bars, Script({0: [Order("market", +1, qty=1)]}), MES, NOCOST).trades[0].pnl
    p3 = run(bars, Script({0: [Order("market", +1, qty=3)]}), MES, NOCOST).trades[0].pnl
    assert p1 == 5.0 and p3 == 15.0


def test_short_side_pnl_sign():
    bars = session("2026-10-05", [FLAT, FLAT, (99, 99, 99, 99)])
    assert run(bars, Script({0: [Order("market", -1)]}), ES, NOCOST).trades[0].pnl == 50.0


# ---------------------------------------------------------------- sessions / risk
def test_session_boundary_flattens_and_drops_pending_orders():
    d1 = session("2026-10-05", [FLAT, FLAT, (101, 101, 101, 101)])
    d2 = session("2026-10-06", [(105, 105, 105, 105), FLAT])
    bars = pd.concat([d1, d2])
    s = Script({0: [Order("market", +1)], 2: [Order("market", +1)]})   # order after the LAST bar of day 1
    r = run(bars, s, ES, NOCOST)
    assert len(r.trades) == 1 and r.trades[0].exit_price == 101.0        # no overnight hold, no carried order
    assert list(r.daily["pnl"]) == [50.0, 0.0]


def test_flatten_time():
    rows = [FLAT] * 5 + [(101, 101, 101, 101)] * 5
    bars = session("2026-10-05", rows, start="14:50")
    t = run(bars, Script({0: [Order("market", +1)]}), ES, NOCOST, flatten_at=time(14, 55)).trades[0]
    assert t.exit_reason == "flatten_time" and t.exit_price == 101.0     # 14:55 bar open


def test_daily_loss_limit_flattens_at_the_limit_and_halts():
    bars = session("2026-10-05", [FLAT, FLAT, (100, 100, 96, 97), FLAT, (100, 100, 100, 100)])
    s = Script({0: [Order("market", +1)], 3: [Order("market", +1)]})
    r = run(bars, s, ES, NOCOST, daily_loss_limit=150)
    assert len(r.trades) == 1 and r.trades[0].exit_reason == "daily_loss_limit"
    assert r.trades[0].exit_price == 97.0 and r.trades[0].pnl == -150.0  # 100 - 150/50
    assert r.daily["pnl"].iloc[0] == -150.0                              # halted: the bar-3 order never fires


def test_daily_low_and_high_track_open_pnl_extremes():
    bars = session("2026-10-05", [FLAT, FLAT, (100, 102, 99, 101), (101, 101, 101, 101)])
    r = run(bars, Script({0: [Order("market", +1)]}), ES, NOCOST)
    d = r.daily.iloc[0]
    assert (d["pnl"], d["low"], d["high"]) == (50.0, -50.0, 100.0)


def test_runs_are_deterministic():
    rng = np.random.default_rng(9)
    p = 100 + np.cumsum(rng.normal(0, 0.5, 300))
    bars = session("2026-10-05", [(x, x + 0.5, x - 0.5, x) for x in p])

    class Alt:
        def on_bar(self, ctx):
            if ctx.position == 0 and ctx.i % 7 == 0:
                return [Order("market", 1 if ctx.i % 2 else -1, stop_loss=ctx.close[-1] - 2 if ctx.i % 2
                              else ctx.close[-1] + 2)]
            return None
    a = run(bars, Alt(), ES, CostModel(2, 1, 1))
    b = run(bars, Alt(), ES, CostModel(2, 1, 1))
    assert [(t.entry_time, t.entry_price, t.exit_price, t.pnl) for t in a.trades] == \
           [(t.entry_time, t.entry_price, t.exit_price, t.pnl) for t in b.trades]
    assert a.daily.equals(b.daily)
