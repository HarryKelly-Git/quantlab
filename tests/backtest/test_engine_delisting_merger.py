"""Delisting with a known MERGER record: exit at last close x (1 + costs.delisting_return_merger)
instead of the flat costs.delisting_return haircut. Backtester and simulate_plan agree, and only a
record effective on/before the delisting session counts (point-in-time)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import BacktestEngine
from quantlab.core.calendar import TradingCalendar
from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import TradePlan
from quantlab.data.panel import DataBundle, build_panel
from quantlab.testing.pit import assert_truncation_invariant

from .test_engine import FixedPlan, _bundle, _one_signal, bt_config  # noqa: F401  (fixture re-export)

N = 20
LAST_BAR = 8                       # AAA's last bar; the delisting rule fires 5 sessions later
CLOSES = [10, 10, 10.2, 10.4, 10.6, 10.8, 11, 11, 11] + [None] * (N - LAST_BAR - 1)
DELIST = LAST_BAR + 5              # execution.delisting_missing_sessions default (not in default.yaml)


def _action(date, action_type="cash_merger", symbol="AAA"):
    return {"symbol": symbol, "ex_date": date, "action_type": action_type, "ratio": np.nan, "amount": 25.0,
            "declared_date": pd.NaT, "available_at": pd.Timestamp(date).tz_localize("America/New_York")
            .tz_convert("UTC") + pd.Timedelta(hours=9, minutes=30), "pit_status": "PIT_CONSERVATIVE",
            "source_id": f"m-{action_type}-{pd.Timestamp(date).date()}", "provider": "test",
            "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")}


# case -> (session index of the merger record or None, is it used?)
CASES = {
    "merger_after_last_bar": (LAST_BAR + 1, True),
    "merger_on_last_bar": (LAST_BAR, True),
    "merger_within_lookback": (LAST_BAR - 3, True),
    "no_record": (None, False),
    "merger_after_delisting_session": (DELIST + 2, False),     # PIT: not yet effective when booked
    "merger_before_lookback": (LAST_BAR - 6, False),            # stock kept trading: not this exit
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_delisting_uses_merger_return_only_with_a_known_record(case, bt_config):  # noqa: F811
    cfg = bt_config.with_overrides({"costs": {"delisting_return_merger": -0.02}})
    costs = CostModel.from_config(cfg)
    assert costs.delisting_return == pytest.approx(-0.30) and costs.delisting_return_merger == pytest.approx(-0.02)
    at, used = CASES[case]
    closes = {"AAA": CLOSES, "SPY": [100.0] * N}
    dates = _bundle(closes).panel.dates
    actions = pd.DataFrame([_action(dates[at])]) if at is not None else None
    b = _bundle(closes, actions=actions)
    strat = FixedPlan(TradePlan(stop_price=None, holding_sessions=30))
    res = BacktestEngine(cfg, b).run({"fixed": _one_signal(b, "AAA", 1)}, {"fixed": strat},
                                     start=dates[0], end=dates[-1])
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    dr = costs.delisting_return_merger if used else costs.delisting_return
    assert t["exit_reason"] == "DELISTED"
    assert pd.Timestamp(t["exit_date"]) == dates[DELIST]
    assert t["exit_ref_price"] == pytest.approx(11 * (1 + dr), abs=1e-9)
    assert res.diagnostics["delistings"] == 1 and res.diagnostics["delistings_merger"] == int(used)
    # one definition everywhere: simulate_plan (shadow outcomes) books the same exit
    ref_plan = strat.plan(type("FS", (), {"panel": b.panel})(), "AAA", dates[1])
    out = simulate_plan(b.panel, "AAA", dates[1], ref_plan, costs, force_close_at_end=True)
    assert out.status == "delisted" and out.exit_reason == "DELISTED"
    assert out.exit_price_raw == pytest.approx(11 * (1 + dr), abs=1e-9)
    assert t["net_ret"] == pytest.approx(out.net_ret, abs=1e-9)


def test_default_merger_return_exits_at_the_last_close(bt_config):  # noqa: F811
    costs = CostModel.from_config(bt_config)
    assert costs.delisting_return_merger == 0.0       # config/default.yaml
    closes = {"AAA": CLOSES, "SPY": [100.0] * N}
    dates = _bundle(closes).panel.dates
    b = _bundle(closes, actions=pd.DataFrame([_action(dates[LAST_BAR + 1], "stock_and_cash_merger")]))
    res = BacktestEngine(bt_config, b).run({"fixed": _one_signal(b, "AAA", 1)},
                                           {"fixed": FixedPlan(TradePlan(stop_price=None, holding_sessions=30))},
                                           start=dates[0], end=dates[-1])
    assert res.trades.iloc[0]["exit_ref_price"] == pytest.approx(11.0, abs=1e-9)


def test_merger_placed_on_calendar_session_and_not_for_other_types():
    """The acquiree never trades again: the record lands on the first CALENDAR session >= its date
    (a weekend date moves to Monday); spin-offs/dividends never set the merger flag; a record dated
    after the panel's last session is not placed."""
    closes = {"AAA": CLOSES, "BBB": [20.0] * N, "SPY": [100.0] * N}
    dates = _bundle(closes).panel.dates
    assert dates[LAST_BAR] == pd.Timestamp("2020-01-13")          # Monday; AAA's last bar
    sat = pd.Timestamp("2020-01-18")                              # no session: next one is Mon 01-20
    acts = pd.DataFrame([_action(sat), _action(dates[3], "spin_off", "BBB"),
                         _action(dates[-1] + pd.Timedelta(days=3), "stock_merger", "BBB")])
    p = _bundle(closes, actions=acts).panel
    m = p.merger
    assert m.dtypes.eq(bool).all()
    assert list(m.index[m["AAA"]]) == [pd.Timestamp("2020-01-20")]
    assert not m["BBB"].any() and not m["SPY"].any()


def test_merger_flag_is_truncation_invariant():
    closes = {"AAA": CLOSES, "SPY": [100.0] * N}
    dates = _bundle(closes).panel.dates
    acts = pd.DataFrame([_action(dates[LAST_BAR + 1]), _action(dates[DELIST + 2], "stock_merger")])
    rows = []
    for sym, cl in closes.items():
        for d, c in zip(dates, cl):
            if c is not None:
                rows.append({"symbol": sym, "date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1e6,
                             "vwap": c, "trade_count": np.nan, "provider": "test",
                             "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    bars = pd.DataFrame(rows)
    cal = TradingCalendar.from_dates(dates)
    bundle = DataBundle(build_panel(bars, acts, calendar=cal), cal, actions=acts,
                        benchmarks={"market": "SPY", "sectors": {}})

    def rebuilt_merger(b: DataBundle) -> pd.DataFrame:
        end = b.panel.dates[-1]
        p = build_panel(bars[bars["date"] <= end], b.actions, calendar=TradingCalendar.from_dates(b.panel.dates))
        return p.merger.astype("float64")

    assert_truncation_invariant(rebuilt_merger, bundle, check_dates=list(dates[LAST_BAR - 1:]), name="panel.merger")
    # and the field truncates with the panel (what shadow outcomes see after panel.truncate)
    assert not bundle.truncate(dates[LAST_BAR]).panel.merger["AAA"].any()
    assert bundle.truncate(dates[LAST_BAR + 1]).panel.merger["AAA"].iloc[-1]
