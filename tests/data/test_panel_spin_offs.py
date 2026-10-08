"""build_panel: a recorded SPIN-OFF's effective session is a NEUTRAL step for the parent (a raw drop
is not booked as a loss), the backtester's share ledger follows the same path as simulate_plan, and
the rule is point-in-time."""
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
from tests.backtest.test_engine import FixedPlan, _one_signal

SPIN = 4                                   # session index of the spin-off ex-date
DATES = pd.bdate_range("2020-01-01", periods=14)
DROP = [10, 10, 10, 10, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.3, 9.3, 9.3, 9.3]      # raw -12% on SPIN
RISE = [10, 10, 10, 10, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3]


def _bars(closes: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for sym, cl in closes.items():
        for d, c in zip(DATES, cl):
            rows.append({"symbol": sym, "date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1e6, "vwap": c, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    return pd.DataFrame(rows)


def _spin(symbol: str = "AAA", at: int = SPIN) -> pd.DataFrame:
    d = DATES[at]
    return pd.DataFrame([{"symbol": symbol, "ex_date": d, "action_type": "spin_off", "ratio": 0.2, "amount": np.nan,
                          "declared_date": pd.NaT,
                          "available_at": (d + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York")
                          .tz_convert("UTC"), "pit_status": "PIT_CONSERVATIVE", "source_id": f"so-{symbol}-{at}",
                          "provider": "test", "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")}])


def _panel(closes, actions=None):
    return build_panel(_bars(closes), actions, calendar=TradingCalendar.from_dates(DATES))


def test_spin_off_drop_is_a_neutral_step():
    p = _panel({"AAA": DROP, "SPY": [100.0] * 14}, _spin())
    a = p.aclose["AAA"]
    assert p.ret["AAA"].iloc[SPIN] == 0.0
    assert a.iloc[SPIN] == pytest.approx(a.iloc[SPIN - 1], rel=1e-12)          # continuous total return
    assert p.ret["AAA"].iloc[SPIN + 1] == pytest.approx(8.9 / 8.8 - 1)         # later moves unchanged
    # the tri-scaled bar of the ex-date is scaled consistently with its close
    assert p.aopen["AAA"].iloc[SPIN] / a.iloc[SPIN] == pytest.approx(p.open["AAA"].iloc[SPIN] / 8.8)
    assert p.ahigh["AAA"].iloc[SPIN] / a.iloc[SPIN] == pytest.approx(1.01)
    assert bool(p.spin_off["AAA"].iloc[SPIN]) and int(p.spin_off.to_numpy().sum()) == 1
    # raw prices, dividend and split_ratio are untouched (level logic and the paper ledger read those)
    assert p.close["AAA"].iloc[SPIN] == 8.8
    assert p.dividend["AAA"].iloc[SPIN] == 0.0 and p.split_ratio["AAA"].iloc[SPIN] == 1.0


def test_same_drop_without_a_record_stays_a_loss():
    p = _panel({"AAA": DROP, "SPY": [100.0] * 14})
    assert p.ret["AAA"].iloc[SPIN] == pytest.approx(-0.12)
    assert not p.spin_off.to_numpy().any()


def test_positive_move_on_spin_off_date_is_unchanged():
    p = _panel({"AAA": RISE, "SPY": [100.0] * 14}, _spin())
    assert p.ret["AAA"].iloc[SPIN] == pytest.approx(0.03)
    assert not p.spin_off.to_numpy().any()


def test_spin_off_record_only_touches_its_own_symbol():
    p = _panel({"AAA": DROP, "BBB": DROP, "SPY": [100.0] * 14}, _spin("AAA"))
    assert p.ret["AAA"].iloc[SPIN] == 0.0
    assert p.ret["BBB"].iloc[SPIN] == pytest.approx(-0.12)


def test_build_panel_with_spin_off_is_truncation_invariant():
    closes = {"AAA": DROP, "BBB": RISE, "SPY": [100.0] * 14}
    bars = _bars(closes)
    acts = pd.concat([_spin("AAA"), _spin("BBB"), _spin("AAA", at=9)], ignore_index=True)
    cal = TradingCalendar.from_dates(DATES)
    bundle = DataBundle(build_panel(bars, acts, calendar=cal), cal, actions=acts,
                        benchmarks={"market": "SPY", "sectors": {}})

    def rebuilt(b: DataBundle) -> pd.DataFrame:
        p = build_panel(bars[bars["date"] <= b.panel.dates[-1]], b.actions,
                        calendar=TradingCalendar.from_dates(b.panel.dates))
        return pd.concat({"ret": p.ret, "spin_off": p.spin_off.astype("float64"),
                          "aclose_ratio": p.aclose / p.aclose.shift(1)}, axis=1)

    assert_truncation_invariant(rebuilt, bundle, check_dates=list(DATES[2:12]), name="build_panel spin-off step")


def _run(config, record: bool, hold: int):
    cfg = config.with_overrides({"backtest": {"sizing": "equal_weight", "max_position_weight": 0.5,
                                              "max_positions": 2, "fractional": True}})
    p = _panel({"AAA": DROP, "SPY": [100.0] * 14}, _spin() if record else None)
    b = DataBundle(p, TradingCalendar.from_dates(DATES), benchmarks={"market": "SPY", "sectors": {}})
    strat = FixedPlan(TradePlan(stop_price=9.5, holding_sessions=hold))      # the raw 8.8 close is below it
    res = BacktestEngine(cfg, b).run({"fixed": _one_signal(b, "AAA", 1)}, {"fixed": strat},
                                     start=DATES[0], end=DATES[-1])
    assert len(res.trades) == 1
    ref_plan = strat.plan(type("FS", (), {"panel": p})(), "AAA", DATES[1])
    costs = CostModel.from_config(cfg)
    out = simulate_plan(p, "AAA", DATES[1], ref_plan, costs, force_close_at_end=True)
    return res.trades.iloc[0], out, costs.one_way_cost_frac(10 * 1e6)       # adv = close x volume


@pytest.mark.parametrize("hold", [2, 6])
def test_without_a_record_the_raw_drop_is_booked_identically(hold, config):
    t, out, _ = _run(config, record=False, hold=hold)
    assert t["exit_reason"] == out.exit_reason == ("TIME" if hold == 2 else "STOP")   # fake stop-out
    assert pd.Timestamp(t["exit_date"]) == out.exit_date
    assert t["net_ret"] == pytest.approx(out.net_ret, abs=1e-9)


# hold 2: the time exit fills AT the ex-date open (the seller keeps the entitlement);
# hold 6: held through the ex-date. Entry at the open of session 2 (10.0); the credit is 10 - 8.8.
@pytest.mark.parametrize("hold,exit_idx", [(2, SPIN), (6, 8)])
def test_spin_off_is_neutral_in_backtester_and_simulate_plan(hold, exit_idx, config):
    t, out, ow = _run(config, record=True, hold=hold)
    px = DROP[exit_idx]                                   # opens == closes in this fixture
    assert t["exit_reason"] == out.exit_reason == "TIME"  # no fake stop: the holder lost nothing
    assert pd.Timestamp(t["exit_date"]) == out.exit_date == DATES[exit_idx]
    assert t["dividends"] == pytest.approx(t["qty"] * 1.2)
    # backtester: the distribution is credited as cash, like a cash dividend
    assert t["net_ret"] == pytest.approx((px * (1 - ow) + 1.2 - 10 * (1 + ow)) / 10, abs=1e-12)
    # simulate_plan: the tri path reinvests it (continuous total return through the ex-date)
    g = px / 8.8 - 1
    assert out.net_ret == pytest.approx(g - ow * (2 + g), abs=1e-12)
    # same second-order gap as for a cash dividend: no reinvestment, no sell cost on the credit
    assert abs(t["net_ret"] - out.net_ret) <= 0.12 * (abs(g) + 2 * ow) + 1e-12
    assert min(t["net_ret"], out.net_ret) > -0.01         # never the raw -12%
