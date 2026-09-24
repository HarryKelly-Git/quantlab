"""CRITICAL EQUIVALENCE: ExitEngine + SimBroker + Ledger, driven session-by-session through
PaperExecutionService exactly as the daily pipeline would, must reproduce core.tradesim.simulate_plan's
net return for the SAME TradePlan over the SAME synthetic path -- for the STOP, TIME and DELISTED
exit cases. This is what makes the bot/human paper books and the backtester comparable
(ARCHITECTURE.md sections 6 and 8).

The two net-return formulas are not bit-identical: tradesim subtracts a flat round-trip cost
fraction from the raw price-ratio return, while the live path bakes the same one-way cost into each
fill's actual price (a real order-execution model). They agree to within O(cost^2), i.e. far inside
the tolerance used below (costs here are ~10bps).
"""
from __future__ import annotations

import pytest

from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import Book, Direction, TradePlan
from quantlab.db.database import open_db
from quantlab.execution.exits import ExitEngine
from quantlab.execution.ledger import Ledger
from quantlab.execution.service import PaperExecutionService
from quantlab.execution.sim_broker import SimBroker

from .panels import build

N_SESSIONS = 16


def _business_dates(n: int, start_year=2024, start_month=2, start_day=1) -> list[str]:
    import pandas as pd
    return [d.date().isoformat() for d in pd.bdate_range(f"{start_year}-{start_month:02d}-{start_day:02d}", periods=n)]


DATES = _business_dates(N_SESSIONS)


def cost_model(one_way_bps: float = 10.0) -> CostModel:
    return CostModel(half_spread_tiers=((0.0, one_way_bps),), slippage_bps=0.0, commission_per_share=0.0,
                     commission_min_per_order=0.0, delisting_return=-0.30)


def _bars(prices: dict[str, float | None], symbol: str) -> list[dict]:
    return [{"symbol": symbol, "date": d, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 5_000_000}
            for d, px in prices.items() if px is not None]


def _run_live(prices_aaa: dict[str, float], panel, plan: TradePlan, *, qty: float = 5.0,
             delisting_missing_sessions: int = 3, cm: CostModel | None = None) -> dict:
    """Submit the entry at DATES[0], then walk every session, letting the exit engine decide when to
    submit the exit -- exactly like the daily pipeline. Returns the closed trade row."""
    cm = cm or cost_model()
    db = open_db(":memory:")
    broker = SimBroker(Book.BOT, cm, starting_cash=1_000_000.0, delisting_missing_sessions=delisting_missing_sessions,
                       max_pending_sessions=50)
    ledger = Ledger(db, Book.BOT, starting_cash=1_000_000.0)
    svc = PaperExecutionService(db, None, Book.BOT, broker, ledger)
    exit_engine = ExitEngine(delisting_missing_sessions=delisting_missing_sessions, default_holding_sessions=20,
                             delisting_return=cm.delisting_return)

    signal_date = DATES[0]
    entry = svc.submit_entry("AAA", qty, plan=plan, session_date=signal_date)
    assert not entry.get("refused"), entry

    trade_id = None
    for s in panel.dates:
        s_str = s.date().isoformat()
        if s_str < signal_date:
            continue
        svc.sync(s_str, panel)
        if trade_id is None:
            open_trades = ledger.open_trades()
            if open_trades:
                trade_id = open_trades[0].trade_id
        if trade_id is not None:
            trade_row = ledger.journal.get_trade(trade_id)
            if trade_row["status"] == "CLOSED":
                return trade_row
            open_trades = ledger.open_trades()
            if open_trades:
                for sig in exit_engine.evaluate(open_trades, s_str, panel):
                    if sig.trade_id == trade_id:
                        svc.submit_exit(trade_id, sig.reason.value, detail=sig.detail, session_date=s_str)
    assert trade_id is not None, "entry never filled within the synthetic panel"
    trade_row = ledger.journal.get_trade(trade_id)
    assert trade_row["status"] == "CLOSED", f"trade never closed within the panel window: {trade_row}"
    return trade_row


def _spy_bars() -> list[dict]:
    return _bars({d: 400.0 for d in DATES}, "SPY")


@pytest.mark.parametrize("tolerance", [1e-3])
def test_equivalence_stop_case(tolerance):
    prices = {d: 100.0 - 2.0 * i for i, d in enumerate(DATES)}
    panel = build(_bars(prices, "AAA") + _spy_bars())
    plan = TradePlan(entry="next_open", stop_price=90.0, target_price=None, holding_sessions=20)
    cm = cost_model()

    ref = simulate_plan(panel, "AAA", DATES[0], plan, cm, direction=Direction.LONG, benchmark="SPY")
    assert ref.status == "complete"
    assert ref.exit_reason == "STOP"

    trade = _run_live(prices, panel, plan, cm=cm)
    assert trade["exit_reason"] == "STOP"
    assert trade["entry_date"] == ref.entry_date.date().isoformat()
    assert trade["exit_date"] == ref.exit_date.date().isoformat()
    # entry_price is the cost-ADJUSTED fill (ref * (1 + one_way)); ref.entry_price_raw is the bare
    # reference open. They legitimately differ by ~one_way (10bps here) -- not bit-equal by design.
    assert trade["entry_price"] == pytest.approx(ref.entry_price_raw, rel=5e-3)
    assert trade["ret"] == pytest.approx(ref.net_ret, abs=tolerance)


@pytest.mark.parametrize("tolerance", [1e-3])
def test_equivalence_time_case(tolerance):
    prices = {d: 100.0 for _, d in enumerate(DATES)}  # flat: no stop/target can ever trigger
    panel = build(_bars(prices, "AAA") + _spy_bars())
    plan = TradePlan(entry="next_open", stop_price=None, target_price=None, holding_sessions=3)
    cm = cost_model()

    ref = simulate_plan(panel, "AAA", DATES[0], plan, cm, direction=Direction.LONG, benchmark="SPY")
    assert ref.status == "complete"
    assert ref.exit_reason == "TIME"

    trade = _run_live(prices, panel, plan, cm=cm)
    assert trade["exit_reason"] == "TIME"
    assert trade["entry_date"] == ref.entry_date.date().isoformat()
    assert trade["exit_date"] == ref.exit_date.date().isoformat()
    assert trade["ret"] == pytest.approx(ref.net_ret, abs=tolerance)


@pytest.mark.parametrize("tolerance", [1e-3])
def test_equivalence_delisted_case(tolerance):
    n_before = 6  # AAA trades for the first 6 sessions, then vanishes while SPY keeps trading
    prices = {DATES[i]: 100.0 + i for i in range(n_before)}
    panel = build(_bars(prices, "AAA") + _spy_bars())
    plan = TradePlan(entry="next_open", stop_price=None, target_price=None, holding_sessions=20)
    cm = cost_model()

    ref = simulate_plan(panel, "AAA", DATES[0], plan, cm, direction=Direction.LONG, benchmark="SPY")
    assert ref.status == "delisted"
    assert ref.exit_reason == "DELISTED"

    trade = _run_live(prices, panel, plan, cm=cm, delisting_missing_sessions=3)
    assert trade["exit_reason"] == "DELISTED"
    assert trade["entry_date"] == ref.entry_date.date().isoformat()
    # tradesim settles AT the last-traded session (it has hindsight) using the BARE settlement price;
    # the live system can only fire a few sessions later (delisting_missing_sessions) and its exit
    # fill is cost-ADJUSTED on top of the same settlement basis -- same basis, ~one_way apart.
    assert trade["exit_price"] == pytest.approx(ref.exit_price_raw, rel=5e-3)
    assert trade["ret"] == pytest.approx(ref.net_ret, abs=tolerance)
