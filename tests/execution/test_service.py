"""PaperExecutionService: SYSTEM_PAUSED blocks new orders (refusal recorded, no order row), the
risk gates (max_daily_orders / max_order_notional) refuse without creating an order, a service bound
to one book can never touch another book's ledger, and submit_entry -> sync -> submit_exit -> sync
round-trips a full paper trade."""
from __future__ import annotations

import pytest

from quantlab.config import load_config
from quantlab.core.costs import CostModel
from quantlab.core.types import Book, SystemState, TradePlan
from quantlab.db.database import open_db, utcnow_iso
from quantlab.execution.ledger import Ledger
from quantlab.execution.service import ExecutionError, PaperExecutionService
from quantlab.execution.sim_broker import SimBroker

from .panels import build

DATES = ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-06", "2024-05-07"]


def zero_cost_model() -> CostModel:
    return CostModel(half_spread_tiers=((0.0, 0.0),), slippage_bps=0.0, commission_per_share=0.0,
                     commission_min_per_order=0.0, delisting_return=-0.30)


def _bars(prices, symbol):
    return [{"symbol": symbol, "date": d, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 5_000_000}
            for d, px in prices.items()]


def make_service(db, book, *, config=None, starting_cash=100_000.0):
    broker = SimBroker(book, zero_cost_model(), starting_cash=starting_cash)
    ledger = Ledger(db, book, starting_cash=starting_cash)
    return PaperExecutionService(db, config, book, broker, ledger), broker, ledger


def pause_system(db, reason="test pause"):
    now = utcnow_iso()
    db.upsert("system_state", {"id": 1, "state": SystemState.PAUSED.value, "reason": reason, "changed_at": now,
                               "changed_by": "test"}, ["id"])


def test_system_paused_blocks_new_entry_orders():
    db = open_db(":memory:")
    svc, broker, ledger = make_service(db, Book.BOT)
    pause_system(db)
    result = svc.submit_entry("AAA", 10, plan=TradePlan(), session_date=DATES[0])
    assert result["refused"] is True
    assert "SYSTEM_PAUSED" in result["reason"]
    assert db.fetchall("SELECT * FROM orders") == []
    refusals = db.fetchall("SELECT * FROM execution_refusals")
    assert len(refusals) == 1 and refusals[0]["reason"] == "SYSTEM_PAUSED"


def test_system_paused_blocks_new_exit_orders():
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices, "AAA") + _bars({d: 400.0 for d in DATES}, "SPY"))
    db = open_db(":memory:")
    svc, broker, ledger = make_service(db, Book.BOT)
    entry = svc.submit_entry("AAA", 10, plan=TradePlan(), session_date=DATES[0])
    assert not entry["refused"]
    svc.sync(DATES[0], panel)
    svc.sync(DATES[1], panel)
    trade_id = ledger.open_trades()[0].trade_id

    pause_system(db)
    result = svc.submit_exit(trade_id, "MANUAL", session_date=DATES[1])
    assert result["refused"] is True
    assert "SYSTEM_PAUSED" in result["reason"]
    trade = ledger.journal.get_trade(trade_id)
    assert trade["status"] == "OPEN"  # untouched


def test_max_daily_orders_gate():
    db = open_db(":memory:")
    cfg = load_config(overrides={"risk": {"max_daily_orders": 1}})
    svc, broker, ledger = make_service(db, Book.BOT, config=cfg)
    first = svc.submit_entry("AAA", 1, plan=TradePlan(), session_date=DATES[0])
    assert not first["refused"]
    second = svc.submit_entry("BBB", 1, plan=TradePlan(), session_date=DATES[0])
    assert second["refused"] is True
    assert "max_daily_orders" in second["reason"]


def test_max_order_notional_gate():
    db = open_db(":memory:")
    cfg = load_config(overrides={"risk": {"max_order_notional": 500.0}})
    svc, broker, ledger = make_service(db, Book.BOT, config=cfg)
    plan = TradePlan(entry_ref_price=100.0)
    result = svc.submit_entry("AAA", 10, plan=plan, session_date=DATES[0])  # notional 1000 > 500
    assert result["refused"] is True
    assert "max_order_notional" in result["reason"]
    assert db.fetchall("SELECT * FROM orders") == []


def test_submit_entry_requires_a_determinable_session():
    db = open_db(":memory:")
    svc, broker, ledger = make_service(db, Book.BOT)
    with pytest.raises(ExecutionError):
        svc.submit_entry("AAA", 10)  # no candidate/human_decision/plan.signal_date/session_date/broker history


def test_ledger_book_mismatch_rejected_at_construction():
    db = open_db(":memory:")
    bot_broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    human_ledger = Ledger(db, Book.HUMAN, starting_cash=100_000.0)
    with pytest.raises(ValueError):
        PaperExecutionService(db, None, Book.BOT, bot_broker, human_ledger)


def test_bot_and_human_services_never_cross_books():
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices, "AAA") + _bars({d: 400.0 for d in DATES}, "SPY"))
    db = open_db(":memory:")
    bot_svc, bot_broker, bot_ledger = make_service(db, Book.BOT, starting_cash=100_000.0)
    human_svc, human_broker, human_ledger = make_service(db, Book.HUMAN, starting_cash=50_000.0)

    bot_svc.submit_entry("AAA", 10, plan=TradePlan(), session_date=DATES[0])
    bot_svc.sync(DATES[0], panel)
    bot_svc.sync(DATES[1], panel)

    assert human_ledger.positions() == []
    assert human_ledger.cash() == pytest.approx(50_000.0)
    assert len(bot_ledger.positions()) == 1
    bot_orders = db.fetchall("SELECT * FROM orders WHERE book='BOT'")
    human_orders = db.fetchall("SELECT * FROM orders WHERE book='HUMAN'")
    assert len(bot_orders) == 1
    assert len(human_orders) == 0


def test_full_entry_and_exit_round_trip_via_sync():
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices, "AAA") + _bars({d: 400.0 for d in DATES}, "SPY"))
    db = open_db(":memory:")
    svc, broker, ledger = make_service(db, Book.BOT)

    entry = svc.submit_entry("AAA", 10, plan=TradePlan(stop_price=90.0, holding_sessions=20),
                             session_date=DATES[0])
    assert not entry["refused"]
    r0 = svc.sync(DATES[0], panel)
    assert r0.filled == []
    r1 = svc.sync(DATES[1], panel)
    assert len(r1.filled) == 1
    trade_id = r1.filled[0]["trade_id"]
    assert ledger.journal.get_trade(trade_id)["status"] == "OPEN"

    exit_result = svc.submit_exit(trade_id, "MANUAL", session_date=DATES[1])
    assert not exit_result["refused"]
    r2 = svc.sync(DATES[2], panel)  # decision_session == DATES[1] -> eligible to fill at DATES[2]'s open
    assert len(r2.filled) == 1
    trade = ledger.journal.get_trade(trade_id)
    assert trade["status"] == "CLOSED"
    assert trade["exit_reason"] == "MANUAL"
