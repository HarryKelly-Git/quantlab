"""Ledger accounting identities: equity = cash + marked positions, cash is the append-only journal
sum, split/dividend handling (incl. the uncredited-dividend audit trail), and idempotent corporate
action application. All data is SYNTHETIC (hand-built panels, no network)."""
from __future__ import annotations

import pytest

from quantlab.core.costs import CostModel
from quantlab.core.types import Book, Side, TradePlan
from quantlab.db.database import open_db
from quantlab.execution.broker import OrderRequest
from quantlab.execution.ledger import Ledger, LedgerError
from quantlab.execution.sim_broker import SimBroker

from .panels import build

DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]


def _bars(prices: dict[str, float], symbol: str = "AAA") -> list[dict]:
    return [{"symbol": symbol, "date": d, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 5_000_000}
            for d, px in prices.items()]


def zero_cost_model() -> CostModel:
    return CostModel(half_spread_tiers=((0.0, 0.0),), slippage_bps=0.0, commission_per_share=0.0,
                     commission_min_per_order=0.0, delisting_return=-0.30)


@pytest.fixture
def db(tmp_path):
    d = open_db(tmp_path / "ledger_test.db")
    yield d
    d.close()


_ORDER_SEQ = iter(range(1_000_000))


def _open_long_trade(ledger: Ledger, broker: SimBroker, panel, symbol="AAA", qty=10, signal_date=DATES[0]) -> dict:
    """Submits + fills one long entry via the real order/fill path, returns the ledger FillResult-ish dict."""
    n = next(_ORDER_SEQ)
    order_id = f"order_test_entry_{ledger.book}_{n}"
    client_order_id = f"c-entry-{ledger.book}-{n}"
    ledger.db.insert("order_intents", {"order_id": order_id, "book": ledger.book, "session_date": signal_date,
                                       "purpose": "entry", "intent_json": "{}", "created_at": "2024-01-01T00:00:00Z"})
    ledger.db.insert("orders", {
        "order_id": order_id, "client_order_id": client_order_id, "book": ledger.book, "broker": "sim",
        "candidate_id": None, "decision_id": None, "human_decision_id": None, "trade_id": None,
        "purpose": "entry", "symbol": symbol, "side": Side.BUY.value, "qty": qty, "order_type": "market",
        "time_in_force": "opg", "limit_price": None, "created_at": "2024-01-01T00:00:00Z",
        "submitted_at": "2024-01-01T00:00:00Z", "status": "accepted", "broker_order_id": "b1",
        "filled_qty": 0, "filled_avg_price": None, "last_update_at": "2024-01-01T00:00:00Z", "raw_json": "{}",
    })
    req = OrderRequest(client_order_id=f"c-broker-entry-{ledger.book}-{n}", symbol=symbol, side=Side.BUY, qty=qty,
                       time_in_force="opg", decision_session=signal_date)
    broker.submit_order(req)
    broker.process_session(DATES[0], panel)
    bo = broker.process_session(DATES[1], panel)[0]
    fr = ledger.apply_fill(order_id=order_id, qty=bo.filled_qty, price=bo.filled_avg_price,
                           session_date=DATES[1], commission=bo.commission, modeled_cost=bo.modeled_cost,
                           filled_at=bo.filled_at, signal_date=signal_date,
                           plan=TradePlan(stop_price=90.0, target_price=None, holding_sessions=20))
    return fr


def test_cash_is_sum_of_append_only_journal(db):
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    assert ledger.cash() == pytest.approx(100_000.0)
    # re-constructing a Ledger for the same book must NOT reseed the starting cash.
    ledger2 = Ledger(db, Book.BOT, starting_cash=999.0)
    assert ledger2.cash() == pytest.approx(100_000.0)


def test_apply_fill_updates_order_position_cash_and_opens_trade(db):
    panel = build(_bars({d: 100.0 for d in DATES}) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)

    fr = _open_long_trade(ledger, broker, panel)
    assert fr.trade_id is not None
    assert ledger.cash() == pytest.approx(100_000.0 - 1_000.0)  # 10 shares @ 100, zero cost
    pos = ledger.get_position("AAA")
    assert pos["qty"] == pytest.approx(10)
    assert pos["avg_cost"] == pytest.approx(100.0)
    order = db.fetchone("SELECT * FROM orders WHERE order_id=?", (fr.order_id,))
    assert order["status"] == "filled"
    assert order["trade_id"] == fr.trade_id
    trade = ledger.journal.get_trade(fr.trade_id)
    assert trade["status"] == "OPEN"
    assert trade["qty"] == pytest.approx(10)
    assert trade["stop_price"] == pytest.approx(90.0)


def test_equity_identity_cash_plus_marked_positions(db):
    prices = {d: 100.0 + i for i, d in enumerate(DATES)}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    _open_long_trade(ledger, broker, panel)

    snap = ledger.mark_to_market(DATES[3], panel)
    expected_price = prices[DATES[3]]
    expected_equity = ledger.cash() + 10 * expected_price
    assert snap["equity"] == pytest.approx(expected_equity)
    assert snap["cash"] == pytest.approx(ledger.cash())
    assert snap["positions_count"] == 1
    assert snap["peak_equity"] >= snap["equity"] - 1e-9


def test_sell_fill_closes_trade_with_realized_pnl(db):
    prices = {DATES[0]: 100.0, DATES[1]: 100.0, DATES[2]: 110.0, DATES[3]: 110.0}
    for d in DATES[4:]:
        prices[d] = 110.0
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    fr = _open_long_trade(ledger, broker, panel)
    trade_id = fr.trade_id

    # Manually build + fill an exit order at 110 on DATES[3] (simulating what the exit engine +
    # execution service would do): sell all 10 shares.
    exit_order_id = "order_test_exit"
    ledger.db.insert("order_intents", {"order_id": exit_order_id, "book": ledger.book,
                                       "session_date": DATES[2], "purpose": "exit", "intent_json": "{}",
                                       "created_at": "2024-01-01T00:00:00Z"})
    ledger.db.insert("orders", {
        "order_id": exit_order_id, "client_order_id": "c-exit", "book": ledger.book, "broker": "sim",
        "candidate_id": None, "decision_id": None, "human_decision_id": None, "trade_id": trade_id,
        "purpose": "exit", "symbol": "AAA", "side": Side.SELL.value, "qty": 10, "order_type": "market",
        "time_in_force": "opg", "limit_price": None, "created_at": "2024-01-01T00:00:00Z",
        "submitted_at": "2024-01-01T00:00:00Z", "status": "accepted", "broker_order_id": "b2",
        "filled_qty": 0, "filled_avg_price": None, "last_update_at": "2024-01-01T00:00:00Z", "raw_json": "{}",
    })
    ledger.apply_fill(order_id=exit_order_id, qty=10, price=110.0, session_date=DATES[3], commission=0.0,
                      modeled_cost=0.0, exit_reason="TARGET", held_sessions=3)

    trade = ledger.journal.get_trade(trade_id)
    assert trade["status"] == "CLOSED"
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["exit_reason"] == "TARGET"
    assert trade["net_pnl"] == pytest.approx((110.0 - 100.0) * 10)
    assert trade["ret"] == pytest.approx((110.0 - 100.0) * 10 / (100.0 * 10))
    assert ledger.get_position("AAA") is None  # flat position removed
    assert ledger.cash() == pytest.approx(100_000.0 - 1000.0 + 1100.0)


def test_exit_without_position_raises(db):
    panel = build(_bars({d: 100.0 for d in DATES}) + _bars({d: 400.0 for d in DATES}, "SPY"))
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    ledger.db.insert("orders", {
        "order_id": "orphan", "client_order_id": "c-orphan", "book": ledger.book, "broker": "sim",
        "candidate_id": None, "decision_id": None, "human_decision_id": None, "trade_id": None,
        "purpose": "exit", "symbol": "AAA", "side": Side.SELL.value, "qty": 1, "order_type": "market",
        "time_in_force": "opg", "limit_price": None, "created_at": "2024-01-01T00:00:00Z",
        "submitted_at": "2024-01-01T00:00:00Z", "status": "accepted", "broker_order_id": None,
        "filled_qty": 0, "filled_avg_price": None, "last_update_at": "2024-01-01T00:00:00Z", "raw_json": "{}",
    })
    with pytest.raises(LedgerError):
        ledger.apply_fill(order_id="orphan", qty=1, price=100.0, session_date=DATES[0])


def test_split_adjusts_position_and_is_idempotent(db):
    # RAW prices halve on the 2-for-1 ex-date (a split the raw bars do not show is not applied).
    panel = build(_bars({d: (100.0 if i < 3 else 50.0) for i, d in enumerate(DATES)}) + _bars({d: 400.0 for d in DATES}, "SPY"),
                 action_rows=[{"symbol": "AAA", "ex_date": DATES[3], "action_type": "split", "ratio": 2.0}])
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    _open_long_trade(ledger, broker, panel)

    applied = ledger.apply_corporate_actions(DATES[3], panel)
    assert any(a["action_type"] == "split" for a in applied)
    pos = ledger.get_position("AAA")
    assert pos["qty"] == pytest.approx(20)
    assert pos["avg_cost"] == pytest.approx(50.0)

    # Re-applying the same session must be a no-op (idempotent on (book, symbol, session, type)).
    applied_again = ledger.apply_corporate_actions(DATES[3], panel)
    assert applied_again == []
    pos = ledger.get_position("AAA")
    assert pos["qty"] == pytest.approx(20)


def test_dividend_credited_and_uncredited_are_both_audited(db):
    panel = build(_bars({d: 100.0 for d in DATES}) + _bars({d: 400.0 for d in DATES}, "SPY"),
                 action_rows=[{"symbol": "AAA", "ex_date": DATES[3], "action_type": "cash_dividend", "amount": 0.5}])
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0, config=None)
    _open_long_trade(ledger, broker, panel)
    cash_before = ledger.cash()

    ledger.apply_corporate_actions(DATES[3], panel)
    assert ledger.cash() == pytest.approx(cash_before + 10 * 0.5)
    row = ledger.db.fetchone("SELECT * FROM ledger_corporate_actions WHERE book=? AND action_type='cash_dividend'",
                            (ledger.book,))
    assert row["credited"] == 1
    assert row["cash_amount"] == pytest.approx(5.0)

    # Uncredited (Alpaca-paper-like) ledger on a fresh book: dividend recorded but NOT paid.
    ledger_nc = Ledger(db, Book.HUMAN, starting_cash=100_000.0)
    broker_nc = SimBroker(Book.HUMAN, zero_cost_model(), starting_cash=100_000.0)
    ledger_nc.credit_dividends = False
    _open_long_trade(ledger_nc, broker_nc, panel)
    cash_before_nc = ledger_nc.cash()
    ledger_nc.apply_corporate_actions(DATES[3], panel)
    assert ledger_nc.cash() == pytest.approx(cash_before_nc)
    row_nc = ledger_nc.db.fetchone(
        "SELECT * FROM ledger_corporate_actions WHERE book=? AND action_type='cash_dividend'", (ledger_nc.book,))
    assert row_nc["credited"] == 0
    assert row_nc["cash_amount"] == pytest.approx(0.0)
    assert row_nc["amount"] == pytest.approx(0.5)  # per-share amount is still recorded


def test_bot_and_human_books_never_mix(db):
    panel = build(_bars({d: 100.0 for d in DATES}) + _bars({d: 400.0 for d in DATES}, "SPY"))
    bot_broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    human_broker = SimBroker(Book.HUMAN, zero_cost_model(), starting_cash=50_000.0)
    bot_ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    human_ledger = Ledger(db, Book.HUMAN, starting_cash=50_000.0)

    _open_long_trade(bot_ledger, bot_broker, panel, qty=10)
    assert human_ledger.positions() == []
    assert human_ledger.cash() == pytest.approx(50_000.0)
    assert bot_ledger.cash() == pytest.approx(100_000.0 - 1000.0)
    assert {p["symbol"] for p in bot_ledger.positions()} == {"AAA"}
