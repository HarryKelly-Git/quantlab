"""SimBroker: fill timing (next open after the decision session), fill prices via CostModel, cash
rejection (no margin), and split/dividend handling on held positions. All data is SYNTHETIC."""
from __future__ import annotations

import pytest

from quantlab.core.costs import CostModel
from quantlab.core.types import Book, OrderStatus, Side
from quantlab.execution.broker import OrderRequest
from quantlab.execution.sim_broker import SimBroker

from .panels import build

DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]


def _bars(prices: dict[str, float], symbol: str = "AAA") -> list[dict]:
    return [{"symbol": symbol, "date": d, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 5_000_000}
            for d, px in prices.items()]


def cost_model(one_way_bps: float = 10.0, commission_per_share: float = 0.0) -> CostModel:
    return CostModel(half_spread_tiers=((0.0, one_way_bps),), slippage_bps=0.0,
                     commission_per_share=commission_per_share, commission_min_per_order=0.0,
                     delisting_return=-0.30)


def make_broker(cm: CostModel | None = None, starting_cash: float = 100_000.0) -> SimBroker:
    return SimBroker(Book.BOT, cm or cost_model(), starting_cash=starting_cash)


def test_entry_fills_at_next_open_not_signal_session():
    prices = {d: 100.0 + i for i, d in enumerate(DATES)}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = make_broker()
    req = OrderRequest(client_order_id="c1", symbol="AAA", side=Side.BUY, qty=10, time_in_force="opg",
                       decision_session=DATES[0])
    bo = broker.submit_order(req)
    assert bo.status is OrderStatus.ACCEPTED

    # Processing the SIGNAL session itself must not fill it (entry is next_open, not same-day).
    changed = broker.process_session(DATES[0], panel)
    assert changed == []
    still = broker.get_order_by_client_id("c1")
    assert still.status is OrderStatus.ACCEPTED

    # The NEXT session fills it, at that session's RAW open.
    changed = broker.process_session(DATES[1], panel)
    assert len(changed) == 1
    filled = changed[0]
    assert filled.status is OrderStatus.FILLED
    assert filled.fill_session == DATES[1]
    expected_price = prices[DATES[1]] * 1.001  # 10bps one-way cost baked into the fill
    assert filled.filled_avg_price == pytest.approx(expected_price, rel=1e-9)
    assert filled.modeled_cost == pytest.approx(abs(expected_price - prices[DATES[1]]) * 10, rel=1e-9)

    account = broker.account()
    assert account.cash == pytest.approx(100_000.0 - expected_price * 10, rel=1e-9)
    positions = {p.symbol: p for p in broker.positions()}
    assert positions["AAA"].qty == pytest.approx(10)
    assert positions["AAA"].avg_entry_price == pytest.approx(expected_price, rel=1e-9)


def test_commission_is_charged_on_top_of_spread_cost():
    prices = {d: 50.0 for d in DATES}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = make_broker(cost_model(one_way_bps=0.0, commission_per_share=0.01))
    req = OrderRequest(client_order_id="c1", symbol="AAA", side=Side.BUY, qty=100, time_in_force="opg",
                       decision_session=DATES[0])
    broker.submit_order(req)
    broker.process_session(DATES[0], panel)
    changed = broker.process_session(DATES[1], panel)
    bo = changed[0]
    assert bo.filled_avg_price == pytest.approx(50.0)
    assert bo.commission == pytest.approx(1.0)  # 100 shares * $0.01
    assert broker.account().cash == pytest.approx(100_000.0 - 50.0 * 100 - 1.0)


def test_buy_rejected_when_cash_insufficient_no_margin():
    prices = {d: 1_000.0 for d in DATES}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = make_broker(starting_cash=500.0)  # cannot afford even 1 share at ~$1000
    broker.submit_order(OrderRequest(client_order_id="c1", symbol="AAA", side=Side.BUY, qty=1,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    changed = broker.process_session(DATES[1], panel)
    bo = changed[0]
    assert bo.status is OrderStatus.REJECTED
    assert "insufficient cash" in bo.reason
    assert broker.account().cash == pytest.approx(500.0)  # untouched
    assert broker.positions() == []


def test_sell_rejected_when_no_position_no_short_selling():
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = make_broker()
    broker.submit_order(OrderRequest(client_order_id="c1", symbol="AAA", side=Side.SELL, qty=5,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    changed = broker.process_session(DATES[1], panel)
    bo = changed[0]
    assert bo.status is OrderStatus.REJECTED
    assert "no short selling" in bo.reason


def test_split_adjusts_held_position_qty_and_avg_cost():
    prices = {d: 100.0 for d in DATES}
    # 2-for-1 split effective on DATES[3].
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"),
                 action_rows=[{"symbol": "AAA", "ex_date": DATES[3], "action_type": "split", "ratio": 2.0}])
    broker = make_broker(cost_model(one_way_bps=0.0))
    broker.submit_order(OrderRequest(client_order_id="c1", symbol="AAA", side=Side.BUY, qty=10,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    broker.process_session(DATES[1], panel)  # fills at 100.0
    pos = {p.symbol: p for p in broker.positions()}["AAA"]
    assert pos.qty == pytest.approx(10) and pos.avg_entry_price == pytest.approx(100.0)

    broker.process_session(DATES[2], panel)
    broker.process_session(DATES[3], panel)  # split ex-date
    pos = {p.symbol: p for p in broker.positions()}["AAA"]
    assert pos.qty == pytest.approx(20)
    assert pos.avg_entry_price == pytest.approx(50.0)


def test_dividend_credited_to_cash_when_configured():
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"),
                 action_rows=[{"symbol": "AAA", "ex_date": DATES[3], "action_type": "cash_dividend", "amount": 0.5}])
    broker = make_broker(cost_model(one_way_bps=0.0), starting_cash=100_000.0)
    broker.submit_order(OrderRequest(client_order_id="c1", symbol="AAA", side=Side.BUY, qty=10,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    broker.process_session(DATES[1], panel)  # fills at 100.0, cash -= 1000
    cash_after_fill = broker.account().cash
    broker.process_session(DATES[2], panel)
    broker.process_session(DATES[3], panel)  # dividend ex-date: +10 * 0.5 = +5
    assert broker.account().cash == pytest.approx(cash_after_fill + 5.0)


def test_dividend_not_credited_when_disabled():
    """SimBroker.credit_dividends=False mirrors Alpaca paper (docs/EXTERNAL-SERVICES.md: paper does
    NOT simulate dividends): cash must be untouched. (Ledger.apply_corporate_actions is the layer
    that still records an uncredited dividend for audit -- see test_ledger.py.)"""
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices) + _bars({d: 400.0 for d in DATES}, "SPY"),
                 action_rows=[{"symbol": "AAA", "ex_date": DATES[3], "action_type": "cash_dividend", "amount": 0.5}])
    broker = SimBroker(Book.BOT, cost_model(one_way_bps=0.0), starting_cash=100_000.0, credit_dividends=False)
    broker.submit_order(OrderRequest(client_order_id="c1", symbol="AAA", side=Side.BUY, qty=10,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    broker.process_session(DATES[1], panel)
    cash_after_fill = broker.account().cash
    broker.process_session(DATES[2], panel)
    broker.process_session(DATES[3], panel)
    assert broker.account().cash == pytest.approx(cash_after_fill)  # not credited
