"""Reconciler: matches when ledger and broker agree, flags a mismatch when they diverge, and
reports broker_unavailable (without pausing anything itself -- see reconcile.py's docstring)."""
from __future__ import annotations

from quantlab.core.costs import CostModel
from quantlab.core.types import Book, Side
from quantlab.db.database import open_db
from quantlab.execution.broker import BrokerError, OrderRequest
from quantlab.execution.ledger import Ledger
from quantlab.execution.reconcile import Reconciler
from quantlab.execution.sim_broker import SimBroker

from .panels import build

DATES = ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]


def zero_cost_model() -> CostModel:
    return CostModel(half_spread_tiers=((0.0, 0.0),), slippage_bps=0.0, commission_per_share=0.0,
                     commission_min_per_order=0.0, delisting_return=-0.30)


def _bars(prices, symbol):
    return [{"symbol": symbol, "date": d, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 5_000_000}
            for d, px in prices.items()]


def _seeded_ledger_and_broker():
    db = open_db(":memory:")
    panel = build(_bars({d: 100.0 for d in DATES}, "AAA") + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    order_id = "order1"
    ledger.db.insert("order_intents", {"order_id": order_id, "book": "BOT", "session_date": DATES[0],
                                       "purpose": "entry", "intent_json": "{}", "created_at": "2024-01-01T00:00:00Z"})
    ledger.db.insert("orders", {
        "order_id": order_id, "client_order_id": "c1", "book": "BOT", "broker": "sim", "candidate_id": None,
        "decision_id": None, "human_decision_id": None, "trade_id": None, "purpose": "entry", "symbol": "AAA",
        "side": Side.BUY.value, "qty": 10, "order_type": "market", "time_in_force": "opg",
        "limit_price": None, "created_at": "2024-01-01T00:00:00Z", "submitted_at": "2024-01-01T00:00:00Z",
        "status": "accepted", "broker_order_id": "b1", "filled_qty": 0, "filled_avg_price": None,
        "last_update_at": "2024-01-01T00:00:00Z", "raw_json": "{}",
    })
    broker.submit_order(OrderRequest(client_order_id="cb1", symbol="AAA", side=Side.BUY, qty=10,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    bo = broker.process_session(DATES[1], panel)[0]
    ledger.apply_fill(order_id=order_id, qty=bo.filled_qty, price=bo.filled_avg_price, session_date=DATES[1],
                      commission=bo.commission, modeled_cost=bo.modeled_cost, filled_at=bo.filled_at,
                      signal_date=DATES[0])
    return ledger, broker


def test_reconcile_ok_when_ledger_and_broker_agree():
    ledger, broker = _seeded_ledger_and_broker()
    result = Reconciler(ledger, broker).reconcile(run_id="run1")
    assert result.ok
    assert result.status == "ok"
    assert result.diffs == []
    row = ledger.db.fetchone("SELECT * FROM reconciliations WHERE run_id='run1'")
    assert row is not None
    assert row["status"] == "ok"


def test_reconcile_flags_mismatch():
    ledger, broker = _seeded_ledger_and_broker()
    # Desync the internal ledger from the broker: an untracked cash adjustment.
    ledger.db.insert("ledger_cash_events", {"book": ledger.book, "session_date": None, "kind": "commission",
                                            "amount": -500.0, "symbol": None, "trade_id": None, "ref_type": "init",
                                            "ref_id": None, "details_json": "{}", "created_at": "2024-01-01T00:00:00Z"})
    result = Reconciler(ledger, broker, cash_tolerance=1.0).reconcile()
    assert not result.ok
    assert result.status == "mismatch"
    cash_diff = next(d for d in result.diffs if d["field"] == "cash")
    assert cash_diff["delta"] == 500.0 or cash_diff["delta"] == -500.0
    row = ledger.db.fetchone("SELECT * FROM reconciliations ORDER BY id DESC LIMIT 1")
    assert row["status"] == "mismatch"


def test_reconcile_position_mismatch_detected():
    ledger, broker = _seeded_ledger_and_broker()
    # Silently change the broker's view without telling the ledger (simulate drift).
    broker.positions_state["AAA"]["qty"] = 999.0
    result = Reconciler(ledger, broker).reconcile()
    assert result.status == "mismatch"
    pos_diff = next(d for d in result.diffs if d["field"] == "position:AAA")
    assert pos_diff["broker"] == 999.0


class _UnavailableBroker:
    name = "broken"

    def is_available(self) -> bool:
        return False

    def account(self):
        raise BrokerError("should not be called")

    def positions(self):
        raise BrokerError("should not be called")


def test_reconcile_reports_broker_unavailable_and_does_not_pause():
    ledger, _broker = _seeded_ledger_and_broker()
    before = ledger.db.fetchone("SELECT state FROM system_state WHERE id=1")
    result = Reconciler(ledger, _UnavailableBroker()).reconcile()
    assert result.status == "broker_unavailable"
    assert not result.ok
    after = ledger.db.fetchone("SELECT state FROM system_state WHERE id=1")
    assert before == after  # Reconciler never touches system_state itself
