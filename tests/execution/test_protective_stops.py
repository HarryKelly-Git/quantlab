"""Broker-held protective stops: the order contract, placement, fills, and above all that an exit
never sells shares a resting stop still holds (no double sell, ever)."""
from __future__ import annotations

from datetime import date

import pytest

from quantlab.config import load_config
from quantlab.core.types import Book, Side, SystemState, TradePlan
from quantlab.db.database import open_db, utcnow_iso
from quantlab.execution import protective_stops
from quantlab.execution.broker import OrderRequest
from quantlab.execution.ledger import Ledger
from quantlab.execution.service import PROTECTIVE_STOP, PaperExecutionService
from quantlab.execution.sim_broker import SimBroker

from ..pipeline.fake_alpaca import FakeAlpacaBroker
from .test_service import zero_cost_model

SESSIONS = [date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 6)]


def open_trade(stop=95.0, qty=10, px=100.0, config=None):
    db = open_db(":memory:")
    broker = FakeAlpacaBroker(SESSIONS)
    ledger = Ledger(db, Book.BOT.value, starting_cash=100_000.0, broker_name="alpaca_paper")
    svc = PaperExecutionService(db, config, Book.BOT.value, broker, ledger)
    svc._sleep = lambda s: None
    r = svc.submit_entry("AAA", qty, plan=TradePlan(entry_ref_price=px, stop_price=stop, holding_sessions=10),
                         session_date="2024-05-01")
    broker.fill(r["client_order_id"], qty, px, "2024-05-02T13:30:05Z")
    svc.sync("2024-05-02")
    trade = db.fetchone("SELECT * FROM trades")
    assert trade["status"] == "OPEN" and trade["stop_price"] == pytest.approx(stop)
    return db, broker, ledger, svc, trade


def stop_orders(broker):
    return [o for o in broker.orders.values() if o["type"] == "stop"]


# -- the order contract ---------------------------------------------------------------------------
def test_stop_order_request_contract():
    ok = OrderRequest(client_order_id="s1", symbol="AAA", side=Side.SELL, qty=10, order_type="stop",
                      time_in_force="gtc", stop_price=95.0)
    assert ok.stop_price == 95.0
    with pytest.raises(ValueError, match="stop_price"):
        OrderRequest(client_order_id="s2", symbol="AAA", side=Side.SELL, qty=10, order_type="stop", time_in_force="gtc")
    with pytest.raises(ValueError, match="gtc"):           # gtc is only for the protective stop
        OrderRequest(client_order_id="s3", symbol="AAA", side=Side.SELL, qty=10, order_type="market", time_in_force="gtc")
    with pytest.raises(ValueError, match="stop_price"):
        OrderRequest(client_order_id="s4", symbol="AAA", side=Side.SELL, qty=10, order_type="market",
                     time_in_force="day", stop_price=95.0)
    with pytest.raises(ValueError, match="whole"):
        OrderRequest(client_order_id="s5", symbol="AAA", side=Side.SELL, qty=1.5, order_type="stop",
                     time_in_force="gtc", stop_price=95.0)


def test_simulated_broker_never_treats_a_stop_as_a_market_order():
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    bo = broker.submit_order(OrderRequest(client_order_id="s1", symbol="AAA", side=Side.SELL, qty=10,
                                          order_type="stop", time_in_force="gtc", stop_price=95.0))
    assert bo.status.value == "rejected" and "stop" in bo.reason


# -- placement ------------------------------------------------------------------------------------
def test_protective_stop_is_one_gtc_stop_sell_for_the_whole_position():
    db, broker, ledger, svc, trade = open_trade()
    res = svc.submit_protective_stop(trade["trade_id"], 95.0049)
    assert not res.get("refused")
    [o] = stop_orders(broker)
    assert (o["side"], o["time_in_force"], float(o["qty"]), o["stop_price"]) == ("sell", "gtc", 10.0, 95.0)
    row = db.fetchone("SELECT * FROM orders WHERE order_type='stop'")
    assert row["purpose"] == "exit" and row["trade_id"] == trade["trade_id"] and row["stop_price"] == 95.0
    assert db.fetchone("SELECT purpose FROM order_intents WHERE order_id=?", (row["order_id"],))["purpose"] == PROTECTIVE_STOP
    # idempotent: asking again changes nothing at the broker
    n = len(broker.submits)
    again = svc.submit_protective_stop(trade["trade_id"], 95.0)
    assert again.get("duplicate") and len(broker.submits) == n


def test_a_filled_stop_closes_the_trade_as_STOP():
    db, broker, ledger, svc, trade = open_trade()
    svc.submit_protective_stop(trade["trade_id"], 95.0)
    [o] = stop_orders(broker)
    broker.fill(o["client_order_id"], 10, 94.80, "2024-05-03T15:12:00Z")      # triggered intraday
    svc.sync("2024-05-03")
    t = db.fetchone("SELECT * FROM trades")
    assert t["status"] == "CLOSED" and t["exit_reason"] == "STOP" and t["exit_price"] == pytest.approx(94.80)
    assert t["exit_date"] == "2024-05-03" and ledger.get_position("AAA") is None


def test_paused_system_places_no_stop():
    db, broker, ledger, svc, trade = open_trade()
    db.upsert("system_state", {"id": 1, "state": SystemState.PAUSED.value, "reason": "t", "changed_at": utcnow_iso(),
                               "changed_by": "test"}, ["id"])
    res = svc.submit_protective_stop(trade["trade_id"], 95.0)
    assert res["refused"] and "SYSTEM_PAUSED" in res["reason"] and stop_orders(broker) == []


def test_protective_stops_do_not_use_the_daily_order_budget():
    cfg = load_config().with_overrides({"risk": {"max_daily_orders": 2}})
    db, broker, ledger, svc, trade = open_trade(config=cfg)          # 1 entry intent on 2024-05-01
    assert not svc.submit_protective_stop(trade["trade_id"], 95.0, session_date="2024-05-01").get("refused")
    r = svc.submit_entry("BBB", 5, plan=TradePlan(entry_ref_price=50.0, stop_price=45.0), session_date="2024-05-01")
    assert not r.get("refused"), r                                   # the stop did not use the 2nd slot


# -- exits never double-sell ------------------------------------------------------------------------
def test_an_exit_cancels_the_stop_first_and_sells_exactly_once():
    db, broker, ledger, svc, trade = open_trade()
    svc.submit_protective_stop(trade["trade_id"], 95.0)
    res = svc.submit_exit(trade["trade_id"], "TIME", session_date="2024-05-03")
    assert not res.get("refused"), res
    [s] = stop_orders(broker)
    assert s["status"] == "canceled"
    live_sells = [o for o in broker.orders.values() if o["side"] == "sell" and o["status"] not in
                  ("canceled", "filled", "expired", "rejected")]
    assert len(live_sells) == 1 and live_sells[0]["type"] == "market" and float(live_sells[0]["qty"]) == 10
    events = [e["event"] for e in db.fetchall("SELECT event FROM trade_events WHERE trade_id=?", (trade["trade_id"],))]
    assert "protective_stop_released" in events and "exit_submitted" in events


def test_an_exit_is_not_sent_while_the_stop_cancel_is_unconfirmed():
    db, broker, ledger, svc, trade = open_trade()
    svc.submit_protective_stop(trade["trade_id"], 95.0)
    broker.cancel_stuck = True
    svc.stop_cancel_wait_seconds = 0.0
    n = len(broker.submits)
    res = svc.submit_exit(trade["trade_id"], "TIME", session_date="2024-05-03")
    assert res["refused"] and "not confirmed" in res["reason"]
    assert len(broker.submits) == n                                  # nothing else was sent
    assert db.fetchone("SELECT status FROM trades")["status"] == "OPEN"


def test_an_exit_is_skipped_when_the_stop_already_filled():
    db, broker, ledger, svc, trade = open_trade()
    svc.submit_protective_stop(trade["trade_id"], 95.0)
    [o] = stop_orders(broker)
    broker.fill(o["client_order_id"], 10, 94.5, "2024-05-03T15:00:00Z")      # not yet synced locally
    n = len(broker.submits)
    res = svc.submit_exit(trade["trade_id"], "TIME", session_date="2024-05-03")
    assert res.get("skipped") and not res.get("refused")
    assert len(broker.submits) == n
    t = db.fetchone("SELECT * FROM trades")
    assert t["status"] == "CLOSED" and t["exit_reason"] == "STOP"


# -- maintenance ------------------------------------------------------------------------------------
def positions(qty=10, price=101.0):
    return {"AAA": {"symbol": "AAA", "qty": qty, "current_price": price}}


def test_maintain_places_keeps_and_skips():
    db, broker, ledger, svc, trade = open_trade()
    out = protective_stops.maintain(svc, broker_positions=positions(), session_date="2024-05-02")
    assert len(out["placed"]) == 1 and len(stop_orders(broker)) == 1
    out = protective_stops.maintain(svc, broker_positions=positions(), session_date="2024-05-02")
    assert out["kept"] == 1 and len(stop_orders(broker)) == 1         # nothing new
    # unknown broker state, fewer shares at the broker, price already through the stop: never placed
    db2, broker2, _, svc2, _ = open_trade()
    for pos, why in ((None, "unknown"), (positions(qty=5), "broker holds"), (positions(price=94.0), "at/below")):
        out = protective_stops.maintain(svc2, broker_positions=pos, session_date="2024-05-02")
        assert out["placed"] == [] and why in out["skipped"][0][1]
    assert stop_orders(broker2) == []


def test_maintain_leaves_a_trade_with_a_working_exit_alone():
    db, broker, ledger, svc, trade = open_trade()
    svc.submit_exit(trade["trade_id"], "TIME", session_date="2024-05-03")
    out = protective_stops.maintain(svc, broker_positions=positions(), session_date="2024-05-03")
    assert out["placed"] == [] and stop_orders(broker) == []


def test_maintain_waits_for_a_working_entry_to_finish():
    """Wash-trade protection: no sell stop while a buy in the same symbol is still working."""
    db = open_db(":memory:")
    broker = FakeAlpacaBroker(SESSIONS)
    ledger = Ledger(db, Book.BOT.value, starting_cash=100_000.0, broker_name="alpaca_paper")
    svc = PaperExecutionService(db, None, Book.BOT.value, broker, ledger)
    r = svc.submit_entry("AAA", 10, plan=TradePlan(entry_ref_price=100.0, stop_price=95.0, holding_sessions=10),
                         session_date="2024-05-01")
    broker.fill(r["client_order_id"], 6, 100.0, "2024-05-02T13:30:05Z")          # 6 of 10 so far
    svc.sync("2024-05-02")
    out = protective_stops.maintain(svc, broker_positions=positions(qty=6), session_date="2024-05-02")
    assert out["placed"] == [] and "still working" in out["skipped"][0][1]
    broker.fill(r["client_order_id"], 4, 100.0, "2024-05-02T13:31:00Z")          # entry complete
    svc.sync("2024-05-02")
    out = protective_stops.maintain(svc, broker_positions=positions(qty=10), session_date="2024-05-02")
    [o] = stop_orders(broker)
    assert len(out["placed"]) == 1 and float(o["qty"]) == 10                      # the WHOLE position


def test_maintain_replaces_the_stop_after_a_split():
    db, broker, ledger, svc, trade = open_trade()
    protective_stops.maintain(svc, broker_positions=positions(), session_date="2024-05-02")
    # 2-for-1 split applied to the position (as Ledger.apply_corporate_actions records it)
    db.execute("UPDATE positions SET qty=20, avg_cost=50 WHERE symbol='AAA'")
    broker.pos["AAA"] = [20.0, 50.0]
    db.insert("ledger_corporate_actions", {"book": "BOT", "symbol": "AAA", "session_date": "2024-05-03",
                                           "action_type": "split", "ratio": 2.0, "amount": None, "qty_before": 10,
                                           "qty_after": 20, "cash_amount": 0.0, "credited": 1,
                                           "trade_id": trade["trade_id"], "created_at": utcnow_iso()})
    out = protective_stops.maintain(svc, broker_positions=positions(qty=20, price=50.5), session_date="2024-05-03")
    assert len(out["replaced"]) == 1
    old, new = sorted(stop_orders(broker), key=lambda o: o["status"] != "canceled")
    assert old["status"] == "canceled" and float(old["qty"]) == 10 and old["stop_price"] == 95.0
    assert new["status"] == "accepted" and float(new["qty"]) == 20 and new["stop_price"] == 47.5


def test_maintain_replacement_budget_prevents_an_order_loop():
    db, broker, ledger, svc, trade = open_trade()
    protective_stops.maintain(svc, broker_positions=positions(), session_date="2024-05-02", max_replacements=1)
    for i, stop in enumerate((94.0, 93.0, 92.0)):        # the desired stop keeps changing
        db.execute("UPDATE trades SET stop_price=?", (stop,))
        protective_stops.maintain(svc, broker_positions=positions(), session_date="2024-05-02", max_replacements=1)
    assert len(stop_orders(broker)) == 2                  # the original + ONE replacement, not four
