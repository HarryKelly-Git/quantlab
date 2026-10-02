"""The stock book's reconciliation must stay OK when the OPT book holds option positions, has paid
premium, or has open orders in the SAME Alpaca paper account -- and must still flag a real stock
mismatch (including a stock position created by an exercise)."""
from __future__ import annotations

import pytest

from quantlab.core.types import Book, OrderStatus, Side
from quantlab.db.database import utcnow_iso
from quantlab.execution.broker import BrokerOrder, BrokerPosition
from quantlab.execution.ledger import Ledger
from quantlab.execution.reconcile import Reconciler, position_asset_class
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.options.book import OPT_ORDER_PREFIX, OptionsLedger, cotenant_cash_offset

from ..pipeline.fake_alpaca import FakeAlpacaBroker
from ..pipeline.test_daily import world  # noqa: F401  (fixture re-export)
from ..pipeline.test_paper_runner import Clock, at, make_runner
from .fake_broker import FakeOptionsBroker
from .helpers import contract

OCC = contract(100).symbol


def _opt_fill(db, qty=1, price=3.0, cid=f"{OPT_ORDER_PREFIX}t1", status=OrderStatus.FILLED):
    """One OPT order + its broker-reported fill through the real OPT ledger code."""
    now = utcnow_iso()
    db.insert("options_structures", {"structure_id": "opt_t", "kind": "LONG_CALL", "underlying": "TGT",
                                     "expiration": "2026-10-16", "multiplier": 100.0, "legs_json": "[]", "qty": qty,
                                     "max_loss_usd": qty * price * 100, "close_by_date": "2026-10-14",
                                     "status": "OPEN", "created_at": now, "updated_at": now}, or_ignore=True)
    row = {"order_id": cid, "client_order_id": cid, "structure_id": "opt_t", "leg_role": "long", "purpose": "open",
           "contract_symbol": OCC, "side": "buy", "qty": qty, "order_type": "limit", "time_in_force": "day",
           "limit_price": price, "multiplier": 100.0, "status": "accepted", "broker": "alpaca_paper",
           "filled_qty": 0.0, "filled_avg_price": None, "created_at": now, "last_update_at": now}
    db.insert("options_orders", row)
    bo = BrokerOrder(client_order_id=cid, broker_order_id="b", status=status, filled_qty=qty, filled_avg_price=price)
    OptionsLedger(db).apply_broker_order(row, bo, "2026-10-01")


def test_asset_class_detection():
    assert position_asset_class(BrokerPosition("AAPL", 1, raw={"asset_class": "us_equity"})) == "us_equity"
    assert position_asset_class(BrokerPosition(OCC, 1, raw={"asset_class": "us_option"})) == "us_option"
    assert position_asset_class(BrokerPosition(OCC, 1)) == "us_option"           # no asset_class: OCC pattern
    assert position_asset_class(BrokerPosition("AAPL", 1)) == "us_equity"


def test_stock_reconcile_ignores_option_positions_and_premium(db):
    broker = FakeOptionsBroker(cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    _opt_fill(db)
    broker.pos[OCC] = 1.0
    broker.cash -= 300.0                                                 # the premium left the shared account
    rec = Reconciler(ledger, broker, cash_offset=lambda: cotenant_cash_offset(db, broker)).reconcile()
    assert rec.ok, rec.diffs
    assert rec.broker_snapshot["positions"] == {} and rec.broker_snapshot["other_books"]["positions"] == {OCC: 1.0}
    # without the offset the premium looks like a stock-ledger cash mismatch ...
    assert [d["field"] for d in Reconciler(ledger, broker).reconcile().diffs] == ["cash"]
    # ... and reconciling every asset class would flag the option position
    both = Reconciler(ledger, broker, asset_classes=("us_equity", "us_option"),
                      cash_offset=lambda: cotenant_cash_offset(db, broker)).reconcile()
    assert [d["field"] for d in both.diffs] == [f"position:{OCC}"]


def test_stock_mismatch_is_still_detected_with_options_present(db):
    broker = FakeOptionsBroker(cash=100_000.0 - 300.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)
    _opt_fill(db)
    broker.pos[OCC] = 1.0
    broker.stock_pos["TGT"] = 100.0                                      # e.g. an exercise created stock
    rec = Reconciler(ledger, broker, cash_offset=lambda: cotenant_cash_offset(db, broker)).reconcile()
    assert not rec.ok and [d["field"] for d in rec.diffs] == ["position:TGT"]


def test_offset_counts_broker_fills_the_opt_ledger_has_not_applied_yet(db):
    broker = FakeOptionsBroker(cash=100_000.0)
    now = utcnow_iso()
    _opt_fill(db, cid=f"{OPT_ORDER_PREFIX}done")                          # applied: -300
    db.insert("options_orders", {"order_id": "o2", "client_order_id": f"{OPT_ORDER_PREFIX}race", "structure_id": "opt_t",
                                 "leg_role": "long", "purpose": "open", "contract_symbol": OCC, "side": "buy", "qty": 2,
                                 "order_type": "limit", "time_in_force": "day", "limit_price": 3.1, "multiplier": 100.0,
                                 "status": "accepted", "broker": "alpaca_paper", "filled_qty": 0.0,
                                 "created_at": now, "last_update_at": now})
    broker.orders[f"{OPT_ORDER_PREFIX}race"] = {"id": "x", "cid": f"{OPT_ORDER_PREFIX}race", "symbol": OCC,
                                                 "side": Side.BUY, "qty": 2, "type": "limit", "tif": "day",
                                                 "limit": 3.1, "status": OrderStatus.ACCEPTED, "filled": 0.0, "avg": None}
    broker.fill(f"{OPT_ORDER_PREFIX}race", 2, 3.1)                         # filled at the broker, not yet synced
    assert cotenant_cash_offset(db) == pytest.approx(-300.0)
    assert cotenant_cash_offset(db, broker) == pytest.approx(-300.0 - 620.0)


def test_paper_runner_stays_active_with_opt_orders_positions_and_fills(world):  # noqa: F811
    ctx = world
    bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [d.date() for d in bundle.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    r, _ = make_runner(ctx, broker, Clock(at(sessions[-30], 12, 0)), bundle=bundle)
    r.start()
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE"
    # the OPT book bought 1 call (filled, applied) and has a second order open at the broker
    _opt_fill(ctx.db)
    broker.pos[OCC] = [1.0, 3.0]
    broker.cash -= 300.0
    cid = f"{OPT_ORDER_PREFIX}open1"
    broker.orders[cid] = {"id": "y", "client_order_id": cid, "symbol": OCC, "side": "buy", "qty": "1",
                          "type": "limit", "time_in_force": "day", "limit_price": 3.0, "status": "accepted",
                          "filled_qty": "0", "filled_avg_price": None}
    now = utcnow_iso()
    ctx.db.insert("options_orders", {"order_id": "o-open1", "client_order_id": cid, "structure_id": "opt_t",
                                     "leg_role": "long", "purpose": "open", "contract_symbol": OCC, "side": "buy",
                                     "qty": 1, "order_type": "limit", "time_in_force": "day", "limit_price": 3.0,
                                     "multiplier": 100.0, "status": "accepted", "broker": "alpaca_paper",
                                     "filled_qty": 0.0, "created_at": now, "last_update_at": now})
    # its fill arrives on the stock runner's trade_updates stream before the OPT book syncs
    broker.cash -= 305.0
    broker.pos[OCC] = [2.0, 3.025]
    broker.orders[cid].update({"status": "filled", "filled_qty": "1", "filled_avg_price": "3.05"})
    r.streams[-1].push({"event": "fill", "timestamp": "t", "order": dict(broker.orders[cid])})
    r.tick()
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE", KillSwitch(ctx.db).state()[1]
    assert r.reconcile("test with options in the account") is True
    # a foreign order that is NOT an OPT order still pauses the system
    r.streams[-1].push({"event": "new", "timestamp": "t", "order": {"id": "z", "client_order_id": "manual-1",
                                                                     "symbol": "TSLA", "side": "buy", "qty": "1",
                                                                     "status": "new", "filled_qty": "0"}})
    r.tick()
    assert KillSwitch(ctx.db).state()[0].value == "SYSTEM_PAUSED"
    r.shutdown("test")
