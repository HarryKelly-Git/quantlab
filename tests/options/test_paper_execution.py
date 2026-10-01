"""OPT paper execution: gated OFF by default, limit/day only, never better than mid, caps, kill
switch, long-leg-first sequencing (never naked), OPT ledger multiplier, closes, expiry guard."""
from __future__ import annotations

from datetime import date

import pytest

from quantlab.core.types import OrderStatus, Side
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.options.book import OPT_ORDER_PREFIX, OptionsLedger, reconcile_options
from quantlab.options.execution import OptionsPaperExecutor, buy_limit, close_by_date, sell_limit
from quantlab.options.settings import OptionsConfigError, OptionsSettings, PaperSettings
from quantlab.options.structures import debit_spread, long_call

from .fake_broker import FakeOptionsBroker
from .helpers import contract, quote

D = date(2026, 10, 1)


def _ctx_db(ctx):
    return ctx.db


def settings(**paper):
    return OptionsSettings(paper_trading=True, paper=PaperSettings(**paper))


def call(k=100, bid=2.90, ask=3.00):
    c = contract(k)
    return long_call(c, quote(c, bid, ask))


def spread():
    lo, hi = contract(100), contract(110)
    return debit_spread(lo, quote(lo, 2.90, 3.00), hi, quote(hi, 0.90, 1.00))


def test_execution_is_gated_off_by_default(ctx):
    s = OptionsSettings.from_config(ctx.config)
    assert s.enabled is True and s.paper_trading is False            # config/default.yaml
    broker = FakeOptionsBroker()
    res = OptionsPaperExecutor(ctx.config, ctx.db, broker).open_structure(call(), 1, session_date=D)
    assert res["ok"] is False and "gated OFF" in res["refused"]
    assert broker.requests == [] and ctx.db.fetchone("SELECT COUNT(*) AS n FROM options_orders")["n"] == 0
    assert ctx.db.fetchone("SELECT reason FROM options_refusals")["reason"].startswith("options.paper_trading is false")


def test_limit_day_order_at_the_ask(ctx):
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=settings())
    res = ex.open_structure(call(), 1, session_date=D)
    assert res["ok"], res
    req = broker.requests[0]
    assert (req.order_type, req.time_in_force, req.limit_price, req.side, req.qty) == ("limit", "day", 3.00, Side.BUY, 1)
    assert req.client_order_id.startswith(OPT_ORDER_PREFIX)
    row = ctx.db.fetchone("SELECT * FROM options_orders")
    assert row["order_type"] == "limit" and row["time_in_force"] == "day" and row["status"] == "accepted"
    with pytest.raises(Exception):                                     # DB refuses a market order outright
        ctx.db.execute("UPDATE options_orders SET order_type='market'")


@pytest.mark.parametrize("bid,ask", [(2.90, 3.00), (0.10, 0.11), (1.01, 1.38), (5.0, 5.0)])
@pytest.mark.parametrize("f", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_limits_are_never_better_than_mid(bid, ask, f):
    mid = (bid + ask) / 2
    b, s = buy_limit(bid, ask, f), sell_limit(bid, ask, f)
    assert mid - 1e-9 <= b <= ask + 1e-9 and bid - 1e-9 <= s <= mid + 1e-9
    assert buy_limit(bid, ask, 0.0) == pytest.approx(ask) and sell_limit(bid, ask, 0.0) == pytest.approx(bid)
    with pytest.raises(ValueError):
        buy_limit(bid, ask, 1.5)


def test_fraction_beyond_mid_is_a_config_error(config):
    with pytest.raises(OptionsConfigError):
        OptionsSettings.from_config(config.with_overrides({"options": {"paper": {"entry_mid_fraction": 1.2}}}))


def test_spread_is_legged_long_first_and_never_naked(ctx):
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=settings(max_premium_per_trade=1000))
    res = ex.open_structure(spread(), 2, session_date=D)
    assert res["ok"] and len(broker.requests) == 1 and broker.requests[0].side is Side.BUY
    ex.sync(D)
    assert len(broker.requests) == 1                                   # long not filled -> short not sent
    long_cid = broker.requests[0].client_order_id
    broker.fill(long_cid, 1, 3.00)                                     # partial, then the day ends
    broker.end_day(long_cid)
    ex.sync(D)
    short = broker.requests[1]
    assert short.side is Side.SELL and short.qty == 1 and short.limit_price == pytest.approx(0.90)
    led = OptionsLedger(ctx.db)
    assert led.positions() == {contract(100).symbol: 1.0}
    assert led.net_trading_cash() == pytest.approx(-300.0)             # 1 x 3.00 x 100: multiplier explicit
    broker.fill(short.client_order_id, 1, 0.90)
    ex.sync(D)
    assert led.positions() == {contract(100).symbol: 1.0, contract(110).symbol: -1.0}
    assert led.net_trading_cash() == pytest.approx(-300.0 + 90.0)
    st = ctx.db.fetchone("SELECT * FROM options_structures")
    assert st["status"] == "OPEN"
    fills = ctx.db.fetchall("SELECT * FROM options_fills ORDER BY created_at")
    assert [f["multiplier"] for f in fills] == [100.0, 100.0]
    assert [f["cash_amount"] for f in fills] == pytest.approx([-300.0, 90.0])
    assert reconcile_options(ctx.db, broker)["ok"]
    # close: the short leg is bought back FIRST, the long leg sold only after
    res = ex.close_structure(st["structure_id"], "horizon", D)
    assert res["ok"] and broker.requests[-1].side is Side.BUY and broker.requests[-1].symbol == contract(110).symbol
    n = len(broker.requests)
    ex.sync(D)
    assert len(broker.requests) == n                                   # long not sold while short is open
    broker.fill(broker.requests[-1].client_order_id, 1, 0.50)
    ex.sync(D)
    assert broker.requests[-1].side is Side.SELL and broker.requests[-1].symbol == contract(100).symbol
    broker.fill(broker.requests[-1].client_order_id, 1, 4.00)
    ex.sync(D)
    assert ctx.db.fetchone("SELECT status FROM options_structures")["status"] == "CLOSED"
    assert led.positions() == {}
    assert led.net_trading_cash() == pytest.approx(-300 + 90 - 50 + 400)


def test_unfilled_entry_is_abandoned(ctx):
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=settings())
    ex.open_structure(call(), 1, session_date=D)
    broker.end_day(broker.requests[0].client_order_id)
    ex.sync(D)
    assert ctx.db.fetchone("SELECT status FROM options_structures")["status"] == "ABANDONED"
    assert OptionsLedger(ctx.db).positions() == {}


def test_caps_and_kill_switch(ctx):
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker,
                              settings=settings(max_premium_per_trade=500, max_premium_total=700, max_open_positions=5))
    assert "max_premium_per_trade" in ex.open_structure(call(), 2, session_date=D)["refused"]   # 2 x 300 = 600
    assert ex.open_structure(call(), 1, session_date=D)["ok"]
    assert ex.open_structure(call(105, 1.0, 1.1), 1, session_date=D)["ok"]                   # 300 + 110
    assert "max_premium_total" in ex.open_structure(call(), 1, session_date=D)["refused"]           # 410 + 300 > 700
    assert "outside 1.." in ex.open_structure(call(), 0, session_date=D)["refused"]
    KillSwitch(ctx.db).pause("test pause", "test")
    n = len(broker.requests)
    assert "SYSTEM_PAUSED" in ex.open_structure(call(105, 1.0, 1.1), 1, session_date=D)["refused"]
    assert len(broker.requests) == n


def test_max_open_positions(ctx):
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=settings(max_open_positions=1))
    assert ex.open_structure(call(), 1, session_date=D)["ok"]
    assert "max_open_positions" in ex.open_structure(call(105, 1.0, 1.1), 1, session_date=D)["refused"]


def test_short_leg_not_sold_when_paused_after_long_fill(ctx):
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=settings())
    ex.open_structure(spread(), 1, session_date=D)
    broker.fill(broker.requests[0].client_order_id, 1, 3.00)
    KillSwitch(ctx.db).pause("test pause", "test")
    ex.sync(D)
    assert len(broker.requests) == 1                                   # nothing sold: plain long call, bounded
    assert ctx.db.fetchone("SELECT status FROM options_structures")["status"] == "OPEN"


def test_never_held_into_expiry(ctx):
    assert close_by_date(date(2026, 10, 16), 2) == date(2026, 10, 14)
    broker = FakeOptionsBroker()
    ex = OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=settings())
    res = ex.open_structure(call(), 1, session_date=date(2026, 10, 14))
    assert "close-by date" in res["refused"] and broker.requests == []
    ok = ex.open_structure(call(), 1, session_date=D, horizon_date=date(2026, 10, 8))
    broker.fill(broker.requests[0].client_order_id, 1, 3.0)
    ex.sync(D)
    assert [r["structure_id"] for r in ex.due_for_close(date(2026, 10, 7))] == []
    assert [r["structure_id"] for r in ex.due_for_close(date(2026, 10, 8))] == [ok["structure_id"]]


def test_opt_reconcile_flags_unknown_positions_and_orders(ctx):
    broker = FakeOptionsBroker()
    broker.pos[contract(100).symbol] = 1.0                              # option position the OPT book never had
    broker.stock_pos["AAA"] = 10.0                                      # stock: not the OPT book's business
    res = reconcile_options(ctx.db, broker)
    assert not res["ok"] and [d["field"] for d in res["diffs"]] == [f"position:{contract(100).symbol}"]
    assert ctx.db.fetchone("SELECT book FROM reconciliations ORDER BY id DESC LIMIT 1")["book"] == "OPT"
