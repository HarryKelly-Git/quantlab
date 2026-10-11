"""Broker-charged fees and untradable assets (verified on the Alpaca paper account, 2026-10-11).

* Alpaca paper charges regulatory/clearing fees (cents) as FEE activities. Unrecorded, they drift
  the broker's cash away from the ledger until the $1 reconciliation tolerance pauses the system.
  The runner now records them in the BOT ledger (once each, from the ledger's start date); a single
  fee above ``paper.runner.broker_fee_max_auto`` is left unrecorded so it still pauses for review.
* A company acquired for cash (WBD, 2026-10-06) turns inactive / not tradable at the broker while the
  position stays. The DELISTED exit must not send an order the broker refuses; it is recorded as a
  refusal and re-evaluated every session, and the position stays matched to the broker's."""
from __future__ import annotations

import pytest

from quantlab.core.types import TradePlan
from quantlab.execution.alpaca_paper import AlpacaPaperBroker
from quantlab.execution.broker import BrokerError
from quantlab.execution.ledger import Ledger
from quantlab.monitoring.killswitch import KillSwitch

from ..execution.test_alpaca_paper import FakeResponse, FakeSession, make_broker  # noqa: F401
from ..execution.test_alpaca_paper import alpaca_env  # noqa: F401  (autouse fixture re-export)
from .fake_alpaca import FakeAlpacaBroker
from .test_daily import world  # noqa: F401  (fixture re-export)
from .test_paper_runner import Clock, _alpaca_service, at, make_runner, setup  # noqa: F401


def _fee(fid: str, day: str, amount: float, sub: str = "CAT") -> dict:
    return {"id": fid, "activity_type": "FEE", "activity_sub_type": sub, "date": day,
            "created_at": f"{day}T23:59:00Z", "net_amount": f"{amount:.2f}",
            "description": f"{sub} fee for proceed of 3 trades on {day} by PA1234567XY", "status": "executed",
            "currency": "USD"}


class FeeBroker(FakeAlpacaBroker):
    """FakeAlpacaBroker that charges fees (cash goes down) and lists them as FEE activities."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fees: list[dict] = []
        self.fees_unavailable = False

    def charge(self, fid: str, day: str, amount: float) -> None:
        self.cash += amount
        self.fees.append({"id": fid, "date": day, "net_amount": amount, "description": "CAT fee", "symbol": None})

    def fee_activities(self, since: str) -> list[dict]:
        if self.fees_unavailable:
            raise BrokerError("activities endpoint down")
        return [f for f in self.fees if f["date"] >= since]


class InactiveBroker(FakeAlpacaBroker):
    def __init__(self, *a, inactive=(), lookup_fails=False, **kw):
        super().__init__(*a, **kw)
        self.inactive, self.lookup_fails = set(inactive), lookup_fails

    def asset_status(self, symbol: str):
        if self.lookup_fails:
            raise BrokerError("assets endpoint down")
        if symbol in self.inactive:
            return {"symbol": symbol, "status": "inactive", "tradable": False}
        return {"symbol": symbol, "status": "active", "tradable": True}


# -- Alpaca client ----------------------------------------------------------------------------------
def test_fee_activities_pages_filters_by_date_and_parses_amounts():
    page1 = [_fee(f"2026100{i}000000000::a{i}", f"2026-10-0{i}", -0.01) for i in range(1, 4)]
    page1[0]["date"] = "2026-09-30"                                   # before `since`: dropped
    page2 = [_fee("20261008000000000::b", "2026-10-08", -0.26, "OCC")]
    b = make_broker(FakeSession([FakeResponse(200, page1), FakeResponse(200, page2)]))
    b_page = 3
    orig = b._request

    def small_pages(method, path, *, params=None, **kw):             # page_size 3 so paging is exercised
        if params and "page_size" in params:
            params["page_size"] = b_page
        return orig(method, path, params=params, **kw)
    b._request = small_pages
    fees = b.fee_activities("2026-10-01")
    assert [f["date"] for f in fees] == ["2026-10-02", "2026-10-03", "2026-10-08"]
    assert fees[-1]["net_amount"] == pytest.approx(-0.26) and fees[-1]["sub_type"] == "OCC"
    calls = b.session.calls
    assert calls[0]["url"].endswith("/v2/account/activities/FEE") and calls[0]["method"] == "GET"
    assert calls[1]["params"]["page_token"] == page1[-1]["id"]
    assert all(c["method"] == "GET" for c in calls)                   # read-only


def test_fee_activity_without_an_amount_is_an_error_not_a_guess():
    bad = _fee("x1", "2026-10-08", -0.01)
    bad["net_amount"] = None
    b = make_broker(FakeSession([FakeResponse(200, [bad])]))
    with pytest.raises(BrokerError, match="without id/amount"):
        b.fee_activities("2026-10-01")


def test_asset_status_maps_fields_and_unknown_symbol_is_none():
    b = make_broker(FakeSession([
        FakeResponse(200, {"symbol": "WBD", "status": "inactive", "tradable": False, "exchange": "NASDAQ"}),
        FakeResponse(404, {"code": 40410000, "message": "asset not found for NOSUCH"}),
    ]))
    assert b.asset_status("WBD") == {"symbol": "WBD", "status": "inactive", "tradable": False}
    assert b.asset_status("NOSUCH") is None
    assert b.session.calls[0]["url"].endswith("/v2/assets/WBD")


# -- ledger -----------------------------------------------------------------------------------------
def test_ledger_books_each_fee_once_from_its_start_and_never_a_large_one(ctx):
    ledger = Ledger(ctx.db, "BOT", starting_cash=99_912.89, broker_name="alpaca_paper")
    since = ledger.initialized_on()
    assert since is not None and len(since) == 10
    assert ledger.latest_broker_fee_date() is None
    fees = [{"id": "old", "date": "2000-01-03", "net_amount": -0.07},
            {"id": "f1", "date": since, "net_amount": -0.01, "description": "CAT"},
            {"id": "f2", "date": since, "net_amount": -0.26, "description": "OCC"},
            {"id": "big", "date": since, "net_amount": -250.0, "description": "unexpected"}]
    res = ledger.book_broker_fees(fees, since=since, max_each=5.0)
    assert [f["id"] for f in res["booked"]] == ["f1", "f2"] and [f["id"] for f in res["too_large"]] == ["big"]
    assert ledger.latest_broker_fee_date() == since
    assert ledger.cash() == pytest.approx(99_912.89 - 0.27)
    again = ledger.book_broker_fees(fees, since=since, max_each=5.0)  # idempotent
    assert again["booked"] == [] and ledger.cash() == pytest.approx(99_912.89 - 0.27)
    rows = ctx.db.fetchall("SELECT kind, ref_type, ref_id FROM ledger_cash_events WHERE kind='broker_fee' ORDER BY id")
    assert [(r["ref_type"], r["ref_id"]) for r in rows] == [("broker_activity", "f1"), ("broker_activity", "f2")]


# -- runner reconciliation --------------------------------------------------------------------------
def test_broker_fees_no_longer_accumulate_into_a_reconciliation_pause(setup):
    ctx, bundle, sessions = setup
    broker = FeeBroker(sessions)
    r, _ = make_runner(ctx, broker, Clock(at(sessions[-30], 12, 0)))
    r.start()
    since = r.ledger.initialized_on()
    for i in range(12):                                               # $1.32 of fees: beyond the $1 tolerance
        broker.charge(f"fee{i}", since, -0.11)
    assert r.reconcile("fees") is True
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE"
    assert r.ledger.cash() == pytest.approx(broker.cash)
    assert r.reconcile("again") is True and r.ledger.cash() == pytest.approx(broker.cash)   # booked once
    r.shutdown("test")


def test_a_large_or_unlisted_cash_change_still_pauses(setup):
    ctx, bundle, sessions = setup
    broker = FeeBroker(sessions)
    r, _ = make_runner(ctx, broker, Clock(at(sessions[-30], 12, 0)))
    r.start()
    broker.charge("big", r.ledger.initialized_on(), -40.0)           # above broker_fee_max_auto (5 USD)
    assert r.reconcile("large fee") is False
    assert "ledger/broker mismatch" in KillSwitch(ctx.db).state()[1]
    KillSwitch(ctx.db).resume("reviewed in test", "human:test")
    broker.fees = []
    broker.fees_unavailable = True                                    # fee list down: WARN, the cash check decides
    assert r.reconcile("fees down") is False
    r.shutdown("test")


def test_fee_booking_can_be_switched_off(setup):
    ctx, bundle, sessions = setup
    ctx.config._data.setdefault("paper", {}).setdefault("runner", {})["book_broker_fees"] = False
    broker = FeeBroker(sessions)
    r, _ = make_runner(ctx, broker, Clock(at(sessions[-30], 12, 0)))
    r.start()
    for i in range(12):
        broker.charge(f"fee{i}", r.ledger.initialized_on(), -0.11)
    assert r.reconcile("off") is False                                # the old behaviour
    r.shutdown("test")


# -- exits on an asset the broker no longer trades --------------------------------------------------
def _open_trade(ctx, broker, symbol="WBD"):
    svc, ledger = _alpaca_service(ctx, broker)
    e = svc.submit_entry(symbol, 64, plan=TradePlan(entry_ref_price=30.95), session_date="2024-05-01")
    svc.apply_broker_order(broker.parse_order(broker.fill(e["client_order_id"], 64, 30.95, "2024-05-02T13:31:00Z")["order"]),
                           "2024-05-02")
    return svc, ledger, ledger.open_trades()[0]


def test_exit_for_an_inactive_asset_is_refused_not_sent(ctx):
    broker = InactiveBroker([])
    svc, ledger, trade = _open_trade(ctx, broker)
    broker.inactive.add("WBD")                                        # acquired for cash after the entry
    n_submits = len(broker.submits)
    r = svc.submit_exit(trade.trade_id, "DELISTED", session_date="2024-05-10")
    assert r["refused"] is True and "status=inactive" in r["reason"]
    assert len(broker.submits) == n_submits                           # no order reached the broker
    assert ledger.get_position("WBD")["qty"] == pytest.approx(64)     # still matches the broker
    ev = ctx.db.fetchall("SELECT event FROM trade_events WHERE trade_id=?", (trade.trade_id,))
    assert "exit_blocked" in [x["event"] for x in ev]
    ref = ctx.db.fetchone("SELECT purpose, symbol FROM execution_refusals ORDER BY rowid DESC LIMIT 1")
    assert (ref["purpose"], ref["symbol"]) == ("exit", "WBD")
    again = svc.submit_exit(trade.trade_id, "DELISTED", session_date="2024-05-13")   # next session: same answer
    assert again["refused"] is True and len(broker.submits) == n_submits


def test_exit_for_an_active_asset_or_an_unknown_answer_is_sent_as_before(ctx):
    broker = InactiveBroker([])
    svc, ledger, trade = _open_trade(ctx, broker, "AAA")
    r = svc.submit_exit(trade.trade_id, "TIME", session_date="2024-05-10")
    assert not r.get("refused") and r["status"] == "accepted"
    broker2 = InactiveBroker([], lookup_fails=True)
    svc2, ledger2, trade2 = _open_trade(ctx, broker2, "BBB")
    r2 = svc2.submit_exit(trade2.trade_id, "TIME", session_date="2024-05-10")
    assert not r2.get("refused") and r2["status"] == "accepted"      # lookup failed: unchanged behaviour
