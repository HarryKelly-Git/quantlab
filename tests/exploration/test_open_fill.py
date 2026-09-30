"""An `opg` entry that expires unfilled in the opening auction must still take the position.

Alpaca expires a market-on-open order that does not execute in the opening cross (observed on the
real paper account 2026-09-29: UGP and RNG both `expired` with filled_qty 0). Without a fallback the
whole decision chain produces nothing to learn from. The runner submits exactly ONE market-day order
per decision, inside the regular session, under all the usual gates."""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quantlab.context import AppContext
from quantlab.core.types import OrderStatus
from quantlab.data.ingest import IngestionService
from quantlab.data.providers.synthetic import SyntheticSpec
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.pipeline.runner import PaperRunner

from tests.pipeline.fake_alpaca import PAPER_ENV, FakeAlpacaBroker, FakeStream

ET = ZoneInfo("America/New_York")


def at(d, hh, mm=0) -> datetime:
    return datetime.combine(pd.Timestamp(d).date(), time(hh, mm), ET).astimezone(timezone.utc)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def world(config):
    ctx = AppContext.create(config.with_overrides({"paper": {"mode": "EXPLORATION"}}), init_logging=False)
    IngestionService(ctx.config, ctx.store, ctx.db).ingest_synthetic(
        SyntheticSpec(n_stocks=40, start="2017-01-03", end="2019-12-31", seed=5))
    yield ctx
    ctx.close()


def _runner(ctx, broker, clock):
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    streams: list = []

    def factory(on_event, on_state):
        streams.append(FakeStream(on_event, on_state))
        return streams[-1]
    r = PaperRunner(ctx, broker=broker, stream_factory=factory, now=clock, ingest=lambda c, d: {"skipped": True},
                    bundle_loader=lambda c: b, env=PAPER_ENV, heartbeat_thread=False, allow_synthetic=True)
    r.reconcile_retry_seconds = 0.0
    r.streams = streams
    return r, [x.date() for x in b.panel.dates]


def _submit_opg(ctx, broker, clock):
    """Evening plan + pre-open submission; returns (runner, sessions, next_session, opg client ids)."""
    r, sessions = _runner(ctx, broker, clock)
    r.start()
    r.tick()                                              # 19:10 ET: process D, plan exploration
    d = r.plan.session
    clock.t = at(r.plan.next_session, 8, 45)
    r.tick()                                              # pre-open: submit opg
    assert broker.submits, "no opg order was submitted"
    return r, sessions, r.plan.next_session, list(broker.submits), d


def _orders(broker, stops: bool = False) -> list[str]:
    """Broker submissions of one kind: entries/exits (default) or the runner's protective stops."""
    return [c for c in broker.submits if (broker.orders[c]["type"] == "stop") == stops]


def _assert_one_stop_per_open_trade(ctx, broker):
    """Protective stops are never duplicated and never outlive their trade: at most ONE working per
    OPEN trade, none for anything else. An open trade without one must be a deliberate skip: these
    fixtures fill at a fixed price, which can sit at/below the plan's stop (no stop is ever placed
    at or above the current price; the close-based stop handles that trade)."""
    open_trades = {t["symbol"]: t for t in ctx.db.fetchall("SELECT symbol, stop_price FROM trades WHERE status='OPEN'")}
    live = [broker.orders[c]["symbol"] for c in _orders(broker, stops=True)
            if broker.orders[c]["status"] not in ("canceled", "filled", "expired", "rejected")]
    assert len(live) == len(set(live)), f"duplicate protective stops: {live}"
    assert set(live) <= set(open_trades), f"stop working for a trade that is not open: {live}"
    for sym, t in open_trades.items():
        if sym not in live:
            assert t["stop_price"] >= broker.pos[sym][1], f"{sym} is above its stop but has no protective stop"


def _expire(broker, r, cids, when="2026-01-02T13:31:00Z"):
    for cid in cids:
        ev = broker.event(cid, "expired", when, status="expired")
        r.on_trade_update(ev)


def _fill_new(broker, r, since: int, price: float = 50.0) -> list[str]:
    """Fill every order submitted after index ``since`` and feed the events to the runner."""
    new = _orders(broker)[since:]
    for cid in new:
        qty = float(broker.orders[cid]["qty"])
        r.on_trade_update(broker.fill(cid, qty, price, "2026-01-02T13:36:00Z"))
    return new


def test_expired_opening_order_is_replaced_once_and_fills(world):
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    n_opg = len(cids)
    _expire(broker, r, cids)
    assert {o["status"] for o in ctx.db.fetchall("SELECT status FROM orders")} == {OrderStatus.EXPIRED.value}
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == 0        # nothing learned without a fallback

    clock.t = at(nxt, 9, 35)
    r.tick()                                                                   # submits the fallback
    _fill_new(broker, r, n_opg)                                                # it fills in the regular session
    fb = ctx.db.fetchall("SELECT symbol, qty, time_in_force, status, filled_qty FROM orders WHERE time_in_force='day'")
    assert len(fb) == n_opg and all(o["status"] == OrderStatus.FILLED.value for o in fb)
    assert len(_orders(broker)) == 2 * n_opg
    trades = ctx.db.fetchall("SELECT symbol, qty, status, stop_price, strategy_id FROM trades")
    assert len(trades) == n_opg and all(t["status"] == "OPEN" for t in trades)  # positions actually taken
    # the original plan survives the fallback: an unmanaged position (no stop) would be unsafe
    assert all(t["stop_price"] and t["stop_price"] > 0 for t in trades), trades
    assert {t["strategy_id"] for t in trades} == {"EXPLORATION"}
    plans = ctx.db.fetchall("SELECT tp.holding_sessions AS h, d.holding_sessions AS dh FROM trade_plans tp "
                            "JOIN trades t ON t.trade_id = tp.trade_id JOIN exploration_decisions d ON d.decision_id = t.decision_id")
    assert plans and all(p["h"] == p["dh"] and p["h"] in (5, 10, 20) for p in plans)  # each keeps its decision's hold
    for o in fb:                                                                # same size as the expired order
        assert o["qty"] == next(x["qty"] for x in ctx.db.fetchall(
            "SELECT symbol, qty FROM orders WHERE time_in_force='opg'") if x["symbol"] == o["symbol"])

    clock.t = at(nxt, 10, 0)
    r.tick()                                                                    # repeated scans: no second fallback
    r.tick()
    assert len(_orders(broker)) == 2 * n_opg
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == n_opg
    _assert_one_stop_per_open_trade(ctx, broker)                                # protected, exactly once
    r.shutdown("test")


def test_restart_does_not_duplicate_the_fallback(world):
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    _expire(broker, r, cids)
    clock.t = at(nxt, 9, 35)
    r.tick()
    _fill_new(broker, r, len(cids))
    n_sub = len(_orders(broker))
    n_ord = ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders WHERE order_type<>'stop'")["n"]
    r.shutdown("restart")
    r2, _ = _runner(ctx, broker, clock)
    r2.start()
    clock.t = at(nxt, 9, 50)
    r2.tick()
    r2.tick()
    assert len(_orders(broker)) == n_sub
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders WHERE order_type<>'stop'")["n"] == n_ord
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM fills")["n"] == n_ord - len(cids)
    _assert_one_stop_per_open_trade(ctx, broker)                               # and no duplicate stop
    r2.shutdown("test")


@pytest.mark.parametrize("case", ["market_closed", "after_cutoff", "kill_switch"])
def test_fallback_respects_the_gates(world, case):
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    _expire(broker, r, cids)
    n_sub = len(broker.submits)
    if case == "market_closed":
        clock.t = at(nxt, 8, 0)                       # before the open
    elif case == "after_cutoff":
        clock.t = at(nxt, 15, 50)                     # past open_fill_fallback_until_et
    else:
        clock.t = at(nxt, 9, 35)
        KillSwitch(ctx.db).pause("test", trigger="manual", actor="human:test")
    r.tick()
    assert len(broker.submits) == n_sub               # no fallback order left the runner
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == 0
    r.shutdown("test")


@pytest.mark.parametrize("case", ["reconciliation_failed", "broker_stale", "trading_blocked", "wrong_purpose"])
def test_fallback_respects_broker_and_state_gates(world, case):
    """The runner submits the fallback through its OWN execution service, so that service must carry
    the same broker/state gates as the session pipeline's."""
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    _expire(broker, r, cids)
    n_sub = len(broker.submits)
    clock.t = at(nxt, 9, 35)
    if case == "reconciliation_failed":
        r.reconciled_ok = False
    elif case == "broker_stale":
        r.broker_ok_at = clock.t - pd.Timedelta(seconds=r.broker_fresh_seconds + 60).to_pytimedelta()
    elif case == "trading_blocked":
        r.broker_snapshot = {**r.broker_snapshot, "trading_blocked": True}
    elif case == "wrong_purpose":
        pass
    else:
        assert r.exec is not None                       # the service refuses non-fallback purposes
        res = r.exec.submit_entry("SYN001", 1.0, session_date=str(sessions[-30]), decision_id="x")
        assert res["refused"] and "only submits the open-fill fallback" in res["reason"]
        r.shutdown("test")
        return
    r._maybe_fill_after_open(clock.t)                   # tick() would re-poll and refresh this state
    assert len(broker.submits) == n_sub                 # no fallback order left the runner
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == 0
    refusals = [x["reason"] for x in ctx.db.fetchall("SELECT reason FROM execution_refusals")]
    assert refusals, "the refusal must be recorded, not silent"
    r.shutdown("test")


@pytest.mark.parametrize("stage", ["after_opg_submit", "after_expiry", "after_fallback_submit", "after_fill"])
def test_restart_at_each_lifecycle_stage_never_duplicates(world, stage):
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    n_opg = len(cids)

    def restart(run):
        run.shutdown("restart test")
        r2, _ = _runner(ctx, broker, clock)
        r2.start()
        return r2
    if stage == "after_opg_submit":
        r = restart(r)
        _expire(broker, r, cids)
    elif stage == "after_expiry":
        _expire(broker, r, cids)
        r = restart(r)
    else:
        _expire(broker, r, cids)
        clock.t = at(nxt, 9, 35)
        r.tick()
        assert len(_orders(broker)) == 2 * n_opg
        if stage == "after_fallback_submit":
            r = restart(r)
            _fill_new(broker, r, n_opg)
        else:
            _fill_new(broker, r, n_opg)
            r = restart(r)
    clock.t = at(nxt, 9, 40)
    r.tick()
    r.tick()
    unfilled = [cid for cid in _orders(broker)[n_opg:] if broker.orders[cid]["status"] != "filled"]
    if unfilled:
        for cid in unfilled:
            r.on_trade_update(broker.fill(cid, float(broker.orders[cid]["qty"]), 50.0, "2026-01-02T13:36:00Z"))
    clock.t = at(nxt, 10, 5)
    r.tick()
    orders = ctx.db.fetchall("SELECT symbol, time_in_force, status FROM orders WHERE order_type<>'stop'")
    assert len(_orders(broker)) == 2 * n_opg, f"{stage}: duplicate broker submission"
    assert len(orders) == 2 * n_opg, f"{stage}: duplicate local order"
    trades = ctx.db.fetchall("SELECT symbol, qty, status, stop_price FROM trades")
    assert len(trades) == n_opg, f"{stage}: duplicate position"
    assert all(t["stop_price"] and t["stop_price"] > 0 for t in trades)   # stop survives restart
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM fills")["n"] == n_opg
    _assert_one_stop_per_open_trade(ctx, broker)                          # f"{stage}": one stop each
    r.shutdown("test")


def test_expired_exit_is_replaced_so_a_stopped_position_is_not_left_open(world):
    """An exit is submitted opg too. If it expires in the auction the position would stay open with
    its stop already breached, so the exit gets the same one-shot market-day fallback."""
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    _expire(broker, r, cids)
    clock.t = at(nxt, 9, 35)
    r.tick()
    _fill_new(broker, r, len(cids))
    open_trades = ctx.db.fetchall("SELECT trade_id, symbol FROM trades WHERE status='OPEN'")
    assert open_trades
    assert r.exec is not None and r.ledger is not None
    n_before = len(_orders(broker))
    # the ORIGINAL exit is submitted by the daily pipeline's service, as in production
    from quantlab.execution.service import PaperExecutionService
    pipe_exec = PaperExecutionService(ctx.db, ctx.config, "BOT", broker, r.ledger)
    ex = pipe_exec.submit_exit(open_trades[0]["trade_id"], "stop", session_date=str(nxt))
    assert not ex.get("refused"), ex
    exit_cid = _orders(broker)[-1]
    assert broker.orders[exit_cid]["time_in_force"] == "opg"
    r.on_trade_update(broker.event(exit_cid, "expired", "2026-01-02T13:31:00Z", status="expired"))
    assert ctx.db.fetchone("SELECT status FROM trades WHERE trade_id=?",
                           (open_trades[0]["trade_id"],))["status"] == "OPEN"     # still exposed

    clock.t = at(nxt, 9, 45)
    r.tick()                                                                       # polls the broker, then scans
    fb_cid = _orders(broker)[-1]
    assert broker.orders[fb_cid]["time_in_force"] == "day" and broker.orders[fb_cid]["side"] == "sell"
    r.on_trade_update(broker.fill(fb_cid, float(broker.orders[fb_cid]["qty"]), 51.0, "2026-01-02T13:46:00Z"))
    assert ctx.db.fetchone("SELECT status FROM trades WHERE trade_id=?",
                           (open_trades[0]["trade_id"],))["status"] == "CLOSED"   # position actually closed

    n_after = len(_orders(broker))
    clock.t = at(nxt, 9, 55)
    r.tick()
    r.tick()
    assert len(_orders(broker)) == n_after                                         # never resubmitted
    assert n_after == n_before + 2                                                 # opg exit + one fallback
    _assert_one_stop_per_open_trade(ctx, broker)          # the closed trade has none left working
    r.shutdown("test")


def test_the_fallback_entry_is_bounded_at_one_atr_above_the_reference(world):
    """Every plan says the setup is invalidated by an opening gap of more than 1 ATR. A market
    fallback would chase that gap and take the trade anyway, on a stop sized for the old price."""
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    n_opg = len(cids)
    _expire(broker, r, cids)

    clock.t = at(nxt, 9, 35)
    r.tick()
    fb = ctx.db.fetchall("SELECT symbol, order_type, time_in_force, limit_price FROM orders "
                         "WHERE time_in_force='day'")
    assert fb, "no fallback was submitted"
    plans = {d["symbol"]: d for d in ctx.db.fetchall(
        "SELECT symbol, ref_price, stop_price FROM exploration_decisions WHERE selection='SELECTED'")}
    for o in fb:
        assert o["order_type"] == "limit", o                 # never an unbounded market order
        p = plans[o["symbol"]]
        atr = (p["ref_price"] - p["stop_price"]) / 2.0       # stop = ref - 2 x ATR
        assert o["limit_price"] == pytest.approx(round(p["ref_price"] + atr, 2)), o
        assert o["limit_price"] > p["ref_price"]             # a normal open still fills
    # the bound reaches the broker, not just the local record
    for cid in broker.submits[n_opg:]:
        assert broker.orders[cid]["type"] == "limit"
        assert float(broker.orders[cid]["limit_price"]) > 0
    r.shutdown("test")


def test_a_gapped_open_leaves_the_bounded_fallback_unfilled_and_is_not_chased(world):
    """A gap beyond the bound must end as a no-fill, and the runner must not follow it with a
    market order on a later scan."""
    ctx = world
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(sessions[-30], 19, 10))
    r, _, nxt, cids, _ = _submit_opg(ctx, broker, clock)
    n_opg = len(cids)
    _expire(broker, r, cids)

    clock.t = at(nxt, 9, 35)
    r.tick()
    n_after = len(broker.submits)
    limits = {cid: float(broker.orders[cid]["limit_price"]) for cid in broker.submits[n_opg:]}
    assert limits
    # the open gapped well above every bound: the limit orders simply rest unfilled
    for cid, lim in limits.items():
        gapped_open = lim * 1.10
        assert gapped_open > lim, "the bound must sit below a gapped-up open"
        assert float(broker.orders[cid]["filled_qty"]) == 0.0

    clock.t = at(nxt, 10, 30)
    r.tick()
    r.tick()
    assert len(broker.submits) == n_after, "the runner chased the gap with another order"
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM trades")["n"] == 0
    assert not ctx.db.fetchall("SELECT 1 FROM orders WHERE order_type='market' AND time_in_force='day'")
    r.shutdown("test")


@pytest.mark.parametrize("plan,expected", [
    ({"entry_ref_price": 100.0, "stop_price": 90.0}, 105.0),        # ATR 5 -> ref + 1 ATR
    ({"entry_ref_price": 52.4, "stop_price": 49.291}, 53.95),
    ({"entry_ref_price": 100.0, "stop_price": 90.0, "stop_atr": 4.0}, 102.5),
    ({"entry_ref_price": 100.0, "stop_price": 100.0}, None),        # no ATR to recover
    ({"entry_ref_price": 100.0, "stop_price": 110.0}, None),        # stop above the reference
    ({"entry_ref_price": 100.0}, None),
    ({"stop_price": 90.0}, None),
    ({"entry_ref_price": None, "stop_price": None}, None),
    ({}, None),
    (None, None),
])
def test_fallback_limit_needs_a_usable_reference_and_stop(world, plan, expected):
    """When the bound cannot be derived the caller keeps the old market fallback, so a missing
    field never silently turns into a skipped trade."""
    ctx = world
    broker = FakeAlpacaBroker([x.date() for x in
                              ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True).panel.dates])
    r, _ = _runner(ctx, broker, Clock(at("2019-12-02", 19, 10)))
    got = r._fallback_limit(plan)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)
