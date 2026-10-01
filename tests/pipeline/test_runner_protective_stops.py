"""The paper runner keeps a broker-held protective stop resting for every open long trade:
placed right after the entry fills, never duplicated, polled without forcing a reconciliation
every minute, and its intraday fill closes the trade as STOP."""
from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from quantlab.core.costs import CostModel
from quantlab.core.types import TradePlan
from quantlab.execution.ledger import Ledger
from quantlab.execution.service import PaperExecutionService
from quantlab.monitoring.killswitch import KillSwitch

from .fake_alpaca import FakeAlpacaBroker
from .test_daily import world  # noqa: F401  (fixture re-export)
from .test_paper_runner import Clock, at, make_runner, setup  # noqa: F401  (fixture re-export)


def _stops(broker):
    return [o for o in broker.orders.values() if o["type"] == "stop"]


def test_runner_places_polls_and_honours_the_protective_stop(setup):  # noqa: F811
    base, bundle, sessions = setup
    ctx = type(base).create(base.config.with_overrides({"execution": {"stop_model": "intraday", "protective_stop": {
        "enabled": True, "distance": 1.75}}}), init_logging=False)     # the feature is opt-in
    broker = FakeAlpacaBroker(sessions)
    d, prev = sessions[-10], sessions[-11]
    clock = Clock(at(d, 10, 0))
    r, _ = make_runner(ctx, broker, clock)
    r._maybe_process = lambda now: None          # this test is about stops, not the session pipeline
    r._maybe_explore = lambda now: None
    r.start()

    sym = next(s for s in bundle.panel.symbols if s != "SPY" and not s.startswith("XL"))
    px = round(float(bundle.panel.close.at[pd.Timestamp(prev), sym]), 2)
    stop = round(px * 0.9, 2)
    svc = PaperExecutionService(ctx.db, ctx.config, "BOT", broker,
                                Ledger(ctx.db, "BOT", config=ctx.config, broker_name="alpaca_paper"))
    e = svc.submit_entry(sym, 10, plan=TradePlan(entry_ref_price=px, stop_price=stop, holding_sessions=10),
                         session_date=str(prev))
    stream = r.streams[-1]
    stream.push(broker.fill(e["client_order_id"], 10, px, at(d, 9, 30).isoformat()))
    r.tick()

    [s] = _stops(broker)
    assert (s["side"], s["time_in_force"], float(s["qty"])) == ("sell", "gtc", 10.0)
    level = CostModel.from_config(ctx.config).broker_stop(px, stop)      # the configured (disaster) level
    assert s["stop_price"] == math.floor(level * 100 + 1e-6) / 100
    msgs = [e["message"] for e in ctx.db.fetchall("SELECT message FROM paper_runner_events")]
    assert any("protective stop placed" in m for m in msgs)

    # later ticks keep ONE stop, and a resting stop does not trigger an "order poll" reconciliation
    def order_poll_reconciles():
        return ctx.db.fetchone("SELECT COUNT(*) AS n FROM paper_runner_events WHERE kind='reconcile' "
                               "AND message LIKE '%order poll%'")["n"]
    n0 = order_poll_reconciles()
    for k in range(3):
        clock.t = clock.t + timedelta(seconds=90)
        r.tick()
    assert len(_stops(broker)) == 1 and order_poll_reconciles() == n0

    # the stop triggers intraday: the fill closes the trade as STOP and the books still reconcile
    stream.push(broker.fill(s["client_order_id"], 10, round(level - 0.05, 2), at(d, 14, 0).isoformat()))
    clock.t = at(d, 14, 1)
    r.tick()
    t = ctx.db.fetchone("SELECT * FROM trades")
    assert t["status"] == "CLOSED" and t["exit_reason"] == "STOP" and t["exit_date"] == str(d)
    assert KillSwitch(ctx.db).state()[0].value == "ACTIVE"
    clock.t = clock.t + timedelta(seconds=90)
    r.tick()
    assert len(_stops(broker)) == 1                    # nothing re-placed for a closed trade
    r.shutdown("test")


def test_runner_places_no_stop_when_disabled(setup):  # noqa: F811
    ctx, bundle, sessions = setup
    assert ctx.config.get("execution.protective_stop.enabled") is False     # the repo default is OFF
    ctx2 = ctx
    broker = FakeAlpacaBroker(sessions)
    d, prev = sessions[-10], sessions[-11]
    clock = Clock(at(d, 10, 0))
    r, _ = make_runner(ctx2, broker, clock)
    r._maybe_process = lambda now: None
    r._maybe_explore = lambda now: None
    r.start()
    sym = next(s for s in bundle.panel.symbols if s != "SPY" and not s.startswith("XL"))
    px = round(float(bundle.panel.close.at[pd.Timestamp(prev), sym]), 2)
    svc = PaperExecutionService(ctx2.db, ctx2.config, "BOT", broker,
                                Ledger(ctx2.db, "BOT", config=ctx2.config, broker_name="alpaca_paper"))
    e = svc.submit_entry(sym, 10, plan=TradePlan(entry_ref_price=px, stop_price=round(px * 0.9, 2)),
                         session_date=str(prev))
    r.streams[-1].push(broker.fill(e["client_order_id"], 10, px, at(d, 9, 30).isoformat()))
    r.tick()
    assert _stops(broker) == []
    r.shutdown("test")
