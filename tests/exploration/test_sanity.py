"""Pre-market sanity checks for PAPER_EXPLORATION: limits, required vs optional information, pre-open
cutoff and revalidation, restart/duplicate safety, paper-only, dashboard lifecycle labels.
SYNTHETIC data and stub / fake brokers only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.dashboard.app import create_app
from quantlab.db.database import to_json, utcnow_iso
from quantlab.discovery import run_discovery
from quantlab.exploration import ExplorationPolicy, paper_mode, plan_exploration, preopen_submit
from quantlab.exploration.engine import OPTIONAL_INPUTS, REQUIRED_AT_SUBMIT, pending_entries

from tests.discovery.world import BENCH, crafted_bundle

ROOT = Path(__file__).resolve().parents[2]
UTC = "UTC"
SESSION, NEXT = "2024-03-22", "2024-03-25"


def _ctx(tmp_path, mode="EXPLORATION"):
    var = tmp_path / "v"
    c = load_config(root=ROOT, overrides={"benchmarks": BENCH, "paper": {"mode": mode}, "project": {
        "var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"), "log_dir": str(var / "l"),
        "report_dir": str(var / "r"), "model_dir": str(var / "m")}})
    return AppContext.create(c, init_logging=False)


def _store(ctx, bars_end=SESSION, with_actions=True, actions=()):
    ctx.store.write("bars", pd.DataFrame([{"symbol": "SPY", "date": bars_end, "open": 1.0, "high": 1.0, "low": 1.0,
                                           "close": 1.0, "volume": 1.0, "vwap": None, "trade_count": None,
                                           "provider": "synthetic", "retrieved_at": "2024-01-01T00:00:00Z"}]),
                    "synthetic", is_synthetic=True)
    if with_actions:
        rows = list(actions) or [{"symbol": "SPY", "ex_date": "2023-06-01", "action_type": "cash_dividend", "ratio": None,
                                  "amount": 1.0, "declared_date": None, "available_at": "2023-06-01T13:30:00Z",
                                  "pit_status": "PIT", "source_id": "a0", "provider": "synthetic",
                                  "retrieved_at": "2024-01-01T00:00:00Z"}]
        ctx.store.write("corporate_actions", pd.DataFrame(rows), "synthetic", is_synthetic=True)


class StubExec:
    """Records submissions; never talks to any broker."""
    book = "BOT"

    def __init__(self):
        self.calls = []

    def submit_entry(self, symbol, qty, **kw):
        self.calls.append({"symbol": symbol, "qty": qty, **kw})
        return {"refused": False, "order_id": f"order-{len(self.calls)}"}


def _decision(ctx, symbol="AAA", qty=10.0, ref=50.0, stop=45.0, session=SESSION, run_id=None, unknowns=None):
    did = f"expl_{symbol}_{session}"
    pre = {"unknowns": unknowns or ["news: UNKNOWN (no article in 90 days)", "fundamentals: UNKNOWN"],
           "invalidation": ["closes below 45"], "features": {}}
    ctx.db.insert("exploration_decisions", {
        "decision_id": did, "session_date": session, "next_session": NEXT, "symbol": symbol, "mode": "EXPLORATION",
        "selection": "SELECTED", "rank": 1, "discovery_run_id": run_id, "discovery_id": None, "origin": "DISCOVERY",
        "setup_type": "Momentum", "setup_class": "TECHNICAL-ONLY", "families": "momentum", "catalyst_families": None,
        "discovery_score": 80.0, "ref_price": ref, "qty": qty, "stop_price": stop, "holding_sessions": 10,
        "strict_blocker": "no validated strategy", "reason": "top-ranked", "info_cutoff_at": "2024-03-22T20:00:00+00:00",
        "pre_trade_json": to_json(pre), "is_synthetic": 1, "created_at": utcnow_iso()})
    return did


def _events(ctx, did):
    return [e["event"] for e in ctx.db.fetchall("SELECT event FROM exploration_events WHERE decision_id=? ORDER BY id", (did,))]


PRE_OPEN = pd.Timestamp(f"{NEXT} 08:45", tz="America/New_York").tz_convert(UTC)


# -- 1. limits ------------------------------------------------------------------------------------------
def test_configured_limits_and_modes():
    cfg = load_config(root=ROOT)                 # repo defaults (tests ignore config/local.yaml)
    p = ExplorationPolicy.from_config(cfg)
    assert (p.max_new_per_session, p.max_position_pct, p.max_open_positions) == (2, 0.02, 5)
    assert paper_mode(cfg) == "STRICT"           # STRICT stays the repo default; EXPLORATION is an explicit opt-in
    assert cfg.get("safety.paper_only") is True


def test_plan_limits_sizing_required_inputs_and_pending_orders(tmp_path):
    ctx = _ctx(tmp_path)
    cb = crafted_bundle()
    _store(ctx)
    run_discovery(ctx, cb, cb.panel.dates[-1], links={})
    res = plan_exploration(ctx, equity=100_000.0, now=PRE_OPEN)
    rows = ctx.db.fetchall("SELECT * FROM exploration_decisions")
    sel = [r for r in rows if r["selection"] == "SELECTED"]
    assert 1 <= len(sel) <= 2                                                  # at most 2 new per session
    for r in sel:
        assert r["qty"] * r["ref_price"] <= 0.02 * 100_000 + 1e-6              # <= 2% of equity
        assert r["qty"] >= 1 and 0 < r["stop_price"] < r["ref_price"]
        pre = json.loads(r["pre_trade_json"])
        assert all(c["passed"] for c in pre["required_inputs"]["at_plan"])     # every REQUIRED input present
        assert set(pre["optional_inputs"]["classes"]) == set(OPTIONAL_INPUTS)
    by = {r["symbol"]: json.loads(r["pre_trade_json"]) for r in rows}
    if "GAPD" in by:                                                           # invalid data -> never selected
        assert not next(c for c in by["GAPD"]["risk_checks"] if c["name"] == "valid_price_volume")["passed"]
    if "ILLQ" in by:
        assert any(not c["passed"] for c in by["ILLQ"]["risk_checks"] if c["name"] in ("basic_liquidity", "research_universe"))
    again = plan_exploration(ctx, equity=100_000.0, now=PRE_OPEN)
    assert again.get("already_planned") and len(ctx.db.fetchall("SELECT 1 FROM exploration_decisions")) == len(rows)
    _ = res
    ctx.close()


def test_working_exploratory_orders_count_toward_the_five_position_cap(tmp_path):
    ctx = _ctx(tmp_path)
    cb = crafted_bundle()
    _store(ctx)
    for i in range(5):                                                         # 5 submitted, not yet filled
        did = _decision(ctx, symbol=f"P{i}", session="2024-03-21")
        ctx.db.insert("orders", {"order_id": f"o{i}", "client_order_id": f"c{i}", "book": "BOT", "broker": "alpaca_paper",
                                 "candidate_id": None, "decision_id": did, "human_decision_id": None, "trade_id": None,
                                 "purpose": "entry", "symbol": f"P{i}", "side": "buy", "qty": 10, "order_type": "market",
                                 "time_in_force": "opg", "limit_price": None, "created_at": utcnow_iso(),
                                 "submitted_at": utcnow_iso(), "status": "accepted", "broker_order_id": None,
                                 "filled_qty": 0, "filled_avg_price": None, "last_update_at": utcnow_iso()})
        ctx.db.insert("exploration_events", {"decision_id": did, "event": "SUBMITTED", "at": utcnow_iso(), "order_id": f"o{i}",
                                             "details_json": "{}", "created_at": utcnow_iso()})
    assert pending_entries(ctx.db)["n"] == 5
    run_discovery(ctx, cb, cb.panel.dates[-1], links={})
    plan_exploration(ctx, equity=100_000.0, now=PRE_OPEN)
    assert not ctx.db.fetchall("SELECT 1 FROM exploration_decisions WHERE session_date=? AND selection='SELECTED'", (SESSION,))
    skipped = ctx.db.fetchall("SELECT pre_trade_json FROM exploration_decisions WHERE session_date=? AND selection='SKIPPED'",
                              (SESSION,))
    assert any(any(c["name"] == "max_open_positions" and not c["passed"] for c in json.loads(s["pre_trade_json"])["risk_checks"])
               for s in skipped)
    ctx.close()


# -- 2 + 3. required vs optional information, pre-open cutoff, revalidation ----------------------------
def test_optional_unknowns_do_not_block_submission(tmp_path):
    ctx = _ctx(tmp_path)
    _store(ctx)
    did = _decision(ctx)
    ex = StubExec()
    r = preopen_submit(ctx, ex, now=PRE_OPEN)
    assert r["submitted"] == 1 and _events(ctx, did) == ["REVALIDATED", "SUBMITTED"]
    assert ex.calls[0]["qty"] == 10.0 and ex.calls[0]["plan"].entry_ref_price == 50.0   # the plan's close, never the open
    ev = json.loads(ctx.db.fetchone("SELECT details_json FROM exploration_events WHERE event='REVALIDATED'")["details_json"])
    assert set(ev["required_checked"]) == set(REQUIRED_AT_SUBMIT)
    ctx.close()


@pytest.mark.parametrize("case", ["no_corporate_action_data", "bars_missing", "after_cutoff", "before_close",
                                  "bad_plan_values", "corporate_action_at_open"])
def test_required_information_missing_or_stale_cancels(tmp_path, case):
    ctx = _ctx(tmp_path)
    now = PRE_OPEN
    kw = {}
    if case == "no_corporate_action_data":
        _store(ctx, with_actions=False)
    elif case == "bars_missing":
        _store(ctx, bars_end="2024-03-21")                                     # decision session's bars not stored
    elif case == "corporate_action_at_open":
        _store(ctx, actions=[{"symbol": "AAA", "ex_date": NEXT, "action_type": "split", "ratio": 2.0, "amount": None,
                              "declared_date": None, "available_at": "2024-03-23T00:00:00Z", "pit_status": "PIT",
                              "source_id": "s1", "provider": "synthetic", "retrieved_at": "2024-01-01T00:00:00Z"}])
    else:
        _store(ctx)
    if case == "after_cutoff":
        now = pd.Timestamp(f"{NEXT} 09:26", tz="America/New_York").tz_convert(UTC)
    if case == "before_close":
        now = pd.Timestamp(f"{SESSION} 15:00", tz="America/New_York").tz_convert(UTC)
    if case == "bad_plan_values":
        kw = {"stop": 55.0}                                                    # stop above the reference price
    did = _decision(ctx, **kw)
    ex = StubExec()
    r = preopen_submit(ctx, ex, now=now)
    assert r["submitted"] == 0 and r["cancelled"] == 1 and ex.calls == []
    assert _events(ctx, did) == ["CANCELLED_PREOPEN"]
    ctx.close()


@pytest.mark.parametrize("before,after,cancel", [("WATCH", "UNKNOWN", True), ("WATCH", "INVALIDATED", True),
                                                 ("WATCH", "REJECTED", True), ("REJECTED", "REJECTED", False),
                                                 ("WATCH", "WATCH", False)])
def test_preopen_recheck_findings_block_submission(tmp_path, before, after, cancel):
    ctx = _ctx(tmp_path)
    _store(ctx)
    did = _decision(ctx, run_id="disc_x")
    ctx.db.insert("preopen_checks", {"discovery_run_id": "disc_x", "discovery_id": "disc_x:AAA", "symbol": "AAA",
                                     "checked_at": (PRE_OPEN - pd.Timedelta(minutes=5)).isoformat(), "next_session": NEXT,
                                     "next_open_at": "2024-03-25T13:30:00+00:00", "status_before": before,
                                     "status_after": after, "reason": "test", "checks_json": "[]", "created_at": utcnow_iso()})
    ex = StubExec()
    r = preopen_submit(ctx, ex, now=PRE_OPEN)
    assert (r["cancelled"], r["submitted"]) == ((1, 0) if cancel else (0, 1))
    _ = did
    ctx.close()


# -- 4. duplicates / restart ----------------------------------------------------------------------------
def test_crash_after_broker_accept_is_adopted_not_resubmitted(tmp_path):
    from tests.pipeline.fake_alpaca import FakeAlpacaBroker
    from quantlab.execution.ledger import Ledger, bind_book
    from quantlab.execution.service import PaperExecutionService
    ctx = _ctx(tmp_path)
    _store(ctx)
    broker = FakeAlpacaBroker([pd.Timestamp(SESSION).date(), pd.Timestamp(NEXT).date()])
    bind_book(ctx.db, "BOT", broker.name)
    ledger = Ledger(ctx.db, "BOT", config=ctx.config, broker_name=broker.name)
    svc = PaperExecutionService(ctx.db, ctx.config, "BOT", broker, ledger)
    did = _decision(ctx)
    svc.submit_entry("AAA", 10.0, decision_id=did, session_date=SESSION)       # accepted, then "crash" before the event
    assert len(broker.submits) == 1
    r = preopen_submit(ctx, svc, now=PRE_OPEN)                                 # restart: the same decision again
    assert len(broker.submits) == 1 and r["submitted"] == 1                    # adopted, never a second order
    assert preopen_submit(ctx, svc, now=PRE_OPEN)["submitted"] == 0            # terminal: skipped
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 1
    ctx.close()


# -- 6. dashboard lifecycle -----------------------------------------------------------------------------
def test_dashboard_shows_planned_vs_submitted_state(tmp_path):
    ctx = _ctx(tmp_path)
    _store(ctx)
    _decision(ctx, symbol="AAA")
    c = TestClient(create_app(ctx))
    page = c.get("/").text
    assert "PAPER MODE: EXPLORATION" in page and "PLANNED (not submitted yet" in page
    assert "actually submitted / filled</b>: 0" in page
    preopen_submit(ctx, StubExec(), now=PRE_OPEN)
    page = c.get("/").text
    assert "SUBMITTED (paper order" in page and "PLANNED (not submitted yet" not in page
    ctx.close()
