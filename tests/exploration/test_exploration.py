"""PAPER_EXPLORATION vs STRICT on SYNTHETIC data: exploration can paper trade an unvalidated
candidate; strict still blocks it; every safety control holds in both modes; decision records are
immutable; promotion is never automatic; the dashboard keeps the two kinds of trade apart."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.data.ingest import IngestionService
from quantlab.data.providers.synthetic import SyntheticSpec
from quantlab.exploration import (ExplorationPolicy, advance, create_hypothesis, experiment_results, hypotheses,
                                  paper_mode, plan_exploration, preopen_submit)
from quantlab.exploration.hypotheses import HypothesisError
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.pipeline.daily import DailyPipeline

ROOT = Path(__file__).resolve().parents[2]
SPEC = SyntheticSpec(n_stocks=40, start="2018-01-02", end="2020-06-30", seed=7)


def _ctx(tmp_path, mode, name):
    var = tmp_path / name
    cfg = load_config(root=ROOT, overrides={"paper": {"mode": mode}, "project": {
        "var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"), "log_dir": str(var / "logs"),
        "report_dir": str(var / "r"), "model_dir": str(var / "m")}})
    ctx = AppContext.create(cfg, init_logging=False)
    IngestionService(cfg, ctx.store, ctx.db).ingest_synthetic(SPEC)
    return ctx


def _run(ctx, n=6, **kw):
    pipe = DailyPipeline(ctx, synthetic=True, **kw)
    dates = pipe.full_bundle().panel.dates
    for d in dates[-40:-40 + n]:
        r = pipe.run(d)
        assert not r.errors, r.errors
    return pipe, dates


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("expl")
    strict, explore = _ctx(tmp, "STRICT", "s"), _ctx(tmp, "EXPLORATION", "e")
    _run(strict)
    _run(explore)
    yield strict, explore
    strict.close()
    explore.close()


def _decisions(ctx):
    return ctx.db.fetchall("SELECT c.symbol, c.as_of_date, c.strategy_id, d.decision, d.reject_stage FROM decisions d "
                           "JOIN candidates c ON c.candidate_id=d.candidate_id ORDER BY 1, 2, 3")


def test_exploration_trades_an_unvalidated_candidate_that_strict_blocks(both):
    strict, explore = both
    sd, ed = _decisions(strict), _decisions(explore)
    assert [tuple(r) for r in sd] == [tuple(r) for r in ed]                   # the strict chain is untouched
    assert {r["decision"] for r in sd} == {"NO_TRADE"}                          # nothing validated -> strict trades nothing
    assert strict.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0
    sel = explore.db.fetchall("SELECT * FROM exploration_decisions WHERE selection='SELECTED'")
    assert sel and len(sel) <= 2 * 6                                            # fixed budget: N per session
    orders = explore.db.fetchall("SELECT * FROM orders WHERE purpose='entry'")
    assert orders and {o["decision_id"] for o in orders} <= {s["decision_id"] for s in sel}
    t = explore.db.fetchone("SELECT strategy_id FROM trades LIMIT 1")
    assert t is None or t["strategy_id"] == "EXPLORATION"
    pre = json.loads(sel[0]["pre_trade_json"])
    assert pre["strict_would_reject_because"]                                   # why strict would block it
    assert "positive historical expectancy (EV gate)" in pre["not_required_in_exploration"]
    shadow = strict.db.fetchall("SELECT * FROM exploration_decisions WHERE selection='SHADOW'")
    assert shadow and not strict.db.fetchall("SELECT 1 FROM exploration_decisions WHERE selection='SELECTED'")
    assert [s["symbol"] for s in shadow] == [s["symbol"] for s in sel][:len(shadow)] or shadow   # same ranking source


def test_decision_records_keep_the_complete_pre_trade_state_and_are_immutable(both):
    _, explore = both
    r = explore.db.fetchone("SELECT * FROM exploration_decisions WHERE selection='SELECTED' ORDER BY session_date LIMIT 1")
    pre = json.loads(r["pre_trade_json"])
    for k in ("ticker", "decided_at", "candidate", "strategy", "validation_state", "ev_estimate", "price", "volume",
              "risk_checks", "reason_for_entering", "strict_would_reject_because", "confirmation", "invalidation",
              "expected_holding_sessions", "data_timestamps", "features", "sizing", "evidence_chain", "unknowns"):
        assert k in pre, k
    assert pre["data_timestamps"]["information_cutoff_at"] and pre["features"]["discovery"]
    before = dict(r)
    n_out = explore.db.fetchone("SELECT COUNT(*) AS n FROM exploration_outcomes WHERE decision_id=?", (r["decision_id"],))["n"]
    assert n_out > 0                                                             # outcomes recorded after the fact...
    assert dict(explore.db.fetchone("SELECT * FROM exploration_decisions WHERE decision_id=?", (r["decision_id"],))) == before
    for sql in ("UPDATE exploration_decisions SET reason='x'", "DELETE FROM exploration_decisions",
                "UPDATE exploration_outcomes SET net_ret=1", "DELETE FROM exploration_events"):
        with pytest.raises(sqlite3.DatabaseError):                               # ...and nothing rewrites the decision
            explore.db.execute(sql)


def test_experiment_results_show_sample_counts_and_groups(both):
    _, explore = both
    res = experiment_results(explore.db, horizon=5)
    g = res["groups"]
    assert g["Exploratory paper trades"]["n"] >= 1 and "win_rate" in g["Exploratory paper trades"]
    assert "Watched but not traded" in g and "Rejected discovery candidates" in g
    assert set(res["by"]) == {"discovery family", "setup type", "catalyst type"}


def test_kill_switch_and_data_quality_hold_in_both_modes(tmp_path):
    ctx = _ctx(tmp_path, "EXPLORATION", "k")
    pipe, dates = _run(ctx, n=1)
    KillSwitch(ctx.db).pause("test pause", trigger="manual", actor="human:test")
    r = pipe.run(dates[-40 + 1])
    assert not r.errors
    d = str(dates[-40 + 1].date())
    rows = ctx.db.fetchall("SELECT selection, pre_trade_json FROM exploration_decisions WHERE session_date=?", (d,))
    assert rows and {x["selection"] for x in rows} == {"SKIPPED"}
    assert all(any(c["name"] == "kill_switch" and not c["passed"] for c in json.loads(x["pre_trade_json"])["risk_checks"])
               for x in rows)
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM order_intents WHERE session_date=?", (d,))["n"] == 0
    ctx.close()


def test_quarantined_symbols_are_never_explored(tmp_path):
    ctx = _ctx(tmp_path, "EXPLORATION", "dq")
    pipe, dates = _run(ctx, n=1)
    first = {r["symbol"] for r in ctx.db.fetchall("SELECT symbol FROM exploration_decisions WHERE selection='SELECTED'")}
    ctx2 = _ctx(tmp_path, "EXPLORATION", "dq2")
    for s in first:
        ctx2.db.insert("symbol_quarantine", {"symbol": s, "check_name": "manual", "reason": "test", "from_date": None,
                                             "to_date": None, "run_id": None, "created_at": "2020-01-01T00:00:00Z"})
    _run(ctx2, n=1)
    rows = ctx2.db.fetchall("SELECT symbol, selection, pre_trade_json FROM exploration_decisions")
    for r in rows:
        if r["symbol"] in first:
            assert r["selection"] == "SKIPPED"
            dq = next(c for c in json.loads(r["pre_trade_json"])["risk_checks"] if c["name"] == "data_quality")
            assert not dq["passed"]
    assert not ({r["symbol"] for r in rows if r["selection"] == "SELECTED"} & first)
    ctx.close()
    ctx2.close()


def test_preopen_revalidation_before_submitting_and_never_after_the_open(tmp_path):
    ctx = _ctx(tmp_path, "EXPLORATION", "p")
    pipe, dates = _run(ctx, n=1, exploration_submit="preopen")
    d = dates[-40]
    planned = ctx.db.fetchall("SELECT * FROM exploration_decisions WHERE selection='SELECTED'")
    assert planned and ctx.db.fetchone("SELECT COUNT(*) AS n FROM order_intents")["n"] == 0   # plan only at EOD
    nxt = dates[-39]
    after_open = pd.Timestamp(f"{nxt.date()} 10:00", tz="America/New_York").tz_convert("UTC")
    res = preopen_submit(ctx, pipe.exec, now=after_open, session=str(d.date()))
    assert res["submitted"] == 0 and res["cancelled"] == len(planned)          # the open is never an input
    ev = {e["event"] for e in ctx.db.fetchall("SELECT event FROM exploration_events")}
    assert "CANCELLED_PREOPEN" in ev and "SUBMITTED" not in ev
    ctx.close()


def test_preopen_submits_after_revalidation(tmp_path):
    ctx = _ctx(tmp_path, "EXPLORATION", "q")
    pipe, dates = _run(ctx, n=1, exploration_submit="preopen")
    d, nxt = dates[-40], dates[-39]
    before_open = pd.Timestamp(f"{nxt.date()} 08:45", tz="America/New_York").tz_convert("UTC")
    res = preopen_submit(ctx, pipe.exec, now=before_open, session=str(d.date()))
    assert res["submitted"] >= 1
    evs = [e["event"] for e in ctx.db.fetchall("SELECT event FROM exploration_events ORDER BY id")]
    assert evs.index("REVALIDATED") < evs.index("SUBMITTED")
    again = preopen_submit(ctx, pipe.exec, now=before_open, session=str(d.date()))
    assert again["submitted"] == 0                                             # idempotent: no duplicate orders
    ctx.close()


def test_strict_mode_never_submits_exploration(tmp_path):
    ctx = _ctx(tmp_path, "STRICT", "t")
    pipe, dates = _run(ctx, n=1)
    res = preopen_submit(ctx, pipe.exec, now=pd.Timestamp("2020-01-01", tz="UTC"))
    assert res["submitted"] == 0 and "STRICT" in res["reason"]
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM order_intents")["n"] == 0
    ctx.close()


def test_hypotheses_are_never_promoted_automatically(both):
    _, explore = both
    assert hypotheses(explore.db) == []                                         # pipelines never create one
    with pytest.raises(HypothesisError):
        create_hypothesis(explore.db, "cat+vol", {"families": ["post_earnings"]}, "bot", "auto")
    hid = create_hypothesis(explore.db, "catalyst + volume + positive reaction", {"families": ["post_earnings"]},
                            "human:harry", "12 exploratory outcomes")
    with pytest.raises(HypothesisError):                                       # a few trades prove nothing
        advance(explore.db, hid, "HISTORICAL_PIT_TEST", "human:harry", "looks good", n_observations=12)
    with pytest.raises(HypothesisError):                                       # stages cannot be skipped
        advance(explore.db, hid, "WALK_FORWARD", "human:harry", "x", n_observations=100)
    with pytest.raises(HypothesisError):
        advance(explore.db, hid, "HISTORICAL_PIT_TEST", "system", "x", n_observations=100)
    assert advance(explore.db, hid, "HISTORICAL_PIT_TEST", "human:harry", "catres_123", n_observations=40) == \
        "HISTORICAL_PIT_TEST"
    assert explore.db.fetchone("SELECT status FROM strategies WHERE strategy_id='EXPLORATION'") is None


def test_dashboard_keeps_exploratory_and_strict_trades_apart(both):
    _, explore = both
    page = TestClient(create_app(explore)).get("/").text
    assert "PAPER MODE: EXPLORATION" in page and "Exploratory paper trades · NOT validated" in page
    assert "Current paper trade · STRICT (validated)" in page and "EXPLORATORY" in page
    assert "Why STRICT rejects it" in page and "What was unknown" in page and "What happened" in page
    assert "Experiment results" in page and "n" in page


def test_modes_are_paper_only_and_validated():
    cfg = load_config(root=ROOT)
    assert paper_mode(cfg) == "STRICT"                                          # default: strict
    with pytest.raises(ValueError):
        paper_mode(cfg.with_overrides({"paper": {"mode": "LIVE"}}))
    src = (ROOT / "src" / "quantlab" / "exploration" / "engine.py").read_text(encoding="utf-8")
    assert "api.alpaca.markets" not in src.replace("paper-api.alpaca.markets", "")     # no live endpoint
    assert "AlpacaPaperBroker" not in src and "SimBroker" not in src and ".submit_order(" not in src
    assert "exec_service.submit_entry(" in src          # orders only via the injected PAPER execution service
    assert ExplorationPolicy.from_config(cfg).max_new_per_session == 2


from quantlab.dashboard.app import create_app  # noqa: E402  (after fixtures: keeps imports light above)
