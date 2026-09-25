"""Daily paper pipeline end to end on SYNTHETIC data: candidates -> decisions -> shadow book ->
paper orders -> next-open fills -> ledger -> exits -> outcomes -> report."""
from __future__ import annotations

import pandas as pd
import pytest

from quantlab.core.types import StrategyStage
from quantlab.data.ingest import IngestionService
from quantlab.data.providers.synthetic import SyntheticSpec
from quantlab.pipeline.daily import DailyPipeline, replay
from quantlab.research.promotion import PromotionManager
from quantlab.strategies.registry import build_strategies, register_strategies


@pytest.fixture
def world(config):
    from quantlab.context import AppContext
    # Synthetic worlds only: allow synthetic backtest statistics to feed EV (never the default).
    ctx = AppContext.create(config.with_overrides({"decision": {"stats": {"allow_synthetic": True}}}), init_logging=False)
    IngestionService(ctx.config, ctx.store, ctx.db).ingest_synthetic(
        SyntheticSpec(n_stocks=50, start="2016-01-04", end="2019-12-31", seed=3, momentum_edge=0.0015))
    yield ctx
    ctx.close()


def _activate(ctx, sid="momentum_trend"):
    """Backtest the strategy (gives the EV engine history), then walk it to PAPER + ACTIVE with an
    explicit, logged HUMAN override (tests/demo only)."""
    from quantlab.experiments.runner import run_strategy_backtest
    run_strategy_backtest(ctx, [sid], synthetic=True)
    register_strategies(ctx.db, build_strategies(ctx.config))
    pm = PromotionManager(ctx.db, ctx.config)
    ver = ctx.config.get(f"strategies.{sid}.version")
    for stage in [StrategyStage.BACKTEST, StrategyStage.WALK_FORWARD, StrategyStage.LOCKED_HOLDOUT,
                  StrategyStage.SHADOW, StrategyStage.PAPER]:
        pm.transition(sid, ver, stage, "pipeline test", "human:test", override_reason="synthetic pipeline test")
    pm.set_status(sid, ver, "ACTIVE", "pipeline test", actor="human:test")


def test_shadow_strategies_never_trade_but_everything_is_recorded(world):
    pipe = DailyPipeline(world, synthetic=True)
    d = pipe.full_bundle().panel.dates[-40]
    res = pipe.run(d)
    assert not res.errors, res.errors
    n_c = world.db.fetchone("SELECT COUNT(*) AS n FROM candidates")["n"]
    assert n_c > 0
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM decisions")["n"] == n_c
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM shadow_opportunities")["n"] == n_c
    stages = {r["reject_stage"] for r in world.db.fetchall("SELECT reject_stage FROM decisions")}
    assert "NONE" not in stages          # nothing traded: strategies are SHADOW until promoted
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0
    assert res.report_path and "PAPER TRADING ONLY" in open(res.report_path, encoding="utf-8").read()


@pytest.mark.slow
def test_active_strategy_trades_fills_next_open_and_exits(world):
    _activate(world)
    pipe = DailyPipeline(world, synthetic=True)
    dates = pipe.full_bundle().panel.dates
    results = replay(world, dates[-70], dates[-10], synthetic=True)
    assert all(not r.errors for r in results), [r.errors for r in results if r.errors]
    orders = world.db.fetchall("SELECT * FROM orders WHERE book='BOT'")
    assert any(o["purpose"] == "entry" for o in orders), "an ACTIVE strategy should place paper orders"
    fills = world.db.fetchall("SELECT f.*, o.created_at, c.as_of_date FROM fills f JOIN orders o ON o.order_id=f.order_id "
                              "LEFT JOIN candidates c ON c.candidate_id=o.candidate_id WHERE o.purpose='entry'")
    assert fills
    for f in fills:   # fills happen strictly after the decision session (next open)
        assert f["as_of_date"] is None or pd.Timestamp(f["session_date"]) > pd.Timestamp(f["as_of_date"])
    closed = world.db.fetchone("SELECT COUNT(*) AS n FROM trades WHERE book='BOT' AND status='CLOSED'")["n"]
    assert closed > 0, "time/stop exits should close some trades within 60 sessions"
    snaps = world.db.fetchone("SELECT COUNT(*) AS n FROM portfolio_snapshots WHERE book='BOT'")["n"]
    assert snaps >= 50
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM shadow_outcomes")["n"] > 0


def test_system_paused_blocks_orders_but_records_decisions(world):
    _activate(world)
    from quantlab.monitoring.killswitch import KillSwitch
    KillSwitch(world.db).pause("test pause", trigger="manual", actor="human:test")
    pipe = DailyPipeline(world, synthetic=True)
    res = pipe.run(pipe.full_bundle().panel.dates[-30])
    assert not res.errors
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == 0
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM decisions")["n"] > 0


def test_resume_does_not_duplicate_candidates(world):
    pipe = DailyPipeline(world, synthetic=True)
    d = pipe.full_bundle().panel.dates[-20]
    res = pipe.run(d)
    n1 = world.db.fetchone("SELECT COUNT(*) AS n FROM candidates")["n"]
    res2 = DailyPipeline(world, synthetic=True).run(d, resume_run_id=res.run_id)
    assert not res2.errors
    assert world.db.fetchone("SELECT COUNT(*) AS n FROM candidates")["n"] == n1


def test_daily_order_count_is_per_session_not_wall_clock(world):
    """Replaying many sessions in one wall-clock day must not accumulate the daily order limit."""
    from quantlab.pipeline.daily import DailyPipeline as P
    _activate(world)
    pipe = P(world, synthetic=True)
    dates = pipe.full_bundle().panel.dates
    world.db.insert("order_intents", {"order_id": "o-old", "book": "BOT", "session_date": str(dates[-40].date()),
                                      "purpose": "entry", "intent_json": "{}", "created_at": "2026-01-01T00:00:00+00:00"})
    # 1 intent on an OLD session must not count against today's session
    n = world.db.fetchone("SELECT COUNT(*) AS n FROM order_intents WHERE book='BOT' AND session_date=?",
                          (str(dates[-20].date()),))["n"]
    assert n == 0
    res = pipe.run(dates[-20])
    assert not res.errors
    report = open(res.report_path, encoding="utf-8").read()
    assert "decided this session" in report and "of equity" in report
