"""The research vertical slice end to end, plus METHOD VALIDATION on synthetic worlds:
a planted edge must be found; a null world must NOT produce a significant result."""
from __future__ import annotations

import sqlite3

import pytest

from quantlab.data.ingest import IngestionService
from quantlab.data.providers.synthetic import SyntheticSpec
from quantlab.experiments.registry import ExperimentRegistry
from quantlab.experiments.report import render_backtest_markdown, write_report
from quantlab.experiments.runner import run_strategy_backtest


def _run(ctx, edge: float, seed: int = 3):
    IngestionService(ctx.config, ctx.store, ctx.db).ingest_synthetic(
        SyntheticSpec(n_stocks=60, start="2016-01-04", end="2021-12-31", seed=seed, momentum_edge=edge))
    return run_strategy_backtest(ctx, ["momentum_trend"], synthetic=True)


@pytest.mark.slow
def test_planted_edge_is_found_and_everything_is_recorded(ctx):
    run = _run(ctx, edge=0.0015)
    assert run.is_synthetic and "SYNTHETIC" in run.labels
    assert run.inference["mean_trade_significant_positive"]
    assert run.information_content.query("horizon == 20")["mean_excess"].iloc[0] > 0
    exp = ExperimentRegistry(ctx.db, ctx.config).get(run.experiment_id)
    assert exp["uses_synthetic_data"] == 1 and exp["config_hash"] == ctx.config.hash
    assert exp["results"][-1]["status"] == "succeeded"
    n = ctx.db.fetchone("SELECT COUNT(*) AS n FROM backtest_trades WHERE experiment_id=?", (run.experiment_id,))["n"]
    assert n == run.inference["n_trades"] > 0
    md = render_backtest_markdown(exp)
    assert "SYNTHETIC DATA" in md and "Verdict" in md and "Deflated Sharpe" in md
    assert write_report(ctx.db, ctx.config, run.experiment_id).is_file()
    with pytest.raises(sqlite3.IntegrityError):
        ctx.db.execute("DELETE FROM experiments")


@pytest.mark.slow
def test_null_world_is_not_significant(ctx):
    run = _run(ctx, edge=0.0)
    assert run.inference["verdict"] in {"NOT_SIGNIFICANT", "INSUFFICIENT_SAMPLE"}
    assert not run.inference["mean_trade_significant_positive"]


def test_failed_experiment_is_recorded(ctx):
    with pytest.raises(Exception):
        run_strategy_backtest(ctx, ["momentum_trend"], synthetic=True)   # no data ingested yet
    # nothing registered before data loads; now force a failure after registration
    reg = ExperimentRegistry(ctx.db, ctx.config)
    with pytest.raises(RuntimeError):
        with reg.run("boom", "backtest") as exp_id:
            raise RuntimeError("boom")
    assert reg.get(exp_id)["results"][-1]["status"] == "failed"
