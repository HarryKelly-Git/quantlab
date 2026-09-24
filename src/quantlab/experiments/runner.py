"""The research vertical slice, end to end:

stored data -> validation -> universe -> features -> strategy signals -> candidates -> backtest
(simulated trades) -> metrics + statistics -> persisted experiment (+ trades, equity) -> report.

Every step reuses the shared components; this module only wires them and records the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestEngine, BacktestResult, save_to_db
from quantlab.context import AppContext
from quantlab.core.types import PitStatus
from quantlab.data.validation import DataValidator, ValidationReport, quarantined_symbols
from quantlab.experiments.registry import ExperimentRegistry
from quantlab.features.base import FeatureSet
from quantlab.logging_setup import get_logger, log_event
from quantlab.strategies.arena import StrategyArena
from quantlab.strategies.registry import build_strategies, register_strategies
from quantlab.universe import UNIVERSE_PIT_STATUS, UniverseEngine
from quantlab.validation import stats as S
from quantlab.validation.holdout import HoldoutGuard

log = get_logger(__name__)


class DataValidationError(RuntimeError):
    pass


@dataclass
class BacktestRun:
    experiment_id: str
    result: BacktestResult
    validation: ValidationReport
    information_content: pd.DataFrame
    inference: dict[str, Any]
    pit_status: PitStatus
    survivorship: dict[str, Any]
    is_synthetic: bool
    labels: list[str] = field(default_factory=list)


def _default_period(ctx: AppContext, dates: pd.DatetimeIndex, start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    hold = pd.Timestamp(ctx.config.get("validation.holdout.start"))
    s = pd.Timestamp(start) if start else dates[0]
    if end:
        e = pd.Timestamp(end)
    else:
        before = dates[dates < hold]
        e = before[-1] if len(before) else dates[-1]
    return s, e


def _fin(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def inference(result: BacktestResult, config, n_variants: int) -> dict[str, Any]:
    """Statistical evidence, not just returns: bootstrap CI of per-trade net return, PSR/DSR of daily
    returns, and a sample-size verdict. MODEL_OUTPUT."""
    tr = result.trade_returns
    daily = result.daily_returns().to_numpy()
    min_trades = int(config.get("validation.min_trades_for_conclusion", 100))
    boot = S.bootstrap_ci(tr, S.stat_mean, config=config)
    sharpe = S.sharpe_inference(daily)
    dsr = S.deflated_sharpe(daily, n_variants)
    verdict = "INSUFFICIENT_SAMPLE" if len(tr) < min_trades else (
        "SIGNIFICANT_AFTER_DEFLATION" if dsr.significant else "NOT_SIGNIFICANT")
    return {
        "n_trades": int(len(tr)),
        "mean_trade_net_ret": _fin(boot.estimate),
        "mean_trade_ci": [_fin(boot.ci_low), _fin(boot.ci_high)],
        "mean_trade_pvalue": _fin(boot.p_value),
        "mean_trade_significant_positive": bool(boot.significant_positive),
        "bootstrap_status": boot.status,
        "psr_vs_zero": sharpe.to_dict() if hasattr(sharpe, "to_dict") else None,
        "deflated_sharpe": dsr.to_dict() if hasattr(dsr, "to_dict") else None,
        "n_variants_tested": n_variants,
        "min_trades_for_conclusion": min_trades,
        "verdict": verdict,
        "info_kind": "MODEL_OUTPUT",
    }


def run_strategy_backtest(ctx: AppContext, strategy_ids: list[str] | None = None, start=None, end=None,
                          synthetic: bool | None = None, holdout_token: str | None = None,
                          name: str | None = None, n_variants_tested: int = 1, hypothesis_id: str | None = None,
                          notes: str = "") -> BacktestRun:
    cfg, db, store = ctx.config, ctx.db, ctx.store
    snapshot = store.snapshot(synthetic=synthetic)
    bundle = store.load_bundle(cfg.section("benchmarks"), snapshot=snapshot, synthetic=synthetic)

    validator = DataValidator(cfg, db)
    vrep = validator.check_bundle(bundle)
    strategies = build_strategies(cfg, only=strategy_ids)
    if not strategies:
        raise ValueError(f"no implemented strategies selected (asked for {strategy_ids})")
    register_strategies(db, strategies)
    s, e = _default_period(ctx, bundle.panel.dates, start, end)
    touches = HoldoutGuard(cfg, db).touches(s, e)
    registry = ExperimentRegistry(db, cfg)
    label = name or f"backtest:{'+'.join(x.strategy_id for x in strategies)}"
    with registry.run(label, "backtest", period=(str(s.date()), str(e.date())), snapshot=snapshot,
                      strategy_versions={x.strategy_id: x.version for x in strategies},
                      n_variants_tested=n_variants_tested, hypothesis_id=hypothesis_id,
                      uses_synthetic=bundle.is_synthetic, touches_holdout=touches, notes=notes) as exp_id:
        validator.record(vrep, run_id=exp_id)
        if not vrep.ok:
            raise DataValidationError("; ".join(f"{c.name}: {c.reason}" for c in vrep.critical_failures))
        exclude = quarantined_symbols(db) | set(vrep.quarantined)
        uni_engine = UniverseEngine(cfg)
        universe = uni_engine.membership(bundle, exclude=exclude)
        fs = FeatureSet(bundle, universe=universe)
        arena = StrategyArena(strategies)
        signals = arena.scores(fs, universe)
        pit = PitStatus.weakest([UNIVERSE_PIT_STATUS] + [x.check_pit(fs) for x in strategies])

        engine = BacktestEngine(cfg, bundle, db)
        result = engine.run(signals, {x.strategy_id: x for x in strategies}, universe, s, e, fs=fs,
                            holdout_token=holdout_token, experiment_id=exp_id)
        save_to_db(db, exp_id, result)
        info = arena.information_content(fs, universe, s, e, horizons=(5, 10, 20),
                                         holdout_start=None if holdout_token else cfg.get("validation.holdout.start"))
        inf = inference(result, cfg, n_variants_tested)
        surv = uni_engine.survivorship_status(bundle)
        labels = list(result.labels)
        registry.finish(exp_id, "succeeded", {
            "metrics": result.metrics, "inference": inf, "diagnostics": result.diagnostics,
            "information_content": info.replace({np.nan: None}).to_dict("records"),
            "validation": [{"check": c.name, "passed": c.passed, "severity": c.severity.value, "reason": c.reason}
                           for c in vrep.checks],
            "quarantined": vrep.quarantined, "pit_status": pit.value, "survivorship": surv, "labels": labels,
        }, conclusion=inf["verdict"])
    log_event(log, "backtest experiment complete", experiment_id=exp_id, verdict=inf["verdict"],
              trades=inf["n_trades"], synthetic=bundle.is_synthetic)
    return BacktestRun(exp_id, result, vrep, info, inf, pit, surv, bundle.is_synthetic, labels)
