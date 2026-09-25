"""Walk-forward evaluation of a FIXED strategy (no parameter search).

For each window from :func:`quantlab.validation.splits.walk_forward_windows`:
  * TRAIN [train_start, train_end]: an in-sample reference backtest on data truncated at train_end
    (nothing is fitted, since the strategy's parameters are fixed; it exists to measure IS -> OOS
    degradation). Its trades are NOT stored as evidence.
  * embargo: ``embargo_sessions`` sessions are skipped after train_end.
  * TEST [test_start, test_end]: an out-of-sample backtest on data truncated at test_end, so the
    engine and every feature physically cannot see anything later. Signals are generated from that
    truncated view; positions still open at test_end are closed at its close. Capital is chained
    from the previous window's ending equity.

The OOS trades are stored in ``backtest_trades`` under segments ``oos:<k>``. The EV engine's
statistics provider prefers exactly these. Per-window and chained equity go to ``backtest_equity``.
The benchmark (SPY buy-and-hold) is chained over the SAME OOS windows, so strategy and benchmark
are exposed to identical periods, embargo gaps included.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest import metrics as M
from quantlab.backtest.engine import BacktestEngine, BacktestResult, save_to_db, tri_equity
from quantlab.context import AppContext
from quantlab.core.types import PitStatus
from quantlab.data.panel import DataBundle
from quantlab.data.validation import DataValidator, quarantine_map
from quantlab.db.database import Database
from quantlab.experiments.registry import ExperimentRegistry
from quantlab.features.base import FeatureSet
from quantlab.logging_setup import get_logger, log_event
from quantlab.strategies.registry import build_strategies, register_strategies
from quantlab.universe import UNIVERSE_PIT_STATUS, UniverseEngine
from quantlab.validation.splits import WalkForwardWindow, walk_forward_windows

log = get_logger(__name__)


class InsufficientHistoryError(RuntimeError):
    """Not a single complete train+embargo+test window fits before the locked holdout."""


class DataNotSuitableError(RuntimeError):
    """The real data failed (or could not pass) the data audit; evaluating it would not be evidence."""


@dataclass
class WindowResult:
    window: WalkForwardWindow
    start_capital: float
    end_capital: float
    oos: BacktestResult
    in_sample_metrics: dict[str, Any]
    benchmark_return: float | None

    @property
    def oos_return(self) -> float:
        return self.end_capital / self.start_capital - 1.0

    def summary(self) -> dict[str, Any]:
        m = self.oos.metrics
        return {**self.window.to_dict(), "oos_trades": int(m.get("n_trades", 0)), "oos_return": self.oos_return,
                "oos_sharpe": m.get("sharpe"), "oos_max_drawdown": m.get("max_drawdown"),
                "benchmark_return": self.benchmark_return,
                "excess_return": None if self.benchmark_return is None else self.oos_return - self.benchmark_return,
                "is_return": self.in_sample_metrics.get("total_return_net"), "is_sharpe": self.in_sample_metrics.get("sharpe"),
                "is_trades": self.in_sample_metrics.get("n_trades")}


@dataclass
class WalkForwardRun:
    experiment_id: str
    strategy_id: str
    windows: list[WindowResult]
    result: BacktestResult                  # aggregated OOS trades + chained OOS equity
    benchmark_equity: pd.DataFrame | None
    inference: dict[str, Any]
    stability: dict[str, Any]
    pit_status: PitStatus
    is_synthetic: bool
    labels: list[str] = field(default_factory=list)


def _signals(view: DataBundle, strategy, cfg, exclude):
    uni = UniverseEngine(cfg).membership(view, exclude=exclude)
    fs = FeatureSet(view, universe=uni)
    return uni, fs, strategy.score(fs, uni), strategy.check_pit(fs)


def _chain_benchmark(panel, symbol: str, windows: list[WindowResult], capital: float) -> pd.DataFrame | None:
    """SPY buy-and-hold over exactly the OOS windows, chained (flat during embargo gaps)."""
    parts, level = [], capital
    for w in windows:
        dates = panel.dates[(panel.dates >= w.window.test_start) & (panel.dates <= w.window.test_end)]
        eq = tri_equity(panel, symbol, dates, level)
        if eq is None:
            return None
        parts.append(eq)
        level = float(eq["equity"].iloc[-1])
    return pd.concat(parts) if parts else None


def _stability(windows: list[WindowResult]) -> dict[str, Any]:
    rets = np.array([w.oos_return for w in windows])
    ex = np.array([w.oos_return - w.benchmark_return for w in windows if w.benchmark_return is not None])
    return {"n_windows": len(windows), "windows_positive": int((rets > 0).sum()),
            "fraction_positive": float((rets > 0).mean()) if len(rets) else None,
            "windows_beating_benchmark": int((ex > 0).sum()) if len(ex) else None,
            "mean_window_return": float(rets.mean()) if len(rets) else None,
            "std_window_return": float(rets.std(ddof=1)) if len(rets) > 1 else None,
            "worst_window_return": float(rets.min()) if len(rets) else None}


def run_walk_forward(ctx: AppContext, strategy_id: str, synthetic: bool | None = None, *,
                     bundle: DataBundle | None = None, snapshot: dict[str, list[str]] | None = None,
                     name: str | None = None, notes: str = "", persist: bool = True) -> WalkForwardRun:
    """Walk-forward evaluate one configured strategy with its FIXED config parameters."""
    from quantlab.experiments.runner import DataValidationError, inference

    cfg, db = ctx.config, ctx.db
    if bundle is None:
        snapshot = snapshot or ctx.store.snapshot(synthetic=synthetic)
        bundle = ctx.store.load_bundle(cfg.section("benchmarks"), snapshot=snapshot, synthetic=synthetic)
    strategies = build_strategies(cfg, only=[strategy_id])
    if not strategies:
        raise ValueError(f"strategy {strategy_id!r} is not implemented/configured")
    strat = strategies[0]
    register_strategies(db, strategies)
    windows = walk_forward_windows(bundle.calendar, cfg)
    period = (str(windows[0].test_start.date()), str(windows[-1].test_end.date())) if windows else None
    registry = ExperimentRegistry(db, cfg)
    capital0 = float(cfg.get("backtest.initial_capital", 100_000))
    with registry.run(name or f"walk_forward:{strategy_id}", "walk_forward", period=period, snapshot=snapshot or {},
                      strategy_versions={strat.strategy_id: strat.version}, n_variants_tested=1,
                      uses_synthetic=bundle.is_synthetic, notes=notes,
                      seeds={"bootstrap": int(cfg.get("validation.bootstrap.seed", 7))}) as exp_id:
        validator = DataValidator(cfg, db)
        vrep = validator.check_bundle(bundle)
        validator.record(vrep, run_id=exp_id)
        if not vrep.ok:
            raise DataValidationError("; ".join(f"{c.name}: {c.reason}" for c in vrep.critical_failures))
        if not windows:
            raise InsufficientHistoryError(
                f"no complete walk-forward window fits in {bundle.panel.dates[0].date()}..{bundle.panel.dates[-1].date()} "
                f"before the holdout ({cfg.get('validation.holdout.start')}) with config validation.walk_forward="
                f"{cfg.section('validation').get('walk_forward')}")
        audit = None
        if not bundle.is_synthetic:
            audit = _audit_real_bundle(ctx, bundle, snapshot or {})
            if audit["status"] not in ("SUITABLE_SMALL_SAMPLE_ONLY", "DATA_LIMITATION"):
                raise DataNotSuitableError(f"data audit {audit['status']}: {audit['reason']}")
        exclude = quarantine_map(db, vrep)
        results: list[WindowResult] = []
        pits = [UNIVERSE_PIT_STATUS]
        capital = capital0
        for w in windows:
            # in-sample reference: data ends at train_end
            tv = bundle.truncate(w.train_end)
            u_is, fs_is, sc_is, _ = _signals(tv, strat, cfg, exclude)
            is_res = BacktestEngine(cfg, tv).run({strat.strategy_id: sc_is}, {strat.strategy_id: strat}, u_is,
                                                 w.train_start, w.train_end, fs=fs_is)
            # out-of-sample: data ends at test_end; nothing later exists for the engine or features
            ov = bundle.truncate(w.test_end)
            u, fs, sc, pit = _signals(ov, strat, cfg, exclude)
            pits.append(pit)
            cfg_k = cfg.with_overrides({"backtest": {"initial_capital": capital}})
            res = BacktestEngine(cfg_k, ov, db).run({strat.strategy_id: sc}, {strat.strategy_id: strat}, u,
                                                     w.test_start, w.test_end, fs=fs, experiment_id=exp_id)
            end_cap = float(res.equity["equity"].iloc[-1])
            b = res.metrics.get("benchmark") or {}
            results.append(WindowResult(w, capital, end_cap, res, is_res.metrics, b.get("benchmark_total_return")))
            if persist:
                save_to_db(db, exp_id, res, segment=w.segment)
            log_event(log, "walk-forward window done", window=w.index, test=[str(w.test_start.date()), str(w.test_end.date())],
                      trades=int(res.metrics.get("n_trades", 0)), oos_return=end_cap / capital - 1)
            capital = end_cap

        trades = pd.concat([r.oos.trades for r in results], ignore_index=True)
        # each window's engine restarts cum_costs at 0: carry the running total across windows
        parts, offset = [], 0.0
        for r in results:
            eq_k = r.oos.equity.copy()
            eq_k["cum_costs"] = eq_k["cum_costs"] + offset
            offset = float(eq_k["cum_costs"].iloc[-1])
            parts.append(eq_k)
        equity = pd.concat(parts)
        bench = _chain_benchmark(bundle.panel, bundle.market_symbol, results, capital0)
        met = M.summary(equity, trades, bench, min_trades=int(cfg.get("validation.min_trades_for_conclusion", 100)))
        agg = BacktestResult(trades, equity, met, {"windows": [w.to_dict() for w in windows]},
                             labels=["SYNTHETIC"] if bundle.is_synthetic else [], benchmark_equity=bench)
        inf = inference(agg, cfg, n_variants=1)
        stab = _stability(results)
        pit = PitStatus.weakest(pits)
        if persist:
            _save_chained_equity(db, exp_id, equity)
        registry.finish(exp_id, "succeeded", {
            "metrics": met, "inference": inf, "stability": stab,
            "windows": [r.summary() for r in results], "pit_status": pit.value,
            "survivorship": UniverseEngine(cfg).survivorship_status(bundle), "labels": agg.labels,
            "validation": [{"check": c.name, "passed": c.passed, "severity": c.severity.value, "reason": c.reason}
                           for c in vrep.checks],
            "quarantined": vrep.quarantined, "data_status": _data_status(db, snapshot or {}, audit),
            "walk_forward_config": cfg.section("validation").get("walk_forward"),
        }, conclusion=inf["verdict"])
    return WalkForwardRun(exp_id, strat.strategy_id, results, agg, bench, inf, stab, pit, bundle.is_synthetic, agg.labels)


def _save_chained_equity(db: Database, exp_id: str, equity: pd.DataFrame) -> None:
    rows = [{"experiment_id": exp_id, "segment": "oos", "date": str(pd.Timestamp(d).date()),
             "equity": float(r["equity"]), "cash": float(r["cash"]),
             "gross_exposure": float(r["gross_exposure"]) if np.isfinite(r["gross_exposure"]) else 0.0,
             "positions": int(r["positions"])} for d, r in equity.iterrows()]
    db.insert_many("backtest_equity", rows)


def _audit_real_bundle(ctx: AppContext, bundle: DataBundle, snapshot: dict[str, list[str]]) -> dict[str, Any]:
    """Run the data-audit checks on exactly the data being evaluated (no symbol-start lateness
    check: in a full universe many stocks list later than the history start)."""
    from quantlab.data.audit import audit_checks, verdict
    from quantlab.db.database import from_json
    bar_ids, act_ids = snapshot.get("bars", []), snapshot.get("corporate_actions", [])
    bars = ctx.store.load("bars", bar_ids) if bar_ids else None
    actions = ctx.store.load("corporate_actions", act_ids) if act_ids else ctx.store.load("corporate_actions", [])
    if bars is None or bars.empty:
        return {"status": "INCONCLUSIVE", "reason": "no bar datasets recorded for this snapshot", "checks": []}
    q = ",".join("?" for _ in bar_ids)
    feeds = {str((from_json(r["params_json"], {}) or {}).get("feed"))
             for r in ctx.db.fetchall(f"SELECT params_json FROM datasets WHERE dataset_id IN ({q})", bar_ids)}
    checks = audit_checks(bundle, bars, actions, next(iter(feeds)) if len(feeds) == 1 else None,
                          [bundle.market_symbol], requested_start=None, requested_end=None)
    status, reason = verdict(checks, ",".join(sorted(feeds)), synthetic=False, corporate_actions_ok=len(actions) > 0)
    return {"status": status, "reason": reason, "feeds": sorted(feeds),
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in checks]}


def _data_status(db: Database, snapshot: dict[str, list[str]], audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance of the price data used: provider(s), feed(s), synthetic flag, and for real data the
    inline audit verdict. IEX-only volume is a DATA_LIMITATION for liquidity-based rules."""
    ids = snapshot.get("bars", [])
    if audit is not None:
        ref_providers = sorted({r["provider"] for r in db.fetchall(
            "SELECT DISTINCT provider FROM datasets WHERE kind='reference' AND is_synthetic=0")})
        survivorship = ("BIASED: universe drawn from a CURRENT listings snapshot (" + ",".join(ref_providers)
                        + "); securities delisted during the test period are absent, which flatters "
                        "long-only results") if ref_providers else "UNKNOWN"
        return {"status": audit["status"], "reason": audit["reason"], "feeds": audit.get("feeds"),
                "survivorship": survivorship,
                "providers": sorted({r["provider"] for r in db.fetchall(
                    f"SELECT provider FROM datasets WHERE dataset_id IN ({','.join('?' for _ in ids)})", ids)}) if ids else []}
    if not ids:
        return {"status": "UNKNOWN", "note": "no bar dataset ids recorded"}
    q = ",".join("?" for _ in ids)
    rows = db.fetchall(f"SELECT provider, params_json, is_synthetic FROM datasets WHERE dataset_id IN ({q})", ids)
    from quantlab.db.database import from_json
    feeds = sorted({str((from_json(r["params_json"], {}) or {}).get("feed")) for r in rows})
    providers = sorted({r["provider"] for r in rows})
    synthetic = any(r["is_synthetic"] for r in rows)
    status = "SYNTHETIC" if synthetic else ("DATA_LIMITATION" if "iex" in feeds else ("OK" if "sip" in feeds else "UNKNOWN"))
    return {"status": status, "providers": providers, "feeds": feeds}
