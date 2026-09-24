"""QuantLab command line. Paper trading only — there is no live-trading command, flag or mode.

    python -m quantlab.cli --help
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from quantlab.context import AppContext


def _ctx(args) -> AppContext:
    from quantlab.config import load_config
    return AppContext.create(load_config(args.config) if args.config else None)


def _data_flag(ctx: AppContext, choice: str) -> bool:
    """'synthetic' | 'real' | 'auto' (real if any real bars were ingested, else synthetic)."""
    if choice == "synthetic":
        return True
    if choice == "real":
        return False
    return not bool(ctx.store.dataset_ids("bars", synthetic=False))


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# -- commands --------------------------------------------------------------------------------------
def cmd_init(args) -> int:
    ctx = _ctx(args)
    from quantlab.strategies.registry import build_strategies, register_strategies, unimplemented
    n = register_strategies(ctx.db, build_strategies(ctx.config, include_disabled=True))
    _print({"db": str(ctx.config.path("project.db_path")), "tables": len(ctx.db.tables()),
            "strategies_registered": n, "configured_but_not_implemented": unimplemented(ctx.config)})
    return 0


def cmd_ingest(args) -> int:
    ctx = _ctx(args)
    from quantlab.data.ingest import IngestionService
    svc = IngestionService(ctx.config, ctx.store, ctx.db)
    run_id = ctx.start_run("ingest", notes="synthetic" if args.synthetic else "providers")
    try:
        if args.synthetic:
            from quantlab.data.providers.synthetic import SyntheticSpec
            rep = svc.ingest_synthetic(SyntheticSpec(n_stocks=args.n_stocks, start=args.start or "2016-01-04",
                                                     end=args.end or "2024-12-31", seed=args.seed,
                                                     momentum_edge=args.momentum_edge, pead_edge=args.pead_edge))
        else:
            syms = args.symbols.split(",") if args.symbols else None
            rep = svc.ingest_all(args.start, args.end, syms, run_id=run_id)
        ctx.finish_run(run_id, "succeeded" if rep.ok else "failed")
        _print({"synthetic": rep.is_synthetic, "ok": rep.ok, "kinds": rep.summary()})
        return 0 if rep.ok else 2
    except Exception as exc:
        ctx.finish_run(run_id, "failed", repr(exc))
        raise


def cmd_validate(args) -> int:
    ctx = _ctx(args)
    from quantlab.data.validation import DataValidator
    syn = _data_flag(ctx, args.data)
    bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=syn), synthetic=syn)
    v = DataValidator(ctx.config, ctx.db)
    rep = v.check_bundle(bundle, expected_last_session=args.expected_last)
    v.record(rep)
    _print({"synthetic": syn, "ok": rep.ok, "quarantined": len(rep.quarantined),
            "checks": [{"check": c.name, "passed": c.passed, "severity": c.severity.value, "reason": c.reason} for c in rep.checks]})
    return 0 if rep.ok else 2


def cmd_universe(args) -> int:
    ctx = _ctx(args)
    from quantlab.universe import UniverseEngine
    syn = _data_flag(ctx, args.data)
    bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=syn), synthetic=syn)
    eng = UniverseEngine(ctx.config)
    as_of = args.as_of or bundle.panel.dates[-1]
    df = eng.explain(bundle, as_of)
    _print({"as_of": str(as_of)[:10], "included": int(df["included"].sum()), "total": len(df),
            "exclusion_reasons": df.loc[~df["included"], "reason"].value_counts().to_dict(),
            "survivorship": eng.survivorship_status(bundle)})
    return 0


def cmd_backtest(args) -> int:
    ctx = _ctx(args)
    from quantlab.experiments.report import write_report
    from quantlab.experiments.runner import run_strategy_backtest
    syn = _data_flag(ctx, args.data)
    run = run_strategy_backtest(ctx, args.strategy or None, args.start, args.end, synthetic=syn,
                                holdout_token=args.holdout_token, n_variants_tested=args.variants, notes=args.notes or "")
    path = write_report(ctx.db, ctx.config, run.experiment_id)
    m = run.result.metrics
    _print({"experiment_id": run.experiment_id, "synthetic": run.is_synthetic, "labels": run.labels,
            "period": [run.result.params["start"], run.result.params["end"]], "trades": m.get("n_trades"),
            "total_return_net": m.get("total_return_net"), "sharpe": m.get("sharpe"), "max_drawdown": m.get("max_drawdown"),
            "verdict": run.inference["verdict"], "pit_status": run.pit_status.value, "report": str(path)})
    return 0


def cmd_experiments(args) -> int:
    ctx = _ctx(args)
    from quantlab.experiments.registry import ExperimentRegistry
    from quantlab.experiments.report import render_backtest_markdown
    reg = ExperimentRegistry(ctx.db, ctx.config)
    if args.show:
        exp = reg.get(args.show)
        if exp is None:
            print(f"unknown experiment {args.show}", file=sys.stderr)
            return 1
        print(render_backtest_markdown(exp))
        return 0
    _print(reg.list(limit=args.limit))
    return 0


def cmd_holdout_unlock(args) -> int:
    ctx = _ctx(args)
    from quantlab.validation.holdout import HoldoutGuard
    token = HoldoutGuard(ctx.config, ctx.db).unlock(args.reason, args.actor, args.start, args.end, args.experiment)
    print("Single-use holdout token (logged permanently; pass with --holdout-token):")
    print(token)
    return 0


def cmd_status(args) -> int:
    ctx = _ctx(args)
    from quantlab.monitoring.killswitch import KillSwitch
    from quantlab.validation.holdout import HoldoutGuard
    state, reason, changed = KillSwitch(ctx.db).state()
    _print({"system_state": state.value, "reason": reason, "changed_at": changed,
            "holdout": HoldoutGuard(ctx.config, ctx.db).describe(),
            "datasets": {k: len(v) for k, v in ctx.store.snapshot().items()},
            "experiments": ctx.db.fetchone("SELECT COUNT(*) AS n FROM experiments")["n"],
            "paper_only": True})
    return 0


def cmd_pause(args) -> int:
    from quantlab.monitoring.killswitch import KillSwitch
    changed = KillSwitch(_ctx(args).db).pause(args.reason, trigger="manual", actor=args.actor)
    print("SYSTEM_PAUSED" + ("" if changed else " (already paused)"))
    return 0


def cmd_resume(args) -> int:
    from quantlab.monitoring.killswitch import KillSwitch
    changed = KillSwitch(_ctx(args).db).resume(args.reason, args.actor)
    print("ACTIVE" + ("" if changed else " (was not paused)"))
    return 0


def cmd_run_daily(args) -> int:
    ctx = _ctx(args)
    from quantlab.pipeline.daily import DailyPipeline
    res = DailyPipeline(ctx, synthetic=_data_flag(ctx, args.data)).run(args.as_of, resume_run_id=args.resume)
    _print({"run_id": res.run_id, "as_of": res.as_of, "system_state": res.system_state, "steps": res.steps,
            "errors": res.errors, "report": res.report_path})
    return 1 if res.errors else 0


def cmd_replay(args) -> int:
    ctx = _ctx(args)
    from quantlab.pipeline.daily import replay
    results = replay(ctx, args.start, args.end, synthetic=_data_flag(ctx, args.data))
    _print({"sessions": len(results), "failed": [r.as_of for r in results if r.errors],
            "last_state": results[-1].system_state if results else None,
            "orders": ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders WHERE book='BOT'")["n"],
            "closed_trades": ctx.db.fetchone("SELECT COUNT(*) AS n FROM trades WHERE book='BOT' AND status='CLOSED'")["n"]})
    return 0


def cmd_strategy(args) -> int:
    """Show strategy stages/status, evaluate promotion evidence, or change stage/status (logged)."""
    ctx = _ctx(args)
    from quantlab.research.promotion import PromotionManager
    from quantlab.strategies.registry import build_strategies, register_strategies
    register_strategies(ctx.db, build_strategies(ctx.config, include_disabled=True))
    pm = PromotionManager(ctx.db, ctx.config)
    if args.action == "list":
        _print(ctx.db.fetchall("SELECT strategy_id, version, family, status, stage FROM strategies ORDER BY strategy_id"))
        return 0
    ver = args.version or ctx.config.get(f"strategies.{args.id}.version")
    if args.action == "evaluate":
        _print(pm.evaluate(args.id, ver).to_dict())
    elif args.action == "promote":
        _print(pm.transition(args.id, ver, args.to_stage, args.reason, args.actor, override_reason=args.override))
    elif args.action == "status":
        _print({"changed": pm.set_status(args.id, ver, args.to_status, args.reason, actor=args.actor)})
    return 0


def cmd_demo(args) -> int:
    """Offline end-to-end run on SYNTHETIC data (planted momentum edge by default)."""
    ctx = _ctx(args)
    from quantlab.data.ingest import IngestionService
    from quantlab.data.providers.synthetic import SyntheticSpec
    from quantlab.experiments.report import write_report
    from quantlab.experiments.runner import run_strategy_backtest
    if not ctx.store.dataset_ids("bars", synthetic=True):
        IngestionService(ctx.config, ctx.store, ctx.db).ingest_synthetic(
            SyntheticSpec(n_stocks=args.n_stocks, momentum_edge=args.momentum_edge, seed=args.seed))
    run = run_strategy_backtest(ctx, ["momentum_trend"], synthetic=True, name="demo:momentum_trend (synthetic)")
    path = write_report(ctx.db, ctx.config, run.experiment_id)
    print(f"SYNTHETIC demo complete. Verdict: {run.inference['verdict']}. Report: {path}")
    return 0


# -- parser ----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantlab", description="QuantLab research lab + PAPER trader (no live money).")
    p.add_argument("--config", help="extra YAML config layered over config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create/migrate the database and register strategies").set_defaults(fn=cmd_init)

    s = sub.add_parser("ingest", help="ingest data from configured providers (or --synthetic)")
    s.add_argument("--synthetic", action="store_true", help="write the SYNTHETIC test world (never market evidence)")
    s.add_argument("--start"), s.add_argument("--end"), s.add_argument("--symbols", help="comma-separated")
    s.add_argument("--n-stocks", type=int, default=120), s.add_argument("--seed", type=int, default=42)
    s.add_argument("--momentum-edge", type=float, default=0.0), s.add_argument("--pead-edge", type=float, default=0.0)
    s.set_defaults(fn=cmd_ingest)

    for name, fn, hlp in [("validate", cmd_validate, "run data integrity checks"),
                          ("universe", cmd_universe, "explain universe membership")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
        if name == "validate":
            s.add_argument("--expected-last", help="session date the data should reach (staleness check)")
        else:
            s.add_argument("--as-of")
        s.set_defaults(fn=fn)

    s = sub.add_parser("backtest", help="run a registered backtest experiment + report")
    s.add_argument("--strategy", action="append", help="strategy id (repeatable); default = all enabled+implemented")
    s.add_argument("--start"), s.add_argument("--end")
    s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
    s.add_argument("--holdout-token", help="single-use token from `holdout-unlock` (only to touch the holdout)")
    s.add_argument("--variants", type=int, default=1, help="number of variants tried (for deflated Sharpe)")
    s.add_argument("--notes")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("experiments", help="list experiments or show one as a report")
    s.add_argument("--show"), s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_experiments)

    s = sub.add_parser("holdout-unlock", help="issue a single-use, permanently logged holdout token")
    s.add_argument("--reason", required=True), s.add_argument("--actor", required=True)
    s.add_argument("--start", required=True), s.add_argument("--end", required=True), s.add_argument("--experiment")
    s.set_defaults(fn=cmd_holdout_unlock)

    sub.add_parser("status", help="system state, holdout access count, datasets").set_defaults(fn=cmd_status)
    s = sub.add_parser("pause", help="SYSTEM_PAUSED: block all new paper orders")
    s.add_argument("--reason", required=True), s.add_argument("--actor", default="human")
    s.set_defaults(fn=cmd_pause)
    s = sub.add_parser("resume", help="resume after human review (actor must be human[:name])")
    s.add_argument("--reason", required=True), s.add_argument("--actor", required=True)
    s.set_defaults(fn=cmd_resume)

    s = sub.add_parser("run-daily", help="run the daily PAPER pipeline for one session (default: latest)")
    s.add_argument("--as-of"), s.add_argument("--resume", help="run_id of a failed run to resume")
    s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
    s.set_defaults(fn=cmd_run_daily)

    s = sub.add_parser("replay", help="run the daily PAPER pipeline over past sessions (forward simulation)")
    s.add_argument("--start", required=True), s.add_argument("--end", required=True)
    s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("strategy", help="list / evaluate / promote / set status (all changes logged)")
    s.add_argument("action", choices=["list", "evaluate", "promote", "status"])
    s.add_argument("--id"), s.add_argument("--version"), s.add_argument("--to-stage"), s.add_argument("--to-status")
    s.add_argument("--reason", default=""), s.add_argument("--actor", default="human")
    s.add_argument("--override", help="human override reason when evidence requirements are unmet (logged)")
    s.set_defaults(fn=cmd_strategy)

    s = sub.add_parser("demo", help="offline end-to-end demo on SYNTHETIC data")
    s.add_argument("--n-stocks", type=int, default=120), s.add_argument("--seed", type=int, default=42)
    s.add_argument("--momentum-edge", type=float, default=0.0015)
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
