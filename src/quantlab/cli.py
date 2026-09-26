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
            kinds = tuple(args.kinds.split(",")) if args.kinds else None
            rep = svc.ingest_all(args.start, args.end, syms, run_id=run_id, kinds=kinds)
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


def cmd_data_audit(args) -> int:
    ctx = _ctx(args)
    from dataclasses import asdict

    from quantlab.data.audit import DEFAULT_SYMBOLS, run_data_audit
    syms = args.symbols.split(",") if args.symbols else list(DEFAULT_SYMBOLS)
    a = run_data_audit(ctx, syms, start=args.start, end=args.end)
    _print({"audit_id": a.audit_id, "status": a.status, "reason": a.reason, "feed": a.feed,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in a.checks]})
    return 0 if a.status in ("SUITABLE_SMALL_SAMPLE_ONLY", "DATA_LIMITATION") else 2


def cmd_walkforward(args) -> int:
    ctx = _ctx(args)
    from quantlab.experiments.report import write_report
    from quantlab.validation.walkforward import run_walk_forward
    if args.data == "real" and not ctx.store.dataset_ids("bars", synthetic=False):
        print("NO REAL DATA: no real bar datasets are stored. Run `data-audit` / `ingest` with Alpaca PAPER keys "
              "first. Synthetic data is never substituted for real evidence.", file=sys.stderr)
        return 2
    run = run_walk_forward(ctx, args.strategy, synthetic=(args.data == "synthetic"), notes=args.notes or "")
    path = write_report(ctx.db, ctx.config, run.experiment_id)
    m = run.result.metrics
    _print({"experiment_id": run.experiment_id, "synthetic": run.is_synthetic, "windows": len(run.windows),
            "oos_trades": m.get("n_trades"), "oos_return_net": m.get("total_return_net"),
            "oos_return_gross": m.get("total_return_gross"), "expectancy": m.get("expectancy"),
            "sharpe": m.get("sharpe"), "max_drawdown": m.get("max_drawdown"),
            "benchmark_return": (m.get("benchmark") or {}).get("benchmark_total_return"),
            "verdict": run.inference["verdict"], "stability": run.stability, "report": str(path)})
    return 0


def cmd_dashboard(args) -> int:
    import uvicorn

    from quantlab.dashboard.app import create_app
    ctx = _ctx(args)
    host = ctx.config.get("dashboard.host", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost") and not args.allow_remote:
        print("refusing to bind the dashboard to a non-local address without --allow-remote", file=sys.stderr)
        return 2
    uvicorn.run(create_app(ctx), host=host, port=args.port or int(ctx.config.get("dashboard.port", 8765)))
    return 0


def cmd_paper(args) -> int:
    """Alpaca PAPER runner: preflight | start | status | stop | order-test."""
    ctx = _ctx(args)
    from quantlab.pipeline import runner as rn
    if args.action == "preflight":
        from quantlab.execution.preflight import run_preflight
        pre = run_preflight(ctx.config, ctx.db)
        d = pre.to_dict()
        d.pop("open_order_client_ids", None)
        _print(d)
        return 0 if pre.ok else 2
    if args.action == "status":
        from quantlab.monitoring.killswitch import KillSwitch
        st = rn.runner_status(ctx.db, float(ctx.config.get("paper.runner.stale_heartbeat_seconds", 120)))
        state, reason, _ = KillSwitch(ctx.db).state()
        sess = st["session"] or {}
        detail = sess.get("detail") or {}
        elig = rn.paper_eligible_strategies(ctx.db)
        _print({"runner": st["state"], "session_id": sess.get("session_id"), "started_at": sess.get("started_at"),
                "heartbeat_age_seconds": st.get("heartbeat_age_seconds"), "phase": sess.get("phase"),
                "stream": sess.get("stream_status"), "system_state": state.value, "system_reason": reason,
                "paper_eligible_strategies": elig or "NO PAPER-ELIGIBLE STRATEGY",
                "broker": detail.get("broker"), "plan": detail.get("plan"), "last_job": detail.get("last_job"),
                "recent_events": ctx.db.fetchall("SELECT at, level, kind, message FROM paper_runner_events "
                                                 "ORDER BY id DESC LIMIT 8")})
        return 0
    if args.action == "stop":
        ids = rn.request_stop(ctx.db, args.reason or "operator stop (quantlab paper stop)")
        print(f"stop requested for {ids}" if ids else "no RUNNING paper runner session")
        return 0
    if args.action == "order-test":
        if not args.confirm:
            print("order-test submits ONE non-marketable paper limit order (1 share at half the last close) and "
                  "cancels it (or with --round-trip: a market BUY then SELL that fill), to verify the order + "
                  "trade_updates path. It is not a strategy order. Re-run with --confirm.", file=sys.stderr)
            return 2
        if args.round_trip:
            _print(rn.round_trip_order_test(ctx, args.symbol, args.qty))
        else:
            _print(rn.connectivity_order_test(ctx, args.symbol))
        return 0
    # start (blocking)
    runner = rn.PaperRunner(ctx)
    try:
        reason = runner.run_forever()
    except rn.RunnerRefused as exc:
        print(f"PAPER RUNNER REFUSED TO START: {exc}", file=sys.stderr)
        return 2
    print(f"paper runner stopped: {reason}")
    return 0


def cmd_discover(args) -> int:
    """Market discovery for one session (research only; reads recorded decisions, never trades)."""
    ctx = _ctx(args)
    from quantlab.data.validation import quarantine_map
    from quantlab.discovery import run_discovery
    syn = _data_flag(ctx, args.data)
    bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=syn),
                                   synthetic=syn)
    as_of = args.as_of or bundle.panel.dates[-1]
    dr = run_discovery(ctx, bundle, as_of, quarantine=quarantine_map(ctx.db))
    a = dr.assessment
    _print({"discovery_run_id": dr.discovery_run_id, "as_of": str(dr.scan.as_of.date()), "synthetic": syn,
            "score_meaning": "discovery score 0-100 = cross-sectional ranking, NOT a probability of profit",
            "funnel": {k: v for k, v in a.funnel.items() if not isinstance(v, list)},
            "why_no_paper_trades": {k: a.blockers[k] for k in ("by_stage", "main", "main_text", "n_discovered",
                                                              "n_eligible")},
            "top": [{"symbol": c["symbol"], "score": round(c["score"], 1) if c["score"] is not None else None,
                     "coverage": c["coverage"], "families": c["fired"], "status": c["status"],
                     "block": c["block_reason"]} for c in a.candidates[:args.top]],
            "coverage": [{"family": r["label"], "coverage": round(r["coverage"], 4), "usable": r["usable"],
                          "role": r["role"], "pit_status": r["pit_status"]} for r in dr.scan.coverage],
            "diagnostics": [f"{x['level']} {x['code']}: {x['message']}" for x in a.diagnostics],
            "outcomes_written": dr.outcomes_written})
    return 0


def cmd_discovery_research(args) -> int:
    """Forward-outcome research on point-in-time discovery replays (research only; never changes rules)."""
    ctx = _ctx(args)
    from quantlab.discovery.research import run_research
    syn = _data_flag(ctx, args.data)
    bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=syn),
                                   synthetic=syn)
    res = run_research(ctx, bundle, args.start, args.end, every=args.every, min_obs=args.min_obs,
                       min_dates=args.min_dates)
    _print({"research_id": res["research_id"], **res["meta"], "report": res["report"],
            "verdicts": {g["group"]: g["verdict"] for g in res["summary"]["groups"]}})
    return 0


def cmd_next_session(args) -> int:
    """OVERNIGHT / NEXT-SESSION mode: scan (end of day) | refresh (overnight catalysts) | preopen | status."""
    ctx = _ctx(args)
    from quantlab.discovery import nextsession as ns
    now = args.now
    if args.action == "scan":
        from quantlab.data.validation import quarantine_map
        from quantlab.discovery import run_discovery
        syn = _data_flag(ctx, args.data)
        bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=syn),
                                       synthetic=syn)
        ms = ns.market_state(now)
        as_of = args.as_of or ms.get("last_completed_session") or bundle.panel.dates[-1]
        if str(as_of) not in {str(d.date()) for d in bundle.panel.dates}:
            print(f"NO BAR FOR {as_of}: the latest stored session is {bundle.panel.dates[-1].date()}. "
                  "Ingest it first; the scan never substitutes older data.", file=sys.stderr)
            return 2
        dr = run_discovery(ctx, bundle, as_of, quarantine=quarantine_map(ctx.db))
        f = dr.assessment.funnel
        _print({"discovery_run_id": dr.discovery_run_id, "decision_session": str(dr.scan.as_of.date()),
                "next_session": (dr.assessment.candidates[0]["setup"] if dr.assessment.candidates else {}) and
                ns.market_state(now).get("next_session"), "funnel": {k: v for k, v in f.items() if not isinstance(v, list)}})
        return 0
    if args.action == "refresh":
        _print(ns.overnight_refresh(ctx, now))
        return 0
    if args.action == "preopen":
        _print(ns.preopen_recheck(ctx, now))
        return 0
    st = ns.next_session_state(ctx, now, top=args.top)
    _print({"market": st["market"], "run": st["run"], "counts": st["counts"], "alerts": st["alerts"],
            "pipeline": st["pipeline"],
            "top": [{"symbol": t["symbol"], "score": t["score"], "origin": t["origin"], "status": t["status"],
                     "setup": t["setup"].get("setup_type"), "confirm": t["setup"].get("confirm"),
                     "invalidate": t["setup"].get("invalidate"), "paper": t["setup"].get("paper")} for t in st["top"]]})
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
    s.add_argument("--kinds", help="comma-separated subset of reference,bars,corporate_actions,events,fundamentals,news")
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

    s = sub.add_parser("data-audit", help="audit REAL data suitability on a small sample (needs PAPER keys + SEC UA)")
    s.add_argument("--symbols", help="comma-separated (default SPY,XLK,AAPL,MSFT)")
    s.add_argument("--start", default="2020-01-01"), s.add_argument("--end")
    s.set_defaults(fn=cmd_data_audit)

    s = sub.add_parser("walkforward", help="walk-forward OOS evaluation of one FIXED strategy")
    s.add_argument("--strategy", required=True)
    s.add_argument("--data", choices=["real", "synthetic"], required=True,
                   help="explicit: synthetic results validate machinery only, never a strategy")
    s.add_argument("--notes")
    s.set_defaults(fn=cmd_walkforward)

    s = sub.add_parser("paper", help="Alpaca PAPER runner: preflight | start | status | stop | order-test")
    s.add_argument("action", choices=["preflight", "start", "status", "stop", "order-test"])
    s.add_argument("--reason", help="stop: reason recorded with the stop request")
    s.add_argument("--symbol", default="SPY", help="order-test: symbol (default SPY)")
    s.add_argument("--confirm", action="store_true", help="order-test: really submit (and cancel) the test order")
    s.add_argument("--round-trip", action="store_true",
                   help="order-test: market BUY then SELL (fills; market hours only; before the runner binds the account)")
    s.add_argument("--qty", type=float, default=1, help="order-test --round-trip: shares (default 1)")
    s.set_defaults(fn=cmd_paper)

    s = sub.add_parser("discover", help="market discovery for one session (research only, never trades)")
    s.add_argument("--as-of"), s.add_argument("--top", type=int, default=15)
    s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
    s.set_defaults(fn=cmd_discover)

    s = sub.add_parser("discovery-research", help="forward outcomes of point-in-time discovery replays (research only)")
    s.add_argument("--start", required=True), s.add_argument("--end", required=True)
    s.add_argument("--every", type=int, default=5, help="sample every N sessions (default 5)")
    s.add_argument("--min-obs", type=int, default=200), s.add_argument("--min-dates", type=int, default=30)
    s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
    s.set_defaults(fn=cmd_discovery_research)

    s = sub.add_parser("next-session", help="overnight / next-session mode: scan | refresh | preopen | status")
    s.add_argument("action", choices=["scan", "refresh", "preopen", "status"])
    s.add_argument("--now", help="observation time (ISO, UTC if no offset); default = wall clock. For dry runs/tests")
    s.add_argument("--as-of", help="scan: decision session (default: last completed session)")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--data", choices=["auto", "synthetic", "real"], default="auto")
    s.set_defaults(fn=cmd_next_session)

    s = sub.add_parser("dashboard", help="serve the read-only dashboard on localhost")
    s.add_argument("--port", type=int), s.add_argument("--allow-remote", action="store_true")
    s.set_defaults(fn=cmd_dashboard)

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
