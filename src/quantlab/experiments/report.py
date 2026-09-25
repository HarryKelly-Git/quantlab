"""Human-readable experiment report (Markdown), rendered from the stored experiment record so any
past experiment can be re-rendered exactly. Never reports total return alone, always states the
sample-size verdict, data provenance, PIT status and whether the data were SYNTHETIC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from quantlab.db.database import Database, from_json, utcnow_iso
from quantlab.experiments.registry import ExperimentRegistry


def _pct(x: Any, digits: int = 2) -> str:
    return "UNKNOWN" if x is None else f"{x * 100:.{digits}f}%"


def _num(x: Any, digits: int = 2) -> str:
    return "UNKNOWN" if x is None else f"{x:.{digits}f}"


def render_backtest_markdown(exp: dict[str, Any]) -> str:
    res = exp["results"][-1] if exp.get("results") else None
    m = (res or {}).get("metrics", {})
    met, inf = m.get("metrics", {}), m.get("inference", {})
    synthetic = bool(exp.get("uses_synthetic_data"))
    lines = [f"# Experiment report — {exp['name']}", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA — NOT MARKET EVIDENCE.** This run validates the machinery only.", ""]
    if exp.get("touches_holdout"):
        lines += ["> **LOCKED HOLDOUT WAS ACCESSED** for this experiment (see holdout_access_log).", ""]
    lines += [
        f"- Experiment: `{exp['experiment_id']}` ({exp['kind']}), created {exp['created_at']}",
        f"- Status: **{(res or {}).get('status', 'NOT FINISHED')}** — verdict: **{(res or {}).get('conclusion') or 'n/a'}**",
        f"- Period: {exp.get('period_start')} → {exp.get('period_end')}",
        f"- Git: `{exp.get('git_commit')}` (dirty={bool(exp.get('git_dirty'))}); config hash `{exp['config_hash']}`",
        f"- Strategies: {from_json(exp.get('strategy_versions_json'), {})}",
        f"- Variants tested (for deflation): {exp.get('n_variants_tested')}",
        f"- Weakest point-in-time status of inputs: **{m.get('pit_status', 'UNKNOWN')}**",
        f"- Survivorship: {m.get('survivorship', {}).get('status', 'UNKNOWN')} — {m.get('survivorship', {}).get('note', '')}",
        "",
    ]
    if res and res.get("status") == "failed":
        lines += ["## Failure", "", f"```\n{m.get('error')}\n```", ""]
        return "\n".join(lines)
    lines += [
        "## Performance (MODEL_OUTPUT, net of modeled costs)", "",
        "| metric | value | metric | value |", "|---|---|---|---|",
        f"| total return (net) | {_pct(met.get('total_return_net'))} | total return (gross) | {_pct(met.get('total_return_gross'))} |",
        f"| CAGR (net) | {_pct(met.get('cagr_net'))} | annual vol | {_pct(met.get('ann_vol'))} |",
        f"| Sharpe | {_num(met.get('sharpe'))} | Sortino | {_num(met.get('sortino'))} |",
        f"| max drawdown | {_pct(met.get('max_drawdown'))} | Calmar | {_num(met.get('calmar'))} |",
        f"| trades | {met.get('n_trades', 0)} | win rate | {_pct(met.get('win_rate'), 1)} |",
        f"| avg win | {_pct(met.get('avg_win'))} | avg loss | {_pct(met.get('avg_loss'))} |",
        f"| expectancy / trade | {_pct(met.get('expectancy'), 3)} | profit factor | {_num(met.get('profit_factor'))} |",
        f"| avg holding (sessions) | {_num(met.get('avg_holding_sessions'), 1)} | avg exposure | {_pct(met.get('avg_exposure'), 1)} |",
        f"| P&L share of top 5 trades | {_pct(met.get('pnl_share_top5'), 1)} | exit reasons | {met.get('exit_reasons', {})} |",
        f"| total costs ($) | {_num(met.get('total_costs'), 0)} | annual turnover | {_num(met.get('turnover_annual'))} |",
        "",
    ]
    b = met.get("benchmark") or {}
    if b:
        lines += ["### Versus benchmark (buy & hold SPY, same period)", "",
                  f"- Benchmark total return: {_pct(b.get('benchmark_total_return'))}; excess: {_pct(b.get('excess_total_return'))}",
                  f"- Beta: {_num(b.get('beta'))}; information ratio: {_num(b.get('information_ratio'))}", ""]
    lines += [
        "## Statistical evidence (MODEL_OUTPUT)", "",
        f"- Mean net return per trade: {_pct(inf.get('mean_trade_net_ret'), 3)}, 95% stationary-bootstrap CI "
        f"[{_pct((inf.get('mean_trade_ci') or [None, None])[0], 3)}, {_pct((inf.get('mean_trade_ci') or [None, None])[1], 3)}], "
        f"one-sided p = {_num(inf.get('mean_trade_pvalue'), 4)} ({inf.get('bootstrap_status')})",
        f"- Probabilistic Sharpe vs 0: {_num((inf.get('psr_vs_zero') or {}).get('psr'), 3)}",
        f"- Deflated Sharpe (n_trials={inf.get('n_variants_tested')}): "
        f"{_num((inf.get('deflated_sharpe') or {}).get('dsr'), 3)} — significant: {(inf.get('deflated_sharpe') or {}).get('significant')}",
        f"- Sample: {inf.get('n_trades')} trades vs minimum {inf.get('min_trades_for_conclusion')} for any conclusion",
        f"- **Verdict: {inf.get('verdict')}**",
        "",
    ]
    windows = m.get("windows") or []
    if windows:
        stab = m.get("stability") or {}
        ds = m.get("data_status") or {}
        lines += ["## Walk-forward windows (out-of-sample; fixed parameters, no search)", "",
                  f"- Data status: **{ds.get('status', 'UNKNOWN')}** (providers {ds.get('providers')}, feeds {ds.get('feeds')})",
                  f"- Survivorship: {ds.get('survivorship', 'UNKNOWN')}",
                  f"- Windows positive: {stab.get('windows_positive')}/{stab.get('n_windows')}; beating SPY: "
                  f"{stab.get('windows_beating_benchmark')}/{stab.get('n_windows')}; worst window {_pct(stab.get('worst_window_return'))}",
                  "",
                  "| # | train | test | OOS trades | OOS return | SPY | excess | IS return (reference) |",
                  "|---|---|---|---|---|---|---|---|"]
        for w in windows:
            lines.append(f"| {w['index']} | {w['train_start']}..{w['train_end']} | {w['test_start']}..{w['test_end']} | "
                         f"{w['oos_trades']} | {_pct(w['oos_return'])} | {_pct(w.get('benchmark_return'))} | "
                         f"{_pct(w.get('excess_return'))} | {_pct(w.get('is_return'))} |")
        lines.append("")
    ic = m.get("information_content") or []
    if ic:
        lines += ["## Signal information content (forward excess return of all signals, next-open entry)", "",
                  "| strategy | horizon | signals | mean excess | hit rate | mean IC | t (per-date) |", "|---|---|---|---|---|---|---|"]
        for r in ic:
            lines.append(f"| {r['strategy_id']} | {r['horizon']} | {r['n_signals']} | {_pct(r.get('mean_excess'), 3)} | "
                         f"{_pct(r.get('hit_rate'), 1)} | {_num(r.get('mean_ic'), 3)} | {_num(r.get('t_stat_dates'))} |")
        lines.append("")
    val = m.get("validation") or []
    if val:
        lines += ["## Data validation", ""]
        lines += [f"- {'PASS' if v['passed'] else 'FAIL'} [{v['severity']}] {v['check']}{': ' + v['reason'] if v['reason'] else ''}"
                  for v in val]
        if m.get("quarantined"):
            lines.append(f"- Quarantined symbols: {len(m['quarantined'])}")
        lines.append("")
    diag = m.get("diagnostics") or {}
    if diag:
        lines += ["## Engine diagnostics", "", ", ".join(f"{k}={v}" for k, v in diag.items()), ""]
    yearly = met.get("by_year") or []
    if yearly:
        lines += ["## By year", "", "| year | return | benchmark | trades |", "|---|---|---|---|"]
        for y in yearly:
            lines.append(f"| {y.get('year')} | {_pct(y.get('return'))} | {_pct(y.get('benchmark_return'))} | {y.get('n_trades', '')} |")
        lines.append("")
    return "\n".join(lines)


def write_report(db: Database, config, experiment_id: str) -> Path:
    exp = ExperimentRegistry(db, config).get(experiment_id)
    if exp is None:
        raise KeyError(experiment_id)
    md = render_backtest_markdown(exp)
    out_dir = config.path("project.report_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"experiment_{experiment_id}.md"
    path.write_text(md, encoding="utf-8")
    db.insert("reports", {"report_id": f"rep_{experiment_id}", "run_id": experiment_id,
                          "as_of_date": exp.get("period_end") or "", "kind": "experiment", "path": str(path),
                          "markdown": md, "created_at": utcnow_iso()}, or_ignore=True)
    return path
