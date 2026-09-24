"""Daily report (Markdown). Every section labels what kind of information it shows:
FACT (data), MODEL_OUTPUT (computed), AI_OPINION (LLM), UNCERTAINTY (unknown)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from quantlab.context import AppContext
from quantlab.db.database import from_json, utcnow_iso
from quantlab.monitoring.killswitch import KillSwitch


def _pct(x: Any) -> str:
    return "UNKNOWN" if x is None else f"{float(x) * 100:.2f}%"


def _money(x: Any) -> str:
    return "UNKNOWN" if x is None else f"${float(x):,.0f}"


def render_daily_report(ctx: AppContext, run_id: str, st: dict[str, Any]) -> str:
    db = ctx.db
    d = st["as_of"]
    view = st.get("view")
    synthetic = bool(view is not None and view.is_synthetic)
    state, reason, _ = KillSwitch(db).state()
    lines = [f"# QuantLab daily report — {d.date()}", "", f"Run `{run_id}` · generated {utcnow_iso()} · PAPER TRADING ONLY", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA — NOT MARKET EVIDENCE.**", ""]
    if state.value != "ACTIVE":
        lines += [f"> **SYSTEM PAUSED** — {reason}. No new paper orders until a human resumes.", ""]

    # MARKET
    rg = st.get("regime") or {}
    lines += ["## Market (MODEL_OUTPUT)", "",
              f"- Regime: **{rg.get('label', 'unknown')}**; SPY vs 200d MA {_pct(rg.get('market_trend_200'))}; "
              f"60d momentum {_pct(rg.get('market_mom_60'))}; 20d vol {_pct(rg.get('market_vol_20'))}; "
              f"drawdown {_pct(rg.get('market_drawdown'))}; breadth>50d {_pct(rg.get('breadth_50'))}", ""]

    # OPPORTUNITIES / DECISIONS
    decisions = st.get("decisions") or []
    trades = [x for x in decisions if x["outcome"].decision.value == "TRADE"]
    lines += ["## Top opportunities (MODEL_OUTPUT; score = strategy ranking, not a probability)", ""]
    cands = sorted(st.get("candidates") or [], key=lambda c: -c.score)[:15]
    if cands:
        lines += ["| symbol | strategy | score | entry ref | stop | hold | bot decision |", "|---|---|---|---|---|---|---|"]
        dec_by = {x["candidate"].candidate_id: x["outcome"] for x in decisions}
        for c in cands:
            o = dec_by.get(c.candidate_id)
            lines.append(f"| {c.symbol} | {c.strategy_id} | {c.score:.3f} | {c.plan.entry_ref_price or 0:.2f} | "
                         f"{(c.plan.stop_price or 0):.2f} | {c.plan.holding_sessions} | "
                         f"{o.decision.value + ' (' + o.reject_stage.value + ')' if o else 'n/a'} |")
    else:
        lines.append("- No strategy produced a candidate today.")
    lines.append("")

    # BOT ACTIONS
    lines += ["## Bot paper actions", ""]
    orders = db.fetchall("SELECT symbol, side, qty, purpose, status FROM orders WHERE book='BOT' AND created_at >= ? "
                         "ORDER BY created_at", (utcnow_iso()[:10],))
    lines += [f"- {o['purpose']} {o['side']} {o['qty']:g} {o['symbol']} — {o['status']}" for o in orders] or ["- No new paper orders."]
    lines += [f"- Decisions: {dict(Counter(x['outcome'].decision.value for x in decisions))}", ""]

    # REJECTIONS
    lines += ["## Rejections (why we did NOT trade)", ""]
    stages = Counter(x["outcome"].reject_stage.value for x in decisions if x["outcome"].decision.value != "TRADE")
    for stage, n in stages.most_common():
        example = next(x for x in decisions if x["outcome"].reject_stage.value == stage)
        lines.append(f"- **{stage}**: {n} — e.g. {example['candidate'].symbol}/{example['candidate'].strategy_id}: "
                     f"{'; '.join(example['outcome'].reasons)[:220]}")
    if not stages:
        lines.append("- none")
    lines.append("")

    # POSITIONS
    mtm = st.get("mtm") or {}
    lines += ["## Bot book (FACT: internal paper ledger)", "",
              f"- Equity {_money(mtm.get('equity'))}, cash {_money(mtm.get('cash'))}, gross exposure "
              f"{_money(mtm.get('gross_exposure'))}, drawdown {_pct(mtm.get('drawdown'))}, open positions {mtm.get('positions_count', 0)}"]
    for t in db.fetchall("SELECT symbol, qty, entry_date, entry_price, stop_price, strategy_id FROM trades "
                         "WHERE book='BOT' AND status='OPEN' ORDER BY entry_date"):
        lines.append(f"  - {t['symbol']} {t['qty']:g} @ {t['entry_price'] or 0:.2f} since {t['entry_date']} "
                     f"(stop {t['stop_price'] or 0:.2f}, {t['strategy_id']})")
    lines.append("")

    # SHADOW
    so = db.fetchone("SELECT COUNT(*) AS n, AVG(excess_ret) AS ex FROM shadow_outcomes")
    lines += ["## Shadow book (MODEL_OUTPUT)", "",
              f"- Opportunities recorded: {db.fetchone('SELECT COUNT(*) AS n FROM shadow_opportunities')['n']}; "
              f"matured outcomes: {so['n']}; mean excess return of matured: {_pct(so['ex'])}",
              "- Counterfactual analysis (rejected vs accepted) needs a larger sample before any conclusion.", ""]

    # AI
    lines += ["## AI activity (AI_OPINION)", ""]
    lines.append("- AI layer disabled (ai.enabled=false): AI decisions are UNKNOWN by design." if not ctx.config.get("ai.enabled", False)
                 else "- See ai_calls / ai_assessments for today's reviews.")
    lines.append("")

    # HEALTH
    lines += ["## System health", "", f"- System state: **{state.value}** {('— ' + reason) if reason else ''}"]
    steps = db.fetchall("SELECT step, status FROM pipeline_steps WHERE run_id=? ORDER BY step_order", (run_id,))
    lines.append("- Pipeline steps: " + ", ".join(f"{s['step']}={s['status']}" for s in steps))
    issues = db.fetchall("SELECT check_name, severity, COUNT(*) AS n FROM data_quality_issues WHERE run_id=? "
                         "GROUP BY check_name, severity", (run_id,))
    lines += [f"- Data issue: {i['check_name']} [{i['severity']}] x{i['n']}" for i in issues] or ["- No data-quality issues recorded."]
    lines.append("")
    return "\n".join(lines)


def write_daily_report(ctx: AppContext, run_id: str, st: dict[str, Any]) -> Path:
    md = render_daily_report(ctx, run_id, st)
    out = ctx.config.path("project.report_dir")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"daily_{st['as_of'].date()}_{run_id}.md"
    path.write_text(md, encoding="utf-8")
    ctx.db.insert("reports", {"report_id": f"rep_{run_id}", "run_id": run_id, "as_of_date": str(st["as_of"].date()),
                              "kind": "daily", "path": str(path), "markdown": md, "created_at": utcnow_iso()},
                  or_ignore=True)
    return path
