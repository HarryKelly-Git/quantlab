"""QuantLab dashboard: server-rendered, localhost by default, PAPER ONLY.

Pages read the audit database. The only writes are the kill switch: pause, and resume, which
requires a declared human actor and a reason. There is no order entry and no live-trading
control anywhere.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from quantlab.context import AppContext
from quantlab.db.database import from_json

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _spark(values: list[float], width: int = 520, height: int = 80) -> str:
    """Inline SVG line for an equity curve (no JS, no external assets)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "<em>not enough history</em>"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i * width / (len(vals) - 1):.1f},{height - (v - lo) / span * height:.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="equity curve">'
            f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{pts}"/></svg>'
            f'<div class="muted">min {lo:,.0f} · max {hi:,.0f} · last {vals[-1]:,.0f}</div>')


def create_app(ctx: AppContext | None = None) -> FastAPI:
    ctx = ctx or AppContext.create()
    app = FastAPI(title="QuantLab (paper only)", docs_url=None, redoc_url=None)
    db = ctx.db

    def render(request: Request, name: str, **kw: Any) -> HTMLResponse:
        from quantlab.monitoring.killswitch import KillSwitch
        state, reason, changed = KillSwitch(db).state()
        return TEMPLATES.TemplateResponse(request, name, {"state": state.value, "state_reason": reason,
                                                          "state_changed": changed, **kw})

    def latest_run() -> dict | None:
        return db.fetchone("SELECT * FROM runs WHERE kind='pipeline' ORDER BY started_at DESC LIMIT 1")

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request):
        run = latest_run()
        steps = db.fetchall("SELECT step, status FROM pipeline_steps WHERE run_id=? ORDER BY step_order",
                            (run["run_id"],)) if run else []
        regime = db.fetchone("SELECT * FROM regime_snapshots ORDER BY as_of_date DESC LIMIT 1")
        books = {b: db.fetchone("SELECT * FROM portfolio_snapshots WHERE book=? ORDER BY as_of_date DESC LIMIT 1", (b,))
                 for b in ("BOT", "HUMAN")}
        counts = {t: db.fetchone(f"SELECT COUNT(*) AS n FROM {t}")["n"] for t in
                  ("candidates", "decisions", "shadow_opportunities", "shadow_outcomes", "orders", "trades",
                   "experiments", "human_decisions", "hypotheses")}
        synthetic = db.fetchone("SELECT MAX(is_synthetic) AS s FROM datasets WHERE kind='bars'")
        return render(request, "overview.html", run=run, steps=steps,
                      regime=from_json(regime["metrics_json"], {}) if regime else None, books=books, counts=counts,
                      synthetic=bool(synthetic and synthetic["s"]))

    @app.get("/signals", response_class=HTMLResponse)
    def signals(request: Request):
        """Latest pipeline run: every candidate with the bot's decision and rejection stage."""
        run = latest_run()
        rows = db.fetchall(
            "SELECT c.as_of_date, c.symbol, c.strategy_id, c.score, c.entry_ref_price, c.stop_price, c.holding_sessions, "
            "c.pit_status, d.decision, d.reject_stage, d.reasons_json FROM candidates c "
            "LEFT JOIN decisions d ON d.candidate_id = c.candidate_id WHERE c.run_id = ? ORDER BY d.decision, c.score DESC",
            (run["run_id"],)) if run else []
        for r in rows:
            r["reasons"] = "; ".join(from_json(r.pop("reasons_json"), []) or [])
        orders = db.fetchall("SELECT created_at, book, purpose, symbol, side, qty, status, filled_avg_price "
                             "FROM orders ORDER BY created_at DESC LIMIT 50")
        return render(request, "signals.html", run=run, rows=rows, orders=orders)

    @app.get("/portfolio/{book}", response_class=HTMLResponse)
    def portfolio(request: Request, book: str):
        book = book.upper()
        if book not in ("BOT", "HUMAN"):
            raise HTTPException(404)
        snaps = db.fetchall("SELECT as_of_date, equity, cash, drawdown, positions_count FROM portfolio_snapshots "
                            "WHERE book=? ORDER BY as_of_date", (book,))
        open_trades = db.fetchall("SELECT * FROM trades WHERE book=? AND status='OPEN' ORDER BY entry_date", (book,))
        closed = db.fetchall("SELECT * FROM trades WHERE book=? AND status='CLOSED' ORDER BY exit_date DESC LIMIT 50", (book,))
        return render(request, "portfolio.html", book=book, spark=_spark([s["equity"] for s in snaps]),
                      last=snaps[-1] if snaps else None, open_trades=open_trades, closed=closed)

    @app.get("/trade/{trade_id}", response_class=HTMLResponse)
    def trade(request: Request, trade_id: str):
        from quantlab.execution.journal import TradeJournal
        return render(request, "trade.html", trade_id=trade_id, explain=TradeJournal(db).explain(trade_id))

    @app.get("/strategies", response_class=HTMLResponse)
    def strategies(request: Request):
        rows = db.fetchall("SELECT strategy_id, version, family, status, stage, updated_at FROM strategies ORDER BY strategy_id")
        exps = db.fetchall("SELECT e.experiment_id, e.name, e.created_at, e.uses_synthetic_data, r.conclusion, r.status "
                           "FROM experiments e LEFT JOIN experiment_results r ON r.experiment_id=e.experiment_id "
                           "ORDER BY e.created_at DESC LIMIT 30")
        log = db.fetchall("SELECT * FROM strategy_status_log ORDER BY id DESC LIMIT 30")
        return render(request, "strategies.html", rows=rows, exps=exps, log=log)

    @app.get("/experiments/{experiment_id}", response_class=HTMLResponse)
    def experiment(request: Request, experiment_id: str):
        from quantlab.experiments.registry import ExperimentRegistry
        from quantlab.experiments.report import render_backtest_markdown
        exp = ExperimentRegistry(db, ctx.config).get(experiment_id)
        if exp is None:
            raise HTTPException(404)
        return render(request, "markdown.html", title=exp["name"], markdown=render_backtest_markdown(exp))

    @app.get("/reports", response_class=HTMLResponse)
    def reports(request: Request):
        rows = db.fetchall("SELECT report_id, run_id, as_of_date, kind, created_at FROM reports ORDER BY created_at DESC LIMIT 100")
        return render(request, "reports.html", rows=rows)

    @app.get("/reports/{run_id}", response_class=HTMLResponse)
    def report(request: Request, run_id: str):
        row = db.fetchone("SELECT markdown, as_of_date FROM reports WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,))
        if row is None:
            raise HTTPException(404)
        return render(request, "markdown.html", title=f"Daily report {row['as_of_date']}", markdown=row["markdown"])

    @app.get("/shadow", response_class=HTMLResponse)
    def shadow(request: Request):
        rows = db.fetchall(
            "SELECT s.as_of_date, s.symbol, s.strategy_id, s.bot_decision, s.reject_stage, s.reject_reason, "
            "o.ret, o.excess_ret, o.status FROM shadow_opportunities s LEFT JOIN shadow_outcomes o "
            "ON o.opportunity_id = s.opportunity_id ORDER BY s.as_of_date DESC LIMIT 200")
        summary = db.fetchall(
            "SELECT s.reject_stage, COUNT(*) AS n, COUNT(o.opportunity_id) AS matured, AVG(o.excess_ret) AS mean_excess "
            "FROM shadow_opportunities s LEFT JOIN shadow_outcomes o ON o.opportunity_id = s.opportunity_id "
            "GROUP BY s.reject_stage ORDER BY n DESC")
        return render(request, "shadow.html", rows=rows, summary=summary)

    @app.get("/research", response_class=HTMLResponse)
    def research(request: Request):
        hyps = db.fetchall("SELECT * FROM hypotheses ORDER BY created_at DESC LIMIT 50")
        ledger = db.fetchall("SELECT * FROM research_ledger ORDER BY entry_id DESC LIMIT 50")
        notes = db.fetchall("SELECT * FROM human_notes ORDER BY created_at DESC LIMIT 50")
        holdout = db.fetchall("SELECT * FROM holdout_access_log ORDER BY id DESC")
        return render(request, "research.html", hyps=hyps, ledger=ledger, notes=notes, holdout=holdout)

    @app.get("/system", response_class=HTMLResponse)
    def system(request: Request):
        runs = db.fetchall("SELECT run_id, kind, as_of_date, status, started_at, git_commit, git_dirty, error "
                           "FROM runs ORDER BY started_at DESC LIMIT 30")
        health = db.fetchall("SELECT * FROM health_checks ORDER BY id DESC LIMIT 50")
        issues = db.fetchall("SELECT * FROM data_quality_issues ORDER BY id DESC LIMIT 50")
        state_log = db.fetchall("SELECT * FROM system_state_log ORDER BY id DESC LIMIT 30")
        datasets = db.fetchall("SELECT dataset_id, kind, provider, row_count, start_date, end_date, is_synthetic, "
                               "retrieved_at FROM datasets ORDER BY created_at DESC LIMIT 30")
        return render(request, "system.html", runs=runs, health=health, issues=issues, state_log=state_log,
                      datasets=datasets)

    @app.post("/system/pause")
    def pause(reason: str = Form(...), actor: str = Form("human")):
        from quantlab.monitoring.killswitch import KillSwitch
        KillSwitch(db).pause(reason, trigger="dashboard", actor=actor)
        return RedirectResponse("/system", status_code=303)

    @app.post("/system/resume")
    def resume(reason: str = Form(...), actor: str = Form(...)):
        from quantlab.monitoring.killswitch import KillSwitch, ResumeNotAllowedError
        try:
            KillSwitch(db).resume(reason, actor)
        except ResumeNotAllowedError as exc:
            raise HTTPException(403, html.escape(str(exc)))
        return RedirectResponse("/system", status_code=303)

    return app


def main() -> None:   # pragma: no cover - manual entry point
    import uvicorn
    ctx = AppContext.create()
    uvicorn.run(create_app(ctx), host=ctx.config.get("dashboard.host", "127.0.0.1"),
                port=int(ctx.config.get("dashboard.port", 8765)))
