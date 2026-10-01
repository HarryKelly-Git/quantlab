"""QuantLab dashboard: server-rendered, localhost by default, PAPER ONLY.

Pages read the audit database. The only writes are the kill switch: pause, and resume, which
requires a declared human actor and a reason. There is no order entry and no live-trading
control anywhere.

``/live`` is the paper-runner view: it re-renders server-side every ``dashboard.live_refresh_seconds``
(a meta refresh, no JavaScript framework) from the runner's heartbeat row and the audit tables.
``/api/live`` returns the same data as JSON.
"""
from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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


_OPEN_ORDER = ("pending_submit", "accepted", "new", "partially_filled", "unknown")


def _age_seconds(ts: str | None, now: datetime) -> float | None:
    if not ts:
        return None
    try:
        return (pd.Timestamp(now) - pd.Timestamp(ts)).total_seconds()
    except (ValueError, TypeError):
        return None


class _BenchCache:
    """SPY closes for the performance comparison, read with a parquet row filter (cheap) and cached."""

    def __init__(self, ttl: float = 300.0):
        self.ttl, self.at, self.key, self.series = ttl, 0.0, None, None

    def closes(self, ctx: AppContext, symbol: str) -> pd.Series | None:
        ids = ctx.store.dataset_ids("bars", synthetic=False) or ctx.store.dataset_ids("bars", synthetic=True)
        key = (symbol, tuple(ids))
        if self.series is not None and self.key == key and time.monotonic() - self.at < self.ttl:
            return self.series
        frames = []
        for dataset_id in ids:
            row = ctx.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (dataset_id,))
            if row is None:
                continue
            try:
                frames.append(pd.read_parquet(ctx.store.data_dir / row["path"], columns=["symbol", "date", "close"],
                                              filters=[("symbol", "==", symbol)]))
            except (OSError, ValueError, KeyError):
                continue
        frames = [f for f in frames if len(f)]
        if not frames:
            return None
        df = pd.concat(frames).drop_duplicates("date", keep="last").sort_values("date")
        self.series = pd.Series(df["close"].astype(float).to_numpy(),
                                index=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").to_numpy())
        self.key, self.at = key, time.monotonic()
        return self.series


_ARM_TARGET = {5: 5, 10: 8, 20: 10}       # the clean-win target studied for each hold arm


def _fnum(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _exploration_live(ctx: AppContext, now: datetime, bpos: dict[str, Any], real: bool) -> dict[str, Any]:
    """What the exploration book is doing right now: open positions with their hold arm, stop and
    big-move forecast; the latest plan through its lifecycle; the hold-arm experiment; the learning
    report. Read-only."""
    from quantlab.exploration import paper_mode
    from quantlab.exploration.engine import learning_report
    db = ctx.db
    today = pd.Timestamp(now).tz_convert("America/New_York").date()

    def pre_of(did: str | None) -> tuple[str | None, dict[str, Any]]:
        r = db.fetchone("SELECT setup_type, pre_trade_json FROM exploration_decisions WHERE decision_id=?", (did,)) \
            if did else None
        return (r["setup_type"], from_json(r["pre_trade_json"], {}) or {}) if r else (None, {})

    def forecast(pre: dict[str, Any], hold: Any) -> dict[str, Any] | None:
        up = pre.get("upside") or {}
        t = _ARM_TARGET.get(int(hold or 0))
        if up.get("state") != "KNOWN" or t is None:
            return None
        return {"target_pct": t, "p_hit": _fnum(up.get(f"p_clean_{t}")), "p_stop": _fnum(up.get("p_stop")),
                "median_days": _fnum(up.get(f"median_days_to_{t}"))}

    positions, gross = [], 0.0
    for t in db.fetchall("SELECT t.trade_id, t.symbol, t.qty, t.entry_date, t.entry_price, t.stop_price, t.decision_id, "
                         "t.strategy_id, t.planned_exit_date, tp.holding_sessions AS hold FROM trades t LEFT JOIN trade_plans tp "
                         "ON tp.trade_id = t.trade_id WHERE t.book='BOT' AND t.status='OPEN' ORDER BY t.entry_date, t.symbol"):
        b = bpos.get(t["symbol"]) or {}
        px, entry, stop = _fnum(b.get("current_price")), _fnum(t["entry_price"]), _fnum(t["stop_price"])
        qty = _fnum(t["qty"]) or 0.0
        setup, pre = pre_of(t["decision_id"])
        held = max(len(pd.bdate_range(str(t["entry_date"])[:10], str(today))) - 1, 0) if t["entry_date"] else None
        hold = int(t["hold"]) if t["hold"] else None
        gross += qty * (px or entry or 0.0)
        positions.append({
            "trade_id": t["trade_id"], "symbol": t["symbol"], "qty": qty, "entry_date": t["entry_date"], "entry_price": entry,
            "current_price": px, "unrealized_pl": _fnum(b.get("unrealized_pl")),
            "unrealized_pct": (px / entry - 1) if px and entry else None, "stop_price": stop,
            "room_to_stop_pct": (px / stop - 1) if px and stop else None, "hold": hold, "sessions_held": held,
            "planned_exit": t["planned_exit_date"], "strategy": t["strategy_id"], "setup": setup,
            "selection_score": _fnum((pre.get("candidate") or {}).get("selection_score")), "forecast": forecast(pre, hold)})

    srow = db.fetchone("SELECT MAX(session_date) AS s FROM exploration_decisions WHERE is_synthetic=?", (0 if real else 1,))
    session = srow["s"] if srow else None
    plan, watched, n_skipped = [], [], 0
    if session:
        for d in db.fetchall("SELECT decision_id, symbol, selection, holding_sessions AS hold, qty, ref_price, stop_price, "
                             "setup_type, next_session, pre_trade_json FROM exploration_decisions WHERE session_date=? "
                             "AND selection IN ('SELECTED','WATCHED_NOT_TRADED') ORDER BY rank", (session,)):
            pre = from_json(d["pre_trade_json"], {}) or {}
            ev = db.fetchone("SELECT event, at FROM exploration_events WHERE decision_id=? ORDER BY rowid DESC LIMIT 1",
                             (d["decision_id"],))
            order = db.fetchone("SELECT status, order_type, time_in_force, filled_qty, filled_avg_price FROM orders "
                                "WHERE decision_id=? AND purpose='entry' ORDER BY created_at DESC LIMIT 1", (d["decision_id"],))
            trade = db.fetchone("SELECT status, entry_price FROM trades WHERE decision_id=?", (d["decision_id"],))
            if trade:
                stage = f"FILLED @ {trade['entry_price']:.2f} ({trade['status']})" if trade["entry_price"] else trade["status"]
            elif order:
                stage = f"ORDER {order['status']} ({order['order_type']}/{order['time_in_force']})"
            else:
                stage = (ev["event"] if ev else "PLANNED") if d["selection"] == "SELECTED" else "watched (not traded)"
            row = {"symbol": d["symbol"], "selection": d["selection"], "stage": stage, "hold": d["hold"], "qty": d["qty"],
                   "ref_price": d["ref_price"], "stop_price": d["stop_price"], "setup": d["setup_type"],
                   "next_session": d["next_session"],
                   "selection_score": _fnum((pre.get("candidate") or {}).get("selection_score")),
                   "forecast": forecast(pre, d["hold"])}
            (plan if d["selection"] == "SELECTED" else watched).append(row)
        n_skipped = (db.fetchone("SELECT COUNT(*) AS n FROM exploration_decisions WHERE session_date=? AND selection='SKIPPED'",
                                 (session,)) or {"n": 0})["n"]
    arms = db.fetchall("SELECT tp.holding_sessions AS hold, t.status, COUNT(*) AS n, AVG(t.ret) AS avg_ret, "
                       "SUM(CASE WHEN t.net_pnl > 0 THEN 1 ELSE 0 END) AS wins FROM trades t JOIN trade_plans tp "
                       "ON tp.trade_id = t.trade_id WHERE t.book='BOT' AND t.strategy_id='EXPLORATION' GROUP BY 1, 2 ORDER BY 1, 2")
    lr = learning_report(db, synthetic=not real)
    learning = {"matured": lr.get("matured", 0), "n_missed": lr.get("n_missed_big_winners", 0),
                "n_failed": lr.get("n_failed_picks", 0), "missed": (lr.get("missed_big_winners") or [])[:6],
                "failed": (lr.get("failed_picks") or [])[:6], "by_hold": lr.get("by_selection_and_hold") or [],
                "note": lr.get("note")}
    return {"mode": paper_mode(ctx.config), "positions": positions, "gross": gross, "plan_session": session,
            "plan": plan, "watched": watched, "skipped": n_skipped, "arms": arms, "learning": learning}


def live_state(ctx: AppContext, bench: "_BenchCache | None" = None, now: datetime | None = None) -> dict[str, Any]:
    """Everything the live page shows. Read-only."""
    from quantlab.data.audit import expected_sessions
    from quantlab.monitoring.killswitch import KillSwitch
    db = ctx.db
    now = now or datetime.now(timezone.utc)
    stale_after = float(ctx.config.get("paper.runner.stale_heartbeat_seconds", 120))

    # SYSTEM ------------------------------------------------------------------------------------
    sess = db.fetchone("SELECT * FROM paper_runner_sessions ORDER BY started_at DESC LIMIT 1")
    detail = from_json(sess["detail_json"], {}) if sess else {}
    hb_age = _age_seconds(sess["last_heartbeat_at"], now) if sess else None
    runner = "NEVER_STARTED" if sess is None else sess["status"]
    if runner == "RUNNING" and (hb_age is None or hb_age > stale_after):
        runner = "STALE (no heartbeat: process not running)"
    state, reason, changed = KillSwitch(db).state()
    strategies = db.fetchall("SELECT strategy_id, version, status, stage, updated_at FROM strategies ORDER BY strategy_id")
    eligible = [r for r in strategies if r["status"] == "ACTIVE" and r["stage"] in ("PAPER", "PROMOTED")]
    last_ok = db.fetchone("SELECT j.as_of_date, j.run_id, j.orders_allowed, j.reason, r.finished_at FROM paper_session_jobs j "
                          "LEFT JOIN runs r ON r.run_id = j.run_id WHERE j.status='succeeded' ORDER BY j.as_of_date DESC LIMIT 1")
    if last_ok is None:
        last_ok = db.fetchone("SELECT as_of_date, run_id, finished_at FROM runs WHERE kind='pipeline' AND status='succeeded' "
                              "ORDER BY finished_at DESC LIMIT 1")
    jobs = db.fetchall("SELECT as_of_date, next_session, status, orders_allowed, attempts, reason, run_id, updated_at "
                       "FROM paper_session_jobs ORDER BY as_of_date DESC LIMIT 5")
    real = bool(db.fetchone("SELECT 1 FROM datasets WHERE kind='bars' AND is_synthetic=0 LIMIT 1"))
    lb = db.fetchone("SELECT MAX(end_date) AS d FROM datasets WHERE kind='bars' AND is_synthetic=?", (0 if real else 1,))
    latest_bar = str(lb["d"])[:10] if lb and lb["d"] else None
    freshness: dict[str, Any] = {"latest_bar": latest_bar, "synthetic": not real, "missing_sessions": None,
                                 "status": "UNKNOWN (no bars)"}
    if latest_bar:
        et = pd.Timestamp(now).tz_convert("America/New_York")
        last_expected = et.normalize().tz_localize(None)
        if et.hour < 16 or (et.hour == 16 and et.minute < 15):
            last_expected -= pd.Timedelta(days=1)
        behind = [d for d in expected_sessions(latest_bar, str(last_expected.date())) if d > pd.Timestamp(latest_bar)]
        freshness.update(missing_sessions=len(behind),
                         status="FRESH" if not behind else f"STALE ({len(behind)} session(s) behind)")
    system = {"runner": runner, "session": sess, "heartbeat_age_seconds": hb_age, "detail": detail,
              "system_state": state.value, "system_reason": reason, "system_changed": changed,
              "last_pipeline": last_ok, "jobs": jobs, "freshness": freshness, "strategies": strategies,
              "eligible": eligible, "no_paper_eligible": not eligible}

    # SIGNALS (latest run that produced candidates) ------------------------------------------------
    run = db.fetchone("SELECT run_id, as_of_date FROM candidates ORDER BY created_at DESC LIMIT 1")
    signals = []
    if run:
        for r in db.fetchall(
                "SELECT c.candidate_id, c.symbol, c.strategy_id, c.score, c.entry_ref_price, c.stop_price, c.features_json, "
                "d.decision, d.reject_stage, d.reasons_json FROM candidates c LEFT JOIN decisions d ON d.candidate_id=c.candidate_id "
                "WHERE c.run_id=? ORDER BY (d.decision='TRADE') DESC, c.score DESC LIMIT 60", (run["run_id"],)):
            feats = from_json(r.pop("features_json"), {}) or {}
            r["features"] = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in list(feats.items())[:4])
            r["reasons"] = "; ".join(from_json(r.pop("reasons_json"), []) or [])[:300]
            signals.append(r)
    stage_counts = db.fetchall("SELECT d.decision, d.reject_stage, COUNT(*) AS n FROM decisions d JOIN candidates c "
                               "ON c.candidate_id=d.candidate_id WHERE c.run_id=? GROUP BY 1,2 ORDER BY n DESC",
                               (run["run_id"],)) if run else []

    # ORDERS ------------------------------------------------------------------------------------
    orders = db.fetchall(
        "SELECT o.created_at, o.symbol, o.side, o.qty, o.purpose, o.status, o.order_id, o.broker_order_id, o.submitted_at, "
        "(SELECT MAX(f.filled_at) FROM fills f WHERE f.order_id=o.order_id) AS filled_at, o.filled_qty, o.filled_avg_price "
        "FROM orders o WHERE o.book='BOT' ORDER BY o.created_at DESC LIMIT 30")
    updates = db.fetchall("SELECT received_at, source, event, client_order_id, broker_order_id, status, filled_qty, "
                          "filled_avg_price FROM broker_order_updates ORDER BY id DESC LIMIT 15")
    refusals = db.fetchall("SELECT created_at, purpose, symbol, qty, reason FROM execution_refusals WHERE book='BOT' "
                           "ORDER BY id DESC LIMIT 10")

    # PORTFOLIO ---------------------------------------------------------------------------------
    snaps = db.fetchall("SELECT as_of_date, equity, cash, gross_exposure, drawdown, positions_count FROM portfolio_snapshots "
                        "WHERE book='BOT' ORDER BY as_of_date")
    cash = db.fetchone("SELECT COALESCE(SUM(amount),0) AS c, COUNT(*) AS n FROM ledger_cash_events WHERE book='BOT'")
    positions = db.fetchall("SELECT p.symbol, p.qty, p.avg_cost, p.trade_id, t.strategy_id, t.entry_date, t.stop_price "
                            "FROM positions p LEFT JOIN trades t ON t.trade_id=p.trade_id WHERE p.book='BOT' ORDER BY p.symbol")
    broker = detail.get("broker") or {}
    bpos = {p["symbol"]: p for p in broker.get("positions") or []}
    for p in positions:
        b = bpos.get(p["symbol"]) or {}
        p["current_price"] = b.get("current_price")
        p["unrealized_pl"] = b.get("unrealized_pl")
    realized = db.fetchone("SELECT COALESCE(SUM(net_pnl),0) AS pnl, COUNT(*) AS n, AVG(ret) AS expectancy, "
                           "AVG(CASE WHEN net_pnl>0 THEN 1.0 ELSE 0.0 END) AS win_rate FROM trades "
                           "WHERE book='BOT' AND status='CLOSED'")
    unreal = sum(float(p["unrealized_pl"]) for p in positions if p.get("unrealized_pl") not in (None, ""))
    ledger_syms = {x["symbol"] for x in positions}
    portfolio = {"ledger_cash": cash["c"] if cash["n"] else None, "last_snapshot": snaps[-1] if snaps else None,
                 "positions": positions, "broker": broker, "realized_pnl": realized["pnl"],
                 "unrealized_pnl": unreal, "external_positions": [p for s_, p in bpos.items() if s_ not in ledger_syms]}

    # PERFORMANCE -------------------------------------------------------------------------------
    perf: dict[str, Any] = {"closed_trades": realized["n"], "expectancy": realized["expectancy"],
                            "win_rate": realized["win_rate"], "equity_points": [s_["equity"] for s_ in snaps],
                            "return": None, "max_drawdown": None, "spy_return": None, "first": None, "last": None}
    if snaps:
        eq = pd.Series([float(s_["equity"]) for s_ in snaps], index=[s_["as_of_date"] for s_ in snaps])
        start_cash = db.fetchone("SELECT amount FROM ledger_cash_events WHERE book='BOT' AND kind='starting_cash' "
                                 "ORDER BY id LIMIT 1")
        base = float(start_cash["amount"]) if start_cash else float(eq.iloc[0])
        perf.update({"first": eq.index[0], "last": eq.index[-1], "return": float(eq.iloc[-1] / base - 1.0),
                     "max_drawdown": float((eq / eq.cummax() - 1.0).min())})
        if len(snaps) >= 2:
            spy = (bench or _BenchCache()).closes(ctx, ctx.config.get("benchmarks.market", "SPY"))
            if spy is not None:
                w = spy.loc[(spy.index >= eq.index[0]) & (spy.index <= eq.index[-1])]
                if len(w) >= 2:
                    perf["spy_return"] = float(w.iloc[-1] / w.iloc[0] - 1.0)

    # AUDIT -------------------------------------------------------------------------------------
    audit: dict[str, Any] = {
        "errors": db.fetchall("SELECT at, level, kind, substr(message,1,400) AS message FROM paper_runner_events "
                              "WHERE level IN ('ERROR','CRITICAL') ORDER BY id DESC LIMIT 10"),
        "failed_steps": db.fetchall("SELECT run_id, step, finished_at, substr(error, -300) AS error FROM pipeline_steps "
                                    "WHERE status='failed' ORDER BY finished_at DESC LIMIT 5"),
        "data_quality": db.fetchall("SELECT created_at, check_name, severity, symbol, session_date, substr(detail,1,200) AS detail "
                                    "FROM data_quality_issues ORDER BY id DESC LIMIT 10"),
        "killswitch": db.fetchall("SELECT changed_at, state, trigger, changed_by, substr(reason,1,300) AS reason "
                                  "FROM system_state_log ORDER BY id DESC LIMIT 10"),
        "events": db.fetchall("SELECT at, level, kind, substr(message,1,300) AS message FROM paper_runner_events "
                              "ORDER BY id DESC LIMIT 25"),
        "trace": [], "trace_candidate": None,
    }
    if signals:
        audit["trace_candidate"] = {k: signals[0][k] for k in ("candidate_id", "symbol", "strategy_id", "decision", "reject_stage")}
        audit["trace"] = db.fetchall("SELECT check_name, passed, severity, substr(reason,1,200) AS reason FROM risk_checks "
                                     "WHERE candidate_id=? ORDER BY id", (signals[0]["candidate_id"],))
    # EXPLORATION: what the bot is actually trading ---------------------------------------------
    try:
        exploration = _exploration_live(ctx, now, bpos, real)
    except Exception as exc:              # the page must still render if this section fails
        exploration = {"error": f"{type(exc).__name__}: {exc}", "mode": None, "positions": [], "plan": [],
                       "watched": [], "arms": [], "learning": {}}
    return {"generated_at": now.isoformat(), "exploration": exploration, "system": system, "signals": signals,
            "signal_run": run,
            "stage_counts": stage_counts, "orders": orders, "order_updates": updates, "refusals": refusals,
            "portfolio": portfolio, "performance": perf, "audit": audit}


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
    def discovery(request: Request, at: str | None = None):
        """The QuantLab terminal: market discovery, next-session setups, current paper trade.
        ``?at=<ISO time>`` views market/next-session state as of that moment (read-only replay)."""
        from quantlab.dashboard.scan import scan_state
        synthetic = db.fetchone("SELECT MAX(is_synthetic) AS s FROM datasets WHERE kind='bars'")
        now = pd.Timestamp(at).to_pydatetime() if at else None
        return render(request, "scan.html", d=scan_state(ctx, live_state(ctx, bench, now=now), now=now),
                      synthetic=bool(synthetic and synthetic["s"]))

    @app.get("/api/scan")
    def api_scan(at: str | None = None):
        from quantlab.dashboard.scan import scan_state
        now = pd.Timestamp(at).to_pydatetime() if at else None
        return JSONResponse(json.loads(json.dumps(scan_state(ctx, now=now), default=str)))

    @app.get("/overview", response_class=HTMLResponse)
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

    bench = _BenchCache()

    @app.get("/live", response_class=HTMLResponse)
    def live(request: Request):
        """Paper-runner live view (server-side auto-refresh). Read-only."""
        data = live_state(ctx, bench)
        return render(request, "live.html", d=data, refresh=int(ctx.config.get("dashboard.live_refresh_seconds", 15)),
                      spark=_spark(data["performance"]["equity_points"]))

    @app.get("/api/live")
    def api_live():
        return JSONResponse(json.loads(json.dumps(live_state(ctx, bench), default=str)))

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
