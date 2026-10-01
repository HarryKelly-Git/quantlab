"""Glue for the CLI: evaluate one thesis from live INDICATIVE quotes and record it; status; marking.
The data client and broker are injectable so tests never touch the network."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.options.book import OptionsLedger, record_evaluation
from quantlab.options.compare import STOCK, Thesis, compare, select_expiration
from quantlab.options.distribution import MoveDistribution, atr, empirical_distribution
from quantlab.options.settings import OptionsSettings


def load_returns_file(path: str | Path) -> MoveDistribution:
    """JSON: a list of returns, or {"end_returns": [...], "weights"?, "max_favourable"?, "max_adverse"?}."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return MoveDistribution(np.asarray(obj, dtype=float), label=f"file:{Path(path).name}")
    return MoveDistribution(np.asarray(obj["end_returns"], dtype=float), weights=obj.get("weights"),
                            max_favourable=obj.get("max_favourable"), max_adverse=obj.get("max_adverse"),
                            label=obj.get("label", f"file:{Path(path).name}"))


def evaluate(ctx: Any, symbol: str, horizon: int, direction: str = "LONG", *, stop: float | None = None,
             spot: float | None = None, returns_file: str | None = None, paper: bool = False, qty: int = 1,
             max_quote_age_minutes: float | None = None, client: Any = None, broker: Any = None,
             now: pd.Timestamp | None = None) -> dict[str, Any]:
    overrides = {"max_quote_age_minutes": max_quote_age_minutes} if max_quote_age_minutes is not None else None
    s = OptionsSettings.from_config(ctx.config, overrides)
    if not s.enabled:
        return {"ok": False, "reason": "options.enabled is false"}
    if client is None:
        from quantlab.options.data import OptionsDataClient
        client = OptionsDataClient(ctx.config)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    today = now.tz_convert("America/New_York").date()
    symbol, direction = symbol.upper(), direction.upper()
    spot_source = "argument"
    if spot is None:
        sp = client.underlying_spot(symbol)
        spot, spot_source = sp.price, f"{sp.source} at {sp.at}"
    bars = client.underlying_daily_bars(symbol, today - timedelta(days=int(s.distribution_lookback_sessions * 1.5) + 30),
                                        today, adjustment="all")
    dist = load_returns_file(returns_file) if returns_file else empirical_distribution(
        bars, horizon, direction, s.distribution_lookback_sessions)
    if stop is None:
        a = atr(bars)
        if a is None:
            return {"ok": False, "reason": "no stop given and ATR(14) UNKNOWN (not enough bars)"}
        stop = spot - s.default_stop_atr * a if direction == "LONG" else spot + s.default_stop_atr * a
    thesis = Thesis(symbol=symbol, direction=direction, horizon_sessions=horizon, spot=float(spot),
                    stop_price=float(stop))
    contracts = client.contracts(symbol, expiration_gte=today + timedelta(days=1),
                                 expiration_lte=today + timedelta(days=int(s.max_expiry_sessions * 1.5) + 7))
    expiry, _ = select_expiration([c.expiration for c in contracts], today, horizon, s.expiry_buffer_sessions,
                                  s.max_expiry_sessions)
    quotes = client.snapshots(symbol, expiration_date=expiry, type_=thesis.option_type) if expiry else {}
    cmp = compare(thesis, dist, contracts, quotes, now, s, session_date=today)
    eid = record_evaluation(ctx.db, cmp, spot_source=spot_source)
    out: dict[str, Any] = {"ok": True, "evaluation_id": eid, **cmp.summary(), "spot_source": spot_source,
                           "expressions": [_row(r) for r in sorted(cmp.results, key=lambda r: (
                               r.expression_id != STOCK, -(r.expected_pnl_per_risk or -1e9)))],
                           "liquidity_rejections": _rejections(cmp)}
    if paper:
        out["paper"] = _paper(ctx, s, cmp, eid, qty, broker)
    return out


def _row(r: Any) -> dict[str, Any]:
    f = lambda v: None if v is None else round(float(v), 4)   # noqa: E731
    return {"expression": r.expression_id, "kind": r.kind, "eligible": r.eligible, "why_not": r.ineligible_reason,
            "entry_cost": f(r.entry_cost), "max_loss": f(r.max_loss_usd), "breakeven": f(r.breakeven),
            "E_pnl_per_risk": f(r.expected_pnl_per_risk), "P_profit": f(r.p_profit), "P_breakeven": f(r.p_breakeven),
            "E_loss_given_loss": f(r.expected_loss_given_loss), "chosen": r.chosen}


def _rejections(cmp: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cmp.liquidity:
        for reason in c.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _paper(ctx: Any, s: OptionsSettings, cmp: Any, eid: str, qty: int, broker: Any) -> dict[str, Any]:
    from quantlab.options.execution import OptionsPaperExecutor
    chosen = cmp.chosen
    if chosen is None or chosen.structure is None:
        return {"ok": False, "refused": f"chosen expression is {cmp.choice}: no options order"}
    if broker is None and s.paper_trading:
        from quantlab.execution.alpaca_paper import AlpacaPaperBroker
        broker = AlpacaPaperBroker(ctx.config)
    return OptionsPaperExecutor(ctx.config, ctx.db, broker, settings=s).open_structure(
        chosen.structure, qty, session_date=cmp.session_date, evaluation_id=eid, horizon_date=cmp.horizon_date)


def status(ctx: Any) -> dict[str, Any]:
    s = OptionsSettings.from_config(ctx.config)
    db = ctx.db
    led = OptionsLedger(db)
    evals = db.fetchall("SELECT evaluation_id, created_at, symbol, direction, horizon_sessions, chosen_expression, "
                        "contracts_checked, contracts_passed FROM options_eval_runs ORDER BY created_at DESC LIMIT 10")
    structs = db.fetchall("SELECT structure_id, kind, underlying, expiration, qty, status, max_loss_usd, close_by_date, "
                          "horizon_date FROM options_structures ORDER BY created_at DESC LIMIT 20")
    choices = db.fetchall("SELECT chosen_expression LIKE 'LONG_%' OR chosen_expression LIKE '%SPREAD%' AS is_option, "
                          "chosen_expression='STOCK' AS is_stock, COUNT(*) AS n FROM options_eval_runs GROUP BY 1, 2")
    marks = db.fetchone("SELECT COUNT(*) AS n, SUM(status='UNKNOWN') AS unknown FROM options_marks")
    return {"enabled": s.enabled, "paper_trading": s.paper_trading, "feed": s.feed,
            "evaluations_total": db.fetchone("SELECT COUNT(*) AS n FROM options_eval_runs")["n"],
            "choices": [dict(r) for r in choices], "recent_evaluations": evals,
            "opt_book": {"allocation": led.allocation(), "net_premium_cash": led.net_trading_cash(),
                         "positions": led.positions(), "structures": structs,
                         "refusals": db.fetchone("SELECT COUNT(*) AS n FROM options_refusals")["n"]},
            "marks": {"rows": marks["n"], "unknown": marks["unknown"] or 0},
            "label": "PAPER research only (MODEL_OUTPUT); not a real-money recommendation"}


def mark(ctx: Any, as_of: date | None = None, client: Any = None) -> dict[str, Any]:
    from quantlab.options.marking import mark_evaluations, mark_positions, outcomes
    if client is None:
        from quantlab.options.data import OptionsDataClient
        client = OptionsDataClient(ctx.config)
    if as_of is None:
        from quantlab.data.audit import expected_last_session
        as_of = expected_last_session().date()
    ev = mark_evaluations(ctx.db, client, as_of)
    pos = mark_positions(ctx.db, client, as_of)
    return {"as_of": str(as_of), "evaluations": ev, "positions": pos, "outcomes": outcomes(ctx.db)[-50:]}


__all__ = ["evaluate", "load_returns_file", "mark", "status"]
