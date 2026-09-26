"""Data for the dashboard's main page (the QuantLab terminal). Read-only; every number comes from the
database (discovery runs, decisions, orders, research summaries), never hard-coded.

Three things are kept visibly separate:
  * CURRENT MARKET OPPORTUNITY  setups found in the last completed session's data;
  * NEXT-SESSION SETUP          the same setups framed as conditional plans for the next session
                                (confirm / invalidate / missing), after overnight refresh + pre-open recheck;
  * VALIDATED PAPER TRADE       only a TRADE decision from the unchanged gates plus a placed paper order.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quantlab.context import AppContext
from quantlab.db.database import from_json
from quantlab.discovery.families import LABELS, SCORED
from quantlab.discovery.status import describe_strategy, strategy_research_status

FUNNEL = (
    ("full_universe", "Scanned", "symbols in the stored data"),
    ("basic", "Basic data/liquidity", "bar, price >= $1, median $ volume >= $1M, >= 60 sessions"),
    ("discovered", "Discovery setups", "at least one family fired"),
    ("strategy_signals", "Strategy signals", "a strategy produced a candidate"),
    ("both", "Both", "discovered AND signalled by a strategy"),
    ("watchlist", "Watchlist", "top high-ranked setups passing data + universe checks"),
    ("validation_pending", "Validation pending", "covered by a strategy that is not validated yet"),
    ("paper_eligible", "Paper eligible", "TRADE decision from the unchanged gates"),
    ("paper_trades", "Paper trades", "paper order placed"),
)
STATUS_ORDER = {"TRADED": 0, "PAPER_ELIGIBLE": 1, "VALIDATION_PENDING": 2, "WATCH": 3, "REJECTED": 4, "DISCOVERED": 5}


def _research_lookup(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    return {g["group"]: g for g in summary.get("groups", [])}


def _evidence_for(fired: list[str], research: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Research evidence for a candidate's family pattern: the matching combination if two of its
    families form one, otherwise its strongest single family. None when no research exists."""
    if not research:
        return None
    from quantlab.discovery.research import COMBOS, FAMILY_LABEL
    f = set(fired)
    for name, parts in COMBOS.items():
        if all(p in f for p in parts):
            g = research.get(name)
            if g:
                return {"group": name, **_ev(g)}
    for fam in fired:
        g = research.get(FAMILY_LABEL.get(fam, fam))
        if g:
            return {"group": FAMILY_LABEL.get(fam, fam), **_ev(g)}
    return None


def _ev(g: dict[str, Any]) -> dict[str, Any]:
    h = (g.get("horizons") or {}).get("5") or (g.get("horizons") or {}).get(5) or {}
    return {"verdict": g.get("verdict"), "n_obs": g.get("n_obs"), "median_net_5d": h.get("median_net"),
            "mean_net_5d": h.get("mean_net"), "vs_baseline_5d": h.get("vs_baseline_mean"),
            "t": h.get("vs_baseline_t_clustered")}


def _tech(factors: dict[str, Any]) -> list[dict[str, str]]:
    f = {k: (v or {}).get("value") for k, v in (factors or {}).items()}
    chips = []
    d50, ma = f.get("dist_ma50"), f.get("ma50_over_ma200")
    if isinstance(d50, (int, float)):
        up = d50 > 0 and isinstance(ma, (int, float)) and ma > 0
        chips.append({"k": "Trend", "v": "up" if up else ("above 50d" if d50 > 0 else "below 50d"), "tone": "ok" if up else "muted"})
    rs = f.get("rs_spy_63")
    if isinstance(rs, (int, float)):
        chips.append({"k": "vs SPY 63d", "v": f"{rs * 100:+.0f} pp", "tone": "ok" if rs > 0 else "bad"})
    rv = f.get("rel_volume_1d")
    if isinstance(rv, (int, float)):
        chips.append({"k": "Rel vol", "v": f"{rv:.1f}x", "tone": "ok" if rv >= 1.5 else "muted"})
    brk = f.get("breakout_55")
    if isinstance(brk, (int, float)):
        chips.append({"k": "55d high", "v": "new high" if brk > 0 else f"{brk:.0%} below", "tone": "ok" if brk > 0 else "muted"})
    z = f.get("ret_z_3d")
    if isinstance(z, (int, float)) and z <= -2:
        chips.append({"k": "3d move", "v": f"z {z:.1f}", "tone": "bad"})
    r20 = f.get("ret_20d")
    if isinstance(r20, (int, float)):
        chips.append({"k": "20d", "v": f"{r20:+.0%}", "tone": "ok" if r20 > 0 else "bad"})
    return chips


def _catalyst(ctx_json: dict[str, Any], overnight: list[dict[str, Any]]) -> dict[str, Any]:
    if overnight:
        return {"state": "PRESENT", "text": f"{len(overnight)} overnight item(s): " +
                ", ".join(sorted({f"{x['kind']} ({x['phase'].lower().replace('_', '-')})" for x in overnight}))}
    known = [k for k in ("news", "earnings") if (ctx_json.get(k) or {}).get("state") == "KNOWN"]
    fired = [k for k in known if (ctx_json.get(k) or {}).get("fired")]
    if fired:
        return {"state": "PRESENT", "text": "; ".join(x for k in fired for x in ctx_json[k].get("reasons", []))}
    if known:
        return {"state": "NONE", "text": "covered by " + " + ".join(known) + ": no catalyst"}
    return {"state": "UNKNOWN", "text": "UNKNOWN: no news/earnings coverage for this symbol"}


def scan_state(ctx: AppContext, live: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
    from quantlab.discovery.nextsession import next_session_state
    db = ctx.db
    now = now or datetime.now(timezone.utc)
    nxt = next_session_state(ctx, now, top=8)
    run = db.fetchone("SELECT * FROM discovery_runs ORDER BY as_of_date DESC, created_at DESC LIMIT 1")
    strategies = strategy_research_status(db, ctx.config)
    rr = db.fetchone("SELECT * FROM discovery_research_runs ORDER BY created_at DESC LIMIT 1")
    research_summary = from_json(rr["summary_json"], {}) if rr else None
    research = _research_lookup(research_summary)
    out: dict[str, Any] = {
        "now": now.isoformat(), "market": nxt["market"], "next": nxt, "live": live, "run": None, "funnel": [],
        "funnel_raw": {}, "top": [], "near_misses": [], "diagnostics": [], "coverage": [], "blockers": {},
        "market_context": {}, "status_counts": {}, "stale": None,
        "strategies": sorted(({**v, "describe": describe_strategy(v)} for v in strategies.values()),
                             key=lambda x: x["strategy_id"]),
        "research": None, "paper": _paper_panel(db), "monitored": None,
    }
    if rr:
        groups = research_summary.get("groups", [])
        ranked = sorted([g for g in groups if not g.get("baseline")],
                        key=lambda g: -(((g.get("horizons") or {}).get("5") or {}).get("vs_baseline_mean") or -9))
        out["research"] = {"research_id": rr["research_id"], "period": f"{rr['period_start']} to {rr['period_end']}",
                           "n_dates": rr["n_dates"], "n_obs": rr["n_observations"], "report_path": rr["report_path"],
                           "baseline": next((_ev(g) | {"group": g["group"], "n_obs": g["n_obs"]} for g in groups
                                             if g.get("baseline")), None),
                           "groups": [{"group": g["group"], "n_obs": g["n_obs"], "n_dates": g["n_dates"],
                                       "verdict_reason": g.get("verdict_reason"), **_ev(g)} for g in ranked],
                           "redundancy": from_json(rr["redundancy_json"], {}),
                           "n_tests": research_summary.get("n_tests"), "z": research_summary.get("bonferroni_z")}
    last_pipe = db.fetchone("SELECT r.run_id, r.as_of_date, ps.output_json FROM runs r LEFT JOIN pipeline_steps ps "
                            "ON ps.run_id=r.run_id AND ps.step='discover' WHERE r.kind='pipeline' "
                            "ORDER BY r.started_at DESC LIMIT 1")
    if last_pipe is not None:
        step_out = from_json(last_pipe["output_json"], {}) or {}
        if step_out.get("error"):
            out["stale"] = f"discovery for {last_pipe['as_of_date']} failed: {step_out['error']}"
        elif run is None or str(run["as_of_date"]) < str(last_pipe["as_of_date"]):
            out["stale"] = (f"latest pipeline session is {last_pipe['as_of_date']} but the latest discovery run is "
                            f"{run['as_of_date'] if run else 'none'}")
    if run is None:
        return out
    f = from_json(run["funnel_json"], {}) or {}
    bl = from_json(run["blockers_json"], {}) or {}
    fam = from_json(run["families_json"], {}) or {}
    out["run"] = {k: run.get(k) for k in ("discovery_run_id", "run_id", "as_of_date", "created_at", "is_synthetic",
                                          "next_session", "info_cutoff_at")}
    out["funnel"] = [{"key": k, "label": lbl, "hint": hint, "n": f.get(k)} for k, lbl, hint in FUNNEL]
    out["funnel_raw"] = f
    out["status_counts"] = f.get("status_counts", {})
    out["blockers"] = bl
    out["near_misses"] = (bl.get("near_misses") or [])[:5]
    out["coverage"] = fam.get("coverage", [])
    out["market_context"] = fam.get("market_context", {})
    out["diagnostics"] = db.fetchall("SELECT level, code, message FROM discovery_diagnostics WHERE discovery_run_id=? "
                                     "ORDER BY CASE level WHEN 'CRITICAL' THEN 0 WHEN 'WARN' THEN 1 ELSE 2 END, id",
                                     (run["discovery_run_id"],))
    overnight: dict[str, list] = {}
    for r in db.fetchall("SELECT symbol, phase, kind, available_at FROM overnight_updates WHERE discovery_run_id=?",
                         (run["discovery_run_id"],)):
        overnight.setdefault(r["symbol"], []).append(dict(r))
    rows = db.fetchall("SELECT * FROM discovery_candidates WHERE discovery_run_id=? "
                       "ORDER BY discovery_score IS NULL, discovery_score DESC LIMIT 400", (run["discovery_run_id"],))
    top = []
    for r in rows:
        fams = from_json(r["families_json"], {}) or {}
        fired = fams.get("fired", [])
        comps = fams.get("components") or {}
        ctxs = from_json(r["catalyst_json"], {}) or {}
        setup = from_json(r.get("setup_json"), {}) or {}
        links = from_json(r["strategy_links_json"], []) or []
        sid = links[0]["strategy_id"] if links else None
        top.append({
            "symbol": r["symbol"], "score": r["discovery_score"], "coverage": r["score_coverage"],
            "origin": r.get("origin") or "DISCOVERY", "families": [LABELS.get(x, x) for x in fired],
            "setup_type": setup.get("setup_type") or " + ".join(LABELS.get(x, x) for x in fired),
            "points": [{"label": LABELS[k], "short": LABELS[k].split("/")[0].split(" ")[0][:8], "value": comps.get(k)}
                       for k in SCORED],
            "reasons": (setup.get("why") or [])[:4], "status": r["status"], "block_stage": r["block_stage"],
            "block_reason": r["block_reason"], "on_watchlist": r["on_watchlist"], "high_quality": r["high_quality"],
            "levels": setup.get("levels") or {}, "tech": _tech((from_json(r["factors_json"], {}) or {}).get("features", {})),
            "catalyst": _catalyst(ctxs, overnight.get(r["symbol"], [])),
            "unknown_context": [LABELS.get(k, k) for k, v in ctxs.items() if (v or {}).get("state") != "KNOWN"],
            "strategy": describe_strategy(strategies.get(sid)) if sid else "no strategy covers this setup",
            "strategy_status": (strategies.get(sid) or {}).get("status") if sid else None,
            "evidence": _evidence_for(fired, research), "setup": setup,
        })
    top.sort(key=lambda x: (STATUS_ORDER.get(x["status"], 9) if x["status"] in ("TRADED", "PAPER_ELIGIBLE") else 5,
                            -(x["score"] if isinstance(x["score"], (int, float)) else -1)))
    out["top"] = top[:15]
    # the strongest candidate being monitored: furthest status, then score
    monitored = sorted(top, key=lambda x: (STATUS_ORDER.get(x["status"], 9) if x["status"] != "REJECTED" else 8,
                                           -(x["score"] if isinstance(x["score"], (int, float)) else -1)))
    out["monitored"] = monitored[0] if monitored else None
    return out


def _paper_panel(db) -> dict[str, Any]:
    """CURRENT PAPER TRADE: open BOT trades from the ledger (source of truth) + the latest orders."""
    trades = db.fetchall("SELECT t.trade_id, t.symbol, t.qty, t.entry_date, t.entry_price, t.stop_price, t.target_price, "
                         "t.planned_exit_date, t.strategy_id FROM trades t WHERE t.book='BOT' AND t.status='OPEN' "
                         "ORDER BY t.entry_date DESC")
    orders = db.fetchall("SELECT created_at, symbol, side, qty, purpose, status, filled_avg_price FROM orders "
                         "WHERE book='BOT' AND status IN ('pending_submit','accepted','new','partially_filled') "
                         "ORDER BY created_at DESC LIMIT 10")
    closed = db.fetchone("SELECT COUNT(*) AS n, COALESCE(SUM(net_pnl),0) AS pnl FROM trades WHERE book='BOT' AND status='CLOSED'")
    return {"open": trades, "working_orders": orders, "closed_n": closed["n"], "realized": closed["pnl"]}


__all__ = ["FUNNEL", "scan_state"]
