"""Data for the dashboard's main page: MARKET DISCOVERY (what looks interesting) kept separate from
TRADE VALIDATION (what the unchanged gates permit). Read-only; every number comes from the latest
persisted discovery run, never hard-coded."""
from __future__ import annotations

from typing import Any

from quantlab.context import AppContext
from quantlab.db.database import from_json
from quantlab.discovery.families import LABELS, SCORED
from quantlab.discovery.status import strategy_research_status

FUNNEL = (
    ("full_universe", "Full universe", "symbols in the stored data"),
    ("basic", "Basic data/liquidity", "bar today, price >= $1, median $ volume >= $1M, >= 60 sessions"),
    ("with_features", "Discovery features", "at least 4 of 5 scored families computable"),
    ("discovered", "Discovery setups", "at least one family fired"),
    ("high_ranked", "Top-N ranking", "discovery score >= threshold"),
    ("watchlist", "Watchlist", "top high-ranked setups passing data + universe checks"),
    ("in_validation", "Validation", "discovered setups (any rank) that a strategy signals: decision chain ran"),
    ("paper_eligible", "Paper eligible", "discovered setups with a TRADE decision from the unchanged gates"),
    ("paper_trades", "Paper trades", "discovered setups with a paper order placed"),
)


def scan_state(ctx: AppContext, live: dict[str, Any] | None = None) -> dict[str, Any]:
    db = ctx.db
    run = db.fetchone("SELECT * FROM discovery_runs ORDER BY as_of_date DESC, created_at DESC LIMIT 1")
    out: dict[str, Any] = {"run": None, "funnel": [], "top": [], "near_misses": [], "diagnostics": [], "coverage": [],
                           "blockers": {}, "market": {}, "status_counts": {}, "strategies": [], "live": live}
    out["strategies"] = sorted(strategy_research_status(db, ctx.config).values(), key=lambda x: x["strategy_id"])
    # is the shown discovery run the latest decided session? (a failed/missing discovery must not
    # silently show yesterday's market as today's)
    last_pipe = db.fetchone("SELECT r.run_id, r.as_of_date, ps.output_json FROM runs r LEFT JOIN pipeline_steps ps "
                            "ON ps.run_id=r.run_id AND ps.step='discover' WHERE r.kind='pipeline' "
                            "ORDER BY r.started_at DESC LIMIT 1")
    out["stale"] = None
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
    out["run"] = {k: run[k] for k in ("discovery_run_id", "run_id", "as_of_date", "created_at", "is_synthetic")}
    out["funnel"] = [{"key": k, "label": lbl, "hint": hint, "n": f.get(k)} for k, lbl, hint in FUNNEL]
    out["funnel_raw"] = f
    out["status_counts"] = f.get("status_counts", {})
    out["blockers"] = bl
    out["near_misses"] = (bl.get("near_misses") or [])[:5]
    out["coverage"] = fam.get("coverage", [])
    out["market"] = fam.get("market_context", {})
    out["fired"] = fam.get("fired", {})
    out["diagnostics"] = db.fetchall("SELECT level, code, message FROM discovery_diagnostics WHERE discovery_run_id=? "
                                     "ORDER BY CASE level WHEN 'CRITICAL' THEN 0 WHEN 'WARN' THEN 1 ELSE 2 END, id",
                                     (run["discovery_run_id"],))
    top = []
    for r in db.fetchall("SELECT * FROM discovery_candidates WHERE discovery_run_id=? "
                         "ORDER BY discovery_score IS NULL, discovery_score DESC LIMIT 15", (run["discovery_run_id"],)):
        fams = from_json(r["families_json"], {}) or {}
        comps = fams.get("components") or {}
        ctxs = from_json(r["catalyst_json"], {}) or {}
        reasons = [x for fam_ in fams.get("fired", []) for x in (fams.get("reasons") or {}).get(fam_, [])]
        top.append({
            "symbol": r["symbol"], "score": r["discovery_score"], "coverage": r["score_coverage"],
            "families": [LABELS.get(x, x) for x in fams.get("fired", [])],
            "points": [{"label": LABELS[k], "value": comps.get(k)} for k in SCORED],
            "reasons": reasons[:4], "status": r["status"], "block_stage": r["block_stage"],
            "block_reason": r["block_reason"], "bias": r["direction_bias"], "on_watchlist": r["on_watchlist"],
            "unknown_context": [LABELS.get(k, k) for k, v in ctxs.items() if (v or {}).get("state") != "KNOWN"],
            "known_context": [LABELS.get(k, k) for k, v in ctxs.items() if (v or {}).get("state") == "KNOWN"],
        })
    out["top"] = top
    return out


__all__ = ["FUNNEL", "scan_state"]
