"""REAL-MONEY REVIEW gate: when is a paper candidate even worth a real-money review?

QuantLab output is paper-trading research: a hypothesis, never a BUY verdict and never personal
financial advice. A candidate is surfaced for the operator's OWN real-money review only when ALL of
these hold (fixed here, never tuned on results):

  1. VALIDATED  its strategy is PAPER_ELIGIBLE (passed the strict validation chain), or its pattern
                is a research hypothesis at STRICT_ELIGIBLE (historical PIT test -> walk-forward ->
                locked holdout -> prospective paper, each moved by a named human);
  2. PROSPECTIVE the same strategy/pattern has >= ``min_closed_paper_trades`` CLOSED paper trades with
                a positive mean net return (forward evidence, not a backtest);
  3. TODAY      the candidate has a TRADE decision from the unchanged strict gates (EV, risk).

Exploratory paper trades never qualify on their own. Anything that qualifies is handed to the
operator's Upside Engine v2 review (primary evidence, fixed output block), which decides.
"""
from __future__ import annotations

from typing import Any

from quantlab.db.database import from_json

MIN_CLOSED_PAPER_TRADES = 30


def _strategy_track(db, sid: str) -> dict[str, Any]:
    r = db.fetchone("SELECT COUNT(*) AS n, AVG(net_pnl / NULLIF(qty * entry_price, 0)) AS mean_ret FROM trades "
                    "WHERE strategy_id=? AND status='CLOSED'", (sid,))
    return {"closed": int(r["n"] or 0), "mean_net_return": r["mean_ret"]}


def review_queue(db, config, run_id: str | None = None) -> dict[str, Any]:
    from quantlab.discovery.status import strategy_research_status
    from quantlab.exploration.hypotheses import hypotheses
    n_min = int(config.get("exploration.min_closed_paper_trades_for_review", MIN_CLOSED_PAPER_TRADES))
    strategies = strategy_research_status(db, config)
    eligible = {k for k, v in strategies.items() if v.get("status") == "PAPER_ELIGIBLE"}
    hyps = hypotheses(db)
    strict_hyps = [h for h in hyps if h["stage"] == "STRICT_ELIGIBLE"]
    run = (db.fetchone("SELECT * FROM discovery_runs WHERE discovery_run_id=?", (run_id,)) if run_id else
           db.fetchone("SELECT * FROM discovery_runs ORDER BY as_of_date DESC, created_at DESC LIMIT 1"))
    queue: list[dict[str, Any]] = []
    if run is not None:
        for c in db.fetchall("SELECT * FROM discovery_candidates WHERE discovery_run_id=? AND status IN "
                             "('PAPER_ELIGIBLE','TRADED')", (run["discovery_run_id"],)):
            links = from_json(c["strategy_links_json"], []) or []
            trade = next((x for x in links if x.get("decision") == "TRADE"), None)
            if trade is None:
                continue
            sid = trade.get("strategy_id")
            fams = set(((from_json(c["families_json"], {}) or {}).get("fired") or [])
                       + [x for x in (c["catalyst_families"] or "").split(",") if x])
            pattern = next((h for h in strict_hyps
                            if set((from_json(h["definition_json"], {}) or {}).get("families", [])) <= fams), None)
            track = _strategy_track(db, sid) if sid else {"closed": 0, "mean_net_return": None}
            checks = {
                "validated": sid in eligible or pattern is not None,
                "prospective": track["closed"] >= n_min and (track["mean_net_return"] or 0) > 0,
                "today_trade_decision": True,
            }
            if all(checks.values()):
                queue.append({"symbol": c["symbol"], "strategy": sid, "pattern": (pattern or {}).get("name"),
                              "track": track, "checks": checks, "evidence_chain": from_json(c["evidence_chain_json"], [])})
    ladder = {"strategies_paper_eligible": sorted(eligible),
              "hypotheses_by_stage": {s: sum(1 for h in hyps if h["stage"] == s) for s in
                                      sorted({h["stage"] for h in hyps})},
              "closed_paper_trades": int(db.fetchone("SELECT COUNT(*) AS n FROM trades WHERE status='CLOSED'")["n"]),
              "min_closed_paper_trades": n_min}
    why_empty = None
    if not queue:
        reasons = []
        if not eligible and not strict_hyps:
            reasons.append("no strategy or pattern has passed the full validation ladder")
        if ladder["closed_paper_trades"] < n_min:
            reasons.append(f"{ladder['closed_paper_trades']} closed paper trade(s) so far (a review needs >= {n_min} "
                           "for the same validated strategy/pattern)")
        if not reasons:
            reasons.append("no validated strategy/pattern produced a TRADE decision this session")
        why_empty = "; ".join(reasons)
    return {"queue": queue, "ladder": ladder, "why_empty": why_empty,
            "note": "Research, not advice: a listed name is only worth the operator's own real-money review under "
                    "the Upside Engine v2 doctrine (primary evidence, fixed output block). Exploratory paper trades "
                    "never qualify."}


__all__ = ["MIN_CLOSED_PAPER_TRADES", "review_queue"]
