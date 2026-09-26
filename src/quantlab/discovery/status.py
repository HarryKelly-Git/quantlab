"""Status models. Strategy research status and candidate status are SEPARATE things.

Strategy research status (derived, read-only; the promotion system in research/promotion.py stays
the only writer of strategies.status/stage):
  DISABLED        not enabled in config, or status RETIRED/PAUSED
  PAPER_ELIGIBLE  status ACTIVE at stage PAPER/PROMOTED (the only status that may place paper orders)
  VALIDATED       a real-data walk-forward concluded SIGNIFICANT_AFTER_DEFLATION (still needs promotion)
  WALK_FORWARD    tested out of sample on real data, not validated (NOT_SIGNIFICANT / INSUFFICIENT_SAMPLE)
  SHADOW          never tested out of sample on real data

Candidate status (one per discovered setup, see engine.py):
  DISCOVERED -> WATCH -> VALIDATION_PENDING -> (REJECTED | PAPER_ELIGIBLE -> TRADED)
A SHADOW strategy can leave a candidate at DISCOVERED, WATCH or VALIDATION_PENDING. It can never
make one PAPER_ELIGIBLE: that needs a TRADE decision from the unchanged decision chain.
"""
from __future__ import annotations

from typing import Any

from quantlab.config import Config
from quantlab.db.database import Database, from_json

STRATEGY_STATUSES = ("SHADOW", "WALK_FORWARD", "VALIDATED", "PAPER_ELIGIBLE", "DISABLED")
CANDIDATE_STATUSES = ("DISCOVERED", "WATCH", "VALIDATION_PENDING", "REJECTED", "PAPER_ELIGIBLE", "TRADED")
POSITIVE_VERDICTS = ("SIGNIFICANT_AFTER_DEFLATION",)


def strategy_research_status(db: Database, config: Config) -> dict[str, dict[str, Any]]:
    """strategy_id -> {status, db_status, stage, version, evidence}. Evidence = latest real-data
    walk-forward (conclusion, trades, expectancy) or None."""
    out: dict[str, dict[str, Any]] = {}
    cfg = config.section("strategies") or {}
    all_rows = db.fetchall("SELECT strategy_id, version, status, stage FROM strategies ORDER BY updated_at")
    by_sid: dict[str, dict[str, Any]] = {}
    for r in all_rows:
        by_sid.setdefault(r["strategy_id"], {})[r["version"]] = r
    for sid in sorted(set(cfg) | set(by_sid)):
        versions = by_sid.get(sid, {})
        conf_ver = (cfg.get(sid) or {}).get("version")
        row = versions.get(conf_ver) or (list(versions.values())[-1] if versions else None)
        enabled = bool((cfg.get(sid) or {}).get("enabled", False))
        wf = db.fetchone(
            "SELECT e.experiment_id, e.created_at, r.conclusion, r.metrics_json FROM experiments e "
            "JOIN experiment_results r ON r.experiment_id=e.experiment_id WHERE e.name=? AND r.status='succeeded' "
            "AND e.uses_synthetic_data=0 ORDER BY e.created_at DESC LIMIT 1", (f"walk_forward:{sid}",))
        evidence = None
        if wf:
            m = from_json(wf["metrics_json"], {}) or {}
            m = m.get("metrics", m) if isinstance(m.get("metrics"), dict) else m
            evidence = {"experiment_id": wf["experiment_id"], "verdict": wf["conclusion"], "trades": m.get("n_trades"),
                        "expectancy": m.get("expectancy"), "sharpe": m.get("sharpe")}
        db_status, stage = (row["status"], row["stage"]) if row else (None, None)
        if not enabled or db_status in ("RETIRED", "PAUSED"):
            status = "DISABLED"
        elif db_status == "ACTIVE" and stage in ("PAPER", "PROMOTED"):
            status = "PAPER_ELIGIBLE"
        elif evidence and evidence["verdict"] in POSITIVE_VERDICTS:
            status = "VALIDATED"
        elif evidence:
            status = "WALK_FORWARD"
        else:
            status = "SHADOW"
        per_version = {}
        for ver, vr in versions.items():
            if not enabled or vr["status"] in ("RETIRED", "PAUSED"):
                vs = "DISABLED"
            elif vr["status"] == "ACTIVE" and vr["stage"] in ("PAPER", "PROMOTED"):
                vs = "PAPER_ELIGIBLE"
            else:
                vs = status if status not in ("DISABLED", "PAPER_ELIGIBLE") else ("WALK_FORWARD" if evidence else "SHADOW")
            per_version[ver] = {"status": vs, "db_status": vr["status"], "stage": vr["stage"], "version": ver}
        out[sid] = {"strategy_id": sid, "status": status, "db_status": db_status, "stage": stage,
                    "version": row["version"] if row else conf_ver, "enabled": enabled, "evidence": evidence,
                    "versions": per_version}
    return out


def describe_strategy(info: dict[str, Any] | None) -> str:
    if not info:
        return "unknown strategy"
    ev = info.get("evidence")
    base = f"{info['strategy_id']} is {info['status']}"
    if ev:
        exp = ev.get("expectancy")
        tail = f"{ev.get('trades')} OOS trades" + (f", {exp * 1e4:+.0f} bps/trade" if isinstance(exp, (int, float)) else "")
        return f"{base} (walk-forward {ev['verdict']}, {tail})"
    return base + " (no out-of-sample test on real data yet)"


__all__ = ["CANDIDATE_STATUSES", "POSITIVE_VERDICTS", "STRATEGY_STATUSES", "describe_strategy",
           "strategy_research_status"]
