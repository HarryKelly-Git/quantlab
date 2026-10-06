"""Alpha-discovery dashboard data (Part 45): read-only summary of research/alpha/ (queue, results,
ledger, data-quality reports). No database and no order path: research files only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3] / "research" / "alpha"


def _load(name: str) -> Any:
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else None


def alpha_state() -> dict[str, Any]:
    queue = _load("queue.json") or []
    ledger_lines = (ROOT / "ledger.jsonl").read_text().splitlines() if (ROOT / "ledger.jsonl").exists() else []
    ledger = [json.loads(x) for x in ledger_lines if x.strip()]
    results_dir = ROOT / "results"
    fams = []
    for f in sorted(results_dir.glob("*.json")) if results_dir.exists() else []:
        if f.stem.endswith("__dev"):
            continue
        r = json.loads(f.read_text())
        fs = r.get("family_stats") or {}
        sel = fs.get("selected") or r.get("selected")
        oos = r.get("oos") or {}
        fams.append({"family": r.get("family", f.stem), "variants": fs.get("n_variants") or len(r.get("variants", {}) or {}),
                     "selected": sel, "spa_p": (fs.get("spa") or {}).get("p_value"), "pbo": (fs.get("pbo") or {}).get("pbo"),
                     "oos_sharpe": oos.get("sharpe"), "oos_t": oos.get("t_mean_nw") or oos.get("weekly_t_nw"),
                     "class": r.get("classification"), "reason": r.get("classification_reason")})
    classes: dict[str, int] = {}
    for h in queue:
        c = h.get("classification") or "QUEUED"
        classes[c] = classes.get(c, 0) + 1
    options = _load("results/options_vol.json") or {}
    base = options.get("baseline", {})
    opt_rows = [{"split": k, "n": v.get("n"), "share_rv_below_iv": v.get("share_rv_below_iv"),
                 "long_straddle_at_ask": v.get("long_straddle_ask_mean"), "long_straddle_at_mid": v.get("long_straddle_mid_mean"),
                 "iron_fly": v.get("iron_fly_mean"), "median_spread_frac": v.get("median_spread_frac")} for k, v in base.items()]
    acc = _load("results/H33_forecast_accuracy_by_liquidity.json") or {}
    acc_rows = [{"subset": sub, "split": sp, "n": v.get("n"), "mse_log_iv": v.get("mse_log_iv"),
                 "mse_log_iv_debiased": v.get("mse_log_iv_debiased"), "mse_log_forecast": v.get("mse_log_forecast"),
                 "t_forecast_in_encompassing": v.get("t_fc")} for sub, d in acc.items() for sp, v in d.items()]
    port = _load("results/H36_portfolio_construction.json") or {}
    port_rows = [{"method": m, "oos_sharpe": (v.get("OOS") or {}).get("sharpe"), "oos_t": (v.get("OOS") or {}).get("t_mean_nw"),
                  "walk_forward_oos_sharpe": (v.get("walk_forward_OOS") or {}).get("sharpe")} for m, v in (port.get("portfolios") or {}).items()]
    q_eq, q_opt = _load("quality_equity.json") or {}, _load("quality_options.json") or {}
    return {
        "n_hypotheses": len(queue), "n_done": sum(1 for h in queue if h.get("status") == "DONE"),
        "n_queued": sum(1 for h in queue if h.get("status") == "QUEUED"), "classes": classes,
        "n_runs": len(ledger), "n_configs": len({r.get("spec_hash") for r in ledger}),
        "n_oos_evaluations": sum(1 for r in ledger if r.get("split") == "OOS"),
        "queue": [{k: h.get(k) for k in ("id", "name", "category", "status", "classification", "result", "next_action")} for h in queue],
        "families": fams, "options_baseline": opt_rows, "forecast_accuracy": acc_rows, "portfolio": port_rows,
        "quality": [{"dataset": "equity bars", **{k: q_eq.get(k) for k in ("rows", "symbols", "duplicates", "high_below_low", "status")}}] +
                   [{"dataset": f"options {m}", "rows": v.get("rows"), "snapshot_timing": v.get("snapshot_timing"), "status": v.get("status")}
                    for m, v in q_opt.items()],
    }
