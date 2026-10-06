"""Persistent research queue + append-only results ledger + A-G classification (Parts 4, 39, 46).

Files (git-tracked, so the record survives this cloud container and is reviewable in pull requests):
  research/alpha/queue.json      one entry per hypothesis: id, name, category, source, statement, data
                                 required, status, priority, expected information value, compute,
                                 result, classification, confidence, next action
  research/alpha/ledger.jsonl    APPEND-ONLY: one line per experiment run (spec + hash, split, metrics,
                                 trial counts, seed, git commit, data manifest, conclusion)

Why files, not the bot's SQLite: the bot's database lives on the operator's PC; this phase runs in the
cloud. ``to_research_ledger`` maps entries onto the existing ``research.ledger`` statuses so they can
be imported there (A -> SUPPORTED needs forward evidence first; B/C/F -> INCONCLUSIVE; D/E/G -> REJECTED).

The OOS-once rule: a hypothesis's OOS split may be evaluated once per locked spec. A second OOS run
with a different spec is refused unless explicitly recorded as a NEW variant (which raises the family's
trial count, so every later statistic is deflated for it).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "research" / "alpha"
QUEUE = DIR / "queue.json"
LEDGER = DIR / "ledger.jsonl"

CLASSES = {
    "A": "STRONG EVIDENCE: positive out-of-sample expectancy with robust statistics",
    "B": "PROMISING: interesting evidence but insufficient data",
    "C": "REGIME-DEPENDENT: works only under identifiable conditions",
    "D": "IMPLEMENTATION PROBLEM: the signal exists but trading it destroys it",
    "E": "NO EDGE: fails realistic testing",
    "F": "DATA-LIMITED: cannot be tested properly yet",
    "G": "OVERFIT: the historical result disappears under robustness testing",
}
TO_LEDGER_STATUS = {"A": "SUPPORTED", "B": "INCONCLUSIVE", "C": "INCONCLUSIVE", "D": "REJECTED", "E": "REJECTED",
                    "F": "INCONCLUSIVE", "G": "REJECTED"}


class OOSReuseError(RuntimeError):
    pass


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet"]) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:          # pragma: no cover
        return "unknown"


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def load_queue() -> list[dict[str, Any]]:
    return json.loads(QUEUE.read_text()) if QUEUE.exists() else []


def save_queue(q: list[dict[str, Any]]) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(_clean(q), indent=1, sort_keys=False) + "\n")


def update_hypothesis(hid: str, **fields: Any) -> None:
    q = load_queue()
    for h in q:
        if h["id"] == hid:
            h.update(fields)
            save_queue(q)
            return
    raise KeyError(hid)


def ledger() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def trial_counts(family: str | None = None) -> dict[str, int]:
    rows = ledger()
    fam = [r for r in rows if family is None or r.get("family") == family]
    return {"runs": len(fam), "configs": len({r["spec_hash"] for r in fam}),
            "global_configs": len({r["spec_hash"] for r in rows})}


def append_run(*, hypothesis_id: str, family: str, spec: dict[str, Any], split: str, metrics: dict[str, Any],
               conclusion: str = "", seed: int | None = None, data: dict[str, Any] | None = None,
               new_variant_ok: bool = False) -> dict[str, Any]:
    h = spec_hash(spec)
    if split == "OOS":
        prior = [r for r in ledger() if r["hypothesis_id"] == hypothesis_id and r["split"] == "OOS"]
        if prior and any(r["spec_hash"] != h for r in prior) and not new_variant_ok:
            raise OOSReuseError(f"{hypothesis_id}: OOS already evaluated with spec(s) "
                                f"{sorted({r['spec_hash'] for r in prior})}; a different spec is a NEW variant")
    row = {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "hypothesis_id": hypothesis_id, "family": family,
           "split": split, "spec": spec, "spec_hash": h, "metrics": metrics, "conclusion": conclusion,
           "seed": seed, "data": data or {}, "git": git_commit()}
    DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(_clean(row), sort_keys=True) + "\n")
    return row


@dataclass
class Evidence:
    """What the Research Director looks at. Every field is filled by a different stage (quant, skeptic,
    statistician, execution, portfolio): no single stage can produce class A on its own."""
    data_ok: bool
    dev_t_net: float | None            # Newey-West t of mean daily net return, TRAIN+VALIDATION
    dev_t_gross: float | None
    oos_t_net: float | None            # None when OOS was not run (only specs that pass dev go to OOS)
    oos_mean_net: float | None
    oos_years_positive_frac: float | None
    deflated_sharpe_prob: float | None # DSR on dev with the family's trial count
    spa_p: float | None                # Hansen SPA across the family's variants (dev)
    pbo: float | None
    survives_2x_costs: bool | None
    survives_top5_removal: bool | None
    regime_conditional_pass: bool | None = None


def classify(e: Evidence) -> tuple[str, str]:
    if not e.data_ok:
        return "F", "data insufficient or failed quality checks"
    dev_pass = e.dev_t_net is not None and e.dev_t_net >= 2
    if not dev_pass:
        # OOS, when it was looked at, is information only: a failed development result is never upgraded
        oos_note = f"; OOS t {e.oos_t_net:.2f}" if e.oos_t_net is not None else ""
        if e.dev_t_gross is not None and e.dev_t_gross >= 2 and (e.dev_t_net is None or e.dev_t_net < 2):
            return "D", "gross signal significant in development, costs remove it" + oos_note
        if e.regime_conditional_pass:
            return "C", "unconditional fails; a pre-specified regime split passes in development" + oos_note
        if e.dev_t_net is not None and 1.0 <= e.dev_t_net < 2.0:
            if e.survives_top5_removal is False:
                return "E", "development t between 1 and 2 but the result depends on the top 5% of trades" + oos_note
            return "B", "development t between 1 and 2" + oos_note
        return "E", "no significant net edge in development" + oos_note
    if e.oos_t_net is None:
        return "B", "passed development; OOS not yet evaluated"
    if e.pbo is not None and e.pbo >= 0.5:
        return "G", f"PBO {e.pbo:.2f} >= 0.5"
    if e.oos_t_net < 1.0:
        return "G", f"passed development but OOS t {e.oos_t_net:.2f} < 1"
    strong = (e.oos_t_net >= 2 and (e.oos_mean_net or 0) > 0 and (e.deflated_sharpe_prob or 0) >= 0.95
              and (e.spa_p is None or e.spa_p < 0.05) and bool(e.survives_2x_costs) and bool(e.survives_top5_removal)
              and (e.oos_years_positive_frac or 0) >= 2 / 3)
    if strong:
        return "A", "OOS t >= 2, DSR >= 0.95, SPA p < 0.05, survives 2x costs and top-5% removal, 2/3 OOS years positive"
    return "B", "passed development; OOS positive but at least one robustness bar not met"
