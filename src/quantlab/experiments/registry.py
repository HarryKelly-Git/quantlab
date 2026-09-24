"""Experiment registry: every research run is registered BEFORE it executes and its outcome is
appended afterwards — including failures, which are research data too.

An ``experiments`` row pins everything needed to reproduce the run: git commit + dirty flag, the
full effective config and its hash, the dataset snapshot (dataset ids per kind), strategy / model /
AI versions, cost assumptions, seeds, the period, how many variants were tried (for deflated
Sharpe), and whether synthetic data or the holdout were involved. Both tables are append-only.
"""
from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from quantlab.config import Config
from quantlab.core.types import new_id
from quantlab.db.database import Database, from_json, git_info, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)


class ExperimentRegistry:
    def __init__(self, db: Database, config: Config, root: str | Path | None = None):
        self.db = db
        self.config = config
        self.root = Path(root) if root else config.root

    def start(self, name: str, kind: str, *, period: tuple[str, str] | None = None,
              snapshot: dict[str, list[str]] | None = None, strategy_versions: dict[str, str] | None = None,
              model_versions: dict[str, str] | None = None, ai_models: dict[str, str] | None = None,
              cost_assumptions: dict[str, Any] | None = None, seeds: dict[str, int] | None = None,
              n_variants_tested: int = 1, hypothesis_id: str | None = None, uses_synthetic: bool = False,
              touches_holdout: bool = False, notes: str = "") -> str:
        if n_variants_tested < 1:
            raise ValueError("n_variants_tested must be >= 1 (count every variant tried, including failures)")
        commit, dirty = git_info(self.root)
        exp_id = new_id("exp")
        self.db.insert("experiments", {
            "experiment_id": exp_id, "name": name, "kind": kind, "hypothesis_id": hypothesis_id,
            "created_at": utcnow_iso(), "git_commit": commit, "git_dirty": None if dirty is None else int(dirty),
            "config_hash": self.config.hash, "config_json": to_json(self.config.as_dict()),
            "dataset_ids_json": to_json(snapshot or {}), "strategy_versions_json": to_json(strategy_versions or {}),
            "model_versions_json": to_json(model_versions or {}), "ai_models_json": to_json(ai_models or {}),
            "cost_assumptions_json": to_json(cost_assumptions if cost_assumptions is not None else self.config.section("costs")),
            "seeds_json": to_json(seeds or {}),
            "period_start": period[0] if period else None, "period_end": period[1] if period else None,
            "n_variants_tested": int(n_variants_tested), "uses_synthetic_data": int(bool(uses_synthetic)),
            "touches_holdout": int(bool(touches_holdout)), "notes": notes,
        })
        log_event(log, "experiment registered", experiment_id=exp_id, name=name, kind=kind, commit=commit,
                  dirty=dirty, synthetic=uses_synthetic)
        return exp_id

    def finish(self, experiment_id: str, status: str, metrics: dict[str, Any] | None = None,
               conclusion: str = "", artifacts_path: str | None = None) -> None:
        if status not in ("succeeded", "failed"):
            raise ValueError("status must be succeeded|failed")
        if self.db.fetchone("SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)) is None:
            raise KeyError(f"unknown experiment {experiment_id}")
        self.db.insert("experiment_results", {
            "experiment_id": experiment_id, "status": status, "metrics_json": to_json(metrics or {}),
            "artifacts_path": artifacts_path, "conclusion": conclusion, "completed_at": utcnow_iso(),
        })

    @contextmanager
    def run(self, name: str, kind: str, **start_kw) -> Iterator[str]:
        """Register, yield the id, and record 'failed' (with the traceback) if the body raises."""
        exp_id = self.start(name, kind, **start_kw)
        try:
            yield exp_id
        except BaseException as exc:
            self.finish(exp_id, "failed", {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]},
                        conclusion="experiment raised; see metrics.error")
            raise

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,))
        if row is None:
            return None
        res = self.db.fetchall("SELECT * FROM experiment_results WHERE experiment_id=? ORDER BY id", (experiment_id,))
        row["results"] = [{**r, "metrics": from_json(r["metrics_json"], {})} for r in res]
        return row

    def list(self, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = ("SELECT e.experiment_id, e.name, e.kind, e.created_at, e.git_commit, e.git_dirty, e.period_start, "
               "e.period_end, e.uses_synthetic_data, e.touches_holdout, e.n_variants_tested, "
               "(SELECT status FROM experiment_results r WHERE r.experiment_id = e.experiment_id ORDER BY id DESC LIMIT 1) AS status "
               "FROM experiments e")
        params: list[Any] = []
        if kind:
            sql += " WHERE e.kind=?"
            params.append(kind)
        sql += " ORDER BY e.created_at DESC LIMIT ?"
        params.append(limit)
        return self.db.fetchall(sql, params)
