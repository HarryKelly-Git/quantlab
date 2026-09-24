"""Application context: the one object CLI commands, the daily pipeline and the dashboard share.

Also owns run bookkeeping: every pipeline/backtest/ingest run gets a ``runs`` row with the git
commit, dirty flag, config hash and full config, so any output can be traced and reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.config import Config, load_config
from quantlab.core.types import SystemState, new_id
from quantlab.data.store import MarketDataStore
from quantlab.db.database import Database, git_info, open_db, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, set_run_id, setup_logging
from quantlab.secrets import load_dotenv

log = get_logger("context")


@dataclass
class AppContext:
    config: Config
    db: Database
    store: MarketDataStore

    @property
    def root(self) -> Path:
        return self.config.root

    @classmethod
    def create(cls, config: Config | None = None, db_path: str | Path | None = None, init_logging: bool = True) -> "AppContext":
        cfg = config or load_config()
        load_dotenv(cfg.root / ".env")
        if init_logging:
            setup_logging(cfg.path("project.log_dir"), cfg.get("logging.level", "INFO"), cfg.get("logging.json_file", True))
        db = open_db(db_path or cfg.path("project.db_path"))
        store = MarketDataStore(cfg.path("project.data_dir"), db)
        ctx = cls(cfg, db, store)
        ctx._ensure_system_state()
        return ctx

    def close(self) -> None:
        self.db.close()

    # -- system state (kill switch storage; logic lives in quantlab.monitoring.killswitch) --------
    def _ensure_system_state(self) -> None:
        if self.db.fetchone("SELECT 1 FROM system_state WHERE id=1") is None:
            now = utcnow_iso()
            self.db.insert("system_state", {"id": 1, "state": SystemState.ACTIVE.value, "reason": "initialized",
                                            "changed_at": now, "changed_by": "system"})
            self.db.insert("system_state_log", {"state": SystemState.ACTIVE.value, "reason": "initialized",
                                                "changed_at": now, "changed_by": "system", "trigger": "init"})

    # -- runs -----------------------------------------------------------------------------------
    def start_run(self, kind: str, mode: str | None = None, as_of_date: str | None = None, notes: str = "") -> str:
        run_id = new_id(kind)
        commit, dirty = git_info(self.root)
        self.db.insert("runs", {
            "run_id": run_id, "kind": kind, "mode": mode, "as_of_date": as_of_date,
            "started_at": utcnow_iso(), "status": "running", "git_commit": commit,
            "git_dirty": None if dirty is None else int(dirty), "config_hash": self.config.hash,
            "config_json": to_json(self.config.as_dict()), "notes": notes,
        })
        set_run_id(run_id)
        log.info("run started kind=%s mode=%s as_of=%s commit=%s dirty=%s", kind, mode, as_of_date, commit, dirty)
        return run_id

    def finish_run(self, run_id: str, status: str = "succeeded", error: str | None = None) -> None:
        self.db.execute(
            "UPDATE runs SET status=?, finished_at=?, error=? WHERE run_id=?",
            (status, utcnow_iso(), error, run_id),
        )
        log.info("run finished status=%s", status)

    def reproducibility_stamp(self) -> dict[str, Any]:
        commit, dirty = git_info(self.root)
        return {"git_commit": commit, "git_dirty": dirty, "config_hash": self.config.hash}
