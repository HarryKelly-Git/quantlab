"""Structured logging. Human-readable console output + JSON-lines file (one object per event).

Every record carries the active ``run_id`` so any log line can be traced to a pipeline run,
backtest or experiment. Secret values are redacted before anything is written.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantlab.secrets import redact

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("quantlab_run_id", default=None)

_CONFIGURED = False


def set_run_id(run_id: str | None) -> contextvars.Token:
    return _run_id.set(run_id)


def current_run_id() -> str | None:
    return _run_id.get()


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra is not None:
            payload["data"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


def setup_logging(log_dir: str | Path | None = None, level: str = "INFO", json_file: bool = True) -> None:
    """Idempotent logging setup for CLI / pipeline / dashboard processes."""
    global _CONFIGURED
    root = logging.getLogger("quantlab")
    root.setLevel(level.upper())
    if _CONFIGURED:
        return
    root.propagate = False
    ctx = _ContextFilter()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s"))
    console.addFilter(ctx)
    root.addHandler(console)

    if json_file and log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "quantlab.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh.addFilter(ctx)
        root.addHandler(fh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("quantlab"):
        name = f"quantlab.{name}"
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **data: Any) -> None:
    """Log a message with structured data attached (lands in the JSON ``data`` field)."""
    logger.log(level, msg, extra={"data": data} if data else None)
