"""SYSTEM_PAUSED kill switch: blocks all new paper orders until a human resumes the system.

WHY it looks like this:
  * Pausing must always be possible and cheap (any trigger, any actor, idempotent). Resuming is
    the dangerous direction, so it requires a HUMAN actor and a non-empty reason: a machine that
    detected a problem must never decide by itself that the problem is gone.
  * Every change is written to the append-only ``system_state_log`` before ``system_state`` is
    updated, inside one transaction. Migration 095 enforces the same rules at the database level
    (logged-first, valid states, human-only resume, no delete), so other writers cannot bypass them.
  * Fail safe: a missing or unreadable ``system_state`` row is reported as SYSTEM_PAUSED, never as
    ACTIVE. Unknown state means no trading.

Actor strings used across monitoring/ and research/: ``"human"``, ``"human:<name>"``,
``"system"``, ``"system:<component>"``, ``"ai"``, ``"ai:<model>"``. Only the explicit ``human``
kind counts as a human; anything else (including free-form names) is not trusted as human review.
"""
from __future__ import annotations

import logging
from typing import Any

from quantlab.core.types import SystemState
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

ACTOR_KINDS = ("human", "ai", "system")


class SystemPausedError(RuntimeError):
    """Raised by :meth:`KillSwitch.assert_trading_allowed` when new orders are blocked."""


class ResumeNotAllowedError(PermissionError):
    """Resuming requires a human actor and an explicit reason."""


def actor_kind(actor: str | None) -> str | None:
    """Return ``human`` / ``ai`` / ``system`` for a well-formed actor string, else None.

    ``"human:harry"`` -> ``"human"``. A bare name like ``"harry"`` is NOT recognized: human review
    must be declared explicitly so an automated caller cannot pass as a human by accident.
    """
    if not isinstance(actor, str):
        return None
    text = actor.strip()
    if not text:
        return None
    kind, sep, name = text.partition(":")
    kind = kind.strip().lower()
    if kind not in ACTOR_KINDS:
        return None
    if sep and not name.strip():
        return None
    return kind


def is_human_actor(actor: str | None) -> bool:
    return actor_kind(actor) == "human"


def is_ai_actor(actor: str | None) -> bool:
    return actor_kind(actor) == "ai"


def require_actor(actor: str | None, what: str = "actor") -> str:
    """Validate an actor string (see module docstring) and return it stripped."""
    if actor_kind(actor) is None:
        raise ValueError(f"{what} must be 'human', 'ai' or 'system' (optionally ':<name>'), got {actor!r}")
    return str(actor).strip()


class KillSwitch:
    """Reads and changes the single-row ``system_state`` table (row created by AppContext)."""

    def __init__(self, db: Database):
        self.db = db

    # -- queries --------------------------------------------------------------------------------
    def state(self) -> tuple[SystemState, str, str | None]:
        """(state, reason, changed_at). Missing/corrupt state is reported as SYSTEM_PAUSED."""
        try:
            row = self.db.fetchone("SELECT state, reason, changed_at FROM system_state WHERE id = 1")
        except Exception as exc:  # unreadable DB => treat as paused (fail safe)
            return SystemState.PAUSED, f"system_state unreadable: {exc}", None
        if row is None:
            return SystemState.PAUSED, "system_state missing (uninitialized or corrupt): treated as SYSTEM_PAUSED", None
        try:
            st = SystemState(row["state"])
        except ValueError:
            return SystemState.PAUSED, f"system_state corrupt (state={row['state']!r}): treated as SYSTEM_PAUSED", row["changed_at"]
        return st, row["reason"] or "", row["changed_at"]

    def is_paused(self) -> bool:
        return self.state()[0] is SystemState.PAUSED

    def assert_trading_allowed(self) -> None:
        """Call immediately before submitting any new paper order."""
        st, reason, _ = self.state()
        if st is not SystemState.ACTIVE:
            raise SystemPausedError(f"SYSTEM_PAUSED: {reason}")

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM system_state_log ORDER BY id DESC LIMIT ?", (int(limit),))

    # -- changes --------------------------------------------------------------------------------
    def pause(self, reason: str, trigger: str, details: dict[str, Any] | None = None, actor: str = "system") -> bool:
        """Pause the system. Returns True if the state changed.

        Idempotent: when already paused, the state row is untouched; a NEW trigger/reason is still
        appended to ``system_state_log`` (so every cause is on record) but an identical repeat is not.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("pause requires a non-empty reason")
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValueError("pause requires a non-empty trigger")
        actor = require_actor(actor)
        reason, trigger = reason.strip(), trigger.strip()
        now = utcnow_iso()
        details_json = to_json(details) if details is not None else None
        with self.db.transaction():
            row = self.db.fetchone("SELECT state FROM system_state WHERE id = 1")
            if row is not None and row["state"] == SystemState.PAUSED.value:
                last = self.db.fetchone("SELECT state, reason, trigger FROM system_state_log ORDER BY id DESC LIMIT 1")
                if last is None or (last["state"], last["reason"], last["trigger"]) != (SystemState.PAUSED.value, reason, trigger):
                    self._log(SystemState.PAUSED, reason, now, actor, trigger, details_json)
                return False
            self._log(SystemState.PAUSED, reason, now, actor, trigger, details_json)
            self._write_state(SystemState.PAUSED, reason, now, actor, exists=row is not None)
        log_event(log, f"SYSTEM_PAUSED: {reason}", level=logging.ERROR, trigger=trigger, actor=actor)
        return True

    def resume(self, reason: str, actor: str) -> bool:
        """Resume trading. Requires a human actor and a reason. Returns True if the state changed."""
        if not is_human_actor(actor):
            raise ResumeNotAllowedError(
                f"resume requires a human actor ('human' or 'human:<name>'), got {actor!r}: human review is required"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ResumeNotAllowedError("resume requires a non-empty reason documenting the human review")
        actor, reason = actor.strip(), reason.strip()
        now = utcnow_iso()
        with self.db.transaction():
            row = self.db.fetchone("SELECT state FROM system_state WHERE id = 1")
            if row is not None and row["state"] == SystemState.ACTIVE.value:
                return False
            self._log(SystemState.ACTIVE, reason, now, actor, "human_resume", None)
            self._write_state(SystemState.ACTIVE, reason, now, actor, exists=row is not None)
        log_event(log, f"system resumed by {actor}: {reason}", level=logging.WARNING, actor=actor)
        return True

    # -- internals ------------------------------------------------------------------------------
    def _log(self, state: SystemState, reason: str, at: str, actor: str, trigger: str, details_json: str | None) -> None:
        self.db.insert("system_state_log", {
            "state": state.value, "reason": reason, "changed_at": at, "changed_by": actor,
            "trigger": trigger, "details_json": details_json,
        })

    def _write_state(self, state: SystemState, reason: str, at: str, actor: str, exists: bool) -> None:
        if exists:
            self.db.execute(
                "UPDATE system_state SET state = ?, reason = ?, changed_at = ?, changed_by = ? WHERE id = 1",
                (state.value, reason, at, actor),
            )
        else:
            self.db.insert("system_state", {"id": 1, "state": state.value, "reason": reason,
                                            "changed_at": at, "changed_by": actor})
