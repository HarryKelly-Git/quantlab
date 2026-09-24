"""The LOCKED holdout: the last line of defence against overfitting by repeated peeking.

Any backtest/evaluation whose period reaches ``validation.holdout.start`` is refused with
:class:`HoldoutLockedError` unless it presents a valid unlock token. Unlocking is deliberately
awkward so it cannot happen by accident:

  * a written reason of at least ``validation.holdout.min_reason_chars`` characters and a named
    actor are required;
  * the unlock covers one explicit period (and optionally one experiment);
  * the token is SINGLE-USE — consumption is enforced by a PRIMARY KEY in the database, so a
    token cannot be replayed even from another process;
  * every unlock is written to ``holdout_access_log`` (append-only, forever) and every use to
    ``holdout_token_uses``; reports show :meth:`HoldoutGuard.access_count`.

Only a hash of the token is persisted, so reading the database never yields a usable token.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantlab.core.calendar import to_session
from quantlab.db.database import Database, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)


class HoldoutLockedError(RuntimeError):
    """Raised when a computation would touch the locked holdout without a valid unlock token."""


@dataclass(frozen=True)
class HoldoutGrant:
    access_log_id: int
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    experiment_id: str | None
    actor: str


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class HoldoutGuard:
    def __init__(self, config, db: Database | None = None):
        self.config = config
        self.db = db
        self.start = to_session(config.get("validation.holdout.start"))
        self.min_reason_chars = int(config.get("validation.holdout.min_reason_chars", 20))

    # -- queries ----------------------------------------------------------------------------------
    def touches(self, start, end) -> bool:
        """True if the period [start, end] reaches the holdout (only ``end`` matters)."""
        return to_session(end) >= self.start

    def access_count(self) -> int:
        if self.db is None:
            return 0
        row = self.db.fetchone("SELECT COUNT(*) AS n FROM holdout_access_log")
        return int(row["n"]) if row else 0

    def accesses(self) -> pd.DataFrame:
        """Every unlock ever granted, with when (if) its token was used. Never contains tokens."""
        if self.db is None:
            return pd.DataFrame(columns=["id", "at", "experiment_id", "period_start", "period_end",
                                         "reason", "actor", "used_at"])
        return self.db.query_df(
            "SELECT l.id, l.at, l.experiment_id, l.period_start, l.period_end, l.reason, l.actor, u.used_at "
            "FROM holdout_access_log l "
            "LEFT JOIN holdout_tokens t ON t.access_log_id = l.id "
            "LEFT JOIN holdout_token_uses u ON u.token_hash = t.token_hash "
            "ORDER BY l.id")

    # -- guard ------------------------------------------------------------------------------------
    def check(self, start, end, token: str | None = None, experiment_id: str | None = None) -> HoldoutGrant | None:
        """Raise HoldoutLockedError if [start, end] touches the holdout without a valid token.

        Returns None when the period is entirely before the holdout; otherwise consumes the token
        (single use) and returns the grant it was issued under.
        """
        s, e = to_session(start), to_session(end)
        if s > e:
            raise ValueError(f"period start {s.date()} is after end {e.date()}")
        if not self.touches(s, e):
            return None
        where = f"[{s.date()}, {e.date()}] reaches the locked holdout (starts {self.start.date()})"
        if not token:
            raise HoldoutLockedError(f"{where}. Unlock explicitly with HoldoutGuard.unlock(reason, actor, "
                                     "start, end) — every unlock is logged permanently.")
        if self.db is None:
            raise HoldoutLockedError(f"{where}: no database available to verify the unlock token")
        h = _hash(token)
        row = self.db.fetchone("SELECT * FROM holdout_tokens WHERE token_hash=?", (h,))
        if row is None:
            log_event(log, "holdout token rejected: unknown", level=30, period=[str(s.date()), str(e.date())])
            raise HoldoutLockedError(f"{where}: unknown holdout token")
        g_start, g_end = to_session(row["period_start"]), to_session(row["period_end"])
        if s < g_start or e > g_end:
            raise HoldoutLockedError(f"{where}: token covers only [{g_start.date()}, {g_end.date()}]")
        if row["experiment_id"] and experiment_id and row["experiment_id"] != experiment_id:
            raise HoldoutLockedError(f"{where}: token was issued for experiment {row['experiment_id']}")
        try:
            with self.db.transaction():
                self.db.insert("holdout_token_uses", {
                    "token_hash": h, "used_at": utcnow_iso(), "period_start": str(s.date()),
                    "period_end": str(e.date()), "experiment_id": experiment_id or row["experiment_id"],
                })
        except sqlite3.IntegrityError as exc:
            raise HoldoutLockedError(f"{where}: holdout token already used (tokens are single-use)") from exc
        log_event(log, "HOLDOUT ACCESSED", level=30, access_log_id=row["access_log_id"], actor=row["actor"],
                  period=[str(s.date()), str(e.date())], experiment_id=experiment_id or row["experiment_id"])
        return HoldoutGrant(int(row["access_log_id"]), g_start, g_end, row["experiment_id"], row["actor"])

    def unlock(self, reason: str, actor: str, start, end, experiment_id: str | None = None) -> str:
        """Issue a single-use token for [start, end] and log the unlock permanently."""
        if self.db is None:
            raise HoldoutLockedError("holdout unlock requires a database (every unlock must be logged)")
        reason = (reason or "").strip()
        actor = (actor or "").strip()
        if len(reason) < self.min_reason_chars:
            raise ValueError(f"holdout unlock needs a written reason of at least {self.min_reason_chars} characters")
        if not actor:
            raise ValueError("holdout unlock needs a named actor")
        s, e = to_session(start), to_session(end)
        if s > e:
            raise ValueError(f"period start {s.date()} is after end {e.date()}")
        if not self.touches(s, e):
            raise ValueError(f"[{s.date()}, {e.date()}] does not touch the holdout; no unlock needed")
        token = "hold_" + secrets.token_urlsafe(24)
        now = utcnow_iso()
        with self.db.transaction():
            cur = self.db.execute(
                "INSERT INTO holdout_access_log (at, experiment_id, period_start, period_end, reason, actor) "
                "VALUES (?, ?, ?, ?, ?, ?)", (now, experiment_id, str(s.date()), str(e.date()), reason, actor))
            self.db.insert("holdout_tokens", {
                "token_hash": _hash(token), "access_log_id": cur.lastrowid, "issued_at": now,
                "period_start": str(s.date()), "period_end": str(e.date()), "experiment_id": experiment_id,
                "actor": actor,
            })
        log_event(log, "HOLDOUT UNLOCKED", level=30, actor=actor, reason=reason,
                  period=[str(s.date()), str(e.date())], experiment_id=experiment_id)
        return token

    def describe(self) -> dict[str, Any]:
        return {"holdout_start": str(self.start.date()), "access_count": self.access_count()}
