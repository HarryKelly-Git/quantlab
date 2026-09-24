"""Research ledger: hypotheses and the append-only record of everything done about them.

Rules (enforced here AND by DB triggers in 001_core.sql / 096_research_guards.sql):
  * A hypothesis must state what would falsify it: title, hypothesis, rationale, required data,
    proposed test and expected failure conditions are all required. It starts PROPOSED.
  * Its text never changes afterwards. :meth:`ResearchLedger.set_status` is the only code path that
    updates a hypothesis (status + updated_at), and it always appends a ``status_change`` entry
    ("STATUS <old> -> <new>: <reason>") first. Ledger entries are append-only; nothing is deleted.
  * AI may propose hypotheses and write entries, but an AI author can never mark a hypothesis
    SUPPORTED: generating ideas is cheap, declaring success needs a human or the system acting on
    recorded evidence.
  * SUPPORTED additionally requires at least one ``result`` entry linked to a registered experiment
    and must come from TESTING. "Supported" without a recorded result would be an opinion.

Authors / sources use the actor convention of :mod:`quantlab.monitoring.killswitch`:
``human``, ``ai``, ``system`` optionally followed by ``:<name>``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantlab.core.types import HypothesisStatus, InfoKind, new_id
from quantlab.db.database import Database, utcnow_iso
from quantlab.logging_setup import get_logger, log_event
from quantlab.monitoring.killswitch import actor_kind, require_actor

log = get_logger(__name__)

ENTRY_TYPES = ("idea", "implementation", "experiment", "result", "conclusion", "next_action", "status_change")
SOURCES = ("human", "ai", "system")

# Allowed status transitions. SUPPORTED is reachable only from TESTING (something must be tested).
TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.PROPOSED: frozenset({HypothesisStatus.TESTING, HypothesisStatus.REJECTED, HypothesisStatus.INCONCLUSIVE}),
    HypothesisStatus.TESTING: frozenset({HypothesisStatus.SUPPORTED, HypothesisStatus.REJECTED, HypothesisStatus.INCONCLUSIVE}),
    HypothesisStatus.SUPPORTED: frozenset({HypothesisStatus.TESTING, HypothesisStatus.REJECTED, HypothesisStatus.INCONCLUSIVE}),
    HypothesisStatus.REJECTED: frozenset({HypothesisStatus.TESTING}),
    HypothesisStatus.INCONCLUSIVE: frozenset({HypothesisStatus.TESTING, HypothesisStatus.REJECTED}),
}


class LedgerError(ValueError):
    """Invalid ledger operation (missing fields, unknown ids, illegal transition)."""


class LedgerPermissionError(PermissionError):
    """The author is not allowed to perform this change (e.g. AI declaring SUPPORTED)."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{field} is required and must be non-empty text")
    return value.strip()


class ResearchLedger:
    def __init__(self, db: Database):
        self.db = db

    # -- hypotheses -----------------------------------------------------------------------------
    def add_hypothesis(
        self,
        title: str,
        hypothesis: str,
        rationale: str,
        required_data: str,
        proposed_test: str,
        expected_failure_conditions: str,
        source: str,
        source_note_id: str | None = None,
        author: str | None = None,
    ) -> str:
        """Record a new PROPOSED hypothesis plus its ``idea`` ledger entry. Returns hypothesis_id.

        ``source`` is the origin kind (human | ai | system). ``author`` (default = source) may add a
        name, e.g. ``ai:claude-sonnet-5``; its kind must match the source.
        """
        fields = {
            "title": _required_text(title, "title"),
            "hypothesis": _required_text(hypothesis, "hypothesis"),
            "rationale": _required_text(rationale, "rationale"),
            "required_data": _required_text(required_data, "required_data"),
            "proposed_test": _required_text(proposed_test, "proposed_test"),
            "expected_failure_conditions": _required_text(expected_failure_conditions, "expected_failure_conditions"),
        }
        if source not in SOURCES:
            raise LedgerError(f"source must be one of {SOURCES}, got {source!r}")
        author = require_actor(author if author is not None else source, "author")
        if actor_kind(author) != source:
            raise LedgerError(f"author {author!r} does not match source {source!r}")
        if source_note_id is not None and self.db.fetchone(
                "SELECT 1 FROM human_notes WHERE note_id = ?", (source_note_id,)) is None:
            raise LedgerError(f"unknown human note {source_note_id!r}")
        hid = new_id("hyp")
        now = utcnow_iso()
        with self.db.transaction():
            self.db.insert("hypotheses", {"hypothesis_id": hid, "created_at": now, "source": source,
                                          "source_note_id": source_note_id, "status": HypothesisStatus.PROPOSED.value,
                                          "updated_at": now, **fields})
            self._append(hid, "idea", f"Hypothesis proposed: {fields['title']}", author, None, now)
        log_event(log, "hypothesis proposed", hypothesis_id=hid, source=source)
        return hid

    def get_hypothesis(self, hypothesis_id: str) -> dict[str, Any]:
        row = self.db.fetchone("SELECT * FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,))
        if row is None:
            raise LedgerError(f"unknown hypothesis {hypothesis_id!r}")
        return row

    def list_hypotheses(self, status: HypothesisStatus | str | None = None) -> list[dict[str, Any]]:
        if status is None:
            return self.db.fetchall("SELECT * FROM hypotheses ORDER BY created_at, hypothesis_id")
        return self.db.fetchall("SELECT * FROM hypotheses WHERE status = ? ORDER BY created_at, hypothesis_id",
                                (HypothesisStatus(status).value,))

    # -- entries --------------------------------------------------------------------------------
    def add_entry(self, hypothesis_id: str | None, entry_type: str, text: str, author: str,
                  experiment_id: str | None = None) -> int:
        """Append a ledger entry. ``status_change`` entries can only be written by :meth:`set_status`.
        ``hypothesis_id`` may be None for lab-wide entries (e.g. a general next_action)."""
        if entry_type not in ENTRY_TYPES:
            raise LedgerError(f"entry_type must be one of {ENTRY_TYPES}, got {entry_type!r}")
        if entry_type == "status_change":
            raise LedgerError("status_change entries are written only by set_status()")
        text = _required_text(text, "text")
        author = require_actor(author, "author")
        if hypothesis_id is not None:
            self.get_hypothesis(hypothesis_id)
        self._check_experiment(experiment_id)
        return self._append(hypothesis_id, entry_type, text, author, experiment_id, utcnow_iso())

    def set_status(self, hypothesis_id: str, status: HypothesisStatus | str, reason: str, author: str,
                   experiment_id: str | None = None) -> int:
        """The ONLY code path that changes a hypothesis. Appends the status_change entry, then
        updates status/updated_at, atomically. Returns the ledger entry id."""
        try:
            new = HypothesisStatus(status)
        except ValueError as exc:
            raise LedgerError(f"unknown hypothesis status {status!r}") from exc
        reason = _required_text(reason, "reason")
        author = require_actor(author, "author")
        self._check_experiment(experiment_id)
        if new is HypothesisStatus.SUPPORTED and actor_kind(author) == "ai":
            raise LedgerPermissionError("AI can generate ideas but cannot declare success: an 'ai' author may never set SUPPORTED")
        with self.db.transaction():
            row = self.get_hypothesis(hypothesis_id)
            old = HypothesisStatus(row["status"])
            if new is old:
                raise LedgerError(f"hypothesis {hypothesis_id} is already {old.value}")
            if new not in TRANSITIONS[old]:
                allowed = ", ".join(sorted(s.value for s in TRANSITIONS[old]))
                raise LedgerError(f"illegal transition {old.value} -> {new.value} (allowed: {allowed})")
            if new is HypothesisStatus.SUPPORTED:
                ev = self.db.fetchone(
                    "SELECT COUNT(*) AS n FROM research_ledger WHERE hypothesis_id = ? AND entry_type = 'result' "
                    "AND experiment_id IS NOT NULL", (hypothesis_id,))
                if not ev or ev["n"] == 0:
                    raise LedgerError("SUPPORTED requires at least one 'result' entry linked to an experiment")
            now = utcnow_iso()
            entry_id = self._append(hypothesis_id, "status_change", f"STATUS {old.value} -> {new.value}: {reason}",
                                    author, experiment_id, now)
            self.db.execute("UPDATE hypotheses SET status = ?, updated_at = ? WHERE hypothesis_id = ?",
                            (new.value, now, hypothesis_id))
        log_event(log, "hypothesis status changed", hypothesis_id=hypothesis_id, old=old.value, new=new.value, author=author)
        return entry_id

    def history(self, hypothesis_id: str) -> list[dict[str, Any]]:
        """All ledger entries for a hypothesis, oldest first."""
        self.get_hypothesis(hypothesis_id)
        return self.db.fetchall("SELECT * FROM research_ledger WHERE hypothesis_id = ? ORDER BY entry_id", (hypothesis_id,))

    # -- export ---------------------------------------------------------------------------------
    def export_markdown(self, path: str | Path | None = None) -> str:
        """Human-readable ledger. Every hypothesis is labeled HYPOTHESIS; AI-written text AI_OPINION."""
        lines = ["# Research ledger", "",
                 f"_Exported {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Hypotheses are "
                 f"{InfoKind.HYPOTHESIS.value} until tested; entries by AI authors are {InfoKind.AI_OPINION.value}._", ""]
        hyps = self.list_hypotheses()
        if not hyps:
            lines.append("_No hypotheses recorded._")
        for h in hyps:
            lines += [
                f"## {h['title']} ({h['hypothesis_id']})",
                f"- **Status:** {h['status']} | **Source:** {h['source']} | **Created:** {h['created_at']} | "
                f"**Updated:** {h['updated_at']}",
                f"- **[{InfoKind.HYPOTHESIS.value}]** {h['hypothesis']}",
                f"- **Rationale:** {h['rationale'] or ''}",
                f"- **Required data:** {h['required_data'] or ''}",
                f"- **Proposed test:** {h['proposed_test'] or ''}",
                f"- **Expected failure conditions:** {h['expected_failure_conditions'] or ''}",
                "",
                "| # | date | type | author | experiment | text |",
                "|---|---|---|---|---|---|",
            ]
            for e in self.history(h["hypothesis_id"]):
                label = f"[{InfoKind.AI_OPINION.value}] " if actor_kind(e["author"]) == "ai" else ""
                text = (label + e["text"]).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {e['entry_id']} | {e['entry_date']} | {e['entry_type']} | {e['author']} | "
                             f"{e['experiment_id'] or ''} | {text} |")
            lines.append("")
        general = self.db.fetchall("SELECT * FROM research_ledger WHERE hypothesis_id IS NULL ORDER BY entry_id")
        if general:
            lines += ["## Lab-wide entries", "", "| # | date | type | author | text |", "|---|---|---|---|---|"]
            for e in general:
                lines.append(f"| {e['entry_id']} | {e['entry_date']} | {e['entry_type']} | {e['author']} | "
                             f"{e['text'].replace('|', chr(92) + '|')} |")
        md = "\n".join(lines) + "\n"
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(md, encoding="utf-8")
        return md

    # -- internals ------------------------------------------------------------------------------
    def _check_experiment(self, experiment_id: str | None) -> None:
        if experiment_id is not None and self.db.fetchone(
                "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)) is None:
            raise LedgerError(f"unknown experiment {experiment_id!r} (register experiments before citing them)")

    def _append(self, hypothesis_id: str | None, entry_type: str, text: str, author: str,
                experiment_id: str | None, now: str) -> int:
        cur = self.db.execute(
            "INSERT INTO research_ledger (hypothesis_id, entry_date, entry_type, text, experiment_id, author, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hypothesis_id, now[:10], entry_type, text, experiment_id, author, now),
        )
        return int(cur.lastrowid)
