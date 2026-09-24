"""Reconcile the internal ledger (source of truth) against a broker's own view of cash/positions.

A mismatch or an unreachable broker is recorded (``reconciliations``) but this module never pauses
the system itself -- ``monitoring/killswitch.py`` is the only place system_state changes, reading
these rows (ARCHITECTURE.md section 10: "A mismatch or unknown broker state trips SYSTEM_PAUSED").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantlab.db.database import to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, PaperBroker
from quantlab.execution.ledger import Ledger
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.reconcile")


@dataclass
class ReconcileResult:
    ok: bool
    status: str  # ok | mismatch | broker_unavailable
    diffs: list[dict[str, Any]] = field(default_factory=list)
    broker_snapshot: dict[str, Any] | None = None
    internal_snapshot: dict[str, Any] | None = None


class Reconciler:
    def __init__(self, ledger: Ledger, broker: PaperBroker, *, cash_tolerance: float = 1.0,
                qty_tolerance: float = 1e-6):
        self.ledger = ledger
        self.broker = broker
        self.cash_tolerance = cash_tolerance
        self.qty_tolerance = qty_tolerance

    def reconcile(self, run_id: str | None = None) -> ReconcileResult:
        book = self.ledger.book
        try:
            if not self.broker.is_available():
                raise BrokerError(f"{self.broker.name} broker reports not available")
            account = self.broker.account()
            broker_positions = self.broker.positions()
        except BrokerError as exc:
            result = ReconcileResult(ok=False, status="broker_unavailable", diffs=[{"error": str(exc)}])
            self._write(run_id, book, result)
            log_event(log, "reconciliation: broker unavailable", book=book, error=str(exc))
            return result

        broker_snapshot = {"cash": account.cash, "equity": account.equity,
                           "positions": {p.symbol: p.qty for p in broker_positions}}
        internal_positions = {p["symbol"]: float(p["qty"]) for p in self.ledger.positions()}
        internal_snapshot = {"cash": self.ledger.cash(), "positions": internal_positions}

        diffs: list[dict[str, Any]] = []
        cash_delta = account.cash - internal_snapshot["cash"]
        if abs(cash_delta) > self.cash_tolerance:
            diffs.append({"field": "cash", "broker": account.cash, "internal": internal_snapshot["cash"],
                          "delta": cash_delta})
        for sym in sorted(set(broker_snapshot["positions"]) | set(internal_positions)):
            b_qty = broker_snapshot["positions"].get(sym, 0.0)
            i_qty = internal_positions.get(sym, 0.0)
            if abs(b_qty - i_qty) > self.qty_tolerance:
                diffs.append({"field": f"position:{sym}", "broker": b_qty, "internal": i_qty,
                              "delta": b_qty - i_qty})

        status = "ok" if not diffs else "mismatch"
        result = ReconcileResult(ok=(status == "ok"), status=status, diffs=diffs,
                                 broker_snapshot=broker_snapshot, internal_snapshot=internal_snapshot)
        self._write(run_id, book, result)
        if diffs:
            log_event(log, "reconciliation mismatch", book=book, diffs=diffs)
        return result

    def _write(self, run_id: str | None, book: str, result: ReconcileResult) -> None:
        self.ledger.db.insert("reconciliations", {
            "run_id": run_id, "book": book, "at": utcnow_iso(), "status": result.status,
            "broker_json": to_json(result.broker_snapshot), "internal_json": to_json(result.internal_snapshot),
            "diffs_json": to_json(result.diffs),
        })
