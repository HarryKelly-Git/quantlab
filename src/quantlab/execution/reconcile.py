"""Reconcile the internal ledger (source of truth) against a broker's own view of cash/positions.

A mismatch or an unreachable broker is recorded (``reconciliations``) but this module never pauses
the system itself -- ``monitoring/killswitch.py`` is the only place system_state changes, reading
these rows (ARCHITECTURE.md section 10: "A mismatch or unknown broker state trips SYSTEM_PAUSED").

Shared account: the options book OPT (``options/``) trades in the same Alpaca paper account. A book
reconciles only the broker positions of its own ``asset_classes`` (stock books: ``us_equity``), and
``cash_offset`` (the other books' net cash moved in the account) is added to the ledger's cash
before the cash comparison. An exercise/assignment that creates a stock position still mismatches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from quantlab.db.database import to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, BrokerPosition, PaperBroker
from quantlab.execution.ledger import Ledger
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.reconcile")

EQUITY_ASSET_CLASS = "us_equity"
OPTION_ASSET_CLASS = "us_option"
_OCC_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,5}\d{6}[CP]\d{8}$")


def position_asset_class(p: BrokerPosition) -> str:
    """Alpaca's ``asset_class`` of a position; without one (simulated brokers) an OCC-format symbol
    is an option and anything else an equity."""
    raw = (p.raw or {}).get("asset_class")
    if raw:
        return str(raw).lower()
    return OPTION_ASSET_CLASS if _OCC_RE.match(str(p.symbol or "")) else EQUITY_ASSET_CLASS


@dataclass
class ReconcileResult:
    ok: bool
    status: str  # ok | mismatch | broker_unavailable
    diffs: list[dict[str, Any]] = field(default_factory=list)
    broker_snapshot: dict[str, Any] | None = None
    internal_snapshot: dict[str, Any] | None = None


class Reconciler:
    def __init__(self, ledger: Ledger, broker: PaperBroker, *, cash_tolerance: float = 1.0,
                qty_tolerance: float = 1e-6, asset_classes: tuple[str, ...] = (EQUITY_ASSET_CLASS,),
                cash_offset: float | Callable[[], float] | None = None):
        self.ledger = ledger
        self.broker = broker
        self.cash_tolerance = cash_tolerance
        self.qty_tolerance = qty_tolerance
        self.asset_classes = tuple(a.lower() for a in asset_classes)
        self.cash_offset = cash_offset

    def reconcile(self, run_id: str | None = None) -> ReconcileResult:
        book = self.ledger.book
        try:
            if not self.broker.is_available():
                raise BrokerError(f"{self.broker.name} broker reports not available")
            account = self.broker.account()
            all_positions = self.broker.positions()
            offset = float(self.cash_offset() if callable(self.cash_offset) else (self.cash_offset or 0.0))
        except BrokerError as exc:
            result = ReconcileResult(ok=False, status="broker_unavailable", diffs=[{"error": str(exc)}])
            self._write(run_id, book, result)
            log_event(log, "reconciliation: broker unavailable", book=book, error=str(exc))
            return result

        broker_positions = [p for p in all_positions if position_asset_class(p) in self.asset_classes]
        broker_snapshot = {"cash": account.cash, "equity": account.equity,
                           "positions": {p.symbol: p.qty for p in broker_positions}}
        others = {p.symbol: p.qty for p in all_positions if position_asset_class(p) not in self.asset_classes}
        if others or offset:
            broker_snapshot["other_books"] = {"positions": others, "cash_offset": offset}
        internal_positions = {p["symbol"]: float(p["qty"]) for p in self.ledger.positions()}
        internal_snapshot = {"cash": self.ledger.cash(), "positions": internal_positions}

        diffs: list[dict[str, Any]] = []
        expected_cash = internal_snapshot["cash"] + offset
        cash_delta = account.cash - expected_cash
        if abs(cash_delta) > self.cash_tolerance:
            diffs.append({"field": "cash", "broker": account.cash, "internal": expected_cash,
                          "delta": cash_delta, **({"other_books_offset": offset} if offset else {})})
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
