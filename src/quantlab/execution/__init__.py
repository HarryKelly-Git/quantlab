"""Paper execution layer: the paper-broker abstraction (SimBroker / AlpacaPaperBroker), the
internal per-book ledger (source of truth), the exit engine, the trade journal, the daily
execution service and broker/ledger reconciliation.

NO LIVE MONEY EVER. The only broker endpoint this package will ever contact is
``https://paper-api.alpaca.markets`` (``quantlab.config.PAPER_TRADING_BASE_URL``), enforced in
:class:`~quantlab.execution.alpaca_paper.AlpacaPaperBroker` before every single request.
"""
from __future__ import annotations

from quantlab.execution.broker import (
    BrokerAccount,
    BrokerAuthError,
    BrokerError,
    BrokerNotConfigured,
    BrokerOrder,
    BrokerPosition,
    BrokerUnavailable,
    LiveTradingForbidden,
    OrderRequest,
    PaperBroker,
)
from quantlab.execution.sim_broker import SimBroker, paper_book
from quantlab.execution.alpaca_paper import AlpacaPaperBroker
from quantlab.execution.ledger import Ledger, LedgerError
from quantlab.execution.exits import ExitEngine, ExitSignal, OpenTrade, build_exit_engine
from quantlab.execution.journal import TradeJournal, JournalError
from quantlab.execution.service import PaperExecutionService, SyncResult
from quantlab.execution.reconcile import Reconciler, ReconcileResult

__all__ = [
    "BrokerAccount", "BrokerAuthError", "BrokerError", "BrokerNotConfigured", "BrokerOrder",
    "BrokerPosition", "BrokerUnavailable", "LiveTradingForbidden", "OrderRequest", "PaperBroker",
    "SimBroker", "paper_book", "AlpacaPaperBroker",
    "Ledger", "LedgerError",
    "ExitEngine", "ExitSignal", "OpenTrade", "build_exit_engine",
    "TradeJournal", "JournalError",
    "PaperExecutionService", "SyncResult",
    "Reconciler", "ReconcileResult",
]
