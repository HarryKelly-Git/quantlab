"""Shared synthetic builders for tests/decision, tests/portfolio and tests/risk."""
from __future__ import annotations

from typing import Any

import pandas as pd

from quantlab.config import Config
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.experiments.registry import ExperimentRegistry


def seed_experiment_trades(db: Database, config: Config, strategy_id: str, version: str,
                          net_rets: list[float], *, segment: str = "oos:w1", gross_rets: list[float] | None = None,
                          start: str = "2019-01-02", uses_synthetic: bool = False, succeeded: bool = True,
                          symbol: str = "AAA", regimes: list[str | None] | None = None,
                          cost: float = 0.001) -> str:
    """Register one experiment + its ``backtest_trades`` rows for one strategy version.

    Trades are 2 sessions apart (signal -> entry -> exit) so they never collide on signal_date.
    """
    reg = ExperimentRegistry(db, config)
    exp_id = reg.start(f"test-{strategy_id}-{segment}-{len(net_rets)}", "backtest", uses_synthetic=uses_synthetic)
    if succeeded:
        reg.finish(exp_id, "succeeded")
    dates = pd.bdate_range(start, periods=max(1, len(net_rets)) * 3)
    rows = []
    for i, net in enumerate(net_rets):
        signal, exit_ = dates[i * 3], dates[i * 3 + 2]
        gross = gross_rets[i] if gross_rets is not None else net + cost
        rows.append({
            "experiment_id": exp_id, "trade_id": f"t{i}", "segment": segment, "strategy_id": strategy_id,
            "strategy_version": version, "symbol": symbol, "signal_date": signal.date().isoformat(),
            "entry_date": signal.date().isoformat(), "exit_date": exit_.date().isoformat(), "exit_reason": "TIME",
            "qty": 100, "entry_price": 10.0, "exit_price": 10.0 * (1 + gross), "gross_ret": gross,
            "cost_ret": cost, "net_ret": net, "pnl": net * 1000, "holding_sessions": 2,
            "regime": (regimes[i] if regimes is not None else None), "sector": None,
        })
    if rows:
        db.insert_many("backtest_trades", rows)
    return exp_id


def seed_shadow_trades(db: Database, strategy_id: str, version: str, rets: list[float], *,
                       start: str = "2019-06-03", holding_sessions: int = 5, symbol: str = "AAA",
                       is_synthetic: bool = False) -> list[str]:
    """Matured ``shadow_opportunities`` + ``shadow_outcomes`` rows (status=complete)."""
    from quantlab.core.types import new_id

    dates = pd.bdate_range(start, periods=max(1, len(rets)) * (holding_sessions + 2))
    ids = []
    for i, ret in enumerate(rets):
        as_of = dates[i * (holding_sessions + 2)]
        exit_ = dates[i * (holding_sessions + 2) + holding_sessions + 1]
        oid = new_id("opp")
        db.insert("shadow_opportunities", {
            "opportunity_id": oid, "candidate_id": new_id("cand"), "as_of_date": as_of.date().isoformat(),
            "symbol": symbol, "strategy_id": strategy_id, "strategy_version": version, "score": 1.0,
            "quant_reasoning": to_json({"direction": "LONG"}), "bot_decision": "NO_TRADE", "reject_stage": "AI",
            "reject_reason": "seeded", "holding_sessions": holding_sessions, "created_at": utcnow_iso(),
            "is_synthetic": int(is_synthetic),
        })
        db.insert("shadow_outcomes", {
            "opportunity_id": oid, "horizon_sessions": holding_sessions, "measured_at": utcnow_iso(),
            "entry_date": as_of.date().isoformat(), "exit_date": exit_.date().isoformat(), "ret": ret,
            "ret_hold": ret, "benchmark_ret": 0.0, "excess_ret": ret, "status": "complete",
        })
        ids.append(oid)
    return ids
