"""Small hand-built builders shared by the human/shadow/counterfactual/compare tests (all synthetic)."""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.calendar import TradingCalendar
from quantlab.core.types import Candidate, Direction, PitStatus, TradePlan, new_id
from quantlab.data.panel import Panel, build_panel
from quantlab.db.database import Database, to_json, utcnow_iso

RETRIEVED = pd.Timestamp("2025-01-01", tz="UTC")


def make_bars(closes: dict[str, list[float | None]], opens: dict[str, list[float | None]] | None = None,
              start: str = "2024-01-01", volume: float = 1e6, provider: str = "test") -> tuple[pd.DataFrame, TradingCalendar]:
    n = len(next(iter(closes.values())))
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for sym, cl in closes.items():
        op = (opens or {}).get(sym, cl)
        for d, c, o in zip(dates, cl, op):
            if c is None:
                continue
            o = c if o is None else o
            rows.append({"symbol": sym, "date": d, "open": o, "high": max(o, c) * 1.01, "low": min(o, c) * 0.99,
                         "close": c, "volume": volume, "vwap": c, "trade_count": np.nan, "provider": provider,
                         "retrieved_at": RETRIEVED})
    return pd.DataFrame(rows), TradingCalendar.from_dates(dates)


def make_panel(closes, opens=None, start="2024-01-01", provider="test") -> tuple[Panel, TradingCalendar]:
    bars, cal = make_bars(closes, opens, start, provider=provider)
    return build_panel(bars, calendar=cal), cal


def make_candidate(symbol: str = "AAA", as_of: date | str | pd.Timestamp = "2024-01-03", strategy_id: str = "momentum_trend",
                   score: float = 1.0, stop: float | None = None, target: float | None = None, hold: int = 5,
                   ref: float | None = 10.0, direction: Direction = Direction.LONG,
                   reasons: list[str] | None = None) -> Candidate:
    return Candidate(
        symbol=symbol, as_of_date=pd.Timestamp(as_of).date(), strategy_id=strategy_id, strategy_version="1.0.0",
        score=score, direction=direction, features={"ret_20d": 0.1},
        plan=TradePlan(entry_ref_price=ref, stop_price=stop, target_price=target, holding_sessions=hold,
                       invalidation="stop or time"),
        risk={"atr14_pct": 0.02, "adv20": 2e7}, pit_status=PitStatus.PIT,
        reasons=reasons if reasons is not None else ["ret_20d=0.1"],
    )


def insert_candidate(db: Database, cand: Candidate, run_id: str | None = "run_1", opportunity_score: float | None = None,
                     rank: int | None = None, is_synthetic: bool = False, created_at: str | None = None) -> str:
    db.insert("candidates", {
        "candidate_id": cand.candidate_id, "run_id": run_id, "as_of_date": cand.as_of_date.isoformat(),
        "created_at": created_at or utcnow_iso(), "symbol": cand.symbol, "strategy_id": cand.strategy_id,
        "strategy_version": cand.strategy_version, "direction": cand.direction.value, "score": cand.score,
        "rank": rank, "opportunity_score": opportunity_score, "features_json": to_json(cand.features),
        "reasons_json": to_json(cand.reasons), "entry_convention": cand.plan.entry,
        "entry_ref_price": cand.plan.entry_ref_price, "stop_price": cand.plan.stop_price,
        "target_price": cand.plan.target_price, "holding_sessions": cand.plan.holding_sessions,
        "invalidation": cand.plan.invalidation, "risk_json": to_json(cand.risk),
        "pit_status": cand.pit_status.value, "is_synthetic": int(is_synthetic),
    })
    return cand.candidate_id


def insert_opportunity_with_outcome(db: Database, *, as_of: str, ret: float | None, symbol: str = "AAA",
                                    strategy_id: str = "s1", bot_decision: str = "NO_TRADE",
                                    reject_stage: str = "AI", ai_decision: str | None = None,
                                    excess: float | None = None, status: str = "complete",
                                    entry_date: str | None = None, is_synthetic: bool = False,
                                    candidate_id: str | None = None, hold: int = 5) -> tuple[str, str]:
    """Seed one shadow opportunity + its outcome directly (for counterfactual/compare tests)."""
    oid = new_id("opp")
    cid = candidate_id or new_id("cand")
    db.insert("shadow_opportunities", {
        "opportunity_id": oid, "candidate_id": cid, "as_of_date": as_of, "symbol": symbol,
        "strategy_id": strategy_id, "strategy_version": "1.0.0", "score": 1.0,
        "quant_reasoning": to_json({"direction": "LONG", "reasons": []}), "ai_decision": ai_decision,
        "bot_decision": bot_decision, "reject_stage": reject_stage, "reject_reason": "seeded",
        "holding_sessions": hold, "created_at": utcnow_iso(), "is_synthetic": int(is_synthetic),
    })
    if status is not None:
        entry = entry_date or (pd.Timestamp(as_of) + pd.offsets.BDay(1)).date().isoformat()
        db.insert("shadow_outcomes", {
            "opportunity_id": oid, "horizon_sessions": hold, "measured_at": utcnow_iso(),
            "entry_date": entry if status != "no_data" else None,
            "exit_date": (pd.Timestamp(entry) + pd.offsets.BDay(hold)).date().isoformat() if status != "no_data" else None,
            "ret": ret, "ret_hold": ret, "excess_ret": excess if excess is not None else ret,
            "benchmark_ret": 0.0, "status": status,
        })
    return oid, cid


def rows_equal(a: dict[str, Any], b: dict[str, Any], keys) -> bool:
    return all(a[k] == b[k] for k in keys)
