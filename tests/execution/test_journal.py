"""TradeJournal.explain end-to-end on a seeded DB: every stage in the decision chain (DATA ->
FEATURES -> STRATEGY -> MODEL -> AI -> OBJECTIONS -> EV -> RISK -> DECISION -> ORDER -> FILL ->
EXIT -> RESULT) must be traceable by joining the append-only audit tables, with missing links
reported as unavailable (UNCERTAINTY), never invented."""
from __future__ import annotations

from quantlab.core.costs import CostModel
from quantlab.core.types import Book, Side, TradePlan
from quantlab.db.database import open_db, to_json, utcnow_iso
from quantlab.execution.journal import EXPLAIN_STAGES
from quantlab.execution.ledger import Ledger
from quantlab.execution.sim_broker import SimBroker
from quantlab.execution.broker import OrderRequest

from .panels import build

DATES = ["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]


def zero_cost_model() -> CostModel:
    return CostModel(half_spread_tiers=((0.0, 0.0),), slippage_bps=0.0, commission_per_share=0.0,
                     commission_min_per_order=0.0, delisting_return=-0.30)


def _bars(prices, symbol):
    return [{"symbol": symbol, "date": d, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 5_000_000}
            for d, px in prices.items()]


def _seed_decision_chain(db) -> dict:
    now = utcnow_iso()
    candidate_id = "cand_test1"
    decision_id = "dec_test1"
    db.insert("candidates", {
        "candidate_id": candidate_id, "run_id": None, "as_of_date": DATES[0], "created_at": now,
        "symbol": "AAA", "strategy_id": "strat_a", "strategy_version": "v1", "direction": "LONG",
        "score": 0.75, "rank": 1, "opportunity_score": 0.5, "features_json": to_json({"mom_12_1": 0.2}),
        "reasons_json": to_json(["12-1 momentum breakout"]), "entry_convention": "next_open",
        "entry_ref_price": 100.0, "stop_price": 90.0, "target_price": None, "holding_sessions": 20,
        "invalidation": "close below 20dma", "risk_json": to_json({"atr": 3.0}), "pit_status": "PIT",
        "is_synthetic": 1,
    })
    db.insert("ml_predictions", {"candidate_id": candidate_id, "model_id": "model_x", "model_version": "v1",
                                 "as_of_date": DATES[0], "symbol": "AAA", "target": "fwd_20d_ret",
                                 "probability": 0.55, "prediction": 0.02, "created_at": now})
    db.insert("ai_assessments", {"assessment_id": "assess1", "candidate_id": candidate_id, "call_id": None,
                                 "role": "researcher", "provider": "anthropic", "model": "claude",
                                 "decision": "ACCEPT", "summary": "supportive evidence",
                                 "assessment_json": to_json({}), "created_at": now})
    db.insert("ai_objections", {"candidate_id": candidate_id, "assessment_id": "assess1", "category": "valuation",
                                "severity": "MINOR_CONCERN", "text": "multiple is stretched", "source": "ai",
                                "created_at": now})
    db.insert("decisions", {"decision_id": decision_id, "candidate_id": candidate_id, "run_id": None, "book": "BOT",
                            "decision": "TRADE", "reject_stage": "NONE", "reasons_json": to_json(["ev positive"]),
                            "ai_decision": "ACCEPT", "ev_json": to_json({"ev": 0.01, "n_obs": 250}),
                            "no_trade_json": None, "sizing_json": to_json({"risk_per_trade": 0.005}),
                            "created_at": now})
    db.insert("risk_checks", {"candidate_id": candidate_id, "decision_id": decision_id, "check_name": "max_positions",
                              "passed": 1, "severity": "CRITICAL", "reason": "ok", "details_json": to_json({}),
                              "created_at": now})
    return {"candidate_id": candidate_id, "decision_id": decision_id}


def test_explain_traces_every_stage_end_to_end():
    db = open_db(":memory:")
    ids = _seed_decision_chain(db)
    prices = {d: 100.0 for d in DATES}
    prices[DATES[2]] = 85.0  # breach the stop on DATES[2]'s close
    panel = build(_bars(prices, "AAA") + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = SimBroker(Book.BOT, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.BOT, starting_cash=100_000.0)

    # entry
    entry_order_id = "order_entry1"
    db.insert("order_intents", {"order_id": entry_order_id, "book": "BOT", "session_date": DATES[0],
                                "purpose": "entry", "intent_json": "{}", "created_at": utcnow_iso()})
    db.insert("orders", {
        "order_id": entry_order_id, "client_order_id": "c-e1", "book": "BOT", "broker": "sim",
        "candidate_id": ids["candidate_id"], "decision_id": ids["decision_id"], "human_decision_id": None,
        "trade_id": None, "purpose": "entry", "symbol": "AAA", "side": Side.BUY.value, "qty": 10,
        "order_type": "market", "time_in_force": "opg", "limit_price": None, "created_at": utcnow_iso(),
        "submitted_at": utcnow_iso(), "status": "accepted", "broker_order_id": "b1", "filled_qty": 0,
        "filled_avg_price": None, "last_update_at": utcnow_iso(), "raw_json": "{}",
    })
    broker.submit_order(OrderRequest(client_order_id="cb-e1", symbol="AAA", side=Side.BUY, qty=10,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    bo = broker.process_session(DATES[1], panel)[0]
    fr = ledger.apply_fill(order_id=entry_order_id, qty=bo.filled_qty, price=bo.filled_avg_price,
                           session_date=DATES[1], commission=bo.commission, modeled_cost=bo.modeled_cost,
                           filled_at=bo.filled_at, signal_date=DATES[0], candidate_id=ids["candidate_id"],
                           decision_id=ids["decision_id"], strategy_id="strat_a", strategy_version="v1",
                           plan=TradePlan(stop_price=90.0, holding_sessions=20))
    trade_id = fr.trade_id

    # exit (stop)
    exit_order_id = "order_exit1"
    db.insert("order_intents", {"order_id": exit_order_id, "book": "BOT", "session_date": DATES[2],
                                "purpose": "exit", "intent_json": to_json({"reason": "STOP"}),
                                "created_at": utcnow_iso()})
    db.insert("orders", {
        "order_id": exit_order_id, "client_order_id": "c-x1", "book": "BOT", "broker": "sim",
        "candidate_id": ids["candidate_id"], "decision_id": ids["decision_id"], "human_decision_id": None,
        "trade_id": trade_id, "purpose": "exit", "symbol": "AAA", "side": Side.SELL.value, "qty": 10,
        "order_type": "market", "time_in_force": "opg", "limit_price": None, "created_at": utcnow_iso(),
        "submitted_at": utcnow_iso(), "status": "accepted", "broker_order_id": "b2", "filled_qty": 0,
        "filled_avg_price": None, "last_update_at": utcnow_iso(), "raw_json": "{}",
    })
    broker.submit_order(OrderRequest(client_order_id="cb-x1", symbol="AAA", side=Side.SELL, qty=10,
                                     time_in_force="opg", decision_session=DATES[2]))
    broker.process_session(DATES[2], panel)
    bo2 = broker.process_session(DATES[3], panel)[0]
    ledger.apply_fill(order_id=exit_order_id, qty=bo2.filled_qty, price=bo2.filled_avg_price,
                      session_date=DATES[3], commission=bo2.commission, modeled_cost=bo2.modeled_cost,
                      filled_at=bo2.filled_at, exit_reason="STOP", held_sessions=2)

    result = ledger.journal.explain(trade_id)
    assert result["trade_id"] == trade_id
    assert result["status"] == "CLOSED"
    assert result["is_synthetic"] is True
    stage_names = [c["stage"] for c in result["chain"]]
    assert stage_names == list(EXPLAIN_STAGES)
    assert result["missing"] == []  # every stage has a record in this fully-seeded example

    stages = result["stages"]
    assert stages["DATA"]["content"]["as_of_date"] == DATES[0]
    assert stages["FEATURES"]["content"] == {"mom_12_1": 0.2}
    assert stages["STRATEGY"]["content"]["strategy_id"] == "strat_a"
    assert stages["MODEL"]["content"][0]["model_id"] == "model_x"
    assert stages["AI"]["content"][0]["decision"] == "ACCEPT"
    assert stages["OBJECTIONS"]["content"][0]["severity"] == "MINOR_CONCERN"
    assert stages["EV"]["content"]["ev"] == 0.01
    assert stages["RISK"]["content"][0]["check_name"] == "max_positions"
    assert stages["DECISION"]["content"]["bot"]["decision"] == "TRADE"
    assert len(stages["ORDER"]["content"]) == 1  # ORDER only lists entry orders (EXIT lists exit orders)
    assert len(stages["FILL"]["content"]) == 2  # both the entry and the exit fill
    assert stages["EXIT"]["content"]["exit_reason"] == "STOP"
    assert stages["RESULT"]["content"]["exit_reason"] == "STOP"
    assert stages["RESULT"]["content"]["net_pnl"] is not None


def test_explain_reports_missing_stages_as_uncertainty_not_invented():
    """A trade with no candidate/decision at all (e.g. a bare human decision) must show MODEL/AI/
    OBJECTIONS/EV/RISK as unavailable, never fabricated."""
    db = open_db(":memory:")
    now = utcnow_iso()
    db.insert("human_decisions", {"decision_id": "hd1", "candidate_id": None, "symbol": "AAA", "action": "BUY",
                                  "conviction": 4, "notes": "looks cheap", "decided_at": now,
                                  "ref_session_date": DATES[0], "ref_price": 100.0,
                                  "fill_convention": "next_open_after_decision", "supersedes_id": None,
                                  "created_at": now})
    prices = {d: 100.0 for d in DATES}
    panel = build(_bars(prices, "AAA") + _bars({d: 400.0 for d in DATES}, "SPY"))
    broker = SimBroker(Book.HUMAN, zero_cost_model(), starting_cash=100_000.0)
    ledger = Ledger(db, Book.HUMAN, starting_cash=100_000.0)
    order_id = "order_hd1"
    db.insert("order_intents", {"order_id": order_id, "book": "HUMAN", "session_date": DATES[0], "purpose": "entry",
                                "intent_json": "{}", "created_at": now})
    db.insert("orders", {
        "order_id": order_id, "client_order_id": "c-hd1", "book": "HUMAN", "broker": "sim", "candidate_id": None,
        "decision_id": None, "human_decision_id": "hd1", "trade_id": None, "purpose": "entry", "symbol": "AAA",
        "side": Side.BUY.value, "qty": 5, "order_type": "market", "time_in_force": "opg", "limit_price": None,
        "created_at": now, "submitted_at": now, "status": "accepted", "broker_order_id": "b1", "filled_qty": 0,
        "filled_avg_price": None, "last_update_at": now, "raw_json": "{}",
    })
    broker.submit_order(OrderRequest(client_order_id="cb-hd1", symbol="AAA", side=Side.BUY, qty=5,
                                     time_in_force="opg", decision_session=DATES[0]))
    broker.process_session(DATES[0], panel)
    bo = broker.process_session(DATES[1], panel)[0]
    fr = ledger.apply_fill(order_id=order_id, qty=bo.filled_qty, price=bo.filled_avg_price, session_date=DATES[1],
                           human_decision_id="hd1", signal_date=DATES[0])
    result = ledger.journal.explain(fr.trade_id)
    assert "MODEL" in result["missing"]
    assert "AI" in result["missing"]
    assert "OBJECTIONS" in result["missing"]
    assert "EV" in result["missing"]
    assert "RISK" in result["missing"]
    assert result["stages"]["MODEL"]["label"] == "UNCERTAINTY"
    assert result["stages"]["DECISION"]["content"]["human"]["action"] == "BUY"
