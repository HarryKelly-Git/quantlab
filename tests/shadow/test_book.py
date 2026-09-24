"""ShadowBook: every candidate is recorded, whatever the decision; rows are immutable."""
from __future__ import annotations

import sqlite3

import pytest

from quantlab.core.types import (
    AIDecision,
    FinalDecision,
    MLPrediction,
    Objection,
    ObjectionSeverity,
    RejectStage,
)
from quantlab.db.database import from_json
from quantlab.shadow import ShadowBook, ShadowBookError, ShadowConflictError

from .support import make_candidate

_NO_TRADE_STAGES = [s for s in RejectStage if s is not RejectStage.NONE]


def test_trade_is_recorded_with_plan_and_direction(db):
    book = ShadowBook(db)
    c = make_candidate("aaa", stop=9.0, target=12.0, hold=7, ref=10.0)
    oid = book.record(c, FinalDecision.TRADE, RejectStage.NONE, "passed all layers", ai_decision=AIDecision.ACCEPT,
                      objections=[Objection("valuation", ObjectionSeverity.MINOR_CONCERN, "rich multiple")],
                      ml=MLPrediction("m1", "1", "positive_excess_return", 0.55, calibrated=True))
    row = db.fetchone("SELECT * FROM shadow_opportunities WHERE opportunity_id=?", (oid,))
    assert row["candidate_id"] == c.candidate_id and row["symbol"] == "AAA"
    assert row["bot_decision"] == "TRADE" and row["reject_stage"] == "NONE" and row["ai_decision"] == "ACCEPT"
    assert (row["stop_price"], row["target_price"], row["holding_sessions"], row["entry_ref_price"]) == (9.0, 12.0, 7, 10.0)
    assert row["as_of_date"] == c.as_of_date.isoformat() and row["is_synthetic"] == 0
    quant = from_json(row["quant_reasoning"])
    assert quant["direction"] == "LONG" and quant["reasons"] == ["ret_20d=0.1"]
    assert from_json(row["objections_json"])[0]["severity"] == "MINOR_CONCERN"
    ml = from_json(row["ml_json"])
    assert ml[0]["probability"] == 0.55 and ml[0]["calibrated"] is True


@pytest.mark.parametrize("stage", _NO_TRADE_STAGES, ids=lambda s: s.value)
def test_every_rejection_stage_is_recorded(db, stage):
    oid = ShadowBook(db).record(make_candidate(), FinalDecision.NO_TRADE, stage, f"stopped at {stage.value}")
    row = db.fetchone("SELECT bot_decision, reject_stage, reject_reason, ai_decision FROM shadow_opportunities "
                      "WHERE opportunity_id=?", (oid,))
    assert row == {"bot_decision": "NO_TRADE", "reject_stage": stage.value, "reject_reason": f"stopped at {stage.value}",
                   "ai_decision": None}


@pytest.mark.parametrize("decision,stage", [(FinalDecision.WATCH, RejectStage.AI), (FinalDecision.WATCH, RejectStage.NONE),
                                            (FinalDecision.UNKNOWN, RejectStage.DATA),
                                            (FinalDecision.UNKNOWN, RejectStage.NONE)])
def test_watch_and_unknown_are_recorded(db, decision, stage):
    oid = ShadowBook(db).record(make_candidate(), decision, stage, "x", ai_decision="UNKNOWN", is_synthetic=True)
    row = db.fetchone("SELECT * FROM shadow_opportunities WHERE opportunity_id=?", (oid,))
    assert row["bot_decision"] == decision.value and row["reject_stage"] == stage.value
    assert row["ai_decision"] == "UNKNOWN" and row["is_synthetic"] == 1


def test_all_candidates_of_a_run_are_kept_traded_or_not(db):
    book = ShadowBook(db)
    ids = [book.record(make_candidate(f"S{i}"), FinalDecision.TRADE if i == 0 else FinalDecision.NO_TRADE,
                       RejectStage.NONE if i == 0 else _NO_TRADE_STAGES[i % len(_NO_TRADE_STAGES)], "r")
           for i in range(8)]
    assert len(set(ids)) == 8
    assert db.fetchone("SELECT COUNT(*) AS n FROM shadow_opportunities")["n"] == 8


@pytest.mark.parametrize("decision,stage", [(FinalDecision.TRADE, RejectStage.AI), (FinalDecision.NO_TRADE, RejectStage.NONE),
                                            ("MAYBE", RejectStage.AI), (FinalDecision.NO_TRADE, "NOWHERE")])
def test_inconsistent_or_invalid_records_are_refused(db, decision, stage):
    with pytest.raises(ShadowBookError):
        ShadowBook(db).record(make_candidate(), decision, stage, "x")
    assert db.fetchone("SELECT COUNT(*) AS n FROM shadow_opportunities")["n"] == 0


def test_invalid_ai_decision_and_objection_refused(db):
    with pytest.raises(ShadowBookError):
        ShadowBook(db).record(make_candidate(), FinalDecision.NO_TRADE, RejectStage.AI, "x", ai_decision="YES")
    with pytest.raises(ShadowBookError):
        ShadowBook(db).record(make_candidate(), FinalDecision.NO_TRADE, RejectStage.AI, "x", objections=["no text"])


def test_rows_are_immutable(db):
    oid = ShadowBook(db).record(make_candidate(), FinalDecision.NO_TRADE, RejectStage.RISK, "too risky")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE shadow_opportunities SET bot_decision='TRADE' WHERE opportunity_id=?", (oid,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM shadow_opportunities WHERE opportunity_id=?", (oid,))
    assert db.fetchone("SELECT bot_decision FROM shadow_opportunities")["bot_decision"] == "NO_TRADE"


def test_rerecord_is_idempotent_but_conflicts_are_refused(db):
    book = ShadowBook(db)
    c = make_candidate()
    oid = book.record(c, FinalDecision.NO_TRADE, RejectStage.EV, "ev too low")
    assert book.record(c, FinalDecision.NO_TRADE, RejectStage.EV, "ev too low (resume)") == oid
    with pytest.raises(ShadowConflictError):
        book.record(c, FinalDecision.TRADE, RejectStage.NONE, "changed my mind")
    assert db.fetchone("SELECT COUNT(*) AS n FROM shadow_opportunities")["n"] == 1


def test_read_helpers_decode_json_and_filter_synthetic(db):
    book = ShadowBook(db)
    real = book.record(make_candidate("R", as_of="2024-02-01"), FinalDecision.NO_TRADE, RejectStage.AI, "x")
    book.record(make_candidate("S", as_of="2024-02-02"), FinalDecision.NO_TRADE, RejectStage.AI, "x", is_synthetic=True)
    got = book.get(real)
    assert got["quant"]["direction"] == "LONG" and got["is_synthetic"] is False
    assert [r["symbol"] for r in book.list()] == ["R"]
    assert [r["symbol"] for r in book.list(include_synthetic=True)] == ["S", "R"]
    assert book.list(start="2024-02-02", include_synthetic=True)[0]["symbol"] == "S"
    assert book.for_candidate(got["candidate_id"])["opportunity_id"] == real
