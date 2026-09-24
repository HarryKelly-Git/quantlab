"""ShadowOutcomeTracker: equals simulate_plan, computed once, never from unseen sessions."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from quantlab.core.costs import CostModel
from quantlab.core.tradesim import simulate_plan
from quantlab.core.types import Direction, FinalDecision, RejectStage, TradePlan, new_id
from quantlab.db.database import utcnow_iso
from quantlab.shadow import ShadowBook, ShadowOutcomeTracker, panel_is_synthetic

from .support import make_candidate, make_panel

N = 20
AAA_CLOSE = [10, 10, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
AAA_OPEN = [10, 10, 10, 10.2, 10.8, 11.3, 11.8, 12.3, 12.8, 13.3, 13.8, 14, 14, 14, 14, 14, 14, 14, 14, 14]


def _world():
    closes = {
        "AAA": AAA_CLOSE,
        "BBB": [None, None, None] + [20.0] * (N - 3),             # no bar on the signal date
        "CCC": [30, 30, 30, 29, 28, 27] + [None] * (N - 6),        # stops trading after session 5
        "DDD": [40, 40, 40, 40, None, None] + [41.0] * (N - 6),   # halted inside the window
        "SPY": [100 + i for i in range(N)],
    }
    opens = {"AAA": AAA_OPEN}
    return make_panel(closes, opens)


def _record(db, symbol="AAA", as_of="2024-01-03", **kw):
    c = make_candidate(symbol, as_of=as_of, **kw)
    return ShadowBook(db).record(c, FinalDecision.NO_TRADE, RejectStage.AI, "test"), c


def _row(db, oid):
    return db.fetchone("SELECT * FROM shadow_outcomes WHERE opportunity_id=?", (oid,))


def test_outcome_equals_simulate_plan_and_hand_computation(db, config):
    panel, cal = _world()
    oid, c = _record(db, stop=9.0, target=12.0, hold=5, ref=10.0)
    assert ShadowOutcomeTracker(db, config).update(panel) == 1
    costs = CostModel.from_config(config)
    d = pd.Timestamp(c.as_of_date)
    exp = simulate_plan(panel, "AAA", d, TradePlan(entry_ref_price=10.0, stop_price=9.0, target_price=12.0,
                                                  holding_sessions=5), costs, Direction.LONG, benchmark="SPY")
    hold = simulate_plan(panel, "AAA", d, TradePlan(entry_ref_price=10.0, holding_sessions=5), costs,
                         Direction.LONG, benchmark="SPY")
    row = _row(db, oid)
    assert row["status"] == "complete" and row["horizon_sessions"] == 5
    assert row["ret"] == pytest.approx(exp.net_ret) and row["ret_hold"] == pytest.approx(hold.net_ret)
    assert row["benchmark_ret"] == pytest.approx(exp.benchmark_ret) and row["excess_ret"] == pytest.approx(exp.excess_ret)
    assert (row["mfe"], row["mae"]) == pytest.approx((exp.mfe, exp.mae))
    # hand check: target 12 hit on the close of session 6 -> exit at the open of session 7 (12.3);
    # entry at the open of session 3 (10.2); adv ~ $10M -> 10bps half-spread + 5bps slippage per side.
    assert row["entry_date"] == cal.sessions[3].date().isoformat() and row["entry_price"] == pytest.approx(10.2)
    assert row["exit_date"] == cal.sessions[7].date().isoformat() and row["exit_price"] == pytest.approx(12.3)
    assert row["hit_target"] == 1 and row["hit_stop"] == 0
    assert row["ret"] == pytest.approx(12.3 / 10.2 - 1 - 15e-4 * (1 + 12.3 / 10.2))  # exact two-leg cost
    # buy-and-hold: 5 sessions held (3..7) -> exit at the open of session 8 (12.8)
    assert row["ret_hold"] == pytest.approx(12.8 / 10.2 - 1 - 15e-4 * (1 + 12.8 / 10.2))
    det = db.fetchone("SELECT * FROM shadow_outcome_details WHERE opportunity_id=?", (oid,))
    assert det["exit_reason"] == "TARGET" and det["hold_exit_reason"] == "TIME" and det["method"].startswith("tradesim")
    assert det["cost_ret"] == pytest.approx(15e-4 * (1 + 12.3 / 10.2)) and det["direction"] == "LONG"
    assert det["panel_last_date"] == cal.sessions[-1].date().isoformat()


def test_idempotent_never_reinserts_or_updates(db, config):
    panel, _ = _world()
    oid, _ = _record(db, hold=5)
    tr = ShadowOutcomeTracker(db, config)
    assert tr.update(panel) == 1
    first = _row(db, oid)
    assert tr.update(panel) == 0 and tr.update(panel) == 0
    assert db.fetchone("SELECT COUNT(*) AS n FROM shadow_outcomes")["n"] == 1
    assert _row(db, oid) == first
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE shadow_outcomes SET ret=1.0")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM shadow_outcomes")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM shadow_outcome_details")


def test_not_computed_before_horizon_matures_and_never_uses_unseen_sessions(db, config):
    panel, cal = _world()
    oid, _ = _record(db, stop=9.0, target=12.0, hold=5)
    tr = ShadowOutcomeTracker(db, config)
    # signal index 2, horizon 5 -> the time-exit fill session (index 8) must be in the panel
    for k in range(0, 8):
        assert tr.update(panel.truncate(cal.sessions[k])) == 0
        assert tr.update(panel, as_of=cal.sessions[k]) == 0
    assert _row(db, oid) is None and tr.last_stats.get("pending", 0) >= 0
    assert tr.update(panel, as_of=cal.sessions[8]) == 1
    truncated_row = _row(db, oid)
    det = db.fetchone("SELECT panel_last_date FROM shadow_outcome_details WHERE opportunity_id=?", (oid,))
    assert det["panel_last_date"] == cal.sessions[8].date().isoformat()
    # the same opportunity measured on the FULL panel gives identical numbers (nothing after the
    # exit influences the outcome), and scrambling every later session changes nothing either.
    full = tr.evaluate(panel, {**db.fetchone("SELECT * FROM shadow_opportunities"), "cand_direction": None})
    for k in ("ret", "ret_hold", "entry_date", "exit_date", "excess_ret", "benchmark_ret", "mfe", "mae", "status"):
        assert full.row[k] == truncated_row[k]
    scrambled, _ = make_panel({"AAA": AAA_CLOSE[:9] + [1.0] * (N - 9), "SPY": [100 + i for i in range(9)] + [5.0] * (N - 9)},
                              {"AAA": AAA_OPEN[:9] + [1.0] * (N - 9)})
    alt = tr.evaluate(scrambled, {**db.fetchone("SELECT * FROM shadow_opportunities"), "cand_direction": None})
    assert alt.row["ret"] == truncated_row["ret"] and alt.row["ret_hold"] == truncated_row["ret_hold"]
    assert alt.row["excess_ret"] == truncated_row["excess_ret"]


def test_no_data_when_no_bar_on_signal_date(db, config):
    panel, _ = _world()
    oid, _ = _record(db, symbol="BBB", hold=5)
    assert ShadowOutcomeTracker(db, config).update(panel) == 1
    row = _row(db, oid)
    assert row["status"] == "no_data" and row["ret"] is None and row["entry_date"] is None


def test_symbol_missing_from_panel_is_skipped_not_no_data(db, config):
    panel, _ = _world()
    oid, _ = _record(db, symbol="ZZZ", hold=5)
    tr = ShadowOutcomeTracker(db, config)
    assert tr.update(panel) == 0
    assert _row(db, oid) is None and tr.last_stats == {"skipped": 1}


def test_delisting_needs_grace_period(db, config):
    panel, cal = _world()
    oid, _ = _record(db, symbol="CCC", hold=5, ref=30.0)
    tr = ShadowOutcomeTracker(db, config)
    grace = tr.delisting_grace
    # last CCC bar is session 5; declared delisted only after `grace` absent sessions
    assert tr.update(panel, as_of=cal.sessions[5 + grace - 1]) == 0
    assert tr.update(panel, as_of=cal.sessions[5 + grace]) == 1
    row = _row(db, oid)
    costs = CostModel.from_config(config)
    exp = simulate_plan(panel, "CCC", cal.sessions[2], TradePlan(entry_ref_price=30.0, holding_sessions=5), costs)
    assert row["status"] == "delisted" and exp.status == "delisted"
    assert row["ret"] == pytest.approx(exp.net_ret)
    # entry at the open of session 3 (29); exit value = last close (27) x (1 - 30%); adv ~ $28M -> 5+5 bps/side
    assert costs.delisting_return == pytest.approx(-0.30)
    assert row["ret"] == pytest.approx(27 * 0.7 / 29 - 1 - 10e-4 * (1 + 27 * 0.7 / 29))


def test_missing_bars_inside_window_wait(db, config):
    panel, cal = _world()
    oid, _ = _record(db, symbol="DDD", hold=5, ref=40.0)
    tr = ShadowOutcomeTracker(db, config)
    # DDD bars missing on sessions 4-5: held sessions 3,6,7,8,9 -> time exit fills at session 10
    assert tr.update(panel, as_of=cal.sessions[8]) == 0
    assert tr.update(panel, as_of=cal.sessions[9]) == 0
    assert tr.update(panel, as_of=cal.sessions[10]) == 1
    assert _row(db, oid)["exit_date"] == cal.sessions[10].date().isoformat()


def test_signal_after_panel_end_is_pending(db, config):
    panel, _ = _world()
    oid, _ = _record(db, as_of="2024-06-03", hold=5)
    assert ShadowOutcomeTracker(db, config).update(panel) == 0
    assert _row(db, oid) is None


def test_unknown_direction_is_never_guessed(db, config):
    panel, _ = _world()
    oid = new_id("opp")
    db.insert("shadow_opportunities", {
        "opportunity_id": oid, "candidate_id": "cand_missing", "as_of_date": "2024-01-03", "symbol": "AAA",
        "strategy_id": "x", "strategy_version": "1", "quant_reasoning": "free text, no direction",
        "bot_decision": "NO_TRADE", "reject_stage": "AI", "holding_sessions": 5, "created_at": utcnow_iso()})
    tr = ShadowOutcomeTracker(db, config)
    assert tr.update(panel) == 0 and tr.last_stats == {"skipped": 1}


def test_short_direction_matches_simulate_plan(db, config):
    panel, _ = _world()
    oid, c = _record(db, hold=4, stop=13.0, direction=Direction.SHORT)
    ShadowOutcomeTracker(db, config).update(panel)
    exp = simulate_plan(panel, "AAA", pd.Timestamp(c.as_of_date), TradePlan(entry_ref_price=10.0, stop_price=13.0,
                        holding_sessions=4), CostModel.from_config(config), Direction.SHORT, benchmark="SPY")
    assert _row(db, oid)["ret"] == pytest.approx(exp.net_ret)


def test_synthetic_and_real_never_mix(db, config):
    panel, _ = _world()
    c = make_candidate("AAA", hold=5)
    oid = ShadowBook(db).record(c, FinalDecision.NO_TRADE, RejectStage.AI, "x", is_synthetic=True)
    tr = ShadowOutcomeTracker(db, config)
    assert not panel_is_synthetic(panel)
    assert tr.update(panel) == 0                       # real panel never evaluates synthetic rows
    assert tr.update(panel, synthetic=True) == 1       # explicitly synthetic run does
    assert _row(db, oid)["status"] == "complete"


def test_default_horizon_used_when_plan_has_none(db, config):
    cfg = config.with_overrides({"shadow": {"default_horizon_sessions": 3}})
    panel, cal = _world()
    oid = new_id("opp")
    db.insert("shadow_opportunities", {
        "opportunity_id": oid, "candidate_id": "c1", "as_of_date": "2024-01-03", "symbol": "AAA", "strategy_id": "x",
        "strategy_version": "1", "quant_reasoning": '{"direction": "LONG"}', "bot_decision": "NO_TRADE",
        "reject_stage": "AI", "holding_sessions": None, "created_at": utcnow_iso()})
    tr = ShadowOutcomeTracker(db, cfg)
    assert tr.update(panel, as_of=cal.sessions[5]) == 0
    assert tr.update(panel, as_of=cal.sessions[6]) == 1
    assert _row(db, oid)["horizon_sessions"] == 3


def test_synthetic_bundle_integration(db, config, bundle):
    """On the synthetic world: every matured opportunity equals a direct simulate_plan call."""
    book = ShadowBook(db)
    panel = bundle.panel
    dates = panel.dates
    syms = [s for s in panel.symbols if s.startswith("SYN0")][:6]
    recorded = []
    rng = np.random.default_rng(3)
    for i, sym in enumerate(syms):
        d = dates[300 + 37 * i]
        close = float(panel.close.at[d, sym]) if np.isfinite(panel.close.at[d, sym]) else None
        c = make_candidate(sym, as_of=d, ref=close, hold=int(rng.integers(3, 30)),
                           stop=close * 0.93 if close else None, target=close * 1.1 if close else None)
        recorded.append((book.record(c, FinalDecision.NO_TRADE, RejectStage.EV, "x", is_synthetic=True), c))
    tr = ShadowOutcomeTracker(db, config)
    n = tr.update(bundle)
    assert n == len(recorded)
    costs = CostModel.from_config(config)
    for oid, c in recorded:
        row = _row(db, oid)
        exp = simulate_plan(panel, c.symbol, pd.Timestamp(c.as_of_date), c.plan, costs, Direction.LONG, benchmark="SPY")
        if exp.status in ("complete", "delisted"):
            assert row["status"] == exp.status and row["ret"] == pytest.approx(exp.net_ret)
        else:
            assert row["status"] == "no_data"
