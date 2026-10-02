"""Stock-vs-option comparator on hand-built distributions and a hand-built chain: a 2-point world
where a call must win, one where the stock must win, one where nothing is worth trading; expiry
selection, the fixed strike grid, path-based stops/breakevens, SHORT theses, and recording."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantlab.options.book import record_evaluation
from quantlab.options.compare import (
    NO_TRADE, STOCK, Thesis, compare, select_expiration, session_offset, sessions_between,
)
from quantlab.options.distribution import MoveDistribution
from quantlab.options.settings import OptionsSettings

from .helpers import NOW, contract, quote

EXP = date(2026, 10, 8)              # exactly 5 NYSE sessions after 2026-10-01
SETTINGS = OptionsSettings(expiry_buffer_sessions=0, stock_round_trip_cost_bps=0.0)
THESIS = Thesis(symbol="TGT", direction="LONG", horizon_sessions=5, spot=100.0, stop_price=90.0)

CALLS = [(90, 10.0, 10.4, 0.90), (95, 5.6, 5.8, 0.70), (100, 2.9, 3.0, 0.50), (105, 1.0, 1.05, 0.30),
         (110, 0.30, 0.32, 0.12), (115, 0.20, 0.40, 0.06)]          # 115: spread too wide


def chain(rows=CALLS, type_="call", exp=EXP):
    cons, qs = [], {}
    for k, b, a, d in rows:
        c = contract(k, type_, exp=exp)
        cons.append(c)
        qs[c.symbol] = quote(c, b, a, iv=0.30, delta=d)
    return cons, qs


def run(dist, thesis=THESIS, rows=CALLS, type_="call", settings=SETTINGS):
    cons, qs = chain(rows, type_)
    early = contract(100, type_, exp=date(2026, 10, 2))                 # 1 session away: too short
    cons.append(early)
    qs[early.symbol] = quote(early, 1.0, 1.05, delta=0.5 if type_ == "call" else -0.5)
    return compare(thesis, dist, cons, qs, NOW, settings)


def by_kind(cmp):
    return {r.expression_id: r for r in cmp.results}


def test_calendar_and_expiry_selection():
    assert session_offset(date(2026, 10, 1), 5) == EXP
    assert sessions_between(date(2026, 10, 1), EXP) == 5
    assert select_expiration([date(2026, 10, 2), EXP, date(2026, 10, 16)], date(2026, 10, 1), 5, 0, 90)[0] == EXP
    assert select_expiration([date(2026, 10, 2), EXP, date(2026, 10, 16)], date(2026, 10, 1), 5, 1, 90)[0] == date(2026, 10, 16)
    assert select_expiration([date(2026, 10, 2)], date(2026, 10, 1), 5, 0, 90)[0] is None


def test_two_point_world_where_the_call_must_win():
    cmp = run(MoveDistribution(np.array([0.20, -0.08])))
    res = by_kind(cmp)
    stock = res[STOCK]
    assert stock.expected_pnl_per_risk == pytest.approx(0.6)            # (0.5*20 - 0.5*8) / 10
    assert cmp.expiration == EXP and cmp.t_remaining_years == 0.0      # held to expiry: exact intrinsic
    kinds = {r.kind for r in cmp.results}
    assert {"LONG_CALL", "CALL_DEBIT_SPREAD"} <= kinds
    c105 = next(r for r in cmp.results if r.kind == "LONG_CALL" and "C00105000" in r.expression_id)
    assert c105.expected_pnl_per_risk == pytest.approx((0.5 * 13.95 - 0.5 * 1.05) / 1.05)
    assert c105.p_profit == pytest.approx(0.5) and c105.max_loss_per_risk == 1.0
    assert c105.expected_loss_given_loss == pytest.approx(-1.0)
    assert cmp.choice == c105.expression_id and c105.chosen and not stock.chosen
    assert "LONG_CALL" in cmp.reason


def test_two_point_world_where_the_stock_must_win():
    cmp = run(np.array([0.015, 0.025]))                                  # plain ndarray input is accepted
    res = by_kind(cmp)
    assert res[STOCK].expected_pnl_per_risk == pytest.approx(0.2)
    best_opt = max((r for r in cmp.results if r.expression_id != STOCK), key=lambda r: r.expected_pnl_per_risk)
    assert best_opt.expected_pnl_per_risk < 0.2 + SETTINGS.min_option_advantage_per_risk
    assert cmp.choice == STOCK and res[STOCK].chosen
    assert cmp.distribution.label == "ndarray"


def test_nothing_worth_trading_is_no_trade():
    cmp = run(MoveDistribution(np.array([-0.05, 0.02])))
    assert cmp.choice == NO_TRADE and not any(r.chosen for r in cmp.results)


def test_option_must_beat_the_stock_by_the_margin():
    # same world as the call-wins test but with an absurd margin: the stock is chosen
    s = OptionsSettings(expiry_buffer_sessions=0, stock_round_trip_cost_bps=0.0, min_option_advantage_per_risk=50.0)
    assert run(MoveDistribution(np.array([0.20, -0.08])), settings=s).choice == STOCK


def test_fixed_grid_and_liquidity_rejections_are_recorded():
    cmp = run(MoveDistribution(np.array([0.20, -0.08])))
    ids = {r.expression_id for r in cmp.results}
    strikes = {int(i.split("C00")[1][:3]) for i in ids if i.startswith("LONG_CALL")}
    assert strikes == {95, 100, 105}                                     # delta .70/.50/.30, ATM, +1 EM
    spread = [i for i in ids if i.startswith("CALL_DEBIT_SPREAD")]
    assert len(spread) == 1 and "C00100000" in spread[0] and "C00105000" in spread[0]
    rejected = {c.symbol[-8:]: c.reasons for c in cmp.liquidity if not c.passed}
    assert rejected == {"00115000": ["SPREAD_TOO_WIDE"]}
    assert all(c.feed == "indicative" for c in cmp.liquidity)


def test_path_based_stop_and_breakeven():
    d = MoveDistribution(np.array([0.20, 0.20]), max_favourable=np.array([0.25, 0.25]),
                         max_adverse=np.array([0.12, 0.0]))
    cmp = run(d)
    st = by_kind(cmp)[STOCK]
    assert st.expected_pnl_per_risk == pytest.approx((0.5 * -10 + 0.5 * 20) / 10)   # path 1 stopped out
    assert st.p_breakeven == pytest.approx(1.0)
    end_only = by_kind(run(MoveDistribution(np.array([0.20, 0.20]))))[STOCK]
    assert end_only.expected_pnl_per_risk == pytest.approx(2.0)
    assert any("optimistic" in n for n in end_only.notes)


def test_short_thesis_uses_puts_and_never_short_stock():
    puts = [(105, 5.6, 5.8, -0.70), (100, 2.9, 3.0, -0.50), (95, 1.0, 1.05, -0.30)]
    th = Thesis(symbol="TGT", direction="SHORT", horizon_sessions=5, spot=100.0, stop_price=110.0)
    cmp = run(MoveDistribution(np.array([-0.20, 0.08])), thesis=th, rows=puts, type_="put")
    st = by_kind(cmp)[STOCK]
    assert st.kind == "SHORT_STOCK" and not st.eligible and st.max_loss_usd == float("inf")
    assert cmp.choice.startswith("LONG_PUT")
    assert all(r.kind in ("SHORT_STOCK", "LONG_PUT", "PUT_DEBIT_SPREAD") for r in cmp.results)


def test_no_qualifying_expiry_leaves_only_the_stock():
    th = Thesis(symbol="TGT", direction="LONG", horizon_sessions=6, spot=100.0, stop_price=90.0)
    cmp = run(MoveDistribution(np.array([0.20, -0.08])), thesis=th)
    assert cmp.expiration is None and [r.expression_id for r in cmp.results] == [STOCK]
    assert cmp.grid_notes[0]["status"] == "NO_EXPIRY"


def test_unknown_iv_before_expiry_makes_a_structure_ineligible():
    s = OptionsSettings(expiry_buffer_sessions=0, stock_round_trip_cost_bps=0.0)
    cons, qs = chain()
    later = [contract(k, exp=date(2026, 10, 16)) for k, *_ in CALLS[:3]]
    q2 = {c.symbol: quote(c, b, a, iv=None, delta=d) for c, (k, b, a, d) in zip(later, CALLS[:3])}
    # bids/asks below any Black-Scholes value -> no model IV either; held past the horizon -> needs IV
    q2 = {k: type(v)(**{**v.__dict__, "bid": 0.10, "ask": 0.105}) for k, v in q2.items()}
    cmp = compare(Thesis("TGT", "LONG", 3, 100.0, 90.0), np.array([0.05, -0.02]), later, q2, NOW, s)
    opts = [r for r in cmp.results if r.expression_id != STOCK]
    assert opts and all(not r.eligible and "UNKNOWN" in (r.ineligible_reason or "") for r in opts)


def test_recording_is_complete_and_append_only(db):
    cmp = run(MoveDistribution(np.array([0.20, -0.08])))
    eid = record_evaluation(db, cmp, spot_source="test")
    run_row = db.fetchone("SELECT * FROM options_eval_runs WHERE evaluation_id=?", (eid,))
    assert run_row["feed"] == "indicative" and run_row["chosen_expression"] == cmp.choice
    rows = db.fetchall("SELECT * FROM options_evaluations WHERE evaluation_id=?", (eid,))
    assert len(rows) == len(cmp.results) and sum(r["chosen"] for r in rows) == 1
    opt = next(r for r in rows if r["kind"] == "LONG_CALL")
    assert opt["quote_time_min"] and opt["feed"] == "indicative" and opt["legs_json"]
    checks = db.fetchall("SELECT * FROM options_liquidity_checks WHERE evaluation_id=?", (eid,))
    assert len(checks) == len(cmp.liquidity) and any(not c["passed"] for c in checks)
    for table in ("options_eval_runs", "options_evaluations", "options_liquidity_checks"):
        with pytest.raises(Exception, match="append-only"):
            db.execute(f"UPDATE {table} SET session_date='x'" if table != "options_liquidity_checks"
                       else f"UPDATE {table} SET passed=1")
        with pytest.raises(Exception, match="append-only"):
            db.execute(f"DELETE FROM {table}")
