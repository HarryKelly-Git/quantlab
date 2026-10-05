"""Prop-firm simulator: every rule checked on hand-built day sequences with known answers."""
from __future__ import annotations

import numpy as np
import pytest

from quantlab.futures.propsim import (Account, PropRules, RulesError, bootstrap_days, load_rules, parametric_days,
                                      simulate, simulate_path)


def rules(**kw) -> PropRules:
    base = dict(name="TEST", start_balance=50_000, profit_target=3_000, max_drawdown=2_000)
    base.update(kw)
    return PropRules(**base)


def days(*rows):
    """rows of (pnl, low, high); a bare number means a flat day with no excursion."""
    rows = [(r, min(r, 0), max(r, 0)) if not isinstance(r, tuple) else r for r in rows]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2]


# ---------------------------------------------------------------- drawdown floors
def test_static_floor_never_moves():
    a = Account(rules(drawdown_type="static"))
    for x in (1500, 1500):
        assert a.step(x, 0, x) == "ok"
    assert a.floor == 48_000
    assert a.step(-4_900, -4_900, 0) == "ok"          # balance 48,100 > 48,000
    assert a.step(-200, -200, 0) == "failed"


def test_trailing_eod_floor_follows_closes_not_intraday_highs():
    a = Account(rules(drawdown_type="trailing_eod"))
    a.step(500, -100, 1_500)                             # intraday high 1,500 does not count
    assert a.floor == 48_500
    a.step(-2_400, -2_400, 0)                            # 48,100 > 48,500? no: 50,500-2,400 = 48,100 < 48,500
    assert a.days == 2


def test_trailing_eod_breach_fails_intraday():
    a = Account(rules(drawdown_type="trailing_eod"))
    a.step(1_000, 0, 1_000)                              # floor 49,000
    assert a.step(500, -1_999, 600) == "ok"              # 51,000 - 1,999 = 49,001: above the floor
    assert a.step(500, -2_500, 600) == "failed"          # 51,500 - 2,500 = 49,000 touches the floor (49,500 now)


def test_trailing_intraday_assumes_high_before_low():
    a = Account(rules(drawdown_type="trailing_intraday"))
    # high +1,500 raises the floor to 49,500 FIRST; the later -100 low leaves 49,900 > floor: survives
    assert a.step(0, -100, 1_500) == "ok"
    assert a.floor == 49_500
    # next day: high +0, low -450 -> 49,550 > 49,500 ok; low -500 -> breach
    assert a.step(0, -450, 0) == "ok"
    assert a.step(0, -500, 0) == "failed"


def test_trail_lock_stops_floor_at_start():
    a = Account(rules(drawdown_type="trailing_eod", trail_lock_profit=0))
    for _ in range(5):
        a.step(1_000, 0, 1_000)
    assert a.floor == 50_000                              # locked at start, not 53,000


# ---------------------------------------------------------------- daily loss limit
def test_dll_stop_day_caps_the_loss_and_continues():
    a = Account(rules(daily_loss_limit=1_000, dll_action="stop_day"))
    assert a.step(-1_800, -1_800, 0) == "ok"
    assert a.balance == 49_000                            # flattened at exactly -1,000


def test_dll_fail_mode_fails():
    a = Account(rules(daily_loss_limit=1_000, dll_action="fail"))
    assert a.step(-1_200, -1_200, 0) == "failed"


def test_floor_breach_beats_dll_stop():
    a = Account(rules(max_drawdown=800, daily_loss_limit=1_000))
    assert a.step(-1_500, -1_500, 0) == "failed"          # the floor (-800) is hit before the DLL (-1,000)


# ---------------------------------------------------------------- evaluation
def test_pass_needs_target_min_days_and_consistency():
    r = rules(min_eval_days=3, eval_consistency=0.5)
    # day 1 makes the whole target in one day: not consistent, and too few days
    res = simulate_path(r, *days(3_000, 100, 100, 3_000), max_evaluations=1)
    assert res.passes == 1 and res.first_pass_day == 4  # best 3,000 <= 0.5 * 6,200 on day 4


def test_failed_eval_pays_reset_fee_and_retries():
    r = rules(eval_fee=150, reset_fee=100, min_eval_days=1)
    res = simulate_path(r, *days(-2_100, 3_000), max_evaluations=2)
    assert res.evaluations == 2 and res.passes == 1
    assert res.fees == 250


def test_recurring_eval_fee():
    r = rules(eval_fee=100, eval_fee_period_days=2)
    res = simulate_path(r, *days(10, 10, 10, 10, 10), max_evaluations=1)
    assert res.fees == 300                                # day 0, day 2, day 4


# ---------------------------------------------------------------- funded + payouts
def test_payout_split_cap_and_keep():
    r = rules(payout_split=0.9, payout_cap=2_000, payout_keep=500, payout_min_profit=1_000,
              min_days_between_payouts=1, activation_fee=50, eval_fee=100)
    res = simulate_path(r, *days(3_000, 4_000), max_evaluations=1)
    # eval passes day 1; funded day 2 profit 4,000 -> withdrawable 3,500 -> capped 2,000 -> trader 1,800
    assert res.n_payouts == 1 and res.payouts == pytest.approx(1_800)
    assert res.fees == 150 and res.net == pytest.approx(1_650)


def test_min_days_between_payouts():
    r = rules(min_days_between_payouts=3)
    res = simulate_path(r, *days(3_000, 500, 500, 500), max_evaluations=1)
    assert res.n_payouts == 1 and res.day_first_payout == 4


def test_funded_failure_buys_a_new_evaluation():
    r = rules(eval_fee=100, activation_fee=0)
    res = simulate_path(r, *days(3_000, -2_500, 3_000), max_evaluations=5)
    assert res.funded_failures == 1 and res.evaluations == 2 and res.fees == 200


def test_max_payouts_ends_the_path():
    r = rules(max_payouts=1)
    res = simulate_path(r, *days(3_000, 1_000, 1_000, 1_000))
    assert res.n_payouts == 1 and res.day_first_payout == 2


# ---------------------------------------------------------------- unknowns, determinism, streams
def test_unknown_fields_refuse_to_simulate(tmp_path):
    p = tmp_path / "firm.yaml"
    p.write_text("name: X\nstart_balance: 50000\nprofit_target: 3000\nmax_drawdown: UNKNOWN\n")
    r = load_rules(p)
    assert r.unknown == ("max_drawdown",)
    with pytest.raises(RulesError):
        simulate_path(r, *days(1))


def test_monte_carlo_is_deterministic_for_a_seed():
    r = rules(eval_fee=100, daily_loss_limit=1_000)
    a = simulate(r, parametric_days(0, 400), n_paths=300, seed=7)
    b = simulate(r, parametric_days(0, 400), n_paths=300, seed=7)
    assert a == b
    c = simulate(r, parametric_days(0, 400), n_paths=300, seed=8)
    assert c != a


def test_no_edge_no_fees_no_payout_structure_is_bounded():
    """With zero fees the trader can never be below zero net: losses are borne by the account."""
    r = rules(eval_fee=0, activation_fee=0)
    out = simulate(r, parametric_days(0, 400), n_paths=200, seed=1)
    assert out["p_net_loss"] == 0.0 and out["expected_fees"] == 0.0


def test_bootstrap_draws_only_history():
    hist = days(100, -50, 30, 20, -10, 5)
    fn = bootstrap_days(*hist, block=2)
    p, lo, hi = fn(np.random.default_rng(0), 50)
    assert set(np.round(p, 6)) <= set(np.round(hist[0], 6)) and len(p) == 50
    assert np.all(lo <= np.minimum(p, 0)) and np.all(hi >= np.maximum(p, 0))
