"""Sprint P5 options execution engine: fill levels, settlement costs, classification, model expectations and
25-delta strangle selection, on hand-made SYNTHETIC numbers."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantlab.alpha import opt_exec as ox


def test_fill_levels_and_mirror():
    bid, ask = np.array([1.0]), np.array([1.2])
    assert ox.fill(bid, ask, "MID")[0] == 1.1
    assert math.isclose(ox.fill(bid, ask, "CONSERVATIVE")[0], 1.15)
    assert math.isclose(ox.fill(bid, ask, "PESSIMISTIC")[0], 1.2)
    assert math.isclose(ox.fill(bid, ask, "WORST_REASONABLE")[0], 1.225)
    assert math.isclose(ox.fill(bid, ask, "PESSIMISTIC", "sell")[0], 1.0)
    assert math.isclose(ox.fill(bid, ask, "WORST_REASONABLE", "sell")[0], 0.975)


def test_long_returns_pay_fees_and_exercise_cost():
    r = ox.long_structure_returns(np.array([4.0]), np.array([4.4]), 2, spot_t=np.array([105.0]),
                                  payoff=np.array([5.0]), itm_legs=np.array([1]), mdv20=np.array([50e6]))
    paid_mid = 4.2 + 2 * 0.0003
    expect = (5.0 - 0.0010 * 105.0) / paid_mid - 1          # 5 bps tier + 5 bps slippage on the full notional
    assert math.isclose(r["MID"]["ret"][0], expect, rel_tol=1e-12)
    assert r["PESSIMISTIC"]["ret"][0] < r["CONSERVATIVE"]["ret"][0] < r["MID"]["ret"][0]
    assert r["MID"]["ret_settle_cost_on_intrinsic"][0] > r["MID"]["ret"][0]


def test_classification():
    d = np.repeat(np.arange(30), 10)
    good = {lv: {"ret": np.full(300, 0.05)} for lv in ox.LEVELS}
    assert ox.classify(ox.summarize_levels(good, d)) == "ROBUST"
    mid_only = {lv: {"ret": np.full(300, 0.02 if lv == "MID" else -0.05)} for lv in ox.LEVELS}
    assert ox.classify(ox.summarize_levels(mid_only, d)) == "EXECUTION-SENSITIVE"
    bad = {lv: {"ret": np.full(300, -0.01)} for lv in ox.LEVELS}
    assert ox.classify(ox.summarize_levels(bad, d)) == "NONVIABLE"


def test_expected_straddle_payoff_matches_closed_form():
    spot, sig, n = np.array([100.0]), np.array([0.30]), np.array([21])
    e = ox.expected_payoff(spot, spot, spot, sig, n, nu=30)[0]
    closed = 100 * 0.30 * math.sqrt(21 / 252) * math.sqrt(2 / math.pi)     # E|S-K| ~ S sigma sqrt(t) sqrt(2/pi)
    assert abs(e / closed - 1) < 0.03
    pp = ox.prob_profit_long(spot, spot, spot, sig, n, 30, paid=np.array([e]))[0]
    assert 0.3 < pp < 0.5


def test_strangle_quote_selection():
    rows = []
    for k, cp, delta in ((105, "C", 0.30), (110, "C", 0.24), (115, "C", 0.15), (95, "P", -0.30), (90, "P", -0.26), (85, "P", -0.12)):
        rows.append({"date": pd.Timestamp("2022-03-01"), "act_symbol": "X", "expiration": pd.Timestamp("2022-04-01"),
                     "cp": cp, "strike": k, "bid": 1.0, "ask": 1.1, "vol": 0.3, "delta": delta})
    q = ox.strangle_quotes(pd.DataFrame(rows))
    assert len(q) == 1 and q["strike_c25"].iloc[0] == 110 and q["strike_p25"].iloc[0] == 90
