"""Momentum-breakout options stage: labels, point-in-time IV, costs, hand-checked real-bar maths."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quantlab.config import load_config
from quantlab.momentum_breakout.options_overlay import (
    APPROXIMATION, REAL_BARS, OverlayParams, evaluate, model_trade, real_bars_trade, realised_vol,
    stock_ret_per_risk, summarize,
)

from ..conftest import ROOT

P = OverlayParams()


def _closes(start="2023-01-02", n=200, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)


def _trade(closes: pd.Series, i_entry=60, i_exit=70, exit_mult=1.10) -> dict:
    s0 = float(closes.iloc[i_entry])
    return {"symbol": "XYZ", "entry_date": closes.index[i_entry], "entry_price": s0,
            "exit_date": closes.index[i_exit], "exit_price": s0 * exit_mult, "stop_price": s0 * 0.97}


def test_config_matches_preregistered_defaults():
    assert OverlayParams.from_config(load_config(root=ROOT)) == OverlayParams()


def test_pre_2024_is_always_approximation_even_if_real_bars_offered():
    c = _closes("2022-01-03")
    fake_real = lambda t: {"symbol": t["symbol"], "price_source": REAL_BARS, "status": "OK"}
    df = evaluate([_trade(c)], {"XYZ": c}, P, real=fake_real)
    assert (df["price_source"] == APPROXIMATION).all()


def test_missing_real_bar_falls_back_to_labelled_approximation():
    c = _closes("2024-03-01")
    t = _trade(c)
    contracts = {"call_d50": "A", "call_d70": "B", "call_spread_long": "A", "call_spread_short": "C",
                 "expiry": t["entry_date"] + pd.Timedelta(days=45)}
    bars = {("A", t["entry_date"]): 3.0, ("A", t["exit_date"]): 5.0, ("B", t["entry_date"]): 6.0}  # B exit missing
    real = lambda tr: real_bars_trade(tr, contracts, lambda s, d: bars.get((s, d)), P)
    df = evaluate([t], {"XYZ": c}, P, real=real)
    assert df.loc[0, "price_source"] == APPROXIMATION


def test_real_bars_hand_computed():
    c = _closes("2024-03-01")
    t = _trade(c)
    e, x = t["entry_date"], t["exit_date"]
    contracts = {"call_d50": "A", "call_d70": "B", "call_spread_long": "A", "call_spread_short": "C",
                 "expiry": e + pd.Timedelta(days=45)}
    bars = {("A", e): 4.0, ("A", x): 6.0, ("B", e): 8.0, ("B", x): 10.0, ("C", e): 1.0, ("C", x): 2.0}
    row = real_bars_trade(t, contracts, lambda s, d: bars.get((s, d)), P)
    assert row["price_source"] == REAL_BARS
    hs = lambda px: max(px * 0.03, 0.025)
    paid, got = 4.0 + hs(4.0), 6.0 - hs(6.0)
    assert row["call_d50_ret_per_risk"] == pytest.approx((got - paid) / paid)
    paid = (4.0 + hs(4.0)) + (-1.0 + hs(1.0))
    got = (6.0 - hs(6.0)) + (-2.0 - hs(2.0))
    assert row["call_spread_ret_per_risk"] == pytest.approx((got - paid) / paid)
    # exit too close to expiry without a forced-exit date -> refused (caller falls back, labelled)
    late = dict(contracts, expiry=x + pd.Timedelta(days=3))
    assert real_bars_trade(t, late, lambda s, d: bars.get((s, d)), P) is None


def test_iv_is_point_in_time():
    c = _closes()
    t = _trade(c)
    a = model_trade(t, c, P)
    c2 = c.copy()
    c2.iloc[61:] *= 3.0                      # rewrite everything after entry
    b = model_trade(t, c2, P)
    assert a["iv"] == pytest.approx(b["iv"])
    assert a["call_d50_strike"] == pytest.approx(b["call_d50_strike"])
    assert a["iv"] == pytest.approx(realised_vol(c, t["entry_date"], 20) * 1.15)


def test_model_shape_and_costs():
    c = _closes()
    flat = model_trade(_trade(c, exit_mult=1.0), c, P)
    up = model_trade(_trade(c, exit_mult=1.20), c, P)
    assert flat["status"] == up["status"] == "OK"
    for k in ("call_d50", "call_d70", "call_spread"):
        assert flat[f"{k}_ret_per_risk"] < 0           # no move: time decay + spread costs lose money
        assert up[f"{k}_ret_per_risk"] > flat[f"{k}_ret_per_risk"]
        assert up[f"{k}_ret_per_risk"] >= -1.0 - 1e-9  # defined risk (spread costs can exceed 0 only slightly)
    # spread: gain is capped by its width, so on a huge move it trails the uncapped ATM call
    huge = model_trade(_trade(c, exit_mult=1.60), c, P)
    assert huge["call_spread_ret_per_risk"] < huge["call_d50_ret_per_risk"]
    dear = model_trade(_trade(c, exit_mult=1.20), c, dataclasses.replace(P, half_spread_frac=0.10))
    assert dear["call_d50_ret_per_risk"] < up["call_d50_ret_per_risk"]
    assert up["stock_ret_per_risk"] == pytest.approx(stock_ret_per_risk(_trade(c, exit_mult=1.20), P))
    assert up["stock_ret_per_risk"] == pytest.approx((0.20 - 0.001) / 0.03)


def test_forced_exit_before_expiry():
    c = _closes()
    long_hold = _trade(c, i_entry=60, i_exit=120)      # ~84 calendar days > 45 - 7
    r = model_trade(long_hold, c, P)
    assert r["status"] == "OK" and r["forced_exit_before_expiry"]
    assert r["option_exit_date"] <= long_hold["entry_date"] + pd.Timedelta(days=38)
    assert r["option_exit_underlying"] == pytest.approx(float(c.loc[r["option_exit_date"]]))


def test_unknown_without_history():
    c = _closes(n=15)
    r = model_trade(_trade(c, i_entry=10, i_exit=12), c, P)
    assert r["status"].startswith("UNKNOWN")


def test_summary_never_pools_sources():
    df = pd.DataFrame([
        {"price_source": APPROXIMATION, "status": "OK", "stock_ret_per_risk": 1.0, "call_d50_ret_per_risk": 0.5},
        {"price_source": REAL_BARS, "status": "OK", "stock_ret_per_risk": -1.0, "call_d50_ret_per_risk": -0.5},
    ])
    s = summarize(df)
    assert set(s.index) == {APPROXIMATION, REAL_BARS}
