"""Payoff / breakeven / max-loss math for every structure, executable-side entry, and the
guarantee that no naked short option can be built."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantlab.core.types import Side
from quantlab.options.occ import build_occ, is_option_symbol, parse_occ
from quantlab.options.pricing import bs_price, implied_vol
from quantlab.options.structures import (
    Leg, NakedShortError, OptionStructure, StructureError, StructureKind, debit_spread, long_call, long_put,
)

from .helpers import contract, quote

GRID = np.linspace(0.0, 400.0, 40001)


def test_long_call_math():
    c = contract(100)
    s = long_call(c, quote(c, 2.90, 3.00))
    assert s.entry_debit == pytest.approx(3.00)            # bought at the ASK, not the 2.95 mid
    assert s.entry_cost == pytest.approx(300.0) and s.max_loss == pytest.approx(300.0)
    assert s.max_gain is None and s.breakeven == pytest.approx(103.0)
    assert s.payoff_at_expiry([90, 100, 103, 110]).tolist() == pytest.approx([-300, -300, 0, 700])
    assert s.payoff_at_expiry(GRID).min() == pytest.approx(-s.max_loss)


def test_long_put_math():
    c = contract(100, "put")
    s = long_put(c, quote(c, 2.40, 2.50))
    assert s.entry_debit == pytest.approx(2.50)
    assert s.breakeven == pytest.approx(97.5)
    assert s.max_gain == pytest.approx(9750.0)
    assert s.payoff_at_expiry([80, 97.5, 120]).tolist() == pytest.approx([1750, 0, -250])
    pay = s.payoff_at_expiry(GRID)
    assert pay.min() == pytest.approx(-s.max_loss) and pay.max() == pytest.approx(s.max_gain)


def test_call_debit_spread_math():
    lo, hi = contract(100), contract(110)
    s = debit_spread(lo, quote(lo, 2.90, 3.00), hi, quote(hi, 0.90, 1.00))
    assert s.kind is StructureKind.CALL_DEBIT_SPREAD
    assert s.entry_debit == pytest.approx(3.00 - 0.90)    # buy at ask, sell at bid
    assert s.max_loss == pytest.approx(210.0) and s.max_gain == pytest.approx(790.0)
    assert s.breakeven == pytest.approx(102.1)
    assert s.payoff_at_expiry([90, 102.1, 105, 120]).tolist() == pytest.approx([-210, 0, 290, 790])
    pay = s.payoff_at_expiry(GRID)
    assert pay.min() == pytest.approx(-s.max_loss) and pay.max() == pytest.approx(s.max_gain)


def test_put_debit_spread_math():
    hi, lo = contract(100, "put"), contract(90, "put")
    s = debit_spread(hi, quote(hi, 3.10, 3.20), lo, quote(lo, 1.00, 1.10))
    assert s.kind is StructureKind.PUT_DEBIT_SPREAD
    assert s.entry_debit == pytest.approx(2.20)
    assert s.max_loss == pytest.approx(220.0) and s.max_gain == pytest.approx(780.0)
    assert s.breakeven == pytest.approx(97.8)
    assert s.payoff_at_expiry([80, 97.8, 110]).tolist() == pytest.approx([780, 0, -220])
    pay = s.payoff_at_expiry(GRID)
    assert pay.min() == pytest.approx(-s.max_loss) and pay.max() == pytest.approx(s.max_gain)


def test_multiplier_is_explicit():
    c = contract(100)
    c2 = type(c)(**{**c.__dict__, "multiplier": 10.0})
    s = long_call(c2, quote(c2, 2.9, 3.0))
    assert s.entry_cost == pytest.approx(30.0) and s.payoff_at_expiry([110])[0] == pytest.approx(70.0)


def _leg(c, side, bid=1.0, ask=1.1):
    return Leg(contract=c, side=side, bid=bid, ask=ask)


def test_no_naked_short_can_be_built():
    c100, c110, p100 = contract(100), contract(110), contract(100, "put")
    with pytest.raises(NakedShortError):                       # lone short call
        OptionStructure(StructureKind.LONG_CALL, (_leg(c100, Side.SELL),))
    with pytest.raises(NakedShortError):                       # credit call spread (short the lower strike)
        OptionStructure(StructureKind.CALL_DEBIT_SPREAD, (_leg(c110, Side.BUY, 0.9, 1.0), _leg(c100, Side.SELL, 2.9, 3.0)))
    with pytest.raises(NakedShortError):                       # a put cannot cover a short call
        OptionStructure(StructureKind.CALL_DEBIT_SPREAD, (_leg(p100, Side.BUY), _leg(c110, Side.SELL)))
    with pytest.raises(NakedShortError):                       # two shorts, one long: one is naked
        OptionStructure(StructureKind.CALL_DEBIT_SPREAD, (_leg(c100, Side.BUY, 2.9, 3.0), _leg(c110, Side.SELL),
                                                          _leg(contract(120), Side.SELL)))
    other_exp = contract(110, exp=date(2026, 11, 20))
    with pytest.raises(NakedShortError):                       # different expiry does not cover
        OptionStructure(StructureKind.CALL_DEBIT_SPREAD, (_leg(c100, Side.BUY, 2.9, 3.0), _leg(other_exp, Side.SELL)))
    with pytest.raises(NakedShortError):
        debit_spread(c110, quote(c110, 0.9, 1.0), c100, quote(c100, 2.9, 3.0))


def test_malformed_structures_are_refused():
    c100, c110 = contract(100), contract(110)
    with pytest.raises(StructureError):                        # debit >= width: no possible gain
        debit_spread(c100, quote(c100, 9.9, 10.5), c110, quote(c110, 0.3, 0.4))
    with pytest.raises(StructureError):                        # no ask -> no executable entry
        long_call(c100, quote(c100, 1.0, None))
    with pytest.raises(StructureError):                        # wrong kind for the legs
        OptionStructure(StructureKind.LONG_PUT, (_leg(c100, Side.BUY),))


def test_exit_values_model_and_intrinsic():
    c = contract(100)
    s = long_call(c, quote(c, 2.9, 3.0), iv=0.30)
    assert s.exit_value([110.0], 0.0, 0.04)[0] == pytest.approx(10.0)          # expiry: intrinsic
    model = s.exit_value([100.0], 30 / 365, 0.04, executable_side=False)[0]
    assert model == pytest.approx(float(bs_price(100.0, 100.0, 30 / 365, 0.30, 0.04, "call")[()]))
    haircut = s.exit_value([100.0], 30 / 365, 0.04, executable_side=True)[0]
    assert haircut == pytest.approx(model - 0.05)                               # sold at model - half-spread
    no_iv = long_call(c, quote(c, 2.9, 3.0))
    with pytest.raises(StructureError):
        no_iv.pnl_at([100.0], 30 / 365, 0.04)


def test_pricing_parity_and_iv_roundtrip():
    S, K, T, r, v = 100.0, 95.0, 0.25, 0.03, 0.4
    c, p = float(bs_price(S, K, T, v, r, "call")[()]), float(bs_price(S, K, T, v, r, "put")[()])
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)
    assert implied_vol(c, S, K, T, r, "call") == pytest.approx(v, abs=1e-4)
    assert implied_vol(0.0001, S, K, T, r, "call") is None                     # below intrinsic: no IV


def test_occ_symbols():
    sym = build_occ("TGT", date(2025, 1, 17), "call", 150)
    assert sym == "TGT250117C00150000" and is_option_symbol(sym) and not is_option_symbol("TGT")
    o = parse_occ(sym)
    assert (o.root, o.expiration, o.type, o.strike) == ("TGT", date(2025, 1, 17), "call", 150.0)
