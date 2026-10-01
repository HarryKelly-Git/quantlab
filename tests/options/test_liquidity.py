"""Liquidity filter: every rule rejects with its reason; UNKNOWN open interest is recorded and fails
only when configured to."""
from __future__ import annotations

from quantlab.options import liquidity as L
from quantlab.options.liquidity import LiquidityRules, check_liquidity

from .helpers import NOW, contract, quote

RULES = LiquidityRules(max_spread_pct=0.10, min_bid=0.10, max_quote_age_minutes=30, min_open_interest=100)


def test_liquid_contract_passes():
    c = contract(100)
    chk = check_liquidity(c, quote(c, 2.90, 3.00), NOW, RULES)
    assert chk.passed and chk.reasons == [] and chk.oi_status == "KNOWN"
    assert chk.spread_pct == (3.00 - 2.90) / 2.95


def test_each_rule_rejects_with_reason():
    c = contract(100)
    cases = {
        L.NO_QUOTE: (c, None),
        L.NOT_TWO_SIDED: (c, quote(c, 0.0, 1.00)),
        L.CROSSED: (c, quote(c, 1.10, 1.00)),
        L.SPREAD_TOO_WIDE: (c, quote(c, 2.50, 3.00)),
        L.BID_BELOW_MIN: (c, quote(c, 0.05, 0.055)),
        L.QUOTE_STALE: (c, quote(c, 2.90, 3.00, age_min=31)),
        L.OI_BELOW_MIN: (contract(100, oi=99), None),
        L.NOT_TRADABLE: (contract(100, tradable=False), None),
    }
    for reason, (con, q) in cases.items():
        q = q if q is not None or reason == L.NO_QUOTE else quote(con, 2.90, 3.00)
        chk = check_liquidity(con, q, NOW, RULES)
        assert not chk.passed and reason in chk.reasons, (reason, chk.reasons)


def test_quote_time_unknown_rejects():
    c = contract(100)
    q = quote(c, 2.9, 3.0)
    q = type(q)(**{**q.__dict__, "quote_time": None})
    chk = check_liquidity(c, q, NOW, RULES)
    assert not chk.passed and L.QUOTE_TIME_UNKNOWN in chk.reasons


def test_unknown_open_interest_is_recorded_and_configurable():
    c = contract(100, oi=None)
    lenient = check_liquidity(c, quote(c, 2.9, 3.0), NOW, RULES)
    assert lenient.passed and lenient.oi_status == "UNKNOWN"
    strict = check_liquidity(c, quote(c, 2.9, 3.0), NOW, LiquidityRules(unknown_open_interest_fails=True))
    assert not strict.passed and strict.reasons == [L.OI_UNKNOWN]


def test_multiple_failures_are_all_recorded():
    c = contract(100, oi=5)
    chk = check_liquidity(c, quote(c, 0.05, 0.50, age_min=120), NOW, RULES)
    assert set(chk.reasons) >= {L.OI_BELOW_MIN, L.QUOTE_STALE, L.SPREAD_TOO_WIDE, L.BID_BELOW_MIN}
