"""Liquidity filter for option contracts. A contract is usable only with a fresh two-sided quote,
a spread within ``max_spread_pct`` of mid, a bid >= ``min_bid`` and (when known) enough open
interest. EVERY failed rule is recorded with its reason; nothing is assumed liquid.

Open interest from Alpaca is a current-only value that is often null. Null OI is recorded as
``UNKNOWN`` and fails only when ``options.unknown_open_interest_fails`` is true.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from quantlab.options.data import OptionContract, OptionQuote

# Reason codes (stable strings: they are stored in options_liquidity_checks.reasons_json)
NO_QUOTE = "NO_QUOTE"
NOT_TWO_SIDED = "NOT_TWO_SIDED"
CROSSED = "CROSSED_OR_LOCKED"
SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
BID_BELOW_MIN = "BID_BELOW_MIN"
QUOTE_TIME_UNKNOWN = "QUOTE_TIME_UNKNOWN"
QUOTE_STALE = "QUOTE_STALE"
OI_BELOW_MIN = "OI_BELOW_MIN"
OI_UNKNOWN = "OI_UNKNOWN"
NOT_TRADABLE = "NOT_TRADABLE"


@dataclass(frozen=True)
class LiquidityRules:
    max_spread_pct: float = 0.10
    min_bid: float = 0.10
    max_quote_age_minutes: float = 30.0
    min_open_interest: float = 100.0
    unknown_open_interest_fails: bool = False

    @classmethod
    def from_settings(cls, s: Any) -> "LiquidityRules":
        return cls(max_spread_pct=s.max_spread_pct, min_bid=s.min_bid, max_quote_age_minutes=s.max_quote_age_minutes,
                   min_open_interest=s.min_open_interest, unknown_open_interest_fails=s.unknown_open_interest_fails)


@dataclass
class LiquidityCheck:
    symbol: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    spread_pct: float | None = None
    quote_time: str | None = None
    quote_age_minutes: float | None = None
    open_interest: float | None = None
    oi_status: str = "UNKNOWN"          # KNOWN | UNKNOWN
    feed: str = "indicative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_liquidity(contract: OptionContract, quote: OptionQuote | None, now: pd.Timestamp,
                    rules: LiquidityRules) -> LiquidityCheck:
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    oi = contract.open_interest
    chk = LiquidityCheck(symbol=contract.symbol, passed=False, open_interest=oi,
                         oi_status="KNOWN" if oi is not None else "UNKNOWN",
                         feed=quote.feed if quote is not None else "indicative")
    reasons = chk.reasons
    if not contract.tradable or contract.status != "active":
        reasons.append(NOT_TRADABLE)
    if oi is None:
        if rules.unknown_open_interest_fails:
            reasons.append(OI_UNKNOWN)
    elif oi < rules.min_open_interest:
        reasons.append(OI_BELOW_MIN)
    if quote is None:
        reasons.append(NO_QUOTE)
        return chk
    chk.bid, chk.ask = quote.bid, quote.ask
    if quote.quote_time is None:
        reasons.append(QUOTE_TIME_UNKNOWN)
    else:
        chk.quote_time = quote.quote_time.isoformat()
        chk.quote_age_minutes = round((now - quote.quote_time).total_seconds() / 60.0, 3)
        if chk.quote_age_minutes > rules.max_quote_age_minutes:
            reasons.append(QUOTE_STALE)
    if not quote.two_sided:
        reasons.append(NOT_TWO_SIDED)
    elif quote.ask <= quote.bid:
        reasons.append(CROSSED)
    else:
        mid = 0.5 * (quote.bid + quote.ask)
        chk.mid = mid
        chk.spread_pct = (quote.ask - quote.bid) / mid
        if chk.spread_pct > rules.max_spread_pct:
            reasons.append(SPREAD_TOO_WIDE)
    if quote.bid is None or quote.bid < rules.min_bid:
        reasons.append(BID_BELOW_MIN)
    chk.passed = not reasons
    return chk
