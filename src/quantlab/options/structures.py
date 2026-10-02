"""Defined-risk option structures: long call, long put, call debit spread, put debit spread.

There is NO way to build a naked short option here: :class:`OptionStructure` validates itself on
construction, and every SELL leg must be covered by a BUY leg of the same underlying, type, expiry
and multiplier whose strike makes the loss bounded by the net debit (call spread: long strike <
short strike; put spread: long strike > short strike). Credit structures and ratio spreads are
refused. Every structure's maximum loss is its net debit x multiplier.

Prices are taken at the EXECUTABLE side: buy at the ask, sell at the bid. Never the midpoint.
Values before expiry come from Black-Scholes under an explicit volatility assumption and are
labelled MODEL (pricing.MODEL_LABEL).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from quantlab.core.types import Side
from quantlab.options.data import OptionContract, OptionQuote
from quantlab.options.pricing import MODEL_LABEL, bs_price, intrinsic


class StructureKind(str, Enum):
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    CALL_DEBIT_SPREAD = "CALL_DEBIT_SPREAD"
    PUT_DEBIT_SPREAD = "PUT_DEBIT_SPREAD"


class NakedShortError(ValueError):
    """A structure would contain a short option whose loss is not bounded by a covering long leg."""


class StructureError(ValueError):
    """A structure is malformed (mixed expiries, no executable price, no possible gain ...)."""


@dataclass(frozen=True)
class Leg:
    contract: OptionContract
    side: Side
    bid: float | None
    ask: float | None
    quote_time: str | None = None
    iv: float | None = None                  # volatility used for MODEL repricing of this leg
    iv_source: str = "UNKNOWN"               # VENDOR | MODEL_FROM_MID | UNKNOWN
    feed: str = "indicative"

    @property
    def sign(self) -> int:
        return 1 if self.side is Side.BUY else -1

    @property
    def entry_price(self) -> float:
        """Opening price at the executable side: buy at the ask, sell at the bid."""
        p = self.ask if self.side is Side.BUY else self.bid
        if p is None or not math.isfinite(p) or p <= 0:
            raise StructureError(f"{self.contract.symbol}: no executable {'ask' if self.side is Side.BUY else 'bid'}")
        return float(p)

    @property
    def half_spread(self) -> float:
        if self.bid is None or self.ask is None or self.ask < self.bid:
            return 0.0
        return 0.5 * (self.ask - self.bid)

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.contract.symbol, "side": self.side.value, "type": self.contract.type,
                "strike": self.contract.strike, "expiration": self.contract.expiration.isoformat(),
                "multiplier": self.contract.multiplier, "bid": self.bid, "ask": self.ask,
                "quote_time": self.quote_time, "iv": self.iv, "iv_source": self.iv_source, "feed": self.feed,
                "open_interest": self.contract.open_interest}


def leg_from_quote(contract: OptionContract, quote: OptionQuote, side: Side, iv: float | None = None,
                   iv_source: str = "UNKNOWN") -> Leg:
    return Leg(contract=contract, side=side, bid=quote.bid, ask=quote.ask,
               quote_time=quote.quote_time.isoformat() if quote.quote_time is not None else None,
               iv=iv, iv_source=iv_source, feed=quote.feed)


def assert_defined_risk(legs: tuple[Leg, ...] | list[Leg]) -> None:
    """Raise NakedShortError unless every SELL leg is covered one-for-one by a BUY leg that bounds
    its loss. Applies to any combination of legs, independent of the declared kind."""
    longs = [lg for lg in legs if lg.side is Side.BUY]
    used: set[int] = set()
    for short in (lg for lg in legs if lg.side is Side.SELL):
        c = short.contract
        cover = None
        for i, lg in enumerate(longs):
            k = lg.contract
            if i in used or (k.underlying, k.type, k.expiration, k.multiplier) != (c.underlying, c.type, c.expiration,
                                                                                   c.multiplier):
                continue
            if (c.type == "call" and k.strike < c.strike) or (c.type == "put" and k.strike > c.strike):
                cover = i
                break
        if cover is None:
            raise NakedShortError(f"short {c.symbol} has no covering long leg: naked short options are never allowed")
        used.add(cover)


@dataclass(frozen=True)
class OptionStructure:
    kind: StructureKind
    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        legs = tuple(self.legs)
        object.__setattr__(self, "legs", legs)
        if not isinstance(self.kind, StructureKind):
            raise StructureError(f"unknown structure kind {self.kind!r}")
        if not legs or not all(isinstance(lg, Leg) and isinstance(lg.side, Side) for lg in legs):
            raise StructureError("a structure needs at least one Leg")
        assert_defined_risk(legs)                               # first: naked shorts are never built
        keys = {(lg.contract.underlying, lg.contract.expiration, lg.contract.multiplier) for lg in legs}
        if len(keys) != 1:
            raise StructureError("all legs must share underlying, expiration and multiplier")
        types = [lg.contract.type for lg in legs]
        sides = [lg.side for lg in legs]
        k = self.kind
        if k in (StructureKind.LONG_CALL, StructureKind.LONG_PUT):
            want = "call" if k is StructureKind.LONG_CALL else "put"
            if len(legs) != 1 or sides != [Side.BUY] or types != [want]:
                raise StructureError(f"{k.value} is exactly one BUY {want}")
        else:
            want = "call" if k is StructureKind.CALL_DEBIT_SPREAD else "put"
            if len(legs) != 2 or sorted(s.value for s in sides) != ["buy", "sell"] or set(types) != {want}:
                raise StructureError(f"{k.value} is one BUY and one SELL {want}")
            lo, sh = self.long_leg.contract.strike, self.short_leg.contract.strike
            if (want == "call" and not lo < sh) or (want == "put" and not lo > sh):
                raise NakedShortError(f"{k.value}: long strike {lo} / short strike {sh} is not a debit spread")
        for lg in legs:
            _ = lg.entry_price                                   # executable prices must exist
        debit = self.entry_debit
        if not debit > 0:
            raise StructureError(f"{k.value}: net debit {debit:.4f} <= 0 (crossed or inconsistent quotes)")
        if self.is_spread and debit >= self.width:
            raise StructureError(f"{k.value}: debit {debit:.4f} >= strike width {self.width}: no possible gain")

    # -- identity -------------------------------------------------------------------------------
    @property
    def is_spread(self) -> bool:
        return len(self.legs) == 2

    @property
    def long_leg(self) -> Leg:
        return next(lg for lg in self.legs if lg.side is Side.BUY)

    @property
    def short_leg(self) -> Leg | None:
        return next((lg for lg in self.legs if lg.side is Side.SELL), None)

    @property
    def option_type(self) -> str:
        return self.long_leg.contract.type

    @property
    def underlying(self) -> str:
        return self.long_leg.contract.underlying

    @property
    def expiration(self):
        return self.long_leg.contract.expiration

    @property
    def multiplier(self) -> float:
        return float(self.long_leg.contract.multiplier)

    @property
    def structure_id(self) -> str:
        return f"{self.kind.value}:" + "/".join(f"{'+' if lg.side is Side.BUY else '-'}{lg.contract.symbol}"
                                                for lg in self.legs)

    @property
    def width(self) -> float:
        return abs(self.long_leg.contract.strike - self.short_leg.contract.strike) if self.short_leg else math.inf

    # -- entry economics (per share unless stated) ----------------------------------------------
    @property
    def entry_debit(self) -> float:
        """Net premium paid per share at the executable side (ask for buys, bid for sells)."""
        return float(sum(lg.sign * lg.entry_price for lg in self.legs))

    @property
    def entry_cost(self) -> float:
        """USD paid to open ONE structure (per-share debit x multiplier)."""
        return self.entry_debit * self.multiplier

    @property
    def max_loss(self) -> float:
        """USD, one structure: the net debit. Bounded by construction."""
        return self.entry_cost

    @property
    def max_gain(self) -> float | None:
        """USD, one structure. None = unbounded (long call)."""
        k, d, m = self.kind, self.entry_debit, self.multiplier
        if k is StructureKind.LONG_CALL:
            return None
        if k is StructureKind.LONG_PUT:
            return (self.long_leg.contract.strike - d) * m
        return (self.width - d) * m

    @property
    def breakeven(self) -> float:
        """Underlying price at expiry where the payoff is zero."""
        k = self.long_leg.contract.strike
        return k + self.entry_debit if self.option_type == "call" else k - self.entry_debit

    # -- payoffs ----------------------------------------------------------------------------------
    def payoff_at_expiry(self, S: Any) -> np.ndarray:
        """USD P&L of ONE structure held to expiry, for underlying price(s) S (FACT given S)."""
        S = np.asarray(S, dtype=float)
        val = sum(lg.sign * intrinsic(S, lg.contract.strike, lg.contract.type) for lg in self.legs)
        return (val - self.entry_debit) * self.multiplier

    def exit_value(self, S: Any, T: float, r: float, iv_multiplier: float = 1.0,
                   executable_side: bool = True) -> np.ndarray:
        """Per-share value of closing the structure at underlying price(s) S with T years left. MODEL.

        Each leg is repriced with Black-Scholes at ``leg.iv * iv_multiplier`` (an explicit
        assumption). With ``executable_side`` the close is charged the entry half-spread: a long
        leg is sold at model - half-spread (floored at 0), a short leg bought back at model +
        half-spread. T <= 0 means intrinsic value (no spread charge at expiry settlement)."""
        S = np.asarray(S, dtype=float)
        total = np.zeros_like(S)
        for lg in self.legs:
            if T > 0:
                if lg.iv is None or not lg.iv > 0:
                    raise StructureError(f"{lg.contract.symbol}: implied volatility UNKNOWN; cannot model an "
                                         "exit before expiry")
                v = bs_price(S, lg.contract.strike, T, lg.iv * iv_multiplier, r, lg.contract.type)
                if executable_side:
                    v = np.maximum(v - lg.half_spread, 0.0) if lg.side is Side.BUY else v + lg.half_spread
            else:
                v = intrinsic(S, lg.contract.strike, lg.contract.type)
            total = total + lg.sign * v
        return total

    def pnl_at(self, S: Any, T: float, r: float, iv_multiplier: float = 1.0, executable_side: bool = True,
               fee_per_contract: float = 0.0) -> np.ndarray:
        """USD P&L of ONE structure closed at S with T years left (MODEL when T > 0). Fees are
        charged per contract per side, on entry and exit."""
        fees = 2.0 * fee_per_contract * len(self.legs)
        return (self.exit_value(S, T, r, iv_multiplier, executable_side) - self.entry_debit) * self.multiplier - fees

    def to_dict(self) -> dict[str, Any]:
        return {"structure_id": self.structure_id, "kind": self.kind.value, "underlying": self.underlying,
                "expiration": self.expiration.isoformat(), "multiplier": self.multiplier,
                "legs": [lg.to_dict() for lg in self.legs], "entry_debit": self.entry_debit,
                "entry_cost": self.entry_cost, "max_loss": self.max_loss, "max_gain": self.max_gain,
                "breakeven": self.breakeven, "model_label": MODEL_LABEL}


# -- builders (thin, readable entry points; the constructor does all validation) ------------------
def long_call(contract: OptionContract, quote: OptionQuote, iv: float | None = None, iv_source: str = "UNKNOWN") -> OptionStructure:
    return OptionStructure(StructureKind.LONG_CALL, (leg_from_quote(contract, quote, Side.BUY, iv, iv_source),))


def long_put(contract: OptionContract, quote: OptionQuote, iv: float | None = None, iv_source: str = "UNKNOWN") -> OptionStructure:
    return OptionStructure(StructureKind.LONG_PUT, (leg_from_quote(contract, quote, Side.BUY, iv, iv_source),))


def debit_spread(long_c: OptionContract, long_q: OptionQuote, short_c: OptionContract, short_q: OptionQuote,
                 long_iv: tuple[float | None, str] = (None, "UNKNOWN"),
                 short_iv: tuple[float | None, str] = (None, "UNKNOWN")) -> OptionStructure:
    kind = StructureKind.CALL_DEBIT_SPREAD if long_c.type == "call" else StructureKind.PUT_DEBIT_SPREAD
    return OptionStructure(kind, (leg_from_quote(long_c, long_q, Side.BUY, *long_iv),
                                  leg_from_quote(short_c, short_q, Side.SELL, *short_iv)))
