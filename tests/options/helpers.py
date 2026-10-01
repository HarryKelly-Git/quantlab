"""Hand-built option contracts / indicative quotes for offline tests (no network, no real data)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from quantlab.options.data import OptionContract, OptionQuote
from quantlab.options.occ import build_occ

NOW = pd.Timestamp("2026-10-01T15:00:00Z")          # 11:00 ET, a Thursday session


def contract(strike: float, type_: str = "call", exp: date = date(2026, 10, 16), oi: float | None = 500.0,
             root: str = "TGT", tradable: bool = True) -> OptionContract:
    return OptionContract(symbol=build_occ(root, exp, type_, strike), underlying=root, expiration=exp,
                          strike=float(strike), type=type_, open_interest=oi, tradable=tradable)


def quote(c: OptionContract, bid: float | None, ask: float | None, *, age_min: float = 1.0, iv: float | None = 0.30,
          delta: float | None = None) -> OptionQuote:
    return OptionQuote(symbol=c.symbol, bid=bid, ask=ask, bid_size=10, ask_size=10,
                       quote_time=NOW - pd.Timedelta(minutes=age_min), vendor_iv=iv, vendor_delta=delta,
                       vendor_greeks={"delta": delta} if delta is not None else None)
