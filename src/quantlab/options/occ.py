"""OCC option symbols (``TGT250117C00150000`` = root, YYMMDD expiry, C/P, strike x 1000 in 8 digits).

Used to recognise option positions in a shared broker account (``execution/reconcile.py`` reads the
broker's ``asset_class`` first and falls back to this pattern) and to cross-check contract metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

OCC_RE = re.compile(r"^([A-Z][A-Z0-9.]{0,5})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


@dataclass(frozen=True)
class OccSymbol:
    root: str
    expiration: date
    type: str          # "call" | "put"
    strike: float


def is_option_symbol(symbol: str | None) -> bool:
    return bool(symbol) and OCC_RE.match(str(symbol)) is not None


def parse_occ(symbol: str) -> OccSymbol:
    m = OCC_RE.match(str(symbol or ""))
    if m is None:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, yy, mm, dd, cp, strike = m.groups()
    return OccSymbol(root=root, expiration=date(2000 + int(yy), int(mm), int(dd)),
                     type="call" if cp == "C" else "put", strike=int(strike) / 1000.0)


def build_occ(root: str, expiration: date, type_: str, strike: float) -> str:
    if type_ not in ("call", "put"):
        raise ValueError("type must be 'call' or 'put'")
    k = round(float(strike) * 1000)
    if k <= 0 or k >= 10 ** 8:
        raise ValueError(f"strike out of OCC range: {strike!r}")
    return f"{root.upper()}{expiration:%y%m%d}{'C' if type_ == 'call' else 'P'}{k:08d}"
