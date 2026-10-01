"""Black-Scholes pricing, greeks and implied volatility. MODEL ONLY.

Everything here is a description, never a fill: QuantLab never fills an option at a model price or
at the quote midpoint. Uses: (1) implied volatility / delta for strike selection when the vendor
snapshot carries no ``impliedVolatility`` / ``greeks`` for a contract, (2) repricing a structure at
an exit date before expiry under an EXPLICIT volatility assumption (``compare.py``).

Known model errors (stated wherever results are shown): US equity options are American (early
exercise ignored), dividends are ignored, volatility is flat (no skew/term structure), the rate is
a config constant.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

MODEL_LABEL = ("MODEL: Black-Scholes (European approximation of American options, no dividends, "
               "flat volatility, constant rate)")

_SQRT2 = math.sqrt(2.0)


try:                                   # scipy ships with scikit-learn; math.erf is the fallback
    from scipy.special import ndtr as _ndtr
except ImportError:  # pragma: no cover
    _erf = np.vectorize(math.erf, otypes=[float])

    def _ndtr(x: Any) -> Any:
        return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / _SQRT2))


def _ncdf(x: Any) -> Any:
    return _ndtr(np.asarray(x, dtype=float))


def _npdf(x: Any) -> Any:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def intrinsic(S: Any, K: float, kind: str) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)


def bs_price(S: Any, K: float, T: float, sigma: float, r: float, kind: str) -> np.ndarray:
    """Per-share option value. ``T`` in years; T <= 0 (or sigma <= 0) gives intrinsic value."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    S = np.asarray(S, dtype=float)
    if T <= 0 or sigma is None or sigma <= 0:
        return intrinsic(S, K, kind)
    sq = sigma * math.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    disc = K * math.exp(-r * T)
    if kind == "call":
        out = S * _ncdf(d1) - disc * _ncdf(d2)
    else:
        out = disc * _ncdf(-d2) - S * _ncdf(-d1)
    return np.where(S > 0, np.maximum(out, 0.0), intrinsic(S, K, kind))


def bs_greeks(S: float, K: float, T: float, sigma: float, r: float, kind: str) -> dict[str, float]:
    """Delta, gamma, vega (per 1.00 vol), theta (per year) for one contract; MODEL."""
    if T <= 0 or sigma <= 0 or S <= 0:
        itm = (S > K) if kind == "call" else (S < K)
        return {"delta": (1.0 if itm else 0.0) * (1 if kind == "call" else -1), "gamma": 0.0, "vega": 0.0,
                "theta": 0.0}
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    pdf = float(_npdf(d1))
    delta = float(_ncdf(d1)) if kind == "call" else float(_ncdf(d1)) - 1.0
    gamma = pdf / (S * sq)
    vega = S * pdf * math.sqrt(T)
    if kind == "call":
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * float(_ncdf(d2))
    else:
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * float(_ncdf(-d2))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(price: float, S: float, K: float, T: float, r: float, kind: str,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-6, max_iter: int = 200) -> float | None:
    """Bisection IV. None when the price is outside the no-arbitrage band (no IV exists) or T <= 0."""
    if price is None or not math.isfinite(price) or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    lower = float(bs_price(S, K, T, lo, r, kind))
    upper = float(bs_price(S, K, T, hi, r, kind))
    if not (lower - 1e-9 <= price <= upper + 1e-9):
        return None
    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        v = float(bs_price(S, K, T, m, r, kind))
        if abs(v - price) < tol:
            return m
        if v < price:
            a = m
        else:
            b = m
    return 0.5 * (a + b)
