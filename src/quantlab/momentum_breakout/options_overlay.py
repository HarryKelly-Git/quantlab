"""Momentum-breakout options stage (docs/MOMENTUM-BREAKOUT-PREREG.md, section 9). PAPER research only.

Question: for the SAME stock trades the frozen stock rule took (same entry day, same exit day), would a
defined-risk call expression have done better per dollar at risk than the stock?

It runs ONLY if the stock level passes (prereg section 7). It never chooses trades; it re-expresses
trades the stock rule already took, so it cannot add selection bias.

Two pricing sources, never mixed silently. Every result row carries ``price_source``:
  * ``APPROXIMATION`` (MODEL): Black-Scholes, IV = trailing 20-session realised vol of the underlying
    (up to and including the entry session only) x ``iv_over_rv``, flat vol, no dividends, European.
    Used for every entry before ``real_bars_from`` (Alpaca option bars start Feb 2024) and for any later
    trade where a real bar is missing. Model strikes are exact (not listed strikes).
  * ``REAL_BARS``: the listed contract's daily option bar VWAP on the entry and exit sessions (Alpaca
    ``/v1beta1/options/bars``). A VWAP is what traded, not an executable bid/ask, so the same spread
    haircut is still charged. Only for entries on/after ``real_bars_from`` with BOTH bars present.

Costs: every option leg pays ``half_spread_frac`` of its price (min ``min_half_spread``) on entry and
again on exit, plus ``fee_per_contract``. Debit spreads pay it on both legs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.options.pricing import bs_price

APPROXIMATION = "APPROXIMATION"
REAL_BARS = "REAL_BARS"
EXPRESSIONS = ("stock", "call_d50", "call_d70", "call_spread")
MULTIPLIER = 100.0


@dataclass(frozen=True)
class OverlayParams:
    """Pre-registered (prereg section 9). Changing one = a new variant: amend the prereg first."""

    real_bars_from: str = "2024-02-01"
    days_to_expiry: int = 45              # calendar days at entry (model); real: first listed expiry >= this
    close_before_expiry_days: int = 7     # forced exit if the stock trade is still open this close to expiry
    rv_window: int = 20
    iv_over_rv: float = 1.15              # sensitivity reported at 1.0 and 1.3
    risk_free_rate: float = 0.04
    half_spread_frac: float = 0.03        # of the leg price, per side
    min_half_spread: float = 0.025        # USD per share, per leg, per side
    fee_per_contract: float = 0.0
    spread_short_em: float = 1.0          # spread: long ATM, short at spot x (1 + EM x this)
    stock_round_trip_bps: float = 10.0    # stock comparison pays costs too (same as options.stock_round_trip_cost_bps)

    @classmethod
    def from_config(cls, config: Any | None) -> "OverlayParams":
        raw = (config.get("momentum_breakout.options", {}) or {}) if config is not None else {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown momentum_breakout.options keys: {sorted(unknown)}")
        return cls(**raw)


def realised_vol(closes: pd.Series, entry: pd.Timestamp, window: int) -> float | None:
    """Annualised close-to-close log vol over the ``window`` returns ending at the entry session (PIT)."""
    c = closes.loc[:entry].dropna()
    if len(c) < window + 1:
        return None
    r = np.log(c.iloc[-(window + 1):]).diff().dropna()
    v = float(r.std(ddof=1) * math.sqrt(252))
    return v if np.isfinite(v) and v > 0 else None


def strike_for_delta(S: float, T: float, sigma: float, r: float, delta: float) -> float:
    """Exact call strike with Black-Scholes delta ``delta`` (model strikes; no listing grid)."""
    from scipy.stats import norm
    d1 = norm.ppf(delta)
    return float(S * math.exp(-(d1 * sigma * math.sqrt(T)) + (r + 0.5 * sigma * sigma) * T))


def _half_spread(px: float, p: OverlayParams) -> float:
    return max(px * p.half_spread_frac, p.min_half_spread)


def _leg_round_trip(entry_mid: float, exit_mid: float, sign: int, p: OverlayParams) -> tuple[float, float]:
    """(entry cash per share paid, exit cash per share received) for one leg, after spread costs."""
    pay = sign * entry_mid + _half_spread(entry_mid, p)
    get = sign * exit_mid - _half_spread(exit_mid, p)
    return pay, get


def _expression_result(legs: list[tuple[float, float, int]], p: OverlayParams) -> dict[str, float]:
    """legs: (entry_mid, exit_mid, +1 long / -1 short). Return per $ of max loss (= net debit paid)."""
    paid = got = 0.0
    for e, x, s in legs:
        a, b = _leg_round_trip(e, x, s, p)
        paid += a
        got += b
    fees = 2 * len(legs) * p.fee_per_contract / MULTIPLIER
    if paid <= 0:
        return {"debit": float("nan"), "ret_per_risk": float("nan")}
    return {"debit": paid, "ret_per_risk": (got - paid - fees) / paid}


def stock_ret_per_risk(trade: dict[str, Any], p: OverlayParams) -> float:
    """Stock P&L after round-trip costs, per $ of initial risk (entry - stop). NaN without a valid stop."""
    S0, S1 = float(trade["entry_price"]), float(trade["exit_price"])
    stop = float(trade.get("stop_price") or float("nan"))
    if not (np.isfinite(stop) and stop < S0):
        return float("nan")
    return (S1 - S0 - S0 * p.stock_round_trip_bps / 1e4) / (S0 - stop)


def model_trade(trade: dict[str, Any], closes: pd.Series, p: OverlayParams,
                iv_over_rv: float | None = None) -> dict[str, Any]:
    """APPROXIMATION pricing of one stock trade. ``closes``: underlying RAW daily closes by session.

    ``trade``: symbol, entry_date, entry_price, exit_date, exit_price, stop_price (stock fills, net of
    nothing). The option exit is on the stock trade's exit session at the stock's exit price, or at the
    underlying's close ``close_before_expiry_days`` before expiry if the stock trade is still open then.
    """
    k = iv_over_rv if iv_over_rv is not None else p.iv_over_rv
    entry, exit_ = pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade["exit_date"])
    S0, S1 = float(trade["entry_price"]), float(trade["exit_price"])
    out: dict[str, Any] = {"symbol": trade["symbol"], "entry_date": entry, "exit_date": exit_,
                           "price_source": APPROXIMATION, "iv_over_rv": k}
    rv = realised_vol(closes, entry, p.rv_window)
    if rv is None:
        out["status"] = "UNKNOWN: not enough history for realised vol"
        return out
    sig, r = rv * k, p.risk_free_rate
    T0 = p.days_to_expiry / 365.0
    last_day = entry + pd.Timedelta(days=p.days_to_expiry - p.close_before_expiry_days)
    forced = exit_ > last_day
    if forced:
        # the option must be closed before expiry: exit at the underlying's RAW close on the last
        # session <= last_day (an OUTCOME, so reading after entry is allowed here)
        c = closes.loc[entry:last_day].dropna()
        if c.empty:
            out["status"] = "UNKNOWN: no underlying close for the forced exit"
            return out
        exit_, S1 = c.index[-1], float(c.iloc[-1])
    T1 = (p.days_to_expiry - (exit_ - entry).days) / 365.0
    out.update(iv=sig, forced_exit_before_expiry=forced, option_exit_date=exit_, option_exit_underlying=S1,
               status="OK")
    em = sig * math.sqrt(T0)
    strikes = {"call_d50": strike_for_delta(S0, T0, sig, r, 0.50), "call_d70": strike_for_delta(S0, T0, sig, r, 0.70)}
    for name, K in strikes.items():
        e, x = float(bs_price(S0, K, T0, sig, r, "call")), float(bs_price(S1, K, T1, sig, r, "call"))
        res = _expression_result([(e, x, +1)], p)
        out[f"{name}_ret_per_risk"], out[f"{name}_strike"] = res["ret_per_risk"], K
    K_lo, K_hi = S0, S0 * (1 + em * p.spread_short_em)
    legs = [(float(bs_price(S0, K_lo, T0, sig, r, "call")), float(bs_price(S1, K_lo, T1, sig, r, "call")), +1),
            (float(bs_price(S0, K_hi, T0, sig, r, "call")), float(bs_price(S1, K_hi, T1, sig, r, "call")), -1)]
    out["call_spread_ret_per_risk"] = _expression_result(legs, p)["ret_per_risk"]
    out["call_spread_strikes"] = (K_lo, K_hi)
    # the stock comparison always uses the stock trade's own exit (its rule), not the option's forced exit
    out["stock_ret_per_risk"] = stock_ret_per_risk(trade, p)
    return out


def real_bars_trade(trade: dict[str, Any], contracts: dict[str, str], vwap: Callable[[str, pd.Timestamp], float | None],
                    p: OverlayParams) -> dict[str, Any] | None:
    """REAL_BARS pricing. ``contracts``: expression -> OCC symbol chosen at entry (single legs) and
    ``call_spread_long`` / ``call_spread_short``. ``vwap(symbol, session)`` -> that session's bar VWAP
    or None. Returns None if ANY needed bar is missing (the caller then uses APPROXIMATION, labelled).
    ``expiry``: the listed expiry; if the stock trade is still open ``close_before_expiry_days`` before
    it, the caller must pass ``trade['option_exit_date']`` (the forced exit session) or this refuses."""
    entry, exit_ = pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade.get("option_exit_date") or trade["exit_date"])
    expiry = contracts.get("expiry")
    if expiry is None or exit_ > pd.Timestamp(expiry) - pd.Timedelta(days=p.close_before_expiry_days):
        return None
    if entry < pd.Timestamp(p.real_bars_from):
        return None
    out: dict[str, Any] = {"symbol": trade["symbol"], "entry_date": entry, "exit_date": exit_,
                           "price_source": REAL_BARS, "status": "OK"}
    for name in ("call_d50", "call_d70"):
        occ = contracts.get(name)
        e, x = (vwap(occ, entry), vwap(occ, exit_)) if occ else (None, None)
        if e is None or x is None:
            return None
        out[f"{name}_ret_per_risk"] = _expression_result([(e, x, +1)], p)["ret_per_risk"]
    lo, hi = contracts.get("call_spread_long"), contracts.get("call_spread_short")
    px = [vwap(s, d) if s else None for s in (lo, hi) for d in (entry, exit_)]
    if any(v is None for v in px):
        return None
    out["call_spread_ret_per_risk"] = _expression_result([(px[0], px[1], +1), (px[2], px[3], -1)], p)["ret_per_risk"]
    out["stock_ret_per_risk"] = stock_ret_per_risk(trade, p)
    return out


def evaluate(trades: list[dict[str, Any]], closes_by_symbol: dict[str, pd.Series], p: OverlayParams | None = None,
             real: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None) -> pd.DataFrame:
    """One row per stock trade. REAL_BARS when ``real(trade)`` returns a row, else APPROXIMATION."""
    p = p or OverlayParams()
    rows = []
    for t in trades:
        row = real(t) if real is not None else None
        if row is None or pd.Timestamp(t["entry_date"]) < pd.Timestamp(p.real_bars_from):
            row = model_trade(t, closes_by_symbol.get(t["symbol"], pd.Series(dtype=float)), p)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean return per $ at risk by price source and expression. Sources are never pooled."""
    ok = df[df["status"] == "OK"]
    cols = [f"{e}_ret_per_risk" for e in EXPRESSIONS if f"{e}_ret_per_risk" in ok]
    return ok.groupby("price_source")[cols].agg(["count", "mean", "median"])
