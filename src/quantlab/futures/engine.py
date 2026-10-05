"""Deterministic 1-minute bar backtester for futures. Research only: places no orders anywhere.

Timing contract (no look-ahead, by construction)
-------------------------------------------------
* Bars are indexed by their START time; bar ``i`` is complete at ``start + 1 min``.
* After bar ``i`` closes the strategy's ``on_bar`` sees ONLY bars ``0..i`` (numpy views truncated at i).
* Any order it returns becomes active from bar ``i + 1``: the earliest possible fill is bar ``i + 1``'s
  open. A signal computed from a completed candle can never trade inside that candle.

Fill rules (conservative, documented)
-------------------------------------
* MARKET: next bar open, ``slippage_ticks`` against us.
* STOP entry (buy above / sell below): triggers when the bar trades at the stop; fills at the worse of
  the stop and the bar open (gaps fill at the open), plus slippage.
* LIMIT (entry or profit target): fills only if the bar trades THROUGH the price by >= 1 tick (a touch is
  not a fill); fills at the limit, or at the open if the bar opens beyond it. No slippage on limits.
* Protective STOP-LOSS: triggers when touched; fills at the worse of the stop and the open, plus slippage.
* Stop-loss and target in the SAME bar: the stop is assumed first (never the favourable order).
* On the entry bar only the stop-loss can trigger; the target cannot (intrabar order is unknown).
* Daily loss limit: when realized + open P&L at the bar's adverse extreme reaches ``-daily_loss_limit``
  the position is flattened at the limit-equivalent price (or the open if already beyond) minus
  slippage, and no new entries are taken that session.
* Flatten at ``flatten_at`` (exchange-local) and at the end of each session's data.
* One position at a time; entry orders are ignored while in a position and expire at session end.
* Partial fills are NOT modelled on 1-minute bars (assumes size within displayed liquidity; small size).

Costs: ``commission_per_side + exchange_fee_per_side`` per contract per side, plus slippage in price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from quantlab.futures.sessions import label_bars


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    tick_size: float
    point_value: float          # $ per 1.00 index point

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


# CME contract specs (tick size / multiplier): ES $50/pt, NQ $20/pt, MES $5/pt, MNQ $2/pt, all 0.25 ticks.
ES = ContractSpec("ES", 0.25, 50.0)
NQ = ContractSpec("NQ", 0.25, 20.0)
MES = ContractSpec("MES", 0.25, 5.0)
MNQ = ContractSpec("MNQ", 0.25, 2.0)


@dataclass(frozen=True)
class CostModel:
    commission_per_side: float = 0.0      # $ per contract per side (broker; set from the real schedule)
    exchange_fee_per_side: float = 0.0    # $ per contract per side (exchange + NFA; set from the real schedule)
    slippage_ticks: int = 1               # adverse ticks on market and stop fills

    def per_side(self, qty: int) -> float:
        return (self.commission_per_side + self.exchange_fee_per_side) * qty


@dataclass(frozen=True)
class Order:
    kind: str                   # market | stop | limit
    side: int                   # +1 long, -1 short
    qty: int = 1
    price: float | None = None  # stop / limit price
    stop_loss: float | None = None
    take_profit: float | None = None
    tag: str = ""


@dataclass
class Trade:
    tag: str
    side: int
    qty: int
    session_date: Any
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0            # net $ after costs


@dataclass
class BarContext:
    """What a strategy may see after bar ``i`` closed: views truncated at ``i`` (inclusive)."""
    i: int
    time: pd.Timestamp
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    labels: pd.DataFrame        # session labels for bars 0..i
    position: int               # +qty / -qty / 0
    session_date: Any
    halted: bool                # daily loss limit hit this session


class Strategy(Protocol):
    def on_bar(self, ctx: BarContext) -> list[Order] | None: ...


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)
    daily: pd.DataFrame | None = None   # session_date -> pnl, low, high (intraday cumulative, $)


def _round_tick(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 10)


def run(bars: pd.DataFrame, strategy: Strategy, spec: ContractSpec, costs: CostModel, *,
        daily_loss_limit: float | None = None, flatten_at: time | None = None, rth_only: bool = True) -> Result:
    """Backtest ``strategy`` on 1-minute ``bars`` (UTC start-time index; open/high/low/close/volume)."""
    bars = bars.sort_index()
    lab = label_bars(bars.index)
    if rth_only:
        keep = lab["is_rth"].to_numpy()
        bars, lab = bars[keep], lab[keep]
    O, H, L, C = (bars[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    V = bars["volume"].to_numpy(float) if "volume" in bars else np.zeros(len(bars))
    idx, sess = bars.index, lab["session_date"].to_numpy()
    loc_min = (lab["local"].dt.hour * 60 + lab["local"].dt.minute).to_numpy()
    flat_min = flatten_at.hour * 60 + flatten_at.minute if flatten_at else None
    tick, pv, slip = spec.tick_size, spec.point_value, costs.slippage_ticks * spec.tick_size
    n = len(bars)
    res = Result()
    pending: list[Order] = []
    pos: Trade | None = None
    sl = tp = None
    entry_bar = -1
    day_real = 0.0
    day_low = day_high = 0.0
    halted = False
    days: dict[Any, list[float]] = {}

    def close_pos(j: int, price: float, reason: str) -> None:
        nonlocal pos, day_real
        assert pos is not None
        pos.exit_time, pos.exit_price, pos.exit_reason = idx[j], price, reason
        gross = (price - pos.entry_price) * pos.side * pos.qty * pv
        pos.pnl = gross - 2 * costs.per_side(pos.qty)
        day_real += gross - costs.per_side(pos.qty)       # entry side was charged at entry
        res.trades.append(pos)
        pos = None

    for j in range(n):
        new_session = j == 0 or sess[j] != sess[j - 1]
        if new_session:
            if j > 0:
                if pos is not None:                           # session ended with a position: flatten at last close
                    close_pos(j - 1, _round_tick(C[j - 1] - pos.side * slip, tick), "session_end")
                    day_low = min(day_low, day_real)
                days[sess[j - 1]] = [day_real, min(day_low, day_real), max(day_high, day_real)]
            pending, day_real, day_low, day_high, halted = [], 0.0, 0.0, 0.0, False
        o, h, l, c = O[j], H[j], L[j], C[j]

        # 1) entries placed after bar j-1 closed (only when flat and not halted)
        if pos is None and not halted and pending:
            fired = None
            for od in pending:
                px = None
                if od.kind == "market":
                    px = o + od.side * slip
                elif od.kind == "stop" and od.price is not None:
                    if (od.side > 0 and h >= od.price) or (od.side < 0 and l <= od.price):
                        px = (max(o, od.price) if od.side > 0 else min(o, od.price)) + od.side * slip
                elif od.kind == "limit" and od.price is not None:
                    if od.side > 0 and l <= od.price - tick:
                        px = min(o, od.price)
                    elif od.side < 0 and h >= od.price + tick:
                        px = max(o, od.price)
                if px is not None:
                    fired = (od, _round_tick(px, tick))
                    break
            if fired:
                od, px = fired
                pos = Trade(od.tag, od.side, od.qty, sess[j], idx[j], px)
                sl, tp, entry_bar = od.stop_loss, od.take_profit, j
                day_real -= costs.per_side(od.qty)
                pending = []

        # 2) exits on bar j
        if pos is not None:
            s, q = pos.side, pos.qty
            adverse = l if s > 0 else h
            # daily loss limit (checked at the adverse extreme, before stop/target: conservative)
            if daily_loss_limit is not None:
                open_at_adverse = (adverse - pos.entry_price) * s * q * pv
                if day_real + open_at_adverse <= -daily_loss_limit:
                    lim_px = pos.entry_price + s * (-daily_loss_limit - day_real) / (q * pv)
                    px = (min(o, lim_px) if s > 0 else max(o, lim_px)) - s * slip
                    close_pos(j, _round_tick(px, tick), "daily_loss_limit")
                    halted = True
            if pos is not None and sl is not None and ((s > 0 and l <= sl) or (s < 0 and h >= sl)):
                px = (min(o, sl) if s > 0 else max(o, sl)) - s * slip
                close_pos(j, _round_tick(px, tick), "stop")
            elif pos is not None and tp is not None and j > entry_bar and (
                    (s > 0 and h >= tp + tick) or (s < 0 and l <= tp - tick)):
                close_pos(j, _round_tick(max(o, tp) if s > 0 else min(o, tp), tick), "target")
            elif pos is not None and flat_min is not None and loc_min[j] >= flat_min:
                close_pos(j, _round_tick(o - s * slip, tick), "flatten_time")
        # intraday equity extremes for the prop simulator (open P&L at the bar's extremes)
        if pos is not None:
            s, q = pos.side, pos.qty
            lo_px, hi_px = (l, h) if s > 0 else (h, l)
            day_low = min(day_low, day_real + (lo_px - pos.entry_price) * s * q * pv)
            day_high = max(day_high, day_real + (hi_px - pos.entry_price) * s * q * pv)
        day_low, day_high = min(day_low, day_real), max(day_high, day_real)

        # 3) strategy sees bars 0..j only; its orders are active from bar j+1
        if flat_min is not None and loc_min[j] >= flat_min:
            pending = []
            continue
        ctx = BarContext(j, idx[j], O[:j + 1], H[:j + 1], L[:j + 1], C[:j + 1], V[:j + 1], lab.iloc[:j + 1],
                         0 if pos is None else pos.side * pos.qty, sess[j], halted)
        out = strategy.on_bar(ctx)
        if out is not None:
            pending = list(out)

    if n:
        if pos is not None:
            close_pos(n - 1, _round_tick(C[n - 1] - pos.side * slip, tick), "session_end")
            day_low = min(day_low, day_real)
        days[sess[n - 1]] = [day_real, min(day_low, day_real), max(day_high, day_real)]
    res.daily = pd.DataFrame.from_dict(days, orient="index", columns=["pnl", "low", "high"]).rename_axis("session_date")
    return res
