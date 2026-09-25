"""Single source of truth for how a TradePlan plays out on daily bars.

Used by: the backtester, shadow-book outcome tracking, counterfactual analysis, human-decision
evaluation and EV calibration — so "what would have happened" means the same thing everywhere.

EXECUTION SEMANTICS (identical in backtest and live paper trading):
  * Decision at the close of the signal session D (information cutoff), entry at the OPEN of D+1.
  * Stops and targets are evaluated on the CLOSE of each held session; a triggered exit executes at
    the NEXT session's open. (No intraday-order assumptions; overnight gaps are paid in full.)
  * Time exit: after ``holding_sessions`` sessions held (entry session counts as 1), exit at the
    next open.
  * A held symbol with no bar for ``costs.delisting_missing_sessions`` consecutive sessions (while
    the market trades) is DELISTED on that session: exit value = last close x (1 + delisting_return).
    Decided only from sessions already seen (no peeking for bars that may come back). If the data
    ends before that -> still OPEN (or END_OF_TEST when forced).
  * All computations run in tri-scaled ("a") prices so splits/dividends during the hold are handled
    exactly; stop/target given in RAW price at D are converted using the ratio to the D close.
  * MFE/MAE use intraday highs/lows for analysis only (they never trigger exits).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.core.costs import CostModel
from quantlab.core.types import Direction, ExitReason, TradePlan
from quantlab.data.panel import Panel


@dataclass
class PlanOutcome:
    symbol: str
    signal_date: pd.Timestamp
    status: str                       # complete | open | no_entry | delisted
    entry_date: pd.Timestamp | None = None
    exit_date: pd.Timestamp | None = None
    exit_reason: str | None = None
    entry_price_raw: float | None = None
    exit_price_raw: float | None = None
    gross_ret: float | None = None    # before costs, direction-adjusted
    cost_ret: float | None = None     # round-trip modeled costs as a fraction
    net_ret: float | None = None
    holding_sessions: int | None = None
    mfe: float | None = None          # max favourable excursion (fraction, intraday, >= 0 typically)
    mae: float | None = None          # max adverse excursion (fraction, intraday, <= 0 typically)
    benchmark_ret: float | None = None
    excess_ret: float | None = None   # net_ret - benchmark_ret over the same entry->exit window

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("signal_date", "entry_date", "exit_date"):
            if d[k] is not None:
                d[k] = pd.Timestamp(d[k]).date().isoformat()
        return d


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def simulate_plan(
    panel: Panel,
    symbol: str,
    signal_date,
    plan: TradePlan,
    costs: CostModel,
    direction: Direction = Direction.LONG,
    benchmark: str | None = "SPY",
    median_dollar_volume: float | None = None,
    force_close_at_end: bool = False,
) -> PlanOutcome:
    d = pd.Timestamp(signal_date)
    out = PlanOutcome(symbol=symbol, signal_date=d, status="no_entry")
    if symbol not in panel.symbols or d not in panel.dates:
        return out
    dates = panel.dates
    n = len(dates)
    i0 = dates.get_loc(d)
    close_raw = panel.close[symbol].to_numpy()
    open_raw = panel.open[symbol].to_numpy()
    aopen = panel.aopen[symbol].to_numpy()
    ahigh = panel.ahigh[symbol].to_numpy()
    alow = panel.alow[symbol].to_numpy()
    aclose = panel.aclose[symbol].to_numpy()
    if not (np.isfinite(close_raw[i0]) and np.isfinite(aclose[i0]) and close_raw[i0] > 0):
        return out

    def next_exec(j: int):
        """First session after j where we can transact: (index, a-price, raw price)."""
        for k in range(j + 1, n):
            if np.isfinite(aopen[k]):
                return k, aopen[k], open_raw[k]
            if np.isfinite(aclose[k]):          # open missing (data issue): transact at close
                return k, aclose[k], close_raw[k]
        return None

    entry = next_exec(i0)
    if entry is None:
        out.status = "open" if i0 == n - 1 else "no_entry"
        return out
    ie, entry_a, entry_raw = entry
    out.entry_date, out.entry_price_raw = dates[ie], _f(entry_raw)

    scale = aclose[i0] / close_raw[i0]
    a_stop = plan.stop_price * scale if plan.stop_price else None
    a_target = plan.target_price * scale if plan.target_price else None
    sign = direction.sign
    adv = median_dollar_volume
    if adv is None:
        dv = panel.dollar_volume[symbol].iloc[max(0, i0 - 19): i0 + 1]
        adv = float(dv.median()) if dv.notna().any() else None
    one_way = costs.one_way_cost_frac(adv)

    hi_ex, lo_ex = 0.0, 0.0
    held, last_valid, missing = 0, ie, 0
    exit_i = exit_a = exit_raw = None
    reason: ExitReason | None = None
    pending_trigger = False
    for j in range(ie, n):
        if not np.isfinite(aclose[j]):
            missing += 1
            if missing >= costs.delisting_missing_sessions:
                exit_i, reason = j, ExitReason.DELISTED
                exit_a = aclose[last_valid] * (1 + costs.delisting_return)
                exit_raw = close_raw[last_valid] * (1 + costs.delisting_return)
                out.status = "delisted"
                break
            continue
        missing = 0
        last_valid = j
        held += 1
        if np.isfinite(ahigh[j]):
            hi_ex = max(hi_ex, ahigh[j] / entry_a - 1)
        if np.isfinite(alow[j]):
            lo_ex = min(lo_ex, alow[j] / entry_a - 1)
        c = aclose[j]
        trig = None
        if a_stop is not None and (c - a_stop) * sign <= 0:
            trig = ExitReason.STOP
        elif a_target is not None and (c - a_target) * sign >= 0:
            trig = ExitReason.TARGET
        elif held >= plan.holding_sessions:
            trig = ExitReason.TIME
        if trig is not None:
            nx = next_exec(j)
            if nx is not None:
                exit_i, exit_a, exit_raw = nx
                reason = trig
            else:
                pending_trigger = True
            break

    if reason is None:
        if force_close_at_end:
            exit_i, exit_a, exit_raw, reason = last_valid, aclose[last_valid], close_raw[last_valid], ExitReason.END_OF_TEST
        else:
            out.status = "open"
            out.holding_sessions = held
            out.mfe, out.mae = (_f(hi_ex), _f(lo_ex)) if sign > 0 else (_f(-lo_ex), _f(-hi_ex))
            out.exit_reason = "PENDING" if pending_trigger else None
            return out

    gross = (exit_a / entry_a - 1) * sign
    # Exact cash accounting per unit of pre-cost entry value: the buy leg costs one_way x entry and
    # the sell leg one_way x exit, so round-trip cost = one_way x (1 + exit/entry) (same for shorts).
    # This is what the share-based backtester and the paper ledgers book.
    cost = one_way * (1 + exit_a / entry_a)
    out.exit_date = dates[exit_i]
    out.exit_reason = reason.value
    out.exit_price_raw = _f(exit_raw)
    out.gross_ret, out.cost_ret, out.net_ret = _f(gross), _f(cost), _f(gross - cost)
    out.holding_sessions = held
    out.mfe, out.mae = (_f(hi_ex), _f(lo_ex)) if sign > 0 else (_f(-lo_ex), _f(-hi_ex))
    if out.status != "delisted":
        out.status = "complete"
    if benchmark and benchmark in panel.symbols:
        b_open = panel.aopen[benchmark].to_numpy()
        b_close = panel.aclose[benchmark].to_numpy()
        b_exit = b_close[exit_i] if reason is ExitReason.DELISTED else b_open[exit_i]
        if np.isfinite(b_open[ie]) and np.isfinite(b_exit):
            out.benchmark_ret = _f(b_exit / b_open[ie] - 1)
            if out.net_ret is not None and out.benchmark_ret is not None:
                out.excess_ret = _f(out.net_ret - out.benchmark_ret)
    return out
