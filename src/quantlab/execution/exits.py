"""Exit engine for open paper trades: STOP / TARGET / TIME / DELISTED / invalidation hooks.

Semantics are IDENTICAL to ``core.tradesim.simulate_plan`` (ARCHITECTURE.md section 6):
  * Stops and targets are evaluated on each held session's CLOSE, and the exit fills at the NEXT
    session's open (the execution service submits the order and the broker fills it).
  * The stop and target are RAW prices as of the signal session D. They are converted to
    tri-scaled space with the ratio aclose[D] / close[D] and compared with aclose. Splits and
    dividends during the hold therefore never fire a stop by mistake.
  * Precedence on the same session: STOP, then TARGET, then TIME (``same_bar_stop_and_target:
    stop_first``). TIME fires once ``holding_sessions`` sessions with a valid close have been held,
    counting the entry session as 1.
  * The engine rescans the whole held path every time and reports the FIRST trigger. If a session
    was skipped, or an exit order expired, the exit is still signalled (``detail.late = True``)
    and is never forgotten.
  * DELISTED: tradesim can see that a symbol never trades again, because it has the future. A
    live system cannot. Here a symbol counts as delisted once it has had no bar for
    ``execution.delisting_missing_sessions`` sessions while the market kept trading. The simulated
    broker settles such an exit at last close x (1 + costs.delisting_return), the same price
    tradesim uses.
  * Invalidation hooks (strategy reversal, thesis invalidation, market risk, portfolio risk) are
    callbacks evaluated only when no mechanical exit fired. A hook that raises is reported in
    ``warnings`` and never invents an exit.

Point-in-time: ``evaluate`` truncates the panel at the session first, so its output at D is the same
on the full history and on ``bundle.truncate(D)`` (tested with assert_truncation_invariant).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.core.types import Direction, ExitReason
from quantlab.data.panel import Panel
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

HOOK_REASONS = frozenset({ExitReason.INVALIDATION, ExitReason.STRATEGY_REVERSAL, ExitReason.THESIS_INVALIDATION,
                          ExitReason.MARKET_RISK, ExitReason.PORTFOLIO_RISK})


@dataclass(frozen=True)
class OpenTrade:
    """What the exit engine needs to know about one open trade (built by ``Ledger.open_trades``)."""

    trade_id: str
    symbol: str
    entry_date: str                       # session of the first entry fill
    signal_date: str | None = None        # decision session whose RAW close the stop/target refer to
    qty: float = 0.0
    stop_price: float | None = None       # RAW price as of signal_date
    target_price: float | None = None     # RAW price as of signal_date
    holding_sessions: int | None = None
    direction: str = "LONG"
    book: str | None = None
    entry_price: float | None = None
    strategy_id: str | None = None
    candidate_id: str | None = None
    decision_id: str | None = None
    human_decision_id: str | None = None


@dataclass
class ExitSignal:
    trade_id: str
    reason: ExitReason
    detail: dict[str, Any] = field(default_factory=dict)


# (trade, session, PIT panel, context) -> None/False (no exit) | True | dict of details (exit)
InvalidationHook = Callable[[OpenTrade, pd.Timestamp, Panel, Any], "dict[str, Any] | bool | None"]


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class ExitEngine:
    def __init__(self, config: Config | None = None, hooks: Iterable[tuple[str, InvalidationHook, ExitReason]] | None = None,
                 *, delisting_missing_sessions: int | None = None, default_holding_sessions: int | None = None,
                 delisting_return: float | None = None):
        def cfg(key: str, default: Any) -> Any:
            return config.get(key, default) if config is not None else default

        self.delisting_missing_sessions = int(delisting_missing_sessions if delisting_missing_sessions is not None
                                              else cfg("execution.delisting_missing_sessions", 5))
        self.default_holding_sessions = int(default_holding_sessions if default_holding_sessions is not None
                                            else cfg("execution.exits.default_holding_sessions", 20))
        self.delisting_return = float(delisting_return if delisting_return is not None
                                      else cfg("costs.delisting_return", -0.30))
        self._hooks: list[tuple[str, InvalidationHook, ExitReason]] = []
        self.warnings: list[str] = []
        for name, fn, reason in hooks or ():
            self.register_hook(name, fn, reason)

    def register_hook(self, name: str, fn: InvalidationHook, reason: ExitReason = ExitReason.INVALIDATION) -> None:
        if reason not in HOOK_REASONS:
            raise ValueError(f"hook reason must be one of {sorted(r.value for r in HOOK_REASONS)}")
        self._hooks.append((name, fn, reason))

    # ------------------------------------------------------------------------------------------
    def evaluate(self, open_trades: Iterable[OpenTrade], session_date, panel: Panel, context: Any = None) -> list[ExitSignal]:
        """Exit signals for ``open_trades`` at the close of ``session_date`` (orders fill next open)."""
        self.warnings = []
        s = to_session(session_date)
        if s not in panel.dates:
            self.warnings.append(f"session {s.date()} not in panel: exits cannot be evaluated")
            log_event(log, "exit evaluation skipped: session missing from panel", session=str(s.date()))
            return []
        p = panel.truncate(s)
        out: list[ExitSignal] = []
        for trade in open_trades:
            sig = self._evaluate_trade(trade, s, p, context)
            if sig is not None:
                out.append(sig)
        return out

    def _evaluate_trade(self, t: OpenTrade, s: pd.Timestamp, p: Panel, context: Any) -> ExitSignal | None:
        sym = t.symbol
        entry = to_session(t.entry_date)
        if entry > s:
            return None
        if sym not in p.symbols:
            self.warnings.append(f"{t.trade_id}: {sym} not in panel; cannot evaluate exits")
            return None
        dates = p.dates
        close = p.close[sym]
        aclose = p.aclose[sym]
        sign = Direction(t.direction).sign

        # Stop/target conversion exactly as tradesim: scale = aclose[D] / close[D] at the signal session.
        sig_d = to_session(t.signal_date) if t.signal_date else None
        if sig_d is None or sig_d not in dates:
            prev = dates[dates < entry]
            sig_d = prev[-1] if len(prev) else None
        scale = None
        if sig_d is not None:
            c0, a0 = _f(close.get(sig_d)), _f(aclose.get(sig_d))
            if c0 is not None and a0 is not None and c0 > 0:
                scale = a0 / c0
        a_stop = t.stop_price * scale if (t.stop_price and scale is not None) else None
        a_target = t.target_price * scale if (t.target_price and scale is not None) else None
        if (t.stop_price or t.target_price) and scale is None:
            self.warnings.append(f"{t.trade_id}: no valid close at signal session {sig_d}; stop/target not evaluable")
        hold = int(t.holding_sessions) if t.holding_sessions else self.default_holding_sessions

        ao = _f(p.aopen[sym].get(entry))
        entry_a = ao if ao is not None else _f(aclose.get(entry))
        path = dates[(dates >= entry) & (dates <= s)]
        ahigh, alow = p.ahigh[sym], p.alow[sym]
        held, hi_ex, lo_ex = 0, 0.0, 0.0
        for j in path:
            c = _f(aclose.get(j))
            if c is None:
                continue
            held += 1
            if entry_a:
                h, lo = _f(ahigh.get(j)), _f(alow.get(j))
                if h is not None:
                    hi_ex = max(hi_ex, h / entry_a - 1)
                if lo is not None:
                    lo_ex = min(lo_ex, lo / entry_a - 1)
            trig = None
            if a_stop is not None and (c - a_stop) * sign <= 0:
                trig = ExitReason.STOP
            elif a_target is not None and (c - a_target) * sign >= 0:
                trig = ExitReason.TARGET
            elif held >= hold:
                trig = ExitReason.TIME
            if trig is not None:
                mfe, mae = (hi_ex, lo_ex) if sign > 0 else (-lo_ex, -hi_ex)
                return ExitSignal(t.trade_id, trig, {
                    "trigger_date": j.date().isoformat(), "evaluated_at": s.date().isoformat(),
                    "late": bool(j < s), "held_sessions": held, "close_raw": _f(close.get(j)),
                    "stop_price": t.stop_price, "target_price": t.target_price, "holding_sessions": hold,
                    "mfe": mfe, "mae": mae, "symbol": sym,
                })

        mfe, mae = (hi_ex, lo_ex) if sign > 0 else (-lo_ex, -hi_ex)
        valid = close.loc[:s].dropna()
        if len(valid):
            last_date = valid.index[-1]
            since = int(((dates > last_date) & (dates <= s)).sum())
            if since >= self.delisting_missing_sessions:
                last_close = float(valid.iloc[-1])
                return ExitSignal(t.trade_id, ExitReason.DELISTED, {
                    "last_trade_date": last_date.date().isoformat(), "last_close_raw": last_close,
                    "settlement_price_raw": last_close * (1 + self.delisting_return),
                    "sessions_without_bar": since, "held_sessions": held, "evaluated_at": s.date().isoformat(),
                    "mfe": mfe, "mae": mae, "symbol": sym,
                })

        for name, fn, reason in self._hooks:
            try:
                res = fn(t, s, p, context)
            except Exception as exc:  # a broken hook must never invent (or suppress) an exit silently
                self.warnings.append(f"{t.trade_id}: hook {name} failed: {exc}")
                log_event(log, "exit hook failed", hook=name, trade_id=t.trade_id, error=str(exc))
                continue
            if res:
                detail = dict(res) if isinstance(res, dict) else {}
                return ExitSignal(t.trade_id, reason, {"hook": name, "evaluated_at": s.date().isoformat(),
                                                       "held_sessions": held, "mfe": mfe, "mae": mae,
                                                       "symbol": sym, **detail})
        return None


# ----------------------------------------------------------------------------------------------
# Ready-made hooks (all PIT: they only see the truncated panel / the supplied context)
# ----------------------------------------------------------------------------------------------
def market_drawdown_hook(market_symbol: str, max_drawdown: float) -> InvalidationHook:
    """MARKET_RISK: exit when the market benchmark is more than ``max_drawdown`` below its running
    peak (tri-scaled, data up to the session only)."""
    def hook(trade: OpenTrade, s: pd.Timestamp, p: Panel, context: Any):
        if market_symbol not in p.symbols:
            return None
        a = p.aclose[market_symbol].loc[:s].dropna()
        if a.empty:
            return None
        dd = float(a.iloc[-1] / a.cummax().iloc[-1] - 1)
        return {"market_drawdown": dd, "threshold": -abs(max_drawdown)} if dd <= -abs(max_drawdown) else None
    return hook


def context_flag_hook(key: str) -> InvalidationHook:
    """Generic hook for decisions made elsewhere (strategy reversal, thesis invalidation, portfolio
    risk): ``context[key]`` is a set/dict of trade_ids or symbols that must be exited. When it is a
    dict, the value is recorded as the detail."""
    def hook(trade: OpenTrade, s: pd.Timestamp, p: Panel, context: Any):
        if not context or key not in context:
            return None
        flagged = context[key]
        for k in (trade.trade_id, trade.symbol):
            if k in flagged:
                val = flagged[k] if isinstance(flagged, dict) else True
                return {"flag": key, "value": val if isinstance(val, (str, int, float, bool)) else str(val)}
        return None
    return hook


def build_exit_engine(config: Config, market_symbol: str | None = None) -> ExitEngine:
    """ExitEngine with the config-enabled standard hooks:
    ``execution.exits.market_risk_max_drawdown`` (null = off) and context-flag hooks for strategy
    reversal / thesis invalidation / portfolio risk (context keys ``strategy_reversal``,
    ``thesis_invalidation``, ``portfolio_risk``)."""
    eng = ExitEngine(config)
    mdd = config.get("execution.exits.market_risk_max_drawdown", None)
    mkt = market_symbol or config.get("benchmarks.market", "SPY")
    if mdd is not None and np.isfinite(float(mdd)):
        eng.register_hook("market_drawdown", market_drawdown_hook(mkt, float(mdd)), ExitReason.MARKET_RISK)
    eng.register_hook("strategy_reversal", context_flag_hook("strategy_reversal"), ExitReason.STRATEGY_REVERSAL)
    eng.register_hook("thesis_invalidation", context_flag_hook("thesis_invalidation"), ExitReason.THESIS_INVALIDATION)
    eng.register_hook("portfolio_risk", context_flag_hook("portfolio_risk"), ExitReason.PORTFOLIO_RISK)
    return eng
