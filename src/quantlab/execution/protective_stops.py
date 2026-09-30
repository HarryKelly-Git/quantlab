"""Broker-held protective stops: one GTC stop-market SELL resting at the broker per open long trade.

Why: the exit engine checks stops only against daily CLOSES (evening pipeline) and exits at the NEXT
open, so a stop breached intraday was held up to a full day longer, and nothing protected a position
while QuantLab was not running. A stop resting at the broker triggers intraday on its own clock.

Rules (``maintain`` is called by the paper runner every ``execution.protective_stop.poll_seconds``):
  * One stop per OPEN long trade with a stop price, for the whole LEDGER position (whole shares).
  * Price = the trade's RAW stop divided by every split applied to the position since entry
    (``ledger_corporate_actions``), rounded down to the cent. Dividends are not adjusted: a raw stop
    is at most one dividend tighter than the exit engine's total-return stop on an ex-date.
  * Never placed while a non-stop exit order is working (those shares are already being sold), while
    an entry order for the symbol is still working (Alpaca's wash-trade protection checks a new sell
    against open buys; docs/EXTERNAL-SERVICES.md item 14), when
    the broker holds fewer shares than the ledger, or when the broker's current price is at or below
    the stop (the setup is already through it: the close-based STOP in the evening pipeline exits it
    at the next open, exactly as before). Those cases are reported as ``skipped`` with the reason.
  * A resting stop whose qty or price no longer matches (split, a later entry fill) is replaced:
    released with a CONFIRMED cancel, then placed again; at most ``max_replacements`` per trade per
    session, so a disagreement can never become an order loop.
  * The close-based STOP in :mod:`quantlab.execution.exits` stays as the backstop.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from quantlab.execution.exits import OpenTrade
from quantlab.execution.service import PROTECTIVE_STOP, PaperExecutionService

_EPS = 1e-9


def split_factor(svc: PaperExecutionService, trade: OpenTrade) -> float:
    """Product of the split ratios applied to this trade's position (1.0 when none)."""
    f = 1.0
    for r in svc.db.fetchall("SELECT ratio FROM ledger_corporate_actions WHERE book=? AND symbol=? AND trade_id=? "
                             "AND action_type='split'", (svc.book, trade.symbol, trade.trade_id)):
        ratio = r["ratio"]
        if ratio is not None and math.isfinite(float(ratio)) and float(ratio) > 0:
            f *= float(ratio)
    return f


def desired_stop(svc: PaperExecutionService, trade: OpenTrade) -> float | None:
    if trade.stop_price is None:
        return None
    raw = float(trade.stop_price)
    if not (math.isfinite(raw) and raw > 0):
        return None
    return raw / split_factor(svc, trade)


def _replacements_today(svc: PaperExecutionService, trade_id: str, session_date: str) -> int:
    """Protective stops of this trade already created for this session beyond the first."""
    n = svc.db.fetchone("SELECT COUNT(*) AS n FROM orders o JOIN order_intents i ON i.order_id=o.order_id "
                        "WHERE o.book=? AND o.trade_id=? AND i.purpose=? AND i.session_date=?",
                        (svc.book, trade_id, PROTECTIVE_STOP, session_date))["n"]
    return max(0, int(n) - 1)


def maintain(svc: PaperExecutionService, *, broker_positions: Mapping[str, Mapping[str, Any]] | None,
             session_date: str, max_replacements: int = 3) -> dict[str, Any]:
    """Bring the broker's resting stops in line with the open trades. ``broker_positions`` maps
    symbol -> {"qty", "current_price"} (the runner's latest broker snapshot); None = unknown, and
    then nothing is placed. Returns what was placed / replaced / kept / skipped / refused."""
    out: dict[str, Any] = {"placed": [], "replaced": [], "kept": 0, "skipped": [], "refused": []}
    for t in svc.ledger.open_trades():
        if t.direction != "LONG":
            out["skipped"].append((t.trade_id, "not a long trade: no protective sell stop"))
            continue
        want = desired_stop(svc, t)
        if want is None:
            out["skipped"].append((t.trade_id, "the trade has no stop price"))
            continue
        if svc.working_exit(t.trade_id) is not None:
            continue                     # an exit is already selling these shares
        if svc.working_entry(t.symbol) is not None:
            # Alpaca's wash-trade protection checks a new sell against open buys in the symbol; the
            # stop follows once the entry is complete (and then covers the whole position)
            out["skipped"].append((t.trade_id, "the entry order is still working"))
            continue
        pos = svc.ledger.get_position(t.symbol)
        qty = float(pos["qty"]) if pos else 0.0
        if qty <= _EPS:
            continue
        if broker_positions is None:
            out["skipped"].append((t.trade_id, "broker positions unknown"))
            continue
        bp = broker_positions.get(t.symbol) or {}
        b_qty, price = bp.get("qty"), bp.get("current_price")
        if b_qty is None or float(b_qty) + _EPS < qty:
            out["skipped"].append((t.trade_id, f"broker holds {b_qty} {t.symbol}, ledger {qty:g}: reconciliation "
                                               "must agree before a stop is placed"))
            continue
        if price is None or not math.isfinite(float(price)):
            out["skipped"].append((t.trade_id, f"no current {t.symbol} price from the broker"))
            continue
        if float(price) <= want:
            out["skipped"].append((t.trade_id, f"{t.symbol} at {float(price):.2f} is already at/below its stop "
                                               f"{want:.2f}: the close-based stop exits it"))
            continue

        active = svc.active_protective_stop(t.trade_id)
        target = math.floor(want * 100 + 1e-6) / 100 if want >= 1.0 else round(want, 4)
        if active is not None:
            same_qty = abs(float(active["qty"]) - round(qty)) <= _EPS
            same_px = abs(float(active["stop_price"] or 0.0) - target) < 0.005
            if same_qty and same_px:
                out["kept"] += 1
                continue
            if _replacements_today(svc, t.trade_id, session_date) >= max_replacements:
                out["skipped"].append((t.trade_id, f"replacement budget ({max_replacements}/session) used"))
                continue
            released = svc.release_protective_stop(t.trade_id, reason="replace: qty/price changed")
            if released == "filled":
                continue                 # the stop executed while we looked: the trade is closed
            if released == "pending":
                out["skipped"].append((t.trade_id, "cancel of the old stop not confirmed yet"))
                continue
            res = svc.submit_protective_stop(t.trade_id, want, session_date=session_date,
                                             detail={"replaces": active["order_id"]})
            (out["refused"] if res.get("refused") else out["replaced"]).append({"trade_id": t.trade_id, **res})
            continue
        res = svc.submit_protective_stop(t.trade_id, want, session_date=session_date)
        (out["refused"] if res.get("refused") else out["placed"]).append({"trade_id": t.trade_id, **res})
    return out
