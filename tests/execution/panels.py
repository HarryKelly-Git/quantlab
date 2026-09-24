"""Small, hand-built SYNTHETIC panels for execution-layer tests (deterministic, no randomness).

Not a test module itself (no test_ prefix) -- shared by the other files in this package.
"""
from __future__ import annotations

import pandas as pd

from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import Panel, build_panel

_UTC_EPOCH = "2020-01-01T00:00:00+00:00"


def make_bars_df(rows: list[dict]) -> pd.DataFrame:
    """rows: {symbol, date, open, high, low, close, volume}. Fills in the rest of the bars schema."""
    df = pd.DataFrame(rows)
    for col in ("vwap", "trade_count"):
        if col not in df.columns:
            df[col] = None
    if "provider" not in df.columns:
        df["provider"] = "synthetic"
    if "retrieved_at" not in df.columns:
        df["retrieved_at"] = _UTC_EPOCH
    return df


def make_actions_df(rows: list[dict]) -> pd.DataFrame:
    """rows: {symbol, ex_date, action_type, ratio=None, amount=None, source_id=None}."""
    if not rows:
        return pd.DataFrame(columns=["symbol", "ex_date", "action_type", "ratio", "amount", "declared_date",
                                     "available_at", "pit_status", "source_id", "provider", "retrieved_at"])
    df = pd.DataFrame(rows)
    for col, default in (("ratio", None), ("amount", None), ("declared_date", None)):
        if col not in df.columns:
            df[col] = default
    if "source_id" not in df.columns:
        df["source_id"] = [f"syn{i}" for i in range(len(df))]
    df["available_at"] = df["ex_date"].astype(str) + "T09:30:00-05:00"
    df["pit_status"] = "PIT_CONSERVATIVE"
    df["provider"] = "synthetic"
    df["retrieved_at"] = _UTC_EPOCH
    return df


def build(bars_rows: list[dict], action_rows: list[dict] | None = None, extra_dates: list[str] | None = None) -> Panel:
    """Build a Panel over the union of bar dates (+ optional extra no-bar sessions, e.g. to model a
    halt/no-bar day the calendar still has)."""
    bars = make_bars_df(bars_rows)
    all_dates = sorted(set(bars["date"].astype(str)) | set(extra_dates or []))
    cal = TradingCalendar.from_dates(all_dates)
    actions = make_actions_df(action_rows or [])
    return build_panel(bars, actions, calendar=cal)
