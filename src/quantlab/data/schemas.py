"""Canonical column contracts for every dataset kind. Providers MUST return these columns.

Point-in-time columns:
  * ``available_at`` — tz-aware UTC timestamp when the information became public. Compared with
    TradingCalendar.cutoff(D) to decide if it may be used for a decision as of session D.
  * ``pit_status``   — a :class:`quantlab.core.types.PitStatus` value describing how well
    ``available_at`` is established.
  * ``retrieved_at`` — when QuantLab downloaded it (UTC). Distinguishes info available at decision
    time from info discovered later.

Price conventions: bars are RAW (unadjusted) prices and share volumes. Split/dividend handling is
done in-house from corporate actions (see :mod:`quantlab.data.panel`) so that level-based filters
(price, dollar volume) never see future reverse splits and returns never see future adjustments.
"""
from __future__ import annotations

import pandas as pd

BARS = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "trade_count", "provider", "retrieved_at"]
BARS_KEY = ["symbol", "date"]

# action_type: "split" -> ratio = new shares per old share (2.0 for 2-for-1, 0.1 for 1-for-10 reverse)
#              "cash_dividend" -> amount = USD per share (pre-split basis on the ex-date)
CORPORATE_ACTIONS = [
    "symbol", "ex_date", "action_type", "ratio", "amount", "declared_date",
    "available_at", "pit_status", "source_id", "provider", "retrieved_at",
]
CORPORATE_ACTIONS_KEY = ["symbol", "ex_date", "action_type", "source_id"]

# security_type: COMMON | ETF | PREFERRED | WARRANT | RIGHT | UNIT | FUND | OTHER | UNKNOWN
REFERENCE = [
    "symbol", "name", "exchange", "security_type", "is_etf", "is_test_issue", "cik", "sic",
    "sector", "industry", "status", "source", "retrieved_at", "pit_status",
]
REFERENCE_KEY = ["symbol", "source"]

# event_type: "earnings_release" (8-K item 2.02), extensible.
# reaction_date: first session whose regular trading can react (see TradingCalendar.reaction_session)
EVENTS = [
    "symbol", "event_type", "event_time", "available_at", "reaction_date", "source_id",
    "pit_status", "provider", "retrieved_at", "payload_json",
]
EVENTS_KEY = ["symbol", "event_type", "source_id"]

FUNDAMENTALS = [
    "symbol", "cik", "concept", "unit", "period_start", "period_end", "fiscal_year", "fiscal_period",
    "form", "value", "accession", "filed_date", "available_at", "pit_status", "provider", "retrieved_at",
]
FUNDAMENTALS_KEY = ["cik", "concept", "unit", "period_start", "period_end", "accession"]

NEWS = [
    "news_id", "symbol", "headline", "summary", "source", "url", "created_at", "updated_at",
    "available_at", "pit_status", "provider", "retrieved_at",
]
NEWS_KEY = ["news_id", "symbol"]

SCHEMAS: dict[str, list[str]] = {
    "bars": BARS,
    "corporate_actions": CORPORATE_ACTIONS,
    "reference": REFERENCE,
    "events": EVENTS,
    "fundamentals": FUNDAMENTALS,
    "news": NEWS,
}
KEYS: dict[str, list[str]] = {
    "bars": BARS_KEY,
    "corporate_actions": CORPORATE_ACTIONS_KEY,
    "reference": REFERENCE_KEY,
    "events": EVENTS_KEY,
    "fundamentals": FUNDAMENTALS_KEY,
    "news": NEWS_KEY,
}
# tz-aware UTC timestamp columns per kind (validated on write)
UTC_COLUMNS: dict[str, list[str]] = {
    "bars": ["retrieved_at"],
    "corporate_actions": ["available_at", "retrieved_at"],
    "reference": ["retrieved_at"],
    "events": ["event_time", "available_at", "retrieved_at"],
    "fundamentals": ["available_at", "retrieved_at"],
    "news": ["created_at", "updated_at", "available_at", "retrieved_at"],
}
# tz-naive session-date columns per kind
DATE_COLUMNS: dict[str, list[str]] = {
    "bars": ["date"],
    "corporate_actions": ["ex_date", "declared_date"],
    "reference": [],
    "events": ["reaction_date"],
    "fundamentals": ["period_start", "period_end", "filed_date"],
    "news": [],
}


class SchemaError(ValueError):
    pass


def empty(kind: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCHEMAS[kind]})


def conform(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate + normalize a provider frame to the canonical schema (column order, dtypes)."""
    if kind not in SCHEMAS:
        raise SchemaError(f"unknown dataset kind {kind!r}")
    missing = [c for c in SCHEMAS[kind] if c not in df.columns]
    if missing:
        raise SchemaError(f"{kind}: missing columns {missing}")
    out = df[SCHEMAS[kind]].copy()
    for c in UTC_COLUMNS[kind]:
        col = pd.to_datetime(out[c], utc=False)
        if len(col.dropna()) and getattr(col.dt, "tz", None) is None:
            raise SchemaError(f"{kind}.{c} must be timezone-aware (UTC)")
        out[c] = pd.to_datetime(out[c], utc=True)
    for c in DATE_COLUMNS[kind]:
        col = pd.to_datetime(out[c])
        if getattr(col.dt, "tz", None) is not None:
            col = col.dt.tz_convert("America/New_York").dt.tz_localize(None)
        out[c] = col.dt.normalize()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper()
    if kind == "bars":
        for c in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if kind == "fundamentals":
        out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")
    if kind == "corporate_actions":
        out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce").astype("float64")
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce").astype("float64")
    return out.reset_index(drop=True)
