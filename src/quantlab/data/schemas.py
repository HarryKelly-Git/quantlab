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
#              "spin_off" -> symbol = the PARENT; ratio = new shares per parent share (informational)
#              "cash_merger" | "stock_merger" | "stock_and_cash_merger" -> symbol = the ACQUIREE,
#                  ex_date = effective date; amount = cash per share, ratio = acquirer shares per
#                  acquiree share (informational). Used as dated events only (data.panel.build_panel).
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

# Alternative "who is trading" disclosures (data/alt_trades.py, docs/ALT-DATA.md): insider Form 4
# filings (SEC EDGAR, providers/sec_form4.py) and House periodic transaction reports (House Clerk,
# providers/house_ptr.py). CONTEXT ONLY: never scored, never a selection, sizing or order input.
#   source            "congress" | "insider"
#   symbol            ticker as disclosed (Form 4 issuerTradingSymbol / the "(TICKER)" of a PTR asset);
#                     None when the disclosure names no ticker (record_status NO_SYMBOL / UNPARSEABLE)
#   actor             reporting owner name / House member name (as published)
#   actor_detail      insider: relationship flags + officer title; congress: "House / <state+district>[ / owner]"
#   side              BUY | SELL | OTHER | UNKNOWN. Insider BUY = open-market purchase (code P), SELL =
#                     open-market sale (code S), every other code (grant, exercise, tax, gift) OTHER.
#                     Congress P = BUY, S / S (partial) = SELL, E (exchange) = OTHER
#   amount_low_usd /  congress: the disclosed value RANGE; insider: shares x price for both when both are
#   amount_high_usd   known; NaN = UNKNOWN (never 0)
#   shares / price    insider only (NaN for congress or when not reported)
#   transaction_date  when the trade happened (NOT when it became public)
#   disclosed_at      insider: the filing's EDGAR acceptance datetime (exact); congress: the FilingDate
#                     (date-only, stored as 00:00 America/New_York of that date)
#   available_at      point in time. Insider: = disclosed_at (PIT). Congress (date-only): the cutoff of
#                     the session AFTER the filing date (PIT_CONSERVATIVE, the repo's date-only rule)
#   record_status     PARSED | NO_SYMBOL (no ticker named) | NO_TRANSACTIONS (filing had no transaction
#                     rows) | UNPARSEABLE (e.g. a scanned paper PTR: recorded, never guessed)
#   record_id         stable per disclosure line: "form4:<accession>:<content hash>[#n]" /
#                     "house:<DocID>:<line no>" / "house:<DocID>:unparseable"; dedupe key = (source, record_id)
ALT_TRADES = [
    "record_id", "source", "symbol", "actor", "actor_detail", "side", "amount_low_usd", "amount_high_usd",
    "shares", "price", "transaction_date", "disclosed_at", "available_at", "pit_status", "record_status",
    "raw_json", "provider", "retrieved_at",
]
ALT_TRADES_KEY = ["source", "record_id"]

SCHEMAS: dict[str, list[str]] = {
    "bars": BARS,
    "corporate_actions": CORPORATE_ACTIONS,
    "reference": REFERENCE,
    "events": EVENTS,
    "fundamentals": FUNDAMENTALS,
    "news": NEWS,
    "alt_trades": ALT_TRADES,
}
KEYS: dict[str, list[str]] = {
    "bars": BARS_KEY,
    "corporate_actions": CORPORATE_ACTIONS_KEY,
    "reference": REFERENCE_KEY,
    "events": EVENTS_KEY,
    "fundamentals": FUNDAMENTALS_KEY,
    "news": NEWS_KEY,
    "alt_trades": ALT_TRADES_KEY,
}
# tz-aware UTC timestamp columns per kind (validated on write)
UTC_COLUMNS: dict[str, list[str]] = {
    "bars": ["retrieved_at"],
    "corporate_actions": ["available_at", "retrieved_at"],
    "reference": ["retrieved_at"],
    "events": ["event_time", "available_at", "retrieved_at"],
    "fundamentals": ["available_at", "retrieved_at"],
    "news": ["created_at", "updated_at", "available_at", "retrieved_at"],
    "alt_trades": ["disclosed_at", "available_at", "retrieved_at"],
}
# tz-naive session-date columns per kind
DATE_COLUMNS: dict[str, list[str]] = {
    "bars": ["date"],
    "corporate_actions": ["ex_date", "declared_date"],
    "reference": [],
    "events": ["reaction_date"],
    "fundamentals": ["period_start", "period_end", "filed_date"],
    "news": [],
    "alt_trades": ["transaction_date"],
}
ALT_SOURCES = ("congress", "insider")
ALT_SIDES = ("BUY", "SELL", "OTHER", "UNKNOWN")
ALT_RECORD_STATUSES = ("PARSED", "NO_SYMBOL", "NO_TRANSACTIONS", "UNPARSEABLE")


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
        no_sym = out["symbol"].isna()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        if kind == "alt_trades":                # a disclosure may name no ticker: None, never "NONE"
            out["symbol"] = out["symbol"].astype("object").where(~no_sym, None)
    if kind == "bars":
        for c in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if kind == "fundamentals":
        out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")
    if kind == "corporate_actions":
        out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce").astype("float64")
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce").astype("float64")
    if kind == "alt_trades":
        for c in ("amount_low_usd", "amount_high_usd", "shares", "price"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        bad_rs = ~out["record_status"].isin(ALT_RECORD_STATUSES)
        if bool(bad_rs.any()):
            raise SchemaError(f"alt_trades.record_status must be one of {ALT_RECORD_STATUSES}")
        bad_src = ~out["source"].isin(ALT_SOURCES)
        if bool(bad_src.any()):
            raise SchemaError(f"alt_trades.source must be one of {ALT_SOURCES}: {sorted(set(out.loc[bad_src, 'source']))[:5]}")
        bad_side = ~out["side"].isin(ALT_SIDES)
        if bool(bad_side.any()):
            raise SchemaError(f"alt_trades.side must be one of {ALT_SIDES}: {sorted(set(out.loc[bad_side, 'side']))[:5]}")
        for c in ("record_id", "actor", "actor_detail", "raw_json", "provider", "pit_status", "record_status"):
            out[c] = out[c].astype("object").where(out[c].notna(), None)
    return out.reset_index(drop=True)
