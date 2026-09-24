"""Nasdaq Trader symbol directory -> security master (``reference`` schema).

Two pipe-delimited text files, refreshed nightly, no auth required:
  * ``nasdaqlisted.txt``  -- symbols listed on Nasdaq itself. Columns: Symbol|Security Name|
    Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares.
  * ``otherlisted.txt``   -- symbols listed on other exchanges, distributed via the Nasdaq
    Trader UTP feed. Columns: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|
    Test Issue|NASDAQ Symbol.

Both files end with a one-column footer line ("File Creation Time: ..."), which is skipped.

This is a CURRENT snapshot only (today's listed/active symbols): there is no history and no
delisting date, so ``pit_status`` is always ``ASSUMED_STATIC`` (see ARCHITECTURE.md PIT rule 8).
``cik``/``sic``/``sector``/``industry`` are unknown from this source and left null; join
:class:`quantlab.data.providers.sec_edgar.SecEdgarProvider` for those.
"""
from __future__ import annotations

import csv
import re
from typing import Any, Callable

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.data import schemas
from quantlab.data.providers.base import ReferenceProvider
from quantlab.data.providers.http import HttpClient, shared_rate_limiter
from quantlab.logging_setup import get_logger, log_event

log = get_logger("data.providers.nasdaq_trader")

# otherlisted.txt "Exchange" single-letter codes we accept (per operating instructions).
_OTHER_EXCHANGE_MAP = {"N": "NYSE", "A": "NYSE_AMERICAN", "P": "NYSE_ARCA", "Z": "CBOE_BZX", "V": "IEX"}


# ---------------------------------------------------------------------------------------------
# security_type classification: ORDERED, table-driven, documented precedence. Never guess COMMON.
# ---------------------------------------------------------------------------------------------
# 1) Name says warrant/right/unit/preferred(-depositary)/notes -> that non-common type. These
#    patterns win even over the ETF flag (e.g. an ETF sponsor's warrants are still warrants).
_NON_COMMON_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("WARRANT", re.compile(r"\bwarrants?\b", re.IGNORECASE)),
    ("RIGHT", re.compile(r"\brights?\b", re.IGNORECASE)),
    ("UNIT", re.compile(r"\bunits?\b", re.IGNORECASE)),
    # Covers plain preferred stock AND "Depositary Shares ... Preferred Stock" (depositary-preferred).
    ("PREFERRED", re.compile(r"\bpreferred\b", re.IGNORECASE)),
    # Debt instruments (notes/bonds/debentures) sometimes listed as if they were equity lines.
    ("OTHER", re.compile(r"\b(notes?|debentures?)\b", re.IGNORECASE)),
]
# 2) ETF flag -> ETF (checked only if nothing above matched).
# 3) Common-stock name patterns -> COMMON (checked only if not ETF).
_COMMON_PATTERNS: list["re.Pattern[str]"] = [
    re.compile(r"\bcommon stock\b", re.IGNORECASE),
    re.compile(r"\bordinary shares\b", re.IGNORECASE),
    re.compile(r"\bamerican depositary shares\b", re.IGNORECASE),
    re.compile(r"\bclass [a-z]\b.*\bcommon\b", re.IGNORECASE),
]
# 4) Otherwise UNKNOWN. NOTE: a bare "Trust" in the name is deliberately NOT a rule here. Many
#    REITs are named "... Trust" and issue plain COMMON shares (caught by pattern 3 if the name
#    also says "Common Stock"), while other "... Trust" issuers are unit investment trusts or
#    grantor trusts whose shares are not common stock. Guessing either way from "Trust" alone
#    would be wrong, so names that say only "Trust" (no common/non-common keyword) fall through
#    to UNKNOWN rather than being guessed as COMMON or as some non-common type.


def classify_security_type(name: str, is_etf: bool) -> str:
    """Classify a listed security's ``security_type`` from its name + ETF flag.

    Precedence (see the module-level comments above for the full rationale):
      1. non-common name keyword (warrant/right/unit/preferred/notes-debentures)
      2. ETF flag
      3. common-stock name keyword
      4. UNKNOWN (never guessed as COMMON)
    """
    n = name or ""
    for label, pattern in _NON_COMMON_PATTERNS:
        if pattern.search(n):
            return label
    if is_etf:
        return "ETF"
    if any(pattern.search(n) for pattern in _COMMON_PATTERNS):
        return "COMMON"
    return "UNKNOWN"


def _flag(raw: dict, key: str) -> bool:
    return str(raw.get(key) or "").strip().upper() == "Y"


def _nasdaq_row(raw: dict) -> dict | None:
    symbol = str(raw.get("Symbol") or "").strip().upper()
    if not symbol:
        return None
    is_etf = _flag(raw, "ETF")
    name = str(raw.get("Security Name") or "").strip()
    return {
        "symbol": symbol, "name": name, "exchange": "NASDAQ",
        "security_type": classify_security_type(name, is_etf),
        "is_etf": is_etf, "is_test_issue": _flag(raw, "Test Issue"),
    }


def _other_row(raw: dict) -> dict | None:
    symbol = str(raw.get("ACT Symbol") or "").strip().upper()
    if not symbol:
        return None
    code = str(raw.get("Exchange") or "").strip().upper()
    exchange = _OTHER_EXCHANGE_MAP.get(code)
    if exchange is None:
        log_event(log, "nasdaq_trader: unrecognized otherlisted exchange code", code=code, symbol=symbol)
        exchange = f"UNKNOWN:{code}" if code else "UNKNOWN"
    is_etf = _flag(raw, "ETF")
    name = str(raw.get("Security Name") or "").strip()
    return {
        "symbol": symbol, "name": name, "exchange": exchange,
        "security_type": classify_security_type(name, is_etf),
        "is_etf": is_etf, "is_test_issue": _flag(raw, "Test Issue"),
    }


def parse_symbol_directory(text: str, row_fn: Callable[[dict], dict | None]) -> list[dict]:
    """Parse one pipe-delimited Nasdaq Trader file, skipping the "File Creation Time" footer."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    rows: list[dict] = []
    for raw in csv.DictReader(lines, delimiter="|"):
        first_value = next(iter(raw.values()), "") or ""
        if str(first_value).startswith("File Creation Time"):
            continue
        row = row_fn(raw)
        if row is not None:
            rows.append(row)
    return rows


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class NasdaqTraderReferenceProvider(ReferenceProvider):
    """Security master from the Nasdaq Trader symbol directory (nasdaqlisted + otherlisted)."""

    name = "nasdaq_trader"
    is_synthetic = False

    def __init__(self, config: Any, http: HttpClient | None = None, clock: Callable[[], pd.Timestamp] | None = None):
        self.config = config
        self.nasdaq_listed_url = str(config.get(
            "providers.nasdaq_trader.nasdaq_listed_url",
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        ))
        self.other_listed_url = str(config.get(
            "providers.nasdaq_trader.other_listed_url",
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        ))
        self._clock = clock or _utcnow
        if http is None:
            rps = float(config.get("providers.nasdaq_trader.max_requests_per_second", 5.0))
            http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("nasdaq-trader", rps),
                                          name="nasdaq_trader")
        self.http = http

    def get_securities(self) -> pd.DataFrame:
        now = pd.Timestamp(self._clock()).tz_convert("UTC")
        nasdaq_rows = parse_symbol_directory(self.http.get_text(self.nasdaq_listed_url), _nasdaq_row)
        other_rows = parse_symbol_directory(self.http.get_text(self.other_listed_url), _other_row)
        rows = nasdaq_rows + other_rows
        if not rows:
            return schemas.empty("reference")
        df = pd.DataFrame(rows)
        for c in ("cik", "sic", "sector", "industry"):
            df[c] = None
        df["status"] = "active"  # both files list only currently tradable symbols (ASSUMED_STATIC)
        df["source"] = self.name
        df["retrieved_at"] = now
        df["pit_status"] = PitStatus.ASSUMED_STATIC.value
        dup = df.duplicated("symbol", keep=False)
        if bool(dup.any()):
            dupe_symbols = sorted(df.loc[dup, "symbol"].unique())
            log_event(log, "nasdaq_trader: symbol present in both nasdaqlisted and otherlisted; "
                           "keeping the nasdaqlisted row", symbols=dupe_symbols[:10], count=len(dupe_symbols))
            df = df.drop_duplicates("symbol", keep="first")
        return schemas.conform("reference", df)
