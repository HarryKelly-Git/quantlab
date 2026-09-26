"""SEC EDGAR (data.sec.gov) -> point-in-time fundamentals + 8-K item 2.02 earnings events.

Facts this module relies on (docs/EXTERNAL-SERVICES.md, "SEC EDGAR" section):
  * No API key. A declared ``User-Agent: <app/org name> <contact email>`` is REQUIRED -- a generic
    or missing UA gets HTTP 403 (HTML "Undeclared Automated Tool" / "Rate Threshold" pages).
  * Fair-access limit is <= 10 req/s PER USER ACROSS ALL MACHINES (config caps us below that);
    one process-wide limiter is shared across every data.sec.gov AND www.sec.gov request.
  * CIKs are zero-padded to 10 digits in every URL; an unpadded CIK 404s (S3 XML NoSuchKey body).
  * ``filings.recent`` covers only the most recent year/1000 filings; older history lives in
    ``filings.files`` pages that must be concatenated in to see a company's full filing history.
  * 8-K item 2.02 ("Results of Operations") is the earnings-release event. Its timestamp is
    ``acceptanceDateTime`` (true UTC) -- NOT ``filingDate`` (a date-only field that rolls forward
    to the next business day for after-5:30pm-ET submissions, so it silently loses same-day
    look-ahead information if used as the availability time).
  * companyfacts numbers are NOT point-in-time by concept name alone: the same (concept, start,
    end) can appear multiple times across filings (restatements), each with its own ``accn`` and
    ``filed`` date. The as-reported/as-known value at any cutoff is chosen by joining ``accn`` to
    the submissions ``acceptanceDateTime`` -- never by the ``frame`` key, which sits on the
    LAST-filed fact for a period and is therefore look-ahead by construction.
  * fy/fp on a companyfacts fact describe the FILING that reported it, not the fact's own period
    (an old comparative fy can show a much later filing year, and 8-K-sourced facts have fy/fp
    both null). Periods must be identified by (start, end) only.
  * There is no standalone fiscal Q4 concept: it is derived as FY minus the 9-month YTD duration
    for the same (concept, unit, start).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, time
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.core.calendar import TradingCalendar, to_session
from quantlab.core.types import PitStatus
from quantlab.data import schemas
from quantlab.data.providers.base import EventProvider, FundamentalsProvider, ProviderNotConfigured
from quantlab.data.providers.http import HttpClient, ProviderResponseError, shared_rate_limiter
from quantlab.logging_setup import get_logger, log_event
from quantlab.secrets import get_secret

log = get_logger("data.providers.sec_edgar")
NY = ZoneInfo("America/New_York")

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

EARNINGS_FORMS = ("8-K", "8-K/A")
EARNINGS_ITEM = "2.02"

# Canonical concept -> ordered fallback chain of (taxonomy, tag). Selection is made PER PERIOD
# (start, end): for each period, the first tag (in this order) that has ANY fact for that period
# supplies ALL of that period's rows (across accessions/restatements); a company that switched
# concepts mid-history is not merged into one Frankenstein series for the same period.
FALLBACK_CHAINS: dict[str, list[tuple[str, str]]] = {
    # SalesRevenueNet (2009-2018ish) -> Revenues (2018+ for some filers) -> RevenueFromContract...
    # (ASC 606, 2019+). Order here matches the operating instructions literally; per-period
    # selection means the literal order mostly only matters as a tie-break.
    "Revenues": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    # ProfitLoss includes noncontrolling interests; used only when NetIncomeLoss is absent.
    "NetIncomeLoss": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "EarningsPerShareDiluted": [("us-gaap", "EarningsPerShareDiluted")],
    "Assets": [("us-gaap", "Assets")],
    "StockholdersEquity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "LongTermDebt": [("us-gaap", "LongTermDebt"), ("us-gaap", "LongTermDebtNoncurrent")],
    "OperatingIncomeLoss": [("us-gaap", "OperatingIncomeLoss")],
    "GrossProfit": [("us-gaap", "GrossProfit")],
    # dei cover-page fact; excluded for multi-class issuers (dimensional facts are not aggregated).
    "SharesOutstanding": [("dei", "EntityCommonStockSharesOutstanding")],
    # cash-flow statement concepts are reported as YTD durations in 10-Qs; free cash flow is only
    # derived from full fiscal-year (FY) durations as known at the time
    "OperatingCashFlow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "Capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")],
}

# Duration classification thresholds (days, inclusive). Quarter/annual ranges are as specified in
# the build instructions; YTD6/YTD9 ranges are this module's own documented assumption (roughly
# 6/9 months +-15-20 days to tolerate 52/53-week fiscal calendars), analogous to the +-30-day
# tolerance EDGAR's own frames API uses for calendar-period matching.
_Q_DAYS = (80, 100)
_YTD6_DAYS = (165, 195)
_YTD9_DAYS = (255, 290)
_FY_DAYS = (350, 380)


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def pad_cik(cik: Any) -> str:
    return f"{int(cik):010d}"


def classify_duration(start: Any, end: Any) -> str:
    """(start, end) -> "I" (instant, no start), "Q", "YTD6", "YTD9", "FY", or "OTHER"."""
    if not start:
        return "I"
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if _Q_DAYS[0] <= days <= _Q_DAYS[1]:
        return "Q"
    if _YTD6_DAYS[0] <= days <= _YTD6_DAYS[1]:
        return "YTD6"
    if _YTD9_DAYS[0] <= days <= _YTD9_DAYS[1]:
        return "YTD9"
    if _FY_DAYS[0] <= days <= _FY_DAYS[1]:
        return "FY"
    return "OTHER"


def _rows_from_columnar(columnar: dict[str, list]) -> list[dict[str, Any]]:
    """SEC submissions ``recent`` / ``files`` pages are parallel (columnar) arrays -> row dicts."""
    keys = list(columnar.keys())
    n = max((len(v) for v in columnar.values() if isinstance(v, list)), default=0)
    return [{k: (columnar[k][i] if i < len(columnar.get(k) or []) else None) for k in keys} for i in range(n)]


def _accn_index(filing_rows: list[dict[str, Any]]) -> tuple[dict[str, pd.Timestamp], dict[str, str]]:
    """accession -> (UTC acceptance timestamp, filing date) built from a company's filing rows."""
    accept: dict[str, pd.Timestamp] = {}
    filed: dict[str, str] = {}
    for row in filing_rows:
        accn = row.get("accessionNumber")
        if not accn:
            continue
        a = row.get("acceptanceDateTime")
        if a:
            ts = pd.Timestamp(a)
            accept[accn] = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        f = row.get("filingDate")
        if f:
            filed[accn] = f
    return accept, filed


def conservative_available_at(filed: Any, calendar: TradingCalendar | None) -> pd.Timestamp:
    """Cutoff of the session after ``filed`` at 16:00 ET, used when acceptanceDateTime is unknown.

    With an injected calendar this uses real trading sessions (skips weekends/holidays). Without
    one it falls back to the next business day (Mon-Fri only; exchange holidays are not known).
    """
    filed_ts = pd.Timestamp(filed)
    if calendar is not None:
        session = to_session(filed_ts)
        nxt = calendar.next_session(session)
        if nxt is None:
            nxt = session + pd.Timedelta(days=1)
        return calendar.cutoff(nxt)
    nxt = pd.bdate_range(filed_ts + pd.Timedelta(days=1), periods=1)[0]
    return pd.Timestamp(datetime.combine(nxt.date(), time(16, 0))).tz_localize(NY).tz_convert("UTC")


def _dedupe_drop_conflicts(df: pd.DataFrame, key_cols: list[str], what: str) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicates; for keys whose rows DISAGREE (real XBRL: one filing reporting the same
    concept/period twice with different values) drop ALL rows of that key: the value is ambiguous,
    so it becomes UNKNOWN instead of a silently picked number or a whole-batch failure."""
    exact = df.drop_duplicates(list(df.columns))
    dup = exact.duplicated(key_cols, keep=False)
    n_conflicts = int(exact.loc[dup, key_cols].drop_duplicates().shape[0]) if bool(dup.any()) else 0
    if n_conflicts:
        sample = exact.loc[dup, key_cols].head(3).to_dict("records")
        log_event(log, f"sec_edgar: ambiguous {what} dropped (UNKNOWN)", level=30, conflicting_keys=n_conflicts,
                  sample=str(sample)[:500])
    return exact.loc[~dup].reset_index(drop=True), n_conflicts


def _dedupe_or_raise(df: pd.DataFrame, key_cols: list[str], what: str) -> pd.DataFrame:
    """Drop exact-duplicate rows; raise if two rows share a key but disagree (never pick silently)."""
    exact = df.drop_duplicates(list(df.columns))
    dup = exact.duplicated(key_cols, keep=False)
    if bool(dup.any()):
        sample = exact.loc[dup, key_cols].head(3).to_dict("records")
        raise ProviderResponseError(f"sec_edgar: conflicting duplicate {what} rows, e.g. {sample}")
    return exact


class SecEdgarProvider(FundamentalsProvider, EventProvider):
    name = "sec_edgar"
    is_synthetic = False

    def __init__(
        self,
        config: Any,
        http: HttpClient | None = None,
        calendar: TradingCalendar | None = None,
        clock: Callable[[], pd.Timestamp] | None = None,
    ):
        self.config = config
        self.ua_env = config.get("providers.sec_edgar.user_agent_env", "QUANTLAB_SEC_USER_AGENT")
        # NOTE: the UA env var is looked up lazily in ``_ua_headers()`` (called by every fetch),
        # never here -- construction must not require it (see registry.build_providers).
        self.calendar = calendar
        self._clock = clock or _utcnow
        if http is None:
            rps = float(config.get("providers.sec_edgar.max_requests_per_second", 8.0))
            http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("sec-edgar", rps), name="sec_edgar")
        self.http = http
        self._cik_by_ticker: dict[str, str] | None = None
        self._submissions_cache: dict[str, dict[str, Any] | None] = {}

    # ------------------------------------------------------------------------------------------
    def _ua_headers(self) -> dict[str, str]:
        ua = get_secret(self.ua_env)
        if not ua:
            raise ProviderNotConfigured(
                f"SEC EDGAR requires environment variable {self.ua_env} "
                f"(format: 'App/Org Name contact@domain.com')"
            )
        return {"User-Agent": ua.reveal(), "Accept-Encoding": "gzip, deflate"}

    def _ticker_map(self) -> dict[str, str]:
        if self._cik_by_ticker is None:
            data = self.http.get_json(TICKERS_URL, headers=self._ua_headers())
            fields = list(data.get("fields") or [])
            rows = data.get("data") or []
            try:
                ci, ti = fields.index("cik"), fields.index("ticker")
            except ValueError:
                raise ProviderResponseError("sec_edgar: company_tickers_exchange.json missing cik/ticker fields")
            mapping: dict[str, str] = {}
            for row in rows:
                ticker = str(row[ti]).strip().upper()
                if ticker:
                    mapping[ticker] = pad_cik(row[ci])
            self._cik_by_ticker = mapping
        return self._cik_by_ticker

    def _resolve_ciks(self, symbols: list[str]) -> tuple[dict[str, str], list[str]]:
        tmap = self._ticker_map()
        found, missing = {}, []
        for s in symbols:
            cik = tmap.get(s)
            if cik:
                found[s] = cik
            else:
                missing.append(s)
        if missing:
            log_event(log, "sec_edgar: symbols not found in company_tickers_exchange.json", symbols=missing)
        return found, missing

    def _submissions(self, cik10: str, since: Any = None) -> dict[str, Any] | None:
        """{"meta": <submissions root minus filings>, "filings": [row dicts, recent+all pages]}.
        ``since``: skip older history pages whose ``filingTo`` ends before it (only the pages; the
        caller still filters rows, because ``filingTo`` is approximate)."""
        key = cik10 if since is None else f"{cik10}|{pd.Timestamp(since).date()}"
        if key not in self._submissions_cache:
            root = self.http.get_json(SUBMISSIONS_URL.format(cik10=cik10), headers=self._ua_headers(),
                                       not_found_ok=True)
            if root is None:
                self._submissions_cache[key] = None
            else:
                recent = dict((root.get("filings") or {}).get("recent") or {})
                rows = _rows_from_columnar(recent)
                for page in (root.get("filings") or {}).get("files") or []:
                    pname = page.get("name")
                    if not pname:
                        continue
                    upto = page.get("filingTo")
                    if since is not None and upto and pd.Timestamp(upto) < pd.Timestamp(pd.Timestamp(since).date()) - pd.Timedelta(days=31):
                        continue
                    page_data = self.http.get_json(SUBMISSIONS_PAGE_URL.format(name=pname),
                                                    headers=self._ua_headers(), not_found_ok=True)
                    if page_data is None:
                        log_event(log, "sec_edgar: submissions history page missing", cik=cik10, page=pname)
                        continue
                    rows.extend(_rows_from_columnar(page_data))
                self._submissions_cache[key] = {"meta": root, "filings": rows}
        return self._submissions_cache[key]

    # ------------------------------------------------------------------------------------------
    # Events: 8-K item 2.02 earnings releases
    # ------------------------------------------------------------------------------------------
    def get_earnings_events(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        start_d, end_d = to_session(start), to_session(end)
        cik_map, unmapped = self._resolve_ciks(syms)
        now = pd.Timestamp(self._clock()).tz_convert("UTC")
        rows: list[dict[str, Any]] = []
        for sym, cik10 in cik_map.items():
            subs = self._submissions(cik10)
            if subs is None:
                continue
            for row in subs["filings"]:
                form = str(row.get("form") or "").strip()
                if form not in EARNINGS_FORMS:
                    continue
                items = [i.strip() for i in str(row.get("items") or "").split(",") if i.strip()]
                if EARNINGS_ITEM not in items:
                    continue
                accn = row.get("accessionNumber")
                accepted = row.get("acceptanceDateTime")
                if not accn or not accepted:
                    continue
                event_time = pd.Timestamp(accepted)
                event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
                sess = to_session(event_time)
                if not (start_d <= sess <= end_d):
                    continue
                reaction = self.calendar.reaction_session(event_time) if self.calendar is not None else pd.NaT
                rows.append({
                    "symbol": sym, "event_type": "earnings_release", "event_time": event_time,
                    "available_at": event_time, "reaction_date": reaction, "source_id": str(accn),
                    "pit_status": PitStatus.PIT.value, "provider": self.name, "retrieved_at": now,
                    "payload_json": json.dumps({"form": form, "items": row.get("items") or ""}),
                })
        out = schemas.conform("events", pd.DataFrame(rows)) if rows else schemas.empty("events")
        out.attrs.update(unmapped_symbols=unmapped)
        return out

    # ------------------------------------------------------------------------------------------
    # Fundamentals: companyfacts -> canonical concepts, PIT via accn->acceptance join
    # ------------------------------------------------------------------------------------------
    def get_fundamentals(self, symbols: list[str], concepts: list[str] | None = None, since: Any = None) -> pd.DataFrame:
        """``since``: keep only facts that became available on/after it (earlier history pages of the
        submissions index are skipped). Comparatives re-reported in later filings are kept with the
        later filing's availability, so prior-period values still exist as they were known then."""
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        cik_map, unmapped = self._resolve_ciks(syms)
        now = pd.Timestamp(self._clock()).tz_convert("UTC")
        chains = FALLBACK_CHAINS if not concepts else {k: v for k, v in FALLBACK_CHAINS.items() if k in concepts}
        all_rows: list[dict[str, Any]] = []
        for sym, cik10 in cik_map.items():
            subs = self._submissions(cik10, since=since)
            accn_accept, accn_filed = _accn_index(subs["filings"]) if subs else ({}, {})
            facts_json = self.http.get_json(COMPANYFACTS_URL.format(cik10=cik10), headers=self._ua_headers(),
                                             not_found_ok=True)
            if facts_json is None:
                log_event(log, "sec_edgar: no XBRL companyfacts for symbol", symbol=sym, cik=cik10)
                continue
            facts_by_tax = facts_json.get("facts") or {}
            per_concept: dict[str, list[dict[str, Any]]] = {}
            for concept, chain in chains.items():
                per_concept[concept] = self._extract_concept(
                    facts_by_tax, chain, concept, sym, cik10, accn_accept, accn_filed, now
                )
            for concept, rows in per_concept.items():
                all_rows.extend(rows)
                all_rows.extend(self._derive_q4(rows))
        if not all_rows:
            return schemas.empty("fundamentals")
        df = pd.DataFrame(all_rows)
        if since is not None:
            s = pd.Timestamp(since)
            s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
            # by the reporting filing's own date: pre-window filings are history, even when their
            # availability had to fall back to a conservative (later) timestamp
            filed = pd.to_datetime(df["filed_date"], errors="coerce")
            df = df[(pd.to_datetime(df["available_at"], utc=True) >= s) & ~(filed < s.tz_convert(None))]
            if df.empty:
                return schemas.empty("fundamentals")
        df, self.last_ambiguous_facts = _dedupe_drop_conflicts(df, schemas.FUNDAMENTALS_KEY, "fundamentals")
        out = schemas.conform("fundamentals", df)
        out.attrs.update(unmapped_symbols=unmapped)
        return out

    def _extract_concept(
        self,
        facts_by_tax: dict[str, Any],
        chain: list[tuple[str, str]],
        concept: str,
        symbol: str,
        cik10: str,
        accn_accept: dict[str, pd.Timestamp],
        accn_filed: dict[str, str],
        now: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        tag_facts: dict[str, list[dict[str, Any]]] = {}
        for taxonomy, tag in chain:
            units = ((facts_by_tax.get(taxonomy) or {}).get(tag) or {}).get("units") or {}
            lst = [dict(f, unit=unit) for unit, facts in units.items() for f in (facts or [])]
            if lst:
                tag_facts[tag] = lst
        # PER PERIOD selection: the first tag (priority order) with ANY fact for a period owns it.
        period_owner: dict[tuple[Any, Any], str] = {}
        for _, tag in chain:
            for f in tag_facts.get(tag, []):
                key = (f.get("start"), f.get("end"))
                period_owner.setdefault(key, tag)
        rows: list[dict[str, Any]] = []
        for _, tag in chain:
            for f in tag_facts.get(tag, []):
                key = (f.get("start"), f.get("end"))
                if period_owner.get(key) != tag:
                    continue
                row = self._fact_row(concept, symbol, cik10, f, accn_accept, accn_filed, now)
                if row is not None:
                    rows.append(row)
        return rows

    def _fact_row(
        self,
        concept: str,
        symbol: str,
        cik10: str,
        f: dict[str, Any],
        accn_accept: dict[str, pd.Timestamp],
        accn_filed: dict[str, str],
        now: pd.Timestamp,
    ) -> dict[str, Any] | None:
        end, accn = f.get("end"), f.get("accn")
        val = f.get("val")
        if end is None or accn is None or val is None:
            log_event(log, "sec_edgar: malformed companyfacts fact skipped", symbol=symbol, concept=concept,
                      accn=accn)
            return None
        start = f.get("start")
        fiscal_period = classify_duration(start, end)
        if accn in accn_accept:
            available_at, pit = accn_accept[accn], PitStatus.PIT
        else:
            filed = accn_filed.get(accn) or f.get("filed")
            if not filed:
                log_event(log, "sec_edgar: fact has no accn join and no 'filed' date; skipped",
                          symbol=symbol, concept=concept, accn=accn)
                return None
            available_at, pit = conservative_available_at(filed, self.calendar), PitStatus.PIT_CONSERVATIVE
        return {
            "symbol": symbol, "cik": cik10, "concept": concept, "unit": f.get("unit"),
            "period_start": start or end, "period_end": end,
            # fiscal_year/fiscal_period describe the derived PERIOD, never the filing's fy/fp
            # (those belong to the reporting filing -- see class docstring).
            "fiscal_year": pd.Timestamp(end).year, "fiscal_period": fiscal_period,
            "form": f.get("form"), "value": val, "accession": str(accn), "filed_date": f.get("filed"),
            "available_at": available_at, "pit_status": pit.value, "provider": self.name, "retrieved_at": now,
        }

    @staticmethod
    def _derive_q4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Standalone fiscal Q4 = FY - YTD9, matched on shared (concept, unit, period_start)."""
        groups: dict[tuple[Any, Any, Any], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"FY": [], "YTD9": []})
        for r in rows:
            if r["fiscal_period"] in ("FY", "YTD9"):
                groups[(r["concept"], r["unit"], r["period_start"])][r["fiscal_period"]].append(r)
        derived: list[dict[str, Any]] = []
        for (concept, unit, start), grp in groups.items():
            for fy in grp["FY"]:
                # ONE Q4 per FY filing: pair it with the YTD9 version known when the FY was filed
                # (else the earliest one). Pairing every restated YTD9 version produced the same key
                # with different availability, which the conflict guard then dropped entirely.
                ok = [y for y in grp["YTD9"] if _Q_DAYS[0] <= (pd.Timestamp(fy["period_end"]) -
                                                               pd.Timestamp(y["period_end"])).days <= _Q_DAYS[1]]
                known = [y for y in ok if pd.Timestamp(y["available_at"]) <= pd.Timestamp(fy["available_at"])]
                pick = (max(known, key=lambda y: pd.Timestamp(y["available_at"])) if known else
                        min(ok, key=lambda y: pd.Timestamp(y["available_at"])) if ok else None)
                for ytd9 in ([pick] if pick is not None else []):
                    fy_pit, ytd9_pit = PitStatus(fy["pit_status"]), PitStatus(ytd9["pit_status"])
                    derived.append({
                        "symbol": fy["symbol"], "cik": fy["cik"], "concept": concept, "unit": unit,
                        "period_start": pd.Timestamp(ytd9["period_end"]) + pd.Timedelta(days=1),
                        "period_end": fy["period_end"], "fiscal_year": pd.Timestamp(fy["period_end"]).year,
                        "fiscal_period": "Q4", "form": fy["form"], "value": fy["value"] - ytd9["value"],
                        "accession": fy["accession"], "filed_date": fy["filed_date"],
                        "available_at": max(fy["available_at"], ytd9["available_at"]),
                        "pit_status": PitStatus.weakest([fy_pit, ytd9_pit]).value,
                        "provider": fy["provider"], "retrieved_at": fy["retrieved_at"],
                    })
        return derived

    # ------------------------------------------------------------------------------------------
    # Reference metadata (not one of the six canonical dataset kinds; a helper for ingestion)
    # ------------------------------------------------------------------------------------------
    def get_company_meta(self, symbols: list[str]) -> pd.DataFrame:
        """symbol/cik/sic/sicDescription/exchanges -- CURRENT snapshot only (ASSUMED_STATIC)."""
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        cik_map, unmapped = self._resolve_ciks(syms)
        now = pd.Timestamp(self._clock()).tz_convert("UTC")
        rows = []
        for sym, cik10 in cik_map.items():
            subs = self._submissions(cik10)
            if subs is None:
                continue
            meta = subs["meta"]
            rows.append({
                "symbol": sym, "cik": cik10, "sic": meta.get("sic"), "sic_description": meta.get("sicDescription"),
                "exchanges": ",".join(x for x in (meta.get("exchanges") or []) if x),
                "pit_status": PitStatus.ASSUMED_STATIC.value, "provider": self.name, "retrieved_at": now,
            })
        df = pd.DataFrame(rows, columns=["symbol", "cik", "sic", "sic_description", "exchanges", "pit_status",
                                          "provider", "retrieved_at"])
        df.attrs.update(unmapped_symbols=unmapped)
        return df
