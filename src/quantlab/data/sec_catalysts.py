"""SEC filings -> point-in-time catalyst events and industry classification for the whole universe.

Everything is stored in the existing ``events`` dataset kind, so ``DataBundle.truncate`` applies the
one availability rule (``available_at <= cutoff(D)``) to every catalyst type:

  earnings_release  8-K / 8-K/A with item 2.02 (Results of Operations). available_at = the SEC
                    acceptanceDateTime (true UTC, docs/EXTERNAL-SERVICES.md SEC items 8-9).
  periodic_report   10-Q / 10-K (+ amendments, 10-KT): when the quarter's numbers became
                    machine-readable (companyfacts facts carry the same accession).
  sec_8k            8-K / 8-K/A with at least one item other than 2.02 / 7.01 / 9.01.
  foreign_report    20-F / 40-F / 6-K. Foreign private issuers file no 8-K item codes, so their
                    earnings timing is UNKNOWN, never "no earnings".
  ownership_13d     SC 13D / SC 13D/A (stakes above 5% with intent to influence).
  registration      S-1 / S-3 (possible share issuance).
  sic_observation   SIC read from a periodic report's OWN SGML header, known from that filing's
                    acceptance time (SEC item 30: the submissions ``sic`` is a current snapshot and
                    is never used historically).
  sec_registrant    one row per symbol: the ticker->CIK mapping and current snapshot fields, with
                    available_at = retrieval time, so it can never influence a historical decision.

Limits recorded here, not hidden:
  * The ticker->CIK map is today's (company_tickers_exchange.json). It matches how bars are keyed
    (current symbols), but a ticker reused by a different company in the past is attributed to the
    current owner. Unmapped symbols stay UNKNOWN.
  * acceptanceDateTime is when the SEC accepted the 8-K. A newswire press release can precede it,
    so availability is a LATE bound: never look-ahead, sometimes one session late.
  * SIC between two header observations is carried forward from the earlier one. Headers are read
    at the first and last periodic report in the window and binary-searched when they differ, so a
    change that reverted between two observations would be missed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.core.calendar import TradingCalendar
from quantlab.core.types import PitStatus
from quantlab.data import schemas
from quantlab.data.providers.http import ProviderError
from quantlab.data.providers.sec_edgar import SecEdgarProvider, pad_cik
from quantlab.logging_setup import get_logger, log_event

log = get_logger("data.sec_catalysts")
NY = ZoneInfo("America/New_York")
HEADER_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{accn}-index-headers.html"

FORM_TYPES: dict[str, str] = {
    "10-Q": "periodic_report", "10-K": "periodic_report", "10-Q/A": "periodic_report",
    "10-K/A": "periodic_report", "10-KT": "periodic_report", "10-KT/A": "periodic_report",
    "20-F": "foreign_report", "20-F/A": "foreign_report", "40-F": "foreign_report", "40-F/A": "foreign_report",
    "6-K": "foreign_report", "6-K/A": "foreign_report",
    "SC 13D": "ownership_13d", "SC 13D/A": "ownership_13d",
    "S-1": "registration", "S-3": "registration",
}
EIGHT_K = ("8-K", "8-K/A")
PERIODIC_FOR_SIC = ("10-K", "10-Q", "10-KT", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A", "40-F/A")
NON_MATERIAL_ITEMS = {"2.02", "7.01", "9.01"}
# 8-K item -> catalyst category (the same vocabulary as the news classifier)
ITEM_CATEGORY: dict[str, str] = {
    "1.01": "contract", "1.02": "contract", "1.03": "bankruptcy", "1.05": "other_material",
    "2.01": "m&a", "5.01": "m&a", "2.03": "financing", "2.04": "financing", "3.02": "financing",
    "2.05": "restructuring", "2.06": "impairment", "3.01": "regulatory", "3.03": "financing",
    "4.01": "accounting", "4.02": "accounting", "5.02": "management", "5.03": "other_material",
    "5.07": "other_material", "8.01": "other_material", "2.02": "earnings",
}
ITEM_LABEL: dict[str, str] = {
    "1.01": "material agreement", "1.02": "agreement terminated", "1.03": "bankruptcy", "1.05": "cybersecurity incident",
    "2.01": "acquisition/disposition completed", "2.02": "results of operations", "2.03": "new financial obligation",
    "2.04": "obligation accelerated", "2.05": "exit/restructuring costs", "2.06": "material impairment",
    "3.01": "delisting notice", "3.02": "unregistered equity sale", "3.03": "security holder rights modified",
    "4.01": "auditor change", "4.02": "non-reliance on prior financials", "5.01": "change in control",
    "5.02": "officer/director change", "5.03": "charter/bylaw amendment", "5.07": "shareholder vote",
    "7.01": "Reg FD disclosure", "8.01": "other events", "9.01": "exhibits",
}
_SIC_RE = re.compile(r"STANDARD INDUSTRIAL CLASSIFICATION:\s*([^\[\r\n<]*?)\s*\[(\d{3,4})\]")
_CIK_RE = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")


def release_timing(ts: pd.Timestamp, calendar: TradingCalendar | None) -> str:
    """PRE_MARKET / INTRADAY / POST_CLOSE relative to the regular session of that ET date,
    NON_SESSION on a weekend/holiday, UNKNOWN beyond the calendar."""
    et = pd.Timestamp(ts).tz_convert(NY)
    day = pd.Timestamp(et.date())
    if calendar is None or day > calendar.sessions[-1] or day < calendar.sessions[0]:
        return "UNKNOWN"
    if day not in calendar.sessions:
        return "NON_SESSION"
    t = et.time()
    if t < time(9, 30):
        return "PRE_MARKET"
    return "INTRADAY" if t < time(16, 0) else "POST_CLOSE"


def parse_header_sic(text: str, cik: str) -> tuple[str, str] | None:
    """(sic, title) for ``cik`` from an SGML header; the filer block that names the CIK wins."""
    blocks = re.split(r"\n\s*(?:FILER|FILED BY|SUBJECT COMPANY):", text)
    want = int(cik)
    candidates = []
    for b in blocks:
        m = _SIC_RE.search(b)
        if not m:
            continue
        ciks = [int(x) for x in _CIK_RE.findall(b)]
        candidates.append((want in ciks, m.group(2), m.group(1).strip()))
    if not candidates:
        return None
    candidates.sort(key=lambda c: not c[0])
    _, sic, title = candidates[0]
    import html
    return None if int(sic) == 0 else (sic.zfill(4), html.unescape(title))


def _utc(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@dataclass
class SecCatalystIngest:
    provider: SecEdgarProvider
    calendar: TradingCalendar | None
    since: pd.Timestamp
    fetch_sic: bool = True
    clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC")
    stats: dict[str, int] = field(default_factory=lambda: {"symbols": 0, "mapped": 0, "no_submissions": 0,
                                                           "headers": 0, "header_failures": 0})

    def _row(self, sym: str, etype: str, ts: pd.Timestamp, source_id: str, payload: dict[str, Any],
             pit: PitStatus = PitStatus.PIT, reaction: bool = True) -> dict[str, Any]:
        react = None
        # before the first stored session the calendar cannot place a reaction: UNKNOWN, never the
        # first session (that would plant a false reaction on it)
        if reaction and self.calendar is not None and ts >= self.calendar.cutoff(self.calendar.sessions[0]) - pd.Timedelta(days=1):
            react = self.calendar.reaction_session(ts)
        return {"symbol": sym, "event_type": etype, "event_time": ts, "available_at": ts,
                "reaction_date": react if react is not None else pd.NaT, "source_id": source_id,
                "pit_status": pit.value, "provider": "sec_edgar", "retrieved_at": self.clock(),
                "payload_json": json.dumps(payload, sort_keys=True)}

    def symbol_rows(self, sym: str, cik10: str) -> list[dict[str, Any]]:
        subs = self.provider._submissions(cik10, since=self.since)
        now = self.clock()
        if subs is None:
            self.stats["no_submissions"] += 1
            return []
        meta = subs["meta"]
        filings = []
        for f in subs["filings"]:
            accepted, accn = f.get("acceptanceDateTime"), f.get("accessionNumber")
            if not accepted or not accn:
                continue
            ts = _utc(accepted)
            if ts < self.since:
                continue
            filings.append((ts, str(f.get("form") or "").strip(), str(accn), f))
        filings.sort(key=lambda x: x[0])
        forms = {form for _, form, _, _ in filings}
        filer = ("DOMESTIC" if forms & {"10-K", "10-Q", "10-KT"} else
                 "FOREIGN" if forms & {"20-F", "40-F", "6-K"} else "UNKNOWN")
        rows = [self._row(sym, "sec_registrant", now, f"registrant:{cik10}", {
            "cik": cik10, "name": meta.get("name"), "sic_current": meta.get("sic"),
            "sic_current_title": meta.get("sicDescription"), "entity_type": meta.get("entityType"),
            "category": meta.get("category"), "fiscal_year_end": meta.get("fiscalYearEnd"),
            "exchanges": [x for x in (meta.get("exchanges") or []) if x], "filer": filer,
            "first_filing_in_window": str(filings[0][0]) if filings else None,
            "n_filings_in_window": len(filings), "since": str(self.since.date())},
            pit=PitStatus.ASSUMED_STATIC, reaction=False)]
        for ts, form, accn, f in filings:
            base = {"cik": cik10, "form": form, "filing_date": f.get("filingDate"), "report_date": f.get("reportDate"),
                    "primary_document": f.get("primaryDocument"), "timing": release_timing(ts, self.calendar),
                    "is_amendment": form.endswith("/A")}
            if form in EIGHT_K:
                items = [i.strip() for i in str(f.get("items") or "").split(",") if i.strip()]
                base["items"] = items
                if "2.02" in items:
                    rows.append(self._row(sym, "earnings_release", ts, accn, base))
                material = [i for i in items if i not in NON_MATERIAL_ITEMS]
                if material:
                    rows.append(self._row(sym, "sec_8k", ts, accn, {
                        **base, "material_items": material,
                        "categories": sorted({ITEM_CATEGORY.get(i, "other_material") for i in material}),
                        "labels": [ITEM_LABEL.get(i, i) for i in material]}))
            elif form in FORM_TYPES:
                rows.append(self._row(sym, FORM_TYPES[form], ts, accn, base))
        if self.fetch_sic:
            rows += self._sic_rows(sym, cik10, [(ts, form, accn) for ts, form, accn, _ in filings
                                                if form in PERIODIC_FOR_SIC])
        return rows

    def _header_sic(self, cik10: str, accn: str) -> tuple[str, str] | None:
        url = HEADER_URL.format(cik=int(cik10), accn_nodash=accn.replace("-", ""), accn=accn)
        try:
            resp = self.provider.http.request(url, headers=self.provider._ua_headers(), not_found_ok=True)
        except ProviderError as exc:
            self.stats["header_failures"] += 1
            log_event(log, "sec header fetch failed", level=30, cik=cik10, accn=accn, error=repr(exc)[:200])
            return None
        self.stats["headers"] += 1
        if resp is None:
            self.stats["header_failures"] += 1
            return None
        return parse_header_sic(resp.text, cik10)

    def _sic_rows(self, sym: str, cik10: str, periodic: list[tuple[pd.Timestamp, str, str]]) -> list[dict[str, Any]]:
        if not periodic:
            return []
        seen: dict[int, tuple[str, str] | None] = {}

        def sic_at(i: int) -> tuple[str, str] | None:
            if i not in seen:
                seen[i] = self._header_sic(cik10, periodic[i][2])
            return seen[i]
        lo, hi = 0, len(periodic) - 1
        a, b = sic_at(lo), sic_at(hi)
        if a is not None and b is not None and a[0] != b[0]:
            while hi - lo > 1:                        # first filing that carries the new code
                mid = (lo + hi) // 2
                m = sic_at(mid)
                if m is None:
                    break
                if m[0] == a[0]:
                    lo = mid
                else:
                    hi = mid
        out = []
        for i, v in sorted(seen.items()):
            if v is None:
                continue
            ts, form, accn = periodic[i]
            out.append(self._row(sym, "sic_observation", ts, f"sic:{accn}",
                                 {"cik": cik10, "sic": v[0], "sic_title": v[1], "form": form, "accession": accn},
                                 reaction=False))
        return out

    def run(self, symbols: Iterable[str], workers: int = 1) -> pd.DataFrame:
        """Rows for ``symbols``. ``workers`` threads share the provider's process-wide SEC rate
        limiter, so parallelism only hides latency and never exceeds the fair-access rate."""
        from concurrent.futures import ThreadPoolExecutor
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        cik_map, unmapped = self.provider._resolve_ciks(syms)
        self.stats["symbols"] += len(syms)
        self.stats["mapped"] += len(cik_map)
        todo = [s for s in syms if s in cik_map]
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for part in pool.map(lambda s: self.symbol_rows(s, cik_map[s]), todo):
                rows += part
        out = schemas.conform("events", pd.DataFrame(rows)) if rows else schemas.empty("events")
        out.attrs.update(unmapped_symbols=unmapped)
        return out


def catalyst_symbols(store, reference: pd.DataFrame | None = None) -> list[str]:
    """Symbols with stored bars that the reference data marks as COMMON stock (not ETF/test)."""
    import pyarrow.parquet as pq
    have: set[str] = set()
    for ds in store.dataset_ids("bars", synthetic=False):
        row = store.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (ds,))
        have |= set(pq.read_table(store.data_dir / row["path"], columns=["symbol"]).column("symbol").to_pylist())
    ref = reference if reference is not None else store.load("reference", synthetic=False)
    if ref.empty:
        return sorted(have)
    common = ref[(ref["security_type"].astype(str).str.upper() == "COMMON")
                 & ~ref["is_etf"].fillna(False).astype(bool) & ~ref["is_test_issue"].fillna(False).astype(bool)]
    return sorted(have & set(common["symbol"]))


def market_calendar(store, market: str = "SPY") -> TradingCalendar | None:
    import pyarrow.parquet as pq
    dates: list[pd.Timestamp] = []
    for ds in store.dataset_ids("bars", synthetic=False):
        row = store.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (ds,))
        t = pq.read_table(store.data_dir / row["path"], columns=["symbol", "date"], filters=[("symbol", "==", market)])
        dates += list(pd.to_datetime(t.column("date").to_pylist()))
    if not dates:
        return None
    # extend past the last stored bar with the NYSE rule calendar (data/audit.py), so filings accepted
    # after the latest ingested session still get a timing class and a reaction session
    from quantlab.data.audit import expected_sessions
    last = max(dates)
    try:
        ahead = expected_sessions(str((last + pd.Timedelta(days=1)).date()),
                                  str((pd.Timestamp.now().normalize() + pd.Timedelta(days=21)).date()))
    except Exception:          # pragma: no cover - the rule calendar is a convenience, bars stay authoritative
        ahead = []
    return TradingCalendar.from_dates(sorted(set(dates) | {pd.Timestamp(x) for x in ahead}))


__all__ = ["SecCatalystIngest", "release_timing", "parse_header_sic", "catalyst_symbols", "market_calendar",
           "ITEM_CATEGORY", "ITEM_LABEL", "FORM_TYPES", "pad_cik"]
