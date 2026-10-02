"""SEC EDGAR Form 4 (insider transactions) -> ``alt_trades`` rows. CONTEXT ONLY (docs/ALT-DATA.md).

Facts this module relies on (docs/ALT-DATA.md, docs/EXTERNAL-SERVICES.md "SEC EDGAR"):
  * No key. A declared ``User-Agent: <app/org> <contact email>`` (env ``QUANTLAB_SEC_USER_AGENT``) is
    required; the process-wide ``sec-edgar`` limiter (<= 10 req/s, config below that) is shared with
    the fundamentals / 8-K provider, so the two can never add up past SEC's fair-access limit.
  * Discovery: the daily form index ``daily-index/<YYYY>/QTR<q>/form.<YYYYMMDD>.idx`` (published in
    the evening of the filing day; 404 on weekends/holidays and before it is published) plus the
    ``getcurrent`` Atom feed for filings not yet in a published index. Both list each Form 4 once per
    filer role, so entries are de-duplicated on the accession number.
  * Each filing's complete submission ``<accession>.txt`` carries the SGML header
    ``<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS`` (EDGAR's clock, US Eastern) and the ownershipDocument XML.
    available_at = that acceptance time (PIT). If the header were ever UTC rather than Eastern, reading
    it as Eastern only makes availability LATER (conservative), never earlier.
  * transactionCode P = open-market purchase -> BUY, S = open-market sale -> SELL, but only in the
    non-derivative table and only with the matching acquired/disposed flag (P+A, S+D). Every other
    code (A grant, M exercise, F tax withholding, G gift, ...) and every derivative-table row is OTHER.
  * A 4/A amendment is a separate disclosure (its own accession); it is recorded, never merged.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.data.providers.base import ProviderNotConfigured
from quantlab.data.providers.http import HttpClient, ProviderResponseError, shared_rate_limiter
from quantlab.secrets import get_secret

NY = ZoneInfo("America/New_York")
DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/form.{ymd}.idx"
CURRENT_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include"
               "&start={start}&count={count}&output=atom")
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{accession}.txt"
FORMS = ("4", "4/A")
PROVIDER = "sec_form4"
_ATOM = "{http://www.w3.org/2005/Atom}"
_IDX_RE = re.compile(r"^(?P<form>\S+(?: \S+)*?)\s{2,}(?P<company>.*?)\s{2,}(?P<cik>\d+)\s{2,}"
                     r"(?P<date>\d{8}|\d{4}-\d{2}-\d{2})\s{2,}(?P<file>edgar/\S+)\s*$")
_ACC_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _num(x: Any) -> float:
    try:
        v = float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")
    return v


def canonical_num(x: Any) -> str:
    """Number text used in record ids, identical for '12500', '12500.0000' and 12500.0."""
    v = _num(x)
    return "" if v != v else f"{v:.10g}"


def form4_record_id(accession: str, table: str, code: str, tx_date: Any, shares: Any, price: Any, ad: str,
                    seen: dict[str, int] | None = None) -> str:
    """Stable id of one Form 4 transaction line (dedupe key with source='insider').

    A content hash of (table, code, date, shares, price, acquired/disposed) inside the accession, plus an
    ordinal ``#n`` for genuinely identical lines in the same filing. The research-parquet import
    (alt_trades.import_insider_parquet) builds the same id, so an overlap never double-counts."""
    d = "" if tx_date is None or pd.isna(tx_date) else str(pd.Timestamp(tx_date).date())
    h = hashlib.sha1("|".join([table, code or "", d, canonical_num(shares), canonical_num(price), ad or ""])
                     .encode()).hexdigest()[:12]
    base = f"form4:{accession}:{h}"
    if seen is None:
        return base
    n = seen.get(base, 0)
    seen[base] = n + 1
    return base if n == 0 else f"{base}#{n}"


# ------------------------------------------------------------------------------------------------
# pure parsers (fixtures in tests/data/alt_fixtures)
# ------------------------------------------------------------------------------------------------
def parse_form_index(text: str, forms: tuple[str, ...] = FORMS) -> list[dict[str, Any]]:
    """EDGAR ``form.<date>.idx`` -> one dict per Form 4 / 4/A filing (accession, cik, date_filed)."""
    out, seen = [], set()
    body = text.split("\n")
    start = next((i + 1 for i, ln in enumerate(body) if ln.startswith("-----")), 0)
    for ln in body[start:]:
        m = _IDX_RE.match(ln.rstrip())
        if not m or m.group("form") not in forms:
            continue
        acc = _ACC_RE.search(m.group("file"))
        if not acc or acc.group(1) in seen:
            continue
        seen.add(acc.group(1))
        ds = m.group("date").replace("-", "")
        out.append({"accession": acc.group(1), "cik": str(int(m.group("cik"))), "form": m.group("form"),
                    "date_filed": f"{ds[:4]}-{ds[4:6]}-{ds[6:]}", "company": m.group("company").strip()})
    return out


def parse_current_atom(xml_text: str) -> list[dict[str, Any]]:
    """EDGAR ``getcurrent`` Atom feed -> one dict per Form 4 / 4/A accession (newest first)."""
    root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    out, seen = [], set()
    for e in root.iter(f"{_ATOM}entry"):
        cat = e.find(f"{_ATOM}category")
        form = (cat.get("term") if cat is not None else "") or ""
        if form not in FORMS:
            continue
        link = e.find(f"{_ATOM}link")
        href = link.get("href") if link is not None else ""
        ident = (e.findtext(f"{_ATOM}id") or "") + " " + (href or "")
        acc = _ACC_RE.search(ident) or _ACC_RE.search((href or "").replace("/", " "))
        cik = re.search(r"/data/(\d+)/", href or "")
        if not acc or not cik or acc.group(1) in seen:
            continue
        seen.add(acc.group(1))
        upd = e.findtext(f"{_ATOM}updated")
        ts = pd.Timestamp(upd) if upd else None
        out.append({"accession": acc.group(1), "cik": str(int(cik.group(1))), "form": form,
                    "date_filed": str(ts.tz_convert(NY).date()) if ts is not None and ts.tzinfo else None,
                    "accepted_feed": ts.tz_convert("UTC").isoformat() if ts is not None and ts.tzinfo else None})
    return out


def parse_acceptance(header_text: str) -> pd.Timestamp | None:
    """``<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS`` (US Eastern) -> UTC timestamp."""
    m = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", header_text)
    if not m:
        return None
    return pd.Timestamp(pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")).tz_localize(NY).tz_convert("UTC")


def extract_ownership_xml(submission: str) -> str | None:
    """The ownershipDocument XML inside a complete submission text file (or the XML itself)."""
    for m in re.finditer(r"<XML>(.*?)</XML>", submission, re.S | re.I):
        if "<ownershipDocument" in m.group(1):
            return m.group(1).strip()
    i = submission.find("<ownershipDocument")
    if i >= 0:
        j = submission.find("</ownershipDocument>", i)
        if j > 0:
            return submission[i:j + len("</ownershipDocument>")]
    return None


def _t(el: ET.Element | None, path: str) -> str | None:
    """Text at ``path`` or at ``path/value`` (Form 4 wraps most values), stripped; None if blank."""
    if el is None:
        return None
    node = el.find(path)
    if node is None:
        return None
    v = node.find("value")
    txt = (v.text if v is not None else node.text) or ""
    txt = txt.strip()
    return txt or None


def _flag(el: ET.Element | None, path: str) -> bool:
    return (_t(el, path) or "").lower() in ("1", "true")


def owner_detail(rel: ET.Element | None) -> str:
    parts = []
    if _flag(rel, "isDirector"):
        parts.append("Director")
    if _flag(rel, "isOfficer"):
        title = _t(rel, "officerTitle")
        parts.append(f"Officer: {title}" if title else "Officer")
    if _flag(rel, "isTenPercentOwner"):
        parts.append("10% owner")
    if _flag(rel, "isOther"):
        other = _t(rel, "otherText")
        parts.append(f"Other: {other}" if other else "Other")
    return "; ".join(parts) or "UNKNOWN"


def parse_form4_xml(xml_text: str, *, accession: str, accepted_at: pd.Timestamp | None, retrieved_at: pd.Timestamp,
                    url: str | None = None, date_filed: str | None = None, calendar=None) -> list[dict[str, Any]]:
    """One ownershipDocument -> alt_trades rows (one per transaction line; one NO_TRANSACTIONS row if none).

    ``accepted_at`` (UTC) is the PIT availability. Without it (never seen in practice) availability falls
    back to the cutoff of the session after ``date_filed`` and the row is PIT_CONSERVATIVE."""
    from quantlab.data.providers.sec_edgar import conservative_available_at
    root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    doc_type = _t(root, "documentType") or "4"
    issuer = root.find("issuer")
    raw_sym = (_t(issuer, "issuerTradingSymbol") or "").upper().strip()
    symbol = raw_sym if _SYMBOL_RE.match(raw_sym) and raw_sym not in ("NONE", "N/A", "NA") else None
    owners = root.findall("reportingOwner")
    names = [(_t(o, "reportingOwnerId/rptOwnerName") or "UNKNOWN") for o in owners]
    details = [owner_detail(o.find("reportingOwnerRelationship")) for o in owners]
    actor = "; ".join(names) or "UNKNOWN"
    detail = " | ".join(details) or "UNKNOWN"
    if accepted_at is not None:
        avail, pit = accepted_at, PitStatus.PIT.value
    else:
        if not date_filed:
            raise ProviderResponseError(f"form 4 {accession}: no acceptance time and no filing date")
        avail, pit = conservative_available_at(date_filed, calendar), PitStatus.PIT_CONSERVATIVE.value
    base = {"source": "insider", "symbol": symbol, "actor": actor[:300], "actor_detail": detail[:300],
            "disclosed_at": avail if accepted_at is not None else pd.Timestamp(date_filed).tz_localize(NY).tz_convert("UTC"),
            "available_at": avail, "pit_status": pit, "provider": PROVIDER, "retrieved_at": retrieved_at}
    meta = {"accession": accession, "form": doc_type, "issuer_cik": _t(issuer, "issuerCik"),
            "issuer_name": _t(issuer, "issuerName"), "issuer_symbol_raw": raw_sym or None,
            "owner_ciks": [_t(o, "reportingOwnerId/rptOwnerCik") for o in owners], "url": url}
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for table, tag in (("nd", "nonDerivativeTable/nonDerivativeTransaction"),
                       ("d", "derivativeTable/derivativeTransaction")):
        for tx in root.findall(tag):
            code = _t(tx, "transactionCoding/transactionCode") or ""
            ad = _t(tx, "transactionAmounts/transactionAcquiredDisposedCode") or ""
            shares = _num(_t(tx, "transactionAmounts/transactionShares"))
            price = _num(_t(tx, "transactionAmounts/transactionPricePerShare"))
            tdate = _t(tx, "transactionDate")
            tdate = pd.Timestamp(tdate[:10]) if tdate else pd.NaT
            side = "OTHER"
            if table == "nd" and code == "P" and ad == "A":
                side = "BUY"
            elif table == "nd" and code == "S" and ad == "D":
                side = "SELL"
            value = shares * price if shares == shares and price == price and side in ("BUY", "SELL") else float("nan")
            rows.append({**base, "record_id": form4_record_id(accession, table, code, tdate, shares, price, ad, seen),
                         "side": side, "amount_low_usd": value, "amount_high_usd": value, "shares": shares,
                         "price": price, "transaction_date": tdate,
                         "record_status": "PARSED" if symbol else "NO_SYMBOL",
                         "raw_json": json.dumps({**meta, "table": table, "code": code, "acquired_disposed": ad,
                                                 "security": _t(tx, "securityTitle")}, separators=(",", ":"))})
    if not rows:
        rows.append({**base, "record_id": f"form4:{accession}:none", "side": "UNKNOWN", "amount_low_usd": float("nan"),
                     "amount_high_usd": float("nan"), "shares": float("nan"), "price": float("nan"),
                     "transaction_date": pd.NaT, "record_status": "NO_TRANSACTIONS",
                     "raw_json": json.dumps(meta, separators=(",", ":"))})
    return rows


def parse_submission(text: str, *, accession: str, retrieved_at: pd.Timestamp, url: str | None = None,
                     date_filed: str | None = None, calendar=None) -> list[dict[str, Any]]:
    xml = extract_ownership_xml(text)
    if xml is None:
        raise ProviderResponseError(f"form 4 {accession}: no ownershipDocument XML in the submission")
    return parse_form4_xml(xml, accession=accession, accepted_at=parse_acceptance(text[:4000]),
                           retrieved_at=retrieved_at, url=url, date_filed=date_filed, calendar=calendar)


# ------------------------------------------------------------------------------------------------
# live provider (network only here; HTTP injectable for tests)
# ------------------------------------------------------------------------------------------------
class SecForm4Provider:
    name = PROVIDER

    def __init__(self, config: Any, http: HttpClient | None = None, calendar=None,
                 clock: Callable[[], pd.Timestamp] | None = None):
        self.config = config
        self.ua_env = config.get("providers.sec_edgar.user_agent_env", "QUANTLAB_SEC_USER_AGENT")
        self.calendar = calendar
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        if http is None:
            rps = float(config.get("providers.sec_edgar.max_requests_per_second", 8.0))
            http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("sec-edgar", rps), name="sec_form4")
        self.http = http

    def _headers(self) -> dict[str, str]:
        ua = get_secret(self.ua_env)
        if not ua:
            raise ProviderNotConfigured(f"SEC EDGAR requires environment variable {self.ua_env} "
                                        f"(format: 'App/Org Name contact@domain.com')")
        return {"User-Agent": ua.reveal(), "Accept-Encoding": "gzip, deflate"}

    def daily_index(self, day: date | str | pd.Timestamp) -> list[dict[str, Any]] | None:
        """Form 4 filings of one day from the daily form index; None = not published (404)."""
        d = pd.Timestamp(day)
        url = DAILY_INDEX_URL.format(year=d.year, q=(d.month - 1) // 3 + 1, ymd=d.strftime("%Y%m%d"))
        resp = self.http.request(url, headers=self._headers(), not_found_ok=True)
        if resp is None:
            return None
        if "html" in resp.content_type:
            raise ProviderResponseError(f"sec_form4: expected the text index from {url}, got HTML")
        return parse_form_index(resp.text)

    def current_filings(self, pages: int = 10, count: int = 100) -> list[dict[str, Any]]:
        """Recent Form 4 filings from the getcurrent Atom feed (newest first, de-duplicated)."""
        out, seen = [], set()
        for p in range(max(int(pages), 0)):
            text = self.http.get_text(CURRENT_URL.format(start=p * count, count=count), headers=self._headers())
            got = parse_current_atom(text)
            new = [g for g in got if g["accession"] not in seen]
            seen.update(g["accession"] for g in new)
            out.extend(new)
            if not got:
                break
        return out

    def fetch_filing(self, cik: str, accession: str, date_filed: str | None = None) -> list[dict[str, Any]]:
        url = FILING_URL.format(cik=int(cik), nodash=accession.replace("-", ""), accession=accession)
        text = self.http.get_text(url, headers=self._headers())
        return parse_submission(text, accession=accession, retrieved_at=self._clock(), url=url,
                                date_filed=date_filed, calendar=self.calendar)


__all__ = ["FORMS", "PROVIDER", "SecForm4Provider", "canonical_num", "extract_ownership_xml", "form4_record_id",
           "owner_detail", "parse_acceptance", "parse_current_atom", "parse_form4_xml", "parse_form_index",
           "parse_submission"]
