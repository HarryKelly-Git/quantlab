"""US House periodic transaction reports (PTRs, House Clerk) -> ``alt_trades`` rows. CONTEXT ONLY.

Facts this module relies on (docs/ALT-DATA.md):
  * The Clerk publishes one index per year, ``financial-pdfs/<YEAR>FD.zip``, holding ``<YEAR>FD.xml``:
    one ``<Member>`` per filing with Prefix/First/Last/Suffix, FilingType ('P' = periodic transaction
    report), StateDst, Year, FilingDate (M/D/YYYY, date only) and DocID. No key, no documented limit:
    we stay modest (``providers.house_clerk.max_requests_per_second``, default 1).
  * Each PTR is a PDF at ``ptr-pdfs/<YEAR>/<DocID>.pdf``. Electronically filed PTRs contain text: one
    table row per transaction, ``[owner] Asset name (TICKER) [XX] <type> <tx date> <notification date>
    <amount range>``, followed by labelled lines (``F S : New``, ``D : description``, ``S O : ...``).
    Paper filings are scanned images (no text): recorded as UNPARSEABLE, never guessed.
  * Type: P = purchase -> BUY, S / S (partial) = sale -> SELL, E = exchange -> OTHER. Only stock rows
    ([ST], or no asset-type tag) get BUY/SELL; options ([OP]) and every other asset class are OTHER
    (a put purchase is bearish: the direction is ambiguous). The original type stays in raw_json.
  * Availability: the index FilingDate is date-only, so available_at = the cutoff of the session AFTER
    it (the repo's date-only rule, PIT_CONSERVATIVE). The trade date is NEVER availability (PTRs arrive
    up to 45 days after the trade).
  * The Senate has no free automated source (its eFD site refuses automated access): not covered.
"""
from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.data.providers.base import ProviderError
from quantlab.data.providers.http import HttpClient, ProviderResponseError, shared_rate_limiter

NY = ZoneInfo("America/New_York")
INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
PROVIDER = "house_clerk"
STOCK_TYPES = ("ST",)


class PdfLibraryMissing(ProviderError):
    """No PDF text library: the congress part is FAILED (retried later), never marked UNPARSEABLE."""


# ------------------------------------------------------------------------------------------------
# index
# ------------------------------------------------------------------------------------------------
def parse_index_xml(xml_bytes: bytes | str) -> list[dict[str, Any]]:
    """``<YEAR>FD.xml`` -> one dict per filing (all filing types; callers keep 'P')."""
    root = ET.fromstring(xml_bytes.encode("utf-8") if isinstance(xml_bytes, str) else xml_bytes)
    out = []
    for m in root.iter("Member"):
        g = lambda k: (m.findtext(k) or "").strip()          # noqa: E731
        fd = g("FilingDate")
        try:
            filed = pd.to_datetime(fd, format="%m/%d/%Y") if fd else None
        except (ValueError, TypeError):
            filed = None
        name = " ".join(x for x in (g("Prefix"), g("First"), g("Last"), g("Suffix")) if x)
        out.append({"doc_id": g("DocID"), "filing_type": g("FilingType"), "name": name or "UNKNOWN",
                    "state_dst": g("StateDst") or None, "year": g("Year") or None,
                    "filing_date": str(filed.date()) if filed is not None else None})
    return out


def read_index_zip(content: bytes) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        xml_name = next((n for n in z.namelist() if n.lower().endswith(".xml")), None)
        if xml_name is None:
            raise ProviderResponseError("house_clerk: index ZIP holds no XML")
        return parse_index_xml(z.read(xml_name))


# ------------------------------------------------------------------------------------------------
# PTR text
# ------------------------------------------------------------------------------------------------
_LABEL_RE = re.compile(r"^\s*(F\s*S|D|S\s*O|C|L|Filing\s+Status|Description|Subholding\s+Of|Comments?|Location)\s*:",
                       re.I)
_TX_RE = re.compile(
    r"(?:(?P<owner>\b(?:SP|JT|DC)\b)\s+)?"
    r"(?P<asset>[^\[\]]{1,240}?)\s*\[(?P<atype>[A-Z]{2})\]\s*"
    r"(?P<type>P|S\s*\(partial\)|S|E)\s+"
    r"(?P<tx>\d{1,2}/\d{1,2}/\d{4})\s*(?P<notif>\d{1,2}/\d{1,2}/\d{4})\s*"
    r"(?P<amount>\$[\d,]+\s*-\s*\$[\d,]+|Over\s+\$[\d,]+|\$[\d,]+)")
_HEADER_RE = re.compile(r"^.*?Gains\s*>\s*\$200\?", re.S)
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-/]{0,9})\)\s*$")


def _amount(text: str) -> tuple[float, float]:
    nums = [float(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", text)]
    if text.strip().lower().startswith("over") and nums:
        return nums[0], float("nan")                         # "Over $50,000,000": upper bound UNKNOWN
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], nums[0]
    return float("nan"), float("nan")


def _date(s: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(s, format="%m/%d/%Y")
    except (ValueError, TypeError):
        return pd.NaT


def parse_ptr_text(text: str) -> list[dict[str, Any]]:
    """Text of an electronically filed PTR -> transactions. Empty list = nothing recognisable.

    Labelled lines (filing status, description, subholding, comments) are dropped first, and the
    remaining lines of each block are joined, so an asset name wrapped across lines still parses
    while a description can never leak into the next row's asset name."""
    text = (text or "").replace("\x00", "")
    m = _HEADER_RE.search(text)
    if m:
        text = text[m.end():]
    blocks, cur = [], []
    for ln in text.splitlines():
        if _LABEL_RE.match(ln):
            if cur:
                blocks.append(" ".join(cur))
            cur = []
            continue
        if ln.strip():
            cur.append(ln.strip())
    if cur:
        blocks.append(" ".join(cur))
    out = []
    for b in blocks:
        b = re.sub(r"\s+", " ", b)
        for t in _TX_RE.finditer(b):
            asset = t.group("asset").strip(" -")
            tick = _TICKER_RE.search(asset)
            typ = re.sub(r"\s+", " ", t.group("type"))
            lo, hi = _amount(t.group("amount"))
            out.append({"owner": t.group("owner"), "asset": asset[:200], "ticker": tick.group(1) if tick else None,
                        "asset_type": t.group("atype"), "type": typ, "transaction_date": _date(t.group("tx")),
                        "notification_date": _date(t.group("notif")), "amount_low_usd": lo, "amount_high_usd": hi,
                        "amount_text": re.sub(r"\s+", " ", t.group("amount"))})
    return out


def side_of(tx_type: str, asset_type: str | None) -> str:
    if asset_type and asset_type not in STOCK_TYPES:
        return "OTHER"
    t = (tx_type or "").upper()
    if t == "P":
        return "BUY"
    if t.startswith("S"):
        return "SELL"
    return "OTHER"


def ptr_rows(filing: dict[str, Any], text: str | None, *, retrieved_at: pd.Timestamp, calendar=None,
             url: str | None = None, why_unparseable: str | None = None) -> list[dict[str, Any]]:
    """One PTR -> alt_trades rows. ``text`` None/empty (scanned paper filing) -> one UNPARSEABLE row."""
    from quantlab.data.providers.sec_edgar import conservative_available_at
    filed = pd.Timestamp(filing["filing_date"])
    base = {"source": "congress", "actor": filing.get("name") or "UNKNOWN",
            "disclosed_at": filed.tz_localize(NY).tz_convert("UTC"),
            "available_at": conservative_available_at(filed, calendar), "pit_status": PitStatus.PIT_CONSERVATIVE.value,
            "shares": float("nan"), "price": float("nan"), "provider": PROVIDER, "retrieved_at": retrieved_at}
    meta = {"doc_id": filing["doc_id"], "filing_date": filing["filing_date"], "state_dst": filing.get("state_dst"),
            "url": url}
    chamber = f"House / {filing.get('state_dst') or 'UNKNOWN'}"
    txs = parse_ptr_text(text) if text else []
    if not txs:
        why = why_unparseable or ("no text layer (scanned paper filing?)" if not (text or "").strip()
                                  else "text present but no transaction row recognised")
        return [{**base, "record_id": f"house:{filing['doc_id']}:unparseable", "symbol": None, "actor_detail": chamber,
                 "side": "UNKNOWN", "amount_low_usd": float("nan"), "amount_high_usd": float("nan"),
                 "transaction_date": pd.NaT, "record_status": "UNPARSEABLE",
                 "raw_json": json.dumps({**meta, "why": why}, separators=(",", ":"))}]
    rows = []
    for i, t in enumerate(txs):
        sym = t["ticker"]
        rows.append({**base, "record_id": f"house:{filing['doc_id']}:{i}", "symbol": sym,
                     "actor_detail": chamber + (f" / owner {t['owner']}" if t["owner"] else ""),
                     "side": side_of(t["type"], t["asset_type"]), "amount_low_usd": t["amount_low_usd"],
                     "amount_high_usd": t["amount_high_usd"], "transaction_date": t["transaction_date"],
                     "record_status": "PARSED" if sym else "NO_SYMBOL",
                     "raw_json": json.dumps({**meta, "asset": t["asset"], "asset_type": t["asset_type"], "type": t["type"],
                                             "owner": t["owner"], "amount": t["amount_text"],
                                             "notification_date": str(t["notification_date"].date())
                                             if pd.notna(t["notification_date"]) else None},
                                            separators=(",", ":"))})
    return rows


def pdf_text(content: bytes) -> str:
    """Text layer of a PDF via pypdf (lazy import). Scanned PDFs return '' (-> UNPARSEABLE)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PdfLibraryMissing("pypdf is not installed in this environment (pip install pypdf); "
                                "congress PTRs are not parsed until it is") from exc
    reader = PdfReader(io.BytesIO(content))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


# ------------------------------------------------------------------------------------------------
# live provider (network only here; HTTP injectable for tests)
# ------------------------------------------------------------------------------------------------
class HousePtrProvider:
    name = PROVIDER

    def __init__(self, config: Any, http: HttpClient | None = None, calendar=None,
                 clock: Callable[[], pd.Timestamp] | None = None, text_of: Callable[[bytes], str] | None = None):
        self.config = config
        self.calendar = calendar
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self._text_of = text_of or pdf_text
        if http is None:
            rps = float(config.get("providers.house_clerk.max_requests_per_second", 1.0))
            ua = config.get("providers.house_clerk.user_agent", "QuantLab research (personal, non-commercial)")
            http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("house-clerk", rps), name="house_clerk",
                                          user_agent=ua)
        self.http = http

    def index(self, year: int) -> list[dict[str, Any]] | None:
        """All filings of ``year`` from the annual index ZIP; None if the ZIP does not exist (404)."""
        resp = self.http.request(INDEX_URL.format(year=int(year)), not_found_ok=True)
        if resp is None:
            return None
        if "html" in resp.content_type:
            raise ProviderResponseError(f"house_clerk: expected the {year} index ZIP, got HTML")
        return read_index_zip(resp.content)

    def ptr(self, filing: dict[str, Any]) -> list[dict[str, Any]]:
        year = filing.get("year") or str(pd.Timestamp(filing["filing_date"]).year)
        url = PTR_URL.format(year=year, doc_id=filing["doc_id"])
        resp = self.http.request(url, not_found_ok=True)
        now = self._clock()
        if resp is None:
            return ptr_rows(filing, None, retrieved_at=now, calendar=self.calendar, url=url,
                            why_unparseable="PDF not found (404)")
        if not resp.content.startswith(b"%PDF"):
            raise ProviderResponseError(f"house_clerk: {url} is not a PDF ({resp.content_type or 'no content-type'})")
        try:
            text = self._text_of(resp.content)
        except PdfLibraryMissing:
            raise
        except Exception as exc:                  # a corrupt / encrypted PDF: recorded, never guessed
            return ptr_rows(filing, None, retrieved_at=now, calendar=self.calendar, url=url,
                            why_unparseable=f"PDF unreadable: {type(exc).__name__}")
        return ptr_rows(filing, text, retrieved_at=now, calendar=self.calendar, url=url)


__all__ = ["HousePtrProvider", "INDEX_URL", "PROVIDER", "PTR_URL", "PdfLibraryMissing", "parse_index_xml",
           "parse_ptr_text", "pdf_text", "ptr_rows", "read_index_zip", "side_of"]
