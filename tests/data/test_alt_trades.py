"""Free alt-data layer (SEC Form 4 + House PTR), offline: parsers, incremental bounded ingest, the
daily refresh hook (never raises, daily scope only) and the research-parquet import."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from quantlab.context import AppContext
from quantlab.data import alt_trades as alt
from quantlab.data import schemas
from quantlab.data.providers import house_ptr as hp
from quantlab.data.providers import sec_form4 as f4
from quantlab.data.providers.http import HttpClient

from .alt_fixtures import (CURRENT_ATOM, FORM4_NO_TICKER, FORM4_SUBMISSION, FORM4_XML, FORM_INDEX, PTR_TEXT,
                           house_index_zip)
from .fakes import FakeClock, FakeResponse, FakeSession

NOW = pd.Timestamp("2024-03-05T23:00:00Z")
UA_ENV = "QUANTLAB_SEC_USER_AGENT"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No real .env is read and no real HTTP request can leave these tests."""
    import quantlab.secrets as secrets
    monkeypatch.setattr(secrets, "load_dotenv", lambda path: [])
    monkeypatch.setenv(UA_ENV, "QuantLab test test@example.com")

    def no_network(*a, **k):
        raise AssertionError("a test tried to reach the network")
    monkeypatch.setattr(requests.Session, "get", no_network)


@pytest.fixture
def actx(config):
    c = AppContext.create(config, init_logging=False)
    yield c
    c.close()


# ---------------------------------------------------------------------------------------------- SEC
def test_form_index_keeps_form4_once_per_accession():
    got = f4.parse_form_index(FORM_INDEX)
    assert [g["accession"] for g in got] == ["0000222222-24-000001", "0000333333-24-000002"]
    assert got[0]["date_filed"] == "2024-03-04" and got[1]["form"] == "4/A"


def test_current_feed_dedupes_roles_and_drops_other_forms():
    got = f4.parse_current_atom(CURRENT_ATOM)
    assert len(got) == 1 and got[0]["accession"] == "0000444444-24-000003" and got[0]["cik"] == "444444"
    assert got[0]["date_filed"] == "2024-03-05"


def test_form4_rows_sides_values_and_exact_acceptance_time():
    rows = f4.parse_submission(FORM4_SUBMISSION, accession="0000222222-24-000001", retrieved_at=NOW)
    df = schemas.conform("alt_trades", pd.DataFrame(rows))
    assert list(df["side"]) == ["BUY", "SELL", "OTHER", "OTHER"]          # P+A, S+D, F (tax), derivative M
    assert set(df["symbol"]) == {"AAA"} and set(df["record_status"]) == {"PARSED"}
    assert df.loc[0, "amount_low_usd"] == 10_000.0 and df.loc[1, "amount_high_usd"] == 6_250.0
    assert np.isnan(df.loc[2, "amount_low_usd"])                         # OTHER: no value claimed
    # 17:30:15 US Eastern (EST in March before DST) = 22:30:15 UTC; availability is exactly that
    assert (df["available_at"] == pd.Timestamp("2024-03-04T22:30:15Z")).all() and set(df["pit_status"]) == {"PIT"}
    assert df.loc[0, "actor"] == "Doe Jane" and df.loc[0, "actor_detail"] == "Director; Officer: Chief Executive Officer"
    assert df["record_id"].is_unique and df["record_id"].str.startswith("form4:0000222222-24-000001:").all()
    again = f4.parse_submission(FORM4_SUBMISSION, accession="0000222222-24-000001", retrieved_at=NOW)
    assert [r["record_id"] for r in again] == list(df["record_id"])      # stable ids: re-runs dedupe


def test_form4_without_a_ticker_is_recorded_not_guessed():
    rows = f4.parse_form4_xml(FORM4_NO_TICKER, accession="x-1", accepted_at=NOW, retrieved_at=NOW)
    assert {r["symbol"] for r in rows} == {None} and {r["record_status"] for r in rows} == {"NO_SYMBOL"}
    df = schemas.conform("alt_trades", pd.DataFrame(rows))
    assert df["symbol"].isna().all()                                     # None, never the string "NONE"


# -------------------------------------------------------------------------------------------- House
def test_house_index_zip():
    got = hp.read_index_zip(house_index_zip())
    p = [g for g in got if g["filing_type"] == "P"]
    assert [g["doc_id"] for g in p] == ["20024001", "8220999"]
    assert p[0]["name"] == "Hon. Pat Example" and p[0]["filing_date"] == "2024-03-01" and p[0]["state_dst"] == "CA12"


def test_ptr_text_parsing():
    txs = hp.parse_ptr_text(PTR_TEXT)
    assert [t["ticker"] for t in txs] == ["AAA", "BBB", "CCC", None]
    a, b, c, t = txs
    assert a["owner"] == "SP" and a["type"] == "P" and (a["amount_low_usd"], a["amount_high_usd"]) == (1001.0, 15000.0)
    assert a["transaction_date"] == pd.Timestamp("2024-02-14") and a["notification_date"] == pd.Timestamp("2024-02-20")
    assert b["asset"].startswith("Beta Holdings Inc. Class A") and b["type"] == "S (partial)"   # wrapped cell
    assert "earnings call" not in b["asset"]                             # a description never leaks into a row
    assert c["owner"] == "JT" and c["asset_type"] == "OP"
    assert t["amount_low_usd"] == 50_000_000.0 and np.isnan(t["amount_high_usd"])   # "Over": upper bound UNKNOWN


def test_ptr_rows_sides_availability_and_unparseable():
    filing = {"doc_id": "20024001", "filing_date": "2024-03-01", "name": "Hon. Pat Example", "state_dst": "CA12"}
    df = schemas.conform("alt_trades", pd.DataFrame(hp.ptr_rows(filing, PTR_TEXT, retrieved_at=NOW)))
    assert list(df["side"]) == ["BUY", "SELL", "OTHER", "OTHER"]          # options and T-bills: OTHER
    assert list(df["record_status"]) == ["PARSED", "PARSED", "PARSED", "NO_SYMBOL"]
    # filed Fri 2024-03-01 (date only): usable from the cutoff of Mon 03-04, 16:00 ET; never the trade date
    assert (df["available_at"] == pd.Timestamp("2024-03-04T21:00:00Z")).all()
    assert set(df["pit_status"]) == {"PIT_CONSERVATIVE"} and df.loc[0, "actor_detail"] == "House / CA12 / owner SP"
    assert list(df["record_id"]) == [f"house:20024001:{i}" for i in range(4)]
    scanned = hp.ptr_rows({**filing, "doc_id": "8220999"}, "", retrieved_at=NOW)
    assert len(scanned) == 1 and scanned[0]["record_status"] == "UNPARSEABLE" and scanned[0]["symbol"] is None
    assert scanned[0]["side"] == "UNKNOWN" and np.isnan(scanned[0]["amount_low_usd"])


def test_house_provider_records_scanned_pdfs_and_fails_without_a_pdf_library(config):
    def handler(url, params, headers):
        if url.endswith("2024FD.zip"):
            return FakeResponse(200, text_body=house_index_zip(), headers={"Content-Type": "application/zip"})
        return FakeResponse(200, text_body=b"%PDF-1.7 fake", headers={"Content-Type": "application/pdf"})
    clock = FakeClock()
    http = HttpClient(session=FakeSession(handler), max_retries=0, sleep=clock.sleep, clock=clock.time, name="house")
    prov = hp.HousePtrProvider(config, http=http, clock=lambda: NOW, text_of=lambda b: "")
    filings = [f for f in prov.index(2024) if f["filing_type"] == "P"]
    assert prov.ptr(filings[1])[0]["record_status"] == "UNPARSEABLE"

    def missing(b):
        raise hp.PdfLibraryMissing("pypdf is not installed")
    prov2 = hp.HousePtrProvider(config, http=http, clock=lambda: NOW, text_of=missing)
    with pytest.raises(hp.PdfLibraryMissing):
        prov2.ptr(filings[0])


# ------------------------------------------------------------------------------------------- ingest
class FakeForm4:
    def __init__(self, fail_listing=False, fail_fetch=False):
        self.http = type("H", (), {"request_count": 0})()
        self.fetched: list[str] = []
        self.fail_listing, self.fail_fetch = fail_listing, fail_fetch

    def daily_index(self, d):
        if self.fail_listing:
            raise requests.ConnectionError("sec unreachable")
        if pd.Timestamp(d) == pd.Timestamp("2024-03-04"):
            return f4.parse_form_index(FORM_INDEX)
        return None

    def current_filings(self, pages=10):
        return f4.parse_current_atom(CURRENT_ATOM)

    def fetch_filing(self, cik, accession, date_filed=None):
        if self.fail_fetch:
            raise requests.ConnectionError("timeout")
        self.fetched.append(accession)
        return f4.parse_form4_xml(FORM4_XML, accession=accession, accepted_at=NOW - pd.Timedelta(hours=1),
                                  retrieved_at=NOW)


class FakeHouse:
    def __init__(self, text_error=None):
        self.http = type("H", (), {"request_count": 0})()
        self.fetched: list[str] = []
        self.text_error = text_error

    def index(self, year):
        return hp.read_index_zip(house_index_zip()) if year == 2024 else None

    def ptr(self, filing):
        if self.text_error:
            raise self.text_error
        self.fetched.append(filing["doc_id"])
        return hp.ptr_rows(filing, PTR_TEXT if filing["doc_id"].startswith("2") else "", retrieved_at=NOW)


def test_ingest_is_incremental_and_dedupes(actx):
    fake = FakeForm4()
    out = alt.ingest_insider(actx, days=5, now=NOW, provider=fake)
    assert out["status"] == "OK" and out["fetched"] == 3 and out["rows"] == 12
    again = alt.ingest_insider(actx, days=5, now=NOW, provider=fake)
    assert again["status"] == "OK" and again["fetched"] == 0 and again["already_stored"] == 3
    assert fake.fetched == ["0000222222-24-000001", "0000333333-24-000002", "0000444444-24-000003"]  # oldest first
    house = FakeHouse()
    out = alt.ingest_congress(actx, days=30, now=NOW, provider=house)
    assert out["status"] == "OK" and out["unparseable_filings"] == 1 and house.fetched == ["20024001", "8220999"]
    assert alt.ingest_congress(actx, days=30, now=NOW, provider=house)["fetched"] == 0
    df = actx.store.load("alt_trades", synthetic=False)
    assert df.duplicated(["source", "record_id"]).sum() == 0 and set(df["source"]) == {"insider", "congress"}


def test_budget_leaves_the_rest_for_the_next_run(actx):
    out = alt.ingest_insider(actx, days=5, now=NOW, provider=FakeForm4(), max_filings=1)
    assert out["status"] == "PARTIAL" and out["fetched"] == 1 and out["remaining"] == 2
    nxt = alt.ingest_insider(actx, days=5, now=NOW, provider=FakeForm4())
    assert nxt["status"] == "OK" and nxt["fetched"] == 2


def test_an_outage_is_reported_never_raised(actx, monkeypatch):
    out = alt.ingest_alt_trades(actx, mode="refresh", now=NOW,
                                providers={"insider": FakeForm4(fail_listing=True),
                                           "congress": FakeHouse(text_error=hp.PdfLibraryMissing("no pypdf"))})
    assert out["insider"]["status"] == "FAILED" and out["congress"]["status"] == "FAILED"
    assert out["status"] == "FAILED" and out["rows"] == "FAILED" and "sec unreachable" in out["error"]
    assert not actx.store.dataset_ids("alt_trades", synthetic=False)
    flaky = alt.ingest_insider(actx, days=5, now=NOW, provider=FakeForm4(fail_fetch=True))
    assert flaky["status"] == "PARTIAL" and flaky["failed"] == 3     # < max_consecutive_errors: retried next run
    # a bug inside the alt part still never escapes the catalyst refresh
    import quantlab.data.catalyst_refresh as cr

    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(alt, "refresh_alt", boom)
    res = cr.refresh_catalysts(actx, pd.Timestamp("2024-03-05"), parts=(), now=NOW)
    assert res["alt"]["status"] == "FAILED" and "bug" in res["alt"]["error"]


def test_without_a_sec_user_agent_insiders_are_skipped(actx, monkeypatch):
    monkeypatch.delenv(UA_ENV, raising=False)
    out = alt.ingest_alt_trades(actx, sources=("insider",), mode="refresh", now=NOW)
    assert out["insider"]["status"] == "SKIPPED" and out["rows"] == "SKIPPED"


def test_alt_runs_in_the_daily_market_wide_refresh_only(actx, monkeypatch):
    import quantlab.data.catalyst_refresh as cr
    calls = []
    monkeypatch.setattr(alt, "refresh_alt", lambda ctx, now=None, providers=None: calls.append(now) or {"rows": 0})
    cr.refresh_catalysts(actx, pd.Timestamp("2024-03-05"), parts=(), now=NOW)                    # daily: symbols=None
    cr.refresh_catalysts(actx, pd.Timestamp("2024-03-05"), symbols=["AAA"], parts=(), now=NOW)   # pre-open scope
    assert len(calls) == 1
    cr.refresh_catalysts(actx, pd.Timestamp("2024-03-05"), symbols=["AAA"], parts=("alt",), now=NOW)
    assert len(calls) == 2


def test_research_parquet_import_matches_live_record_ids(actx, tmp_path):
    live = f4.parse_form4_xml(FORM4_XML, accession="0000222222-24-000001", accepted_at=NOW, retrieved_at=NOW)
    pq = pd.DataFrame({"accession": ["0000222222-24-000001"] * 2, "filing_date": pd.to_datetime(["2024-03-04"] * 2),
                       "trans_date": pd.to_datetime(["2024-03-01"] * 2), "issuer_cik": ["111111"] * 2,
                       "issuer_symbol": ["AAA"] * 2, "owner_cik": ["0000222222"] * 2,
                       "relationship": ["Director,Officer"] * 2, "title": ["CEO"] * 2, "code": ["P", "S"],
                       "shares": [1000.0, 500.0], "price": [10.0, 12.5], "value": [10_000.0, 6_250.0]})
    rows = alt.insider_parquet_rows(pq)
    assert list(rows["record_id"]) == [r["record_id"] for r in live[:2]]       # an overlap never double-counts
    assert list(rows["side"]) == ["BUY", "SELL"] and rows.loc[0, "actor_detail"] == "Director; Officer: CEO"
    # date-only filing date: usable from the cutoff of the session after it (Tue 03-05 16:00 ET)
    assert (rows["available_at"] == pd.Timestamp("2024-03-05T21:00:00Z")).all()
    path = tmp_path / "insider.parquet"
    pq.to_parquet(path)
    out = alt.import_insider_parquet(actx, path)
    assert out["rows"] == 2 and alt.stored_filing_ids(actx.store, "insider") == {"0000222222-24-000001"}


def test_read_helpers_show_unknown_amounts_and_usability(actx):
    alt.ingest_congress(actx, days=30, now=NOW, provider=FakeHouse())
    alt.ingest_insider(actx, days=5, now=NOW, provider=FakeForm4())
    frame = alt.load_recent(actx.store, since_days=60, now=NOW)
    recs = alt.recent_disclosures(actx.store, frame=frame, limit=50, now=NOW)
    assert recs and all(r["symbol"] for r in recs)                        # filings without a ticker are not listed
    other = next(r for r in recs if r["source"] == "insider" and r["side"] == "OTHER")
    assert other["amount"] == "UNKNOWN"                                   # never shown as $0
    assert any(r["amount"] == "$1,001 - $15,000" for r in recs if r["source"] == "congress")
    st = {r["source"]: r for r in alt.source_status(actx.store, actx.config)}
    assert st["insider"]["datasets"] == 1 and st["congress"]["last_pull_detail"]["unparseable"] == 1
