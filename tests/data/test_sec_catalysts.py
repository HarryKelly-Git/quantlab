"""SEC catalyst ingestion (earnings timing, 8-K items, foreign filers, point-in-time SIC), news
market-wide ingestion and headline classification. Offline: every HTTP call is faked."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.data.news_classify import classify_frame, classify_headline, guidance_direction
from quantlab.data.providers.http import HttpClient
from quantlab.data.providers.sec_edgar import SecEdgarProvider
from quantlab.data.sec_catalysts import SecCatalystIngest, parse_header_sic, release_timing
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession

CONFIG = {"providers": {"sec_edgar": {"user_agent_env": "TEST_SEC_UA", "max_requests_per_second": 8}}}
TICKERS = {"fields": ["cik", "name", "ticker", "exchange"],
           "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"], [1046179, "Taiwan Semi", "TSM", "NYSE"],
                    [1111, "ShellCo", "SHEL", "Nasdaq"]]}
CAL = TradingCalendar.from_dates(pd.bdate_range("2024-01-02", "2024-12-31"))


def _subs(rows, sic="3571"):
    cols = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items", "primaryDocument"]
    return {"cik": "0000320193", "name": "X", "sic": sic, "sicDescription": "CURRENT TITLE", "entityType": "operating",
            "fiscalYearEnd": "0928", "exchanges": ["Nasdaq"], "tickers": ["X"],
            "filings": {"recent": {c: [r.get(c, "") for r in rows] for c in cols}, "files": []}}


AAPL_ROWS = [
    {"accessionNumber": "a-8k-post", "acceptanceDateTime": "2024-05-02T20:30:41.000Z", "form": "8-K", "items": "2.02,9.01",
     "filingDate": "2024-05-02"},
    {"accessionNumber": "a-8k-pre", "acceptanceDateTime": "2024-08-01T11:00:00.000Z", "form": "8-K", "items": "2.02",
     "filingDate": "2024-08-01"},
    {"accessionNumber": "a-8ka", "acceptanceDateTime": "2024-08-05T15:00:00.000Z", "form": "8-K/A", "items": "2.02",
     "filingDate": "2024-08-05"},
    {"accessionNumber": "a-8k-fd", "acceptanceDateTime": "2024-06-03T14:00:00.000Z", "form": "8-K", "items": "7.01,9.01"},
    {"accessionNumber": "a-8k-ceo", "acceptanceDateTime": "2024-06-10T21:00:00.000Z", "form": "8-K", "items": "5.02"},
    {"accessionNumber": "a-10q-1", "acceptanceDateTime": "2024-02-02T21:00:00.000Z", "form": "10-Q", "reportDate": "2023-12-30"},
    {"accessionNumber": "a-10q-2", "acceptanceDateTime": "2024-05-03T21:00:00.000Z", "form": "10-Q", "reportDate": "2024-03-30"},
    {"accessionNumber": "a-10q-3", "acceptanceDateTime": "2024-08-02T21:00:00.000Z", "form": "10-Q", "reportDate": "2024-06-29"},
    {"accessionNumber": "a-old", "acceptanceDateTime": "2019-05-02T20:30:41.000Z", "form": "8-K", "items": "2.02"},
]
TSM_ROWS = [{"accessionNumber": "t-6k", "acceptanceDateTime": "2024-04-18T06:00:00.000Z", "form": "6-K"},
            {"accessionNumber": "t-20f", "acceptanceDateTime": "2024-04-20T10:00:00.000Z", "form": "20-F"}]
HEADER = ("<SEC-HEADER>\nFILER:\n COMPANY DATA:\n  COMPANY CONFORMED NAME: X\n  CENTRAL INDEX KEY: {cik}\n"
          "  STANDARD INDUSTRIAL CLASSIFICATION: {title} [{sic}]\n</SEC-HEADER>")


def _provider(monkeypatch, fake_clock, header_sic=None, calls=None):
    monkeypatch.setenv("TEST_SEC_UA", "QuantLab test@example.com")
    header_sic = header_sic or {}

    def handler(url, params, headers):
        if calls is not None:
            calls.append(url)
        if "company_tickers_exchange" in url:
            return FakeResponse(json_body=TICKERS)
        if "CIK0000320193.json" in url:
            return FakeResponse(json_body=_subs(AAPL_ROWS))
        if "CIK0001046179.json" in url:
            return FakeResponse(json_body=_subs(TSM_ROWS, sic="3674"))
        if "CIK0000001111.json" in url:
            return FakeResponse(status_code=404, headers={"Content-Type": "application/xml"}, text_body="<Error/>")
        if "index-headers" in url:
            accn = url.rsplit("/", 1)[-1].replace("-index-headers.html", "")
            cik = "0000320193" if accn.startswith("a-") else "0001046179"
            sic, title = header_sic.get(accn, ("3571", "ELECTRONIC COMPUTERS"))
            return FakeResponse(text_body=HEADER.format(cik=cik, sic=sic, title=title), headers={"Content-Type": "text/html"})
        raise AssertionError(url)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    return SecEdgarProvider(FakeConfig(CONFIG), http=http, calendar=CAL,
                            clock=lambda: pd.Timestamp("2024-09-01T12:00:00Z"))


def _ingest(monkeypatch, fake_clock, **kw):
    prov = _provider(monkeypatch, fake_clock, **kw)
    ing = SecCatalystIngest(prov, CAL, pd.Timestamp("2020-01-01", tz="UTC"),
                            clock=lambda: pd.Timestamp("2024-09-01T12:00:00Z"))
    return ing, ing.run(["AAPL", "TSM", "SHEL", "NOPE"], workers=2)


def test_earnings_timestamps_timing_and_reaction_are_point_in_time(monkeypatch, fake_clock):
    ing, df = _ingest(monkeypatch, fake_clock)
    er = df[df["event_type"] == "earnings_release"].set_index("source_id")
    post = er.loc["a-8k-post"]
    assert post["available_at"] == pd.Timestamp("2024-05-02T20:30:41Z")          # acceptance, not filing date
    assert json.loads(post["payload_json"])["timing"] == "POST_CLOSE"
    assert post["reaction_date"] == pd.Timestamp("2024-05-03")                    # after the close -> next session
    pre = er.loc["a-8k-pre"]
    assert json.loads(pre["payload_json"])["timing"] == "PRE_MARKET" and pre["reaction_date"] == pd.Timestamp("2024-08-01")
    assert json.loads(er.loc["a-8ka"]["payload_json"])["is_amendment"] is True
    assert "a-old" not in er.index                                               # before `since`
    k8 = df[df["event_type"] == "sec_8k"].set_index("source_id")
    assert "a-8k-ceo" in k8.index and "management" in json.loads(k8.loc["a-8k-ceo"]["payload_json"])["categories"]
    assert "a-8k-fd" not in k8.index                                             # 7.01/9.01 only: not material
    assert "a-8k-post" not in k8.index                                           # 2.02 + 9.01: earnings only
    assert df.attrs["unmapped_symbols"] == ["NOPE"]                              # explicit, never silently dropped
    tsm = df[df["symbol"] == "TSM"]
    assert set(tsm["event_type"]) >= {"foreign_report", "sec_registrant"} and "earnings_release" not in set(tsm["event_type"])
    reg = json.loads(tsm[tsm["event_type"] == "sec_registrant"].iloc[0]["payload_json"])
    assert reg["filer"] == "FOREIGN"


def test_current_snapshot_is_never_historical_and_sic_changes_are_found(monkeypatch, fake_clock):
    calls: list[str] = []
    _, df = _ingest(monkeypatch, fake_clock, calls=calls,
                    header_sic={"a-10q-1": ("3571", "COMPUTERS"), "a-10q-2": ("7372", "SOFTWARE"), "a-10q-3": ("7372", "SOFTWARE")})
    reg = df[(df["symbol"] == "AAPL") & (df["event_type"] == "sec_registrant")].iloc[0]
    assert reg["available_at"] == pd.Timestamp("2024-09-01T12:00:00Z")          # the snapshot is only known NOW
    assert reg["pit_status"] == "ASSUMED_STATIC"
    sic = df[(df["symbol"] == "AAPL") & (df["event_type"] == "sic_observation")].sort_values("available_at")
    codes = [json.loads(p)["sic"] for p in sic["payload_json"]]
    assert codes == ["3571", "7372", "7372"]                                     # the change point was located
    assert sic.iloc[1]["available_at"] == pd.Timestamp("2024-05-03T21:00:00Z")   # known from that filing's acceptance
    assert all(pd.isna(x) for x in sic["reaction_date"])


def test_header_parser_picks_the_filers_own_block():
    text = ("FILER:\n CENTRAL INDEX KEY: 0000000099\n STANDARD INDUSTRIAL CLASSIFICATION: BANKS [6022]\n"
            "FILER:\n CENTRAL INDEX KEY: 0000320193\n STANDARD INDUSTRIAL CLASSIFICATION: ELECTRONIC COMPUTERS [3571]\n")
    assert parse_header_sic(text, "0000320193") == ("3571", "ELECTRONIC COMPUTERS")
    assert parse_header_sic("STANDARD INDUSTRIAL CLASSIFICATION: [0000]", "1") is None       # blank SIC = UNKNOWN
    hdr = "CENTRAL INDEX KEY: 1\nSTANDARD INDUSTRIAL CLASSIFICATION: SHIP &amp; BOAT BUILDING [3730]"
    assert parse_header_sic(hdr, "1") == ("3730", "SHIP & BOAT BUILDING")                    # HTML entities decoded


def test_timing_classes_and_events_before_the_calendar():
    assert release_timing(pd.Timestamp("2024-05-02T20:30:00Z"), CAL) == "POST_CLOSE"
    assert release_timing(pd.Timestamp("2024-05-02T12:00:00Z"), CAL) == "PRE_MARKET"
    assert release_timing(pd.Timestamp("2024-05-02T15:00:00Z"), CAL) == "INTRADAY"
    assert release_timing(pd.Timestamp("2024-05-04T15:00:00Z"), CAL) == "NON_SESSION"
    assert release_timing(pd.Timestamp("2019-05-02T15:00:00Z"), CAL) == "UNKNOWN"


def test_ingestion_rerun_does_not_duplicate_rows(monkeypatch, fake_clock, ctx):
    _, a = _ingest(monkeypatch, fake_clock)
    ds1 = ctx.store.write("events", a, "sec_edgar", params={"chunk": 0})
    ds1b = ctx.store.write("events", a, "sec_edgar", params={"chunk": 0})
    assert ds1 == ds1b                                                           # identical content: one dataset
    b = a.assign(retrieved_at=pd.Timestamp("2024-09-02T12:00:00Z"))             # resumed chunk, later retrieval
    ctx.store.write("events", b, "sec_edgar", params={"chunk": 0, "resumed": True})
    loaded = ctx.store.load("events", synthetic=False)
    assert len(loaded) == len(a)                                                 # deduplicated on the key
    assert loaded.duplicated(["symbol", "event_type", "source_id"]).sum() == 0


def test_news_tags_are_counted_before_any_symbol_filter(ctx):
    t = pd.Timestamp("2024-03-01T15:00:00Z")
    rows = []
    for nid, tags, head in (("n1", ["AAPL"], "Apple Unveils New Laptop"), ("n2", ["AAPL", "MSFT", "SPY"], "Tech Stocks Moving"),
                            ("n3", ["ZZZ.X"], "ZZZ Announces Agreement To Acquire Q")):
        for s in tags:
            rows.append({"news_id": nid, "symbol": s, "headline": head, "summary": "", "source": "benzinga", "url": "",
                         "created_at": t, "updated_at": t + pd.Timedelta(hours=2 if nid == "n1" else 0), "available_at": t,
                         "pit_status": "PIT_CONSERVATIVE" if nid == "n1" else "PIT", "provider": "alpaca", "retrieved_at": t})
    ctx.store.write("news", pd.DataFrame(rows), "alpaca")
    n = ctx.store.load("news", synthetic=False)
    assert dict(zip(n["news_id"], n["n_tags"])) == {"n1": 1.0, "n2": 3.0, "n3": 1.0}
    c = classify_frame(n[n["symbol"] == "AAPL"])                                 # a symbol-filtered view keeps n_tags
    by = dict(zip(c["news_id"], zip(c["category"], c["company_specific"], c["material"])))
    assert by["n1"] == ("product", True, True) and by["n2"][1] is False
    assert "ZZZ.X" in set(n["symbol"])                                           # unmapped tags are kept, not dropped
    assert (n["available_at"] == n["created_at"]).all()


@pytest.mark.parametrize("head,cat", [
    ("Teleflex Q2 EPS $3.39 Beats $3.34 Estimate, Sales $704.50M Miss", "earnings"),
    ("Kura Sushi USA Sees FY22 Sales $137M-$142M Vs. $135.85M Est.", "guidance"),
    ("Weatherford Raises Guidance", "guidance"),
    ("Morgan Stanley Maintains Overweight on Apple, Raises Price Target to $200", "analyst"),
    ("Looking Into Tesla's Recent Short Interest", "market_commentary"),
    ("Royal Gold Announces Agreement To Acquire Great Bear Royalties", "m&a"),
    ("Agile Therapeutics Announces Pricing of $24M Upsized Public Offering", "financing"),
    ("WD-40 Company Appoints Sara Hyzer As Chief Financial Officer", "management"),
    ("Nyxoah Announces CE Mark Approval For Genio", "regulatory"),
    ("President Trump Says Tariffs Are Coming", "other"),
    ("Stocks Move Higher Following Release Of Fed Minutes", "other"),
])
def test_headline_rules(head, cat):
    assert classify_headline(head) == cat


def test_guidance_direction_is_only_read_from_explicit_verbs():
    assert guidance_direction("Weatherford Raises Guidance") == "UP"
    assert guidance_direction("XYZ Lowers FY24 Outlook") == "DOWN"
    assert guidance_direction("XYZ Reaffirms FY24 Guidance") == "REAFFIRM"
    assert guidance_direction("XYZ Sees FY24 EPS $1.20-$1.30") == "UNKNOWN"
