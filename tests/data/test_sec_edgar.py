"""SecEdgarProvider: ticker->CIK, submissions incl. files pages, 8-K 2.02 events, companyfacts PIT."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.data.providers.base import ProviderNotConfigured
from quantlab.data.providers.http import HttpClient
from quantlab.data.providers.sec_edgar import SecEdgarProvider, classify_duration, conservative_available_at, pad_cik
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession

BASE_CONFIG = {"providers": {"sec_edgar": {"user_agent_env": "TEST_SEC_UA", "max_requests_per_second": 8}}}

TICKERS_JSON = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [320193, "Apple Inc.", "AAPL", "Nasdaq"],
        [1018724, "Amazon.com Inc", "AMZN", "Nasdaq"],
    ],
}


def make_provider(handler, fake_clock, monkeypatch, calendar=None, set_ua=True):
    if set_ua:
        monkeypatch.setenv("TEST_SEC_UA", "QuantLab test@example.com")
    config = FakeConfig(BASE_CONFIG)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    return SecEdgarProvider(config, http=http, calendar=calendar, clock=lambda: pd.Timestamp("2024-09-01T12:00:00Z"))


def route(routes: dict):
    """routes: {url_substring: json_body_or_callable(params)->json_body}."""
    def handler(url, params, headers):
        assert "User-Agent" in headers and headers["User-Agent"]
        for substr, body in routes.items():
            if substr in url:
                resolved = body(params) if callable(body) else body
                if resolved is None:
                    return FakeResponse(status_code=404, headers={"Content-Type": "application/xml"},
                                         text_body="<Error><Code>NoSuchKey</Code></Error>")
                return FakeResponse(json_body=resolved)
        raise AssertionError(f"unhandled URL in test: {url}")
    return handler


# --------------------------------------------------------------------------------------------
# Construction never requires the UA env var; it is required only at fetch time
# --------------------------------------------------------------------------------------------
def test_construction_never_requires_user_agent(fake_clock):
    config = FakeConfig(BASE_CONFIG)
    SecEdgarProvider(config, http=HttpClient(session=FakeSession(lambda *a: FakeResponse()), clock=fake_clock.time,
                                             sleep=fake_clock.sleep))


def test_fetch_without_user_agent_raises_not_configured(fake_clock, monkeypatch):
    provider = make_provider(route({"company_tickers_exchange.json": TICKERS_JSON}), fake_clock, monkeypatch,
                              set_ua=False)
    with pytest.raises(ProviderNotConfigured):
        provider.get_fundamentals(["AAPL"])


def test_pad_cik():
    assert pad_cik(320193) == "0000320193"
    assert pad_cik("320193") == "0000320193"


# --------------------------------------------------------------------------------------------
# Submissions: recent + files pages concatenated
# --------------------------------------------------------------------------------------------
AAPL_CIK = "0000320193"

RECENT = {
    "accessionNumber": ["0000320193-24-000010", "0000320193-24-000011"],
    "filingDate": ["2024-08-02", "2024-01-04"],
    "reportDate": ["2024-06-29", ""],
    "acceptanceDateTime": ["2024-08-01T18:03:34.000Z", "2024-01-03T21:30:33.000Z"],
    "form": ["10-Q", "8-K"],
    "items": ["", "2.02,9.01"],
}
OLD_PAGE = {
    "accessionNumber": ["0000320193-15-000001"],
    "filingDate": ["2015-07-24"],
    "reportDate": [""],
    "acceptanceDateTime": ["2015-07-24T16:30:00.000Z"],
    "form": ["10-K"],
    "items": [""],
}


def submissions_root(files_pages=True):
    return {
        "cik": "320193", "sic": "3571", "sicDescription": "Electronic Computers",
        "exchanges": ["Nasdaq"],
        "filings": {
            "recent": RECENT,
            "files": [{"name": "CIK0000320193-submissions-001.json", "filingCount": 1,
                      "filingFrom": "2015-07-24", "filingTo": "2015-07-24"}] if files_pages else [],
        },
    }


def test_submissions_merge_recent_and_files_pages(fake_clock, monkeypatch):
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"CIK{AAPL_CIK}-submissions-001.json": OLD_PAGE,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(),
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    cik_map, unmapped = provider._resolve_ciks(["AAPL"])
    assert cik_map == {"AAPL": AAPL_CIK}
    subs = provider._submissions(AAPL_CIK)
    accns = {row["accessionNumber"] for row in subs["filings"]}
    assert accns == {"0000320193-24-000010", "0000320193-24-000011", "0000320193-15-000001"}


def test_unmapped_symbol_logged_not_crashed(fake_clock, monkeypatch):
    handler = route({"company_tickers_exchange.json": TICKERS_JSON,
                     f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False)})
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_earnings_events(["AAPL", "NOSUCHTICKER"], date(2024, 1, 1), date(2024, 1, 10))
    assert "NOSUCHTICKER" in out.attrs["unmapped_symbols"]


# --------------------------------------------------------------------------------------------
# 8-K item 2.02 earnings events: after-close acceptance -> next-session reaction
# --------------------------------------------------------------------------------------------
def test_8k_202_after_close_reacts_next_session(fake_clock, monkeypatch):
    handler = route({"company_tickers_exchange.json": TICKERS_JSON,
                     f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False)})
    calendar = TradingCalendar.business_days("2024-01-02", "2024-01-10")
    provider = make_provider(handler, fake_clock, monkeypatch, calendar=calendar)
    out = provider.get_earnings_events(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_type"] == "earnings_release"
    assert row["source_id"] == "0000320193-24-000011"
    assert row["pit_status"] == "PIT"
    # accepted 2024-01-03T21:30:33Z = 16:30:33 ET, i.e. after the 16:00 ET cutoff -> reacts 01-04
    assert row["reaction_date"] == pd.Timestamp("2024-01-04")
    assert row["available_at"] == row["event_time"]


def test_8k_202_premarket_reacts_same_session(fake_clock, monkeypatch):
    recent = dict(RECENT)
    recent["acceptanceDateTime"] = ["2024-08-01T18:03:34.000Z", "2024-01-03T12:00:00.000Z"]  # 07:00 ET pre-market
    root = submissions_root(files_pages=False)
    root["filings"]["recent"] = recent
    handler = route({"company_tickers_exchange.json": TICKERS_JSON, f"/submissions/CIK{AAPL_CIK}.json": root})
    calendar = TradingCalendar.business_days("2024-01-02", "2024-01-10")
    provider = make_provider(handler, fake_clock, monkeypatch, calendar=calendar)
    out = provider.get_earnings_events(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
    assert out.iloc[0]["reaction_date"] == pd.Timestamp("2024-01-03")


def test_8k_202_reaction_date_nat_without_calendar(fake_clock, monkeypatch):
    handler = route({"company_tickers_exchange.json": TICKERS_JSON,
                     f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False)})
    provider = make_provider(handler, fake_clock, monkeypatch, calendar=None)
    out = provider.get_earnings_events(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
    assert pd.isna(out.iloc[0]["reaction_date"])


def test_8k_item_matching_is_exact_token_not_substring(fake_clock, monkeypatch):
    recent = dict(RECENT)
    recent["items"] = ["", "12.02,9.01"]  # must NOT match "2.02" as a substring
    root = submissions_root(files_pages=False)
    root["filings"]["recent"] = recent
    handler = route({"company_tickers_exchange.json": TICKERS_JSON, f"/submissions/CIK{AAPL_CIK}.json": root})
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_earnings_events(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
    assert len(out) == 0


# --------------------------------------------------------------------------------------------
# Duration classification (pure function)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("start,end,expected", [
    (None, "2024-06-30", "I"),
    ("2024-04-01", "2024-06-30", "Q"),          # 91 days
    ("2024-01-01", "2024-06-30", "YTD6"),       # 182 days
    ("2024-01-01", "2024-09-30", "YTD9"),       # 274 days
    ("2023-10-01", "2024-09-29", "FY"),         # ~365 days (52/53-week tolerance)
    ("2024-01-01", "2024-01-15", "OTHER"),      # 15 days: none of the buckets
])
def test_classify_duration(start, end, expected):
    assert classify_duration(start, end) == expected


def test_conservative_available_at_next_session_with_calendar():
    calendar = TradingCalendar.business_days("2024-01-02", "2024-01-10")
    ts = conservative_available_at("2024-01-03", calendar)  # Wed -> next session Thu 2024-01-04 16:00 ET
    assert ts == calendar.cutoff("2024-01-04")


def test_conservative_available_at_next_business_day_without_calendar():
    ts = conservative_available_at("2024-01-05", None)  # Fri -> next business day is Monday 2024-01-08
    assert ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M") == "2024-01-08 16:00"


# --------------------------------------------------------------------------------------------
# companyfacts: restatement rows, PIT via accn join vs PIT_CONSERVATIVE fallback, Q4 derivation
# --------------------------------------------------------------------------------------------
def _companyfacts(units_by_tag: dict[str, dict]) -> dict:
    facts = {"us-gaap": {}}
    for tag, units in units_by_tag.items():
        facts["us-gaap"][tag] = {"label": tag, "units": units}
    return {"cik": 320193, "entityName": "Apple Inc.", "facts": facts}


def test_companyfacts_restatement_rows_both_kept(fake_clock, monkeypatch):
    # Same (start, end) reported twice under NetIncomeLoss: original + restated. Both are real,
    # distinct accessions and must both survive (point-in-time resolution happens downstream).
    facts = _companyfacts({"NetIncomeLoss": {"USD": [
        {"start": "2016-09-25", "end": "2017-09-30", "val": 48351000000, "accn": "0000320193-17-000009",
         "fy": 2017, "fp": "FY", "form": "10-K", "filed": "2017-11-03"},
        {"start": "2016-09-25", "end": "2017-09-30", "val": 48350000000, "accn": "0000320193-18-000070",
         "fy": 2018, "fp": "FY", "form": "10-K/A", "filed": "2018-11-05"},
    ]}})
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False),
        f"companyfacts/CIK{AAPL_CIK}.json": facts,
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_fundamentals(["AAPL"], concepts=["NetIncomeLoss"])
    ni = out[out["concept"] == "NetIncomeLoss"]
    assert len(ni) == 2
    assert set(ni["value"]) == {48351000000.0, 48350000000.0}
    assert set(ni["accession"]) == {"0000320193-17-000009", "0000320193-18-000070"}
    # neither accession is in our submissions fixture -> both fall back to filed-date PIT_CONSERVATIVE
    assert (ni["pit_status"] == "PIT_CONSERVATIVE").all()


def test_companyfacts_accn_join_gives_pit_else_conservative_fallback(fake_clock, monkeypatch):
    # accn 0000320193-24-000010 IS in our submissions fixture (acceptanceDateTime known) -> PIT.
    # accn 0000999999-24-000099 is NOT -> falls back to its own 'filed' date -> PIT_CONSERVATIVE.
    facts = _companyfacts({"Assets": {"USD": [
        {"end": "2024-06-29", "val": 1000.0, "accn": "0000320193-24-000010", "fy": 2024, "fp": "Q3",
         "form": "10-Q", "filed": "2024-08-02"},
        {"end": "2023-09-30", "val": 900.0, "accn": "0000999999-24-000099", "fy": 2023, "fp": "FY",
         "form": "10-K", "filed": "2023-11-03"},
    ]}})
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False),
        f"companyfacts/CIK{AAPL_CIK}.json": facts,
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_fundamentals(["AAPL"], concepts=["Assets"])
    by_accn = out.set_index("accession")
    assert by_accn.loc["0000320193-24-000010", "pit_status"] == "PIT"
    assert by_accn.loc["0000320193-24-000010", "available_at"] == pd.Timestamp("2024-08-01T18:03:34Z")
    assert by_accn.loc["0000999999-24-000099", "pit_status"] == "PIT_CONSERVATIVE"
    # 2023-11-03 is a Friday -> next business day Monday 2023-11-06, 16:00 ET (no calendar injected)
    conservative = by_accn.loc["0000999999-24-000099", "available_at"]
    assert conservative.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M") == "2023-11-06 16:00"
    assert by_accn.loc["0000999999-24-000099", "fiscal_period"] == "I"


def test_companyfacts_fallback_chain_per_period(fake_clock, monkeypatch):
    # SalesRevenueNet used for FY2016; Revenues used for FY2017. Per-period selection must not
    # blend the two concepts into a look-alike single series for the same period.
    facts = _companyfacts({
        "Revenues": {"USD": [
            {"start": "2016-09-25", "end": "2017-09-30", "val": 229234000000.0, "accn": "0000320193-17-000009",
             "form": "10-K", "filed": "2017-11-03"},
        ]},
        "SalesRevenueNet": {"USD": [
            {"start": "2015-09-27", "end": "2016-09-24", "val": 215639000000.0, "accn": "0000320193-16-000008",
             "form": "10-K", "filed": "2016-10-26"},
        ]},
    })
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False),
        f"companyfacts/CIK{AAPL_CIK}.json": facts,
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_fundamentals(["AAPL"], concepts=["Revenues"])
    assert len(out) == 2
    assert (out["concept"] == "Revenues").all()  # canonical name, regardless of underlying XBRL tag
    assert set(out["value"]) == {229234000000.0, 215639000000.0}


def test_q4_derived_as_fy_minus_ytd9(fake_clock, monkeypatch):
    facts = _companyfacts({"NetIncomeLoss": {"USD": [
        {"start": "2023-01-01", "end": "2023-12-31", "val": 400.0, "accn": "0000320193-24-000001",
         "form": "10-K", "filed": "2024-02-01"},
        {"start": "2023-01-01", "end": "2023-09-30", "val": 300.0, "accn": "0000320193-23-000050",
         "form": "10-Q", "filed": "2023-11-01"},
    ]}})
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False),
        f"companyfacts/CIK{AAPL_CIK}.json": facts,
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_fundamentals(["AAPL"], concepts=["NetIncomeLoss"])
    q4 = out[out["fiscal_period"] == "Q4"]
    assert len(q4) == 1
    row = q4.iloc[0]
    assert row["value"] == pytest.approx(100.0)
    assert row["period_start"] == pd.Timestamp("2023-10-01")
    assert row["period_end"] == pd.Timestamp("2023-12-31")
    # weakest (most conservative) of the two contributing PIT statuses, since neither accn is
    # in our submissions fixture here both are PIT_CONSERVATIVE
    assert row["pit_status"] == "PIT_CONSERVATIVE"


def test_dei_shares_outstanding_uses_dei_taxonomy(fake_clock, monkeypatch):
    facts_json = {
        "cik": 320193, "entityName": "Apple Inc.",
        "facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2024-07-17", "val": 15000000000, "accn": "0000320193-24-000010", "form": "10-Q",
             "filed": "2024-08-02"},
        ]}}}},
    }
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False),
        f"companyfacts/CIK{AAPL_CIK}.json": facts_json,
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_fundamentals(["AAPL"], concepts=["SharesOutstanding"])
    assert len(out) == 1
    assert out.iloc[0]["unit"] == "shares"
    assert out.iloc[0]["pit_status"] == "PIT"  # accn matches submissions fixture


def test_no_companyfacts_for_symbol_skips_without_crash(fake_clock, monkeypatch):
    handler = route({
        "company_tickers_exchange.json": TICKERS_JSON,
        f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False),
        f"companyfacts/CIK{AAPL_CIK}.json": None,  # 404
    })
    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_fundamentals(["AAPL"])
    assert len(out) == 0


# --------------------------------------------------------------------------------------------
# get_company_meta
# --------------------------------------------------------------------------------------------
def test_get_company_meta(fake_clock, monkeypatch):
    handler = route({"company_tickers_exchange.json": TICKERS_JSON,
                     f"/submissions/CIK{AAPL_CIK}.json": submissions_root(files_pages=False)})
    provider = make_provider(handler, fake_clock, monkeypatch)
    meta = provider.get_company_meta(["AAPL"])
    assert meta.iloc[0]["sic"] == "3571"
    assert meta.iloc[0]["pit_status"] == "ASSUMED_STATIC"
    assert meta.iloc[0]["exchanges"] == "Nasdaq"
