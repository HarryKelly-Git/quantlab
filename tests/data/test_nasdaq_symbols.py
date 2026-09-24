"""NasdaqTraderReferenceProvider: pipe-delimited parsing + table-driven security-type classification."""
from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.providers.http import HttpClient
from quantlab.data.providers.nasdaq_symbols import (
    NasdaqTraderReferenceProvider,
    classify_security_type,
    parse_symbol_directory,
)
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession

NASDAQ_LISTED_TEXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "QQQ|Invesco QQQ Trust Series 1|Q|N|N|100|Y|N\n"
    "ZTEST|Nasdaq Test Stock|Q|Y|N|100|N|N\n"
    "File Creation Time: 0925202508:00\n"
)
OTHER_LISTED_TEXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "IBM|International Business Machines Corp Common Stock|N|IBM|N|100|N|IBM\n"
    "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
    "BRK.A|Berkshire Hathaway Inc Class A Common Stock|N|BRK.A|N|100|N|BRK.A\n"
    "IEXW|Some IEX-Listed Warrant|V|IEXW|N|100|N|IEXW\n"
    "File Creation Time: 0925202508:00\n"
)

BASE_CONFIG = {"providers": {"nasdaq_trader": {
    "nasdaq_listed_url": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other_listed_url": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}}}


# --------------------------------------------------------------------------------------------
# Table-driven security_type classification (ordered precedence)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,is_etf,expected", [
    ("Foo Corp Warrants", False, "WARRANT"),
    ("Foo Corp Rights", False, "RIGHT"),
    ("Foo Acquisition Corp Unit", False, "UNIT"),
    ("Foo Corp 6.50% Preferred Stock Series A", False, "PREFERRED"),
    ("Foo Corp Depositary Shares Each Representing 1/1000th Preferred Stock", False, "PREFERRED"),
    ("Foo Corp 4.5% Senior Notes due 2030", False, "OTHER"),
    ("Foo Momentum ETF", True, "ETF"),
    ("Apple Inc. Common Stock", False, "COMMON"),
    ("Some Foreign Co Ordinary Shares", False, "COMMON"),
    ("Some ADR American Depositary Shares", False, "COMMON"),
    ("Foo Corp Class A Common Stock", False, "COMMON"),
    # A bare "Trust" name with no common/non-common keyword must never be guessed either way.
    ("Weird Grantor Trust", False, "UNKNOWN"),
    ("Totally Ambiguous Security", False, "UNKNOWN"),
    # ETF flag loses to an explicit non-common keyword (an ETF sponsor's warrants are warrants).
    ("Foo ETF Trust Warrants", True, "WARRANT"),
])
def test_classify_security_type_table(name, is_etf, expected):
    assert classify_security_type(name, is_etf) == expected


def test_reit_trust_with_common_stock_wording_is_common():
    # The caveat is about a bare "Trust"; a REIT that also literally says "Common Stock" still
    # classifies as COMMON via the common-stock pattern, not because of the word "Trust".
    assert classify_security_type("Realty Income Trust Common Stock", False) == "COMMON"


# --------------------------------------------------------------------------------------------
# Parsing: footer skipped, columns mapped, exchange codes
# --------------------------------------------------------------------------------------------
def test_parse_nasdaq_listed_skips_footer_and_maps_fields():
    from quantlab.data.providers.nasdaq_symbols import _nasdaq_row
    rows = parse_symbol_directory(NASDAQ_LISTED_TEXT, _nasdaq_row)
    assert {r["symbol"] for r in rows} == {"AAPL", "QQQ", "ZTEST"}
    aapl = next(r for r in rows if r["symbol"] == "AAPL")
    assert aapl["exchange"] == "NASDAQ"
    assert aapl["security_type"] == "COMMON"
    assert aapl["is_etf"] is False
    qqq = next(r for r in rows if r["symbol"] == "QQQ")
    assert qqq["is_etf"] is True
    assert qqq["security_type"] == "ETF"
    ztest = next(r for r in rows if r["symbol"] == "ZTEST")
    assert ztest["is_test_issue"] is True


def test_parse_other_listed_exchange_codes():
    from quantlab.data.providers.nasdaq_symbols import _other_row
    rows = parse_symbol_directory(OTHER_LISTED_TEXT, _other_row)
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["IBM"]["exchange"] == "NYSE"
    assert by_symbol["SPY"]["exchange"] == "NYSE_ARCA"
    assert by_symbol["SPY"]["security_type"] == "ETF"
    assert by_symbol["BRK.A"]["exchange"] == "NYSE"
    assert by_symbol["BRK.A"]["security_type"] == "COMMON"
    assert by_symbol["IEXW"]["exchange"] == "IEX"
    assert by_symbol["IEXW"]["security_type"] == "WARRANT"


def test_footer_line_produces_no_row():
    rows = parse_symbol_directory(NASDAQ_LISTED_TEXT, lambda raw: raw)
    assert len(rows) == 3  # AAPL, QQQ, ZTEST -- not the footer


# --------------------------------------------------------------------------------------------
# get_securities(): end to end via a fake HTTP client
# --------------------------------------------------------------------------------------------
def test_get_securities_end_to_end(fake_clock):
    def handler(url, params, headers):
        if "nasdaqlisted" in url:
            return FakeResponse(text_body=NASDAQ_LISTED_TEXT, headers={"Content-Type": "text/plain"})
        assert "otherlisted" in url
        return FakeResponse(text_body=OTHER_LISTED_TEXT, headers={"Content-Type": "text/plain"})

    config = FakeConfig(BASE_CONFIG)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    provider = NasdaqTraderReferenceProvider(config, http=http, clock=lambda: pd.Timestamp("2024-01-06T12:00:00Z"))
    df = provider.get_securities()
    assert set(df["symbol"]) == {"AAPL", "QQQ", "ZTEST", "IBM", "SPY", "BRK.A", "IEXW"}
    assert (df["source"] == "nasdaq_trader").all()
    assert (df["pit_status"] == "ASSUMED_STATIC").all()
    assert (df["status"] == "active").all()
    assert df["cik"].isna().all()


def test_get_securities_dedupes_symbol_listed_in_both_files(fake_clock):
    nasdaq_text = NASDAQ_LISTED_TEXT  # includes AAPL
    other_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "AAPL|Apple duplicate listing artifact|N|AAPL|N|100|N|AAPL\n"
        "File Creation Time: 0925202508:00\n"
    )

    def handler(url, params, headers):
        if "nasdaqlisted" in url:
            return FakeResponse(text_body=nasdaq_text)
        return FakeResponse(text_body=other_text)

    config = FakeConfig(BASE_CONFIG)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    provider = NasdaqTraderReferenceProvider(config, http=http, clock=lambda: pd.Timestamp("2024-01-06T12:00:00Z"))
    df = provider.get_securities()
    assert (df["symbol"] == "AAPL").sum() == 1
    assert df.loc[df["symbol"] == "AAPL", "exchange"].iloc[0] == "NASDAQ"  # nasdaqlisted row kept
