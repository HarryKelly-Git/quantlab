"""AlpacaDataProvider.get_daily_bars: pagination, SIP->IEX fallback, DST session mapping, auth errors."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantlab.core.types import PitStatus
from quantlab.data.providers.alpaca_data import AlpacaDataProvider, bar_time_to_session
from quantlab.data.providers.base import ProviderError, ProviderNotConfigured
from quantlab.data.providers.http import HttpClient, ProviderAuthError
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession

BASE_CONFIG = {
    "providers": {
        "alpaca": {
            "key_id_env": "TEST_ALPACA_KEY", "secret_env": "TEST_ALPACA_SECRET",
            "data_base_url": "https://data.alpaca.markets", "feed": "iex", "feed_fallback": None,
            "asof": "-",
        },
    },
}


def make_provider(handler, fake_clock, overrides=None, set_creds=True, monkeypatch=None):
    if set_creds:
        assert monkeypatch is not None
        monkeypatch.setenv("TEST_ALPACA_KEY", "AKITEST")
        monkeypatch.setenv("TEST_ALPACA_SECRET", "SEKRIT")
    data = dict(BASE_CONFIG)
    if overrides:
        data = {**BASE_CONFIG, "providers": {**BASE_CONFIG["providers"], "alpaca": {**BASE_CONFIG["providers"]["alpaca"], **overrides}}}
    config = FakeConfig(data)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    return AlpacaDataProvider(config, http=http, clock=lambda: pd.Timestamp("2024-01-10T12:00:00Z"))


def test_construction_never_requires_credentials(fake_clock):
    """Registry contract: building the provider must not touch credentials."""
    config = FakeConfig(BASE_CONFIG)
    AlpacaDataProvider(config, http=HttpClient(session=FakeSession(lambda *a: FakeResponse()), clock=fake_clock.time,
                                               sleep=fake_clock.sleep))


def test_fetch_without_credentials_raises_not_configured(fake_clock, monkeypatch):
    provider = make_provider(lambda *a: FakeResponse(), fake_clock, set_creds=False, monkeypatch=monkeypatch)
    with pytest.raises(ProviderNotConfigured):
        provider.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 2))


def test_pagination_follows_token_past_a_short_page(fake_clock, monkeypatch):
    page1 = {"bars": {"AAPL": [{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000,
                                "n": 10, "vw": 1.2}]},
             "next_page_token": "tok1"}  # short page (1 bar) despite a large limit -- NOT the end
    page2 = {"bars": {"MSFT": [{"t": "2024-01-02T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 500,
                                "n": 5, "vw": 10.1}]},
             "next_page_token": None}
    calls = []

    def handler(url, params, headers):
        calls.append(params)
        assert "/v2/stocks/bars" in url
        if len(calls) == 1:
            assert "page_token" not in params
            return FakeResponse(json_body=page1)
        assert params.get("page_token") == "tok1"
        return FakeResponse(json_body=page2)

    provider = make_provider(handler, fake_clock, monkeypatch=monkeypatch)
    out = provider.get_daily_bars(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 2))
    assert len(calls) == 2
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]
    assert (out["date"] == pd.Timestamp("2024-01-02")).all()
    assert set(out["provider"]) == {"alpaca:iex"}
    assert out.attrs["feed"] == "iex"


def test_sip_403_falls_back_to_iex_and_records_feed(fake_clock, monkeypatch):
    def handler(url, params, headers):
        if params.get("feed") == "sip":
            return FakeResponse(status_code=403, json_body={
                "code": 42210000, "message": "subscription does not permit querying recent SIP data"})
        assert params.get("feed") == "iex"
        return FakeResponse(json_body={
            "bars": {"AAPL": [{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000,
                               "n": 10, "vw": 1.2}]},
            "next_page_token": None,
        })

    provider = make_provider(handler, fake_clock, overrides={"feed": "sip", "feed_fallback": "iex"},
                              monkeypatch=monkeypatch)
    out = provider.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 2))
    assert provider.last_feed_used == "iex"
    assert list(out["provider"]) == ["alpaca:iex"]
    assert out.attrs["feed"] == "iex"
    assert out.attrs["feeds_used"] == ["iex"]


def test_sip_403_without_fallback_configured_raises(fake_clock, monkeypatch):
    def handler(url, params, headers):
        return FakeResponse(status_code=403, json_body={"code": 42210000, "message": "no SIP subscription"})

    provider = make_provider(handler, fake_clock, overrides={"feed": "sip", "feed_fallback": None},
                              monkeypatch=monkeypatch)
    with pytest.raises(ProviderError):
        provider.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 2))


def test_html_401_body_does_not_crash_json_parsing(fake_clock, monkeypatch):
    def handler(url, params, headers):
        return FakeResponse(status_code=401, headers={"Content-Type": "text/html"},
                             text_body="<html><title>Unauthorized</title></html>")

    provider = make_provider(handler, fake_clock, monkeypatch=monkeypatch)
    with pytest.raises(ProviderAuthError):
        provider.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 2))


# --------------------------------------------------------------------------------------------
# DST-safe daily-bar-t -> NY session-date conversion (pure function, both DST regimes)
# --------------------------------------------------------------------------------------------
def test_bar_time_to_session_winter_and_summer_dst():
    t = pd.Series(["2024-01-03T05:00:00Z", "2024-06-03T04:00:00Z"])  # EST (winter) / EDT (summer) midnight NY
    sessions = bar_time_to_session(t)
    assert list(sessions) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-06-03")]
    assert sessions.dt.tz is None


def test_bar_time_to_session_rejects_non_midnight_timestamp():
    t = pd.Series(["2024-01-03T13:00:00Z"])  # not NY midnight -> ambiguous, must refuse to guess
    with pytest.raises(ProviderError):
        bar_time_to_session(t)


def test_conflicting_duplicate_bars_raise_not_silently_pick(fake_clock, monkeypatch):
    def handler(url, params, headers):
        return FakeResponse(json_body={
            "bars": {"AAPL": [
                {"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000, "n": 10, "vw": 1.2},
                {"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 9.9, "v": 1000, "n": 10, "vw": 1.2},
            ]},
            "next_page_token": None,
        })

    provider = make_provider(handler, fake_clock, monkeypatch=monkeypatch)
    with pytest.raises(ProviderError):
        provider.get_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 1, 2))
