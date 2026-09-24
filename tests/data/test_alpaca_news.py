"""AlpacaDataProvider.get_news: pagination, symbol fan-out, PIT vs PIT_CONSERVATIVE revision rule."""
from __future__ import annotations

from datetime import date

import pandas as pd

from quantlab.data.providers.alpaca_data import AlpacaDataProvider
from quantlab.data.providers.http import HttpClient
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession

BASE_CONFIG = {"providers": {"alpaca": {"key_id_env": "TEST_ALPACA_KEY", "secret_env": "TEST_ALPACA_SECRET",
                                        "data_base_url": "https://data.alpaca.markets", "feed": "iex",
                                        "feed_fallback": None, "asof": "-"}}}


def make_provider(handler, fake_clock, monkeypatch):
    monkeypatch.setenv("TEST_ALPACA_KEY", "AKITEST")
    monkeypatch.setenv("TEST_ALPACA_SECRET", "SEKRIT")
    config = FakeConfig(BASE_CONFIG)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    return AlpacaDataProvider(config, http=http, clock=lambda: pd.Timestamp("2024-01-06T12:00:00Z"))


def test_pagination_and_pit_status_by_revision_tolerance(fake_clock, monkeypatch):
    page1 = {"news": [{"id": 1, "headline": "H1", "summary": "s1", "source": "benzinga", "url": "",
                       "created_at": "2024-01-05T10:00:00Z", "updated_at": "2024-01-05T10:00:00Z",
                       "symbols": ["AAPL"]}],
             "next_page_token": "tok"}
    page2 = {"news": [{"id": 2, "headline": "H2", "summary": "s2", "source": "benzinga", "url": "",
                       "created_at": "2024-01-05T10:00:00Z", "updated_at": "2024-01-05T10:05:00Z",
                       "symbols": ["AAPL"]}],
             "next_page_token": None}
    calls = []

    def handler(url, params, headers):
        assert "/v1beta1/news" in url
        calls.append(params)
        if len(calls) == 1:
            assert "page_token" not in params
            return FakeResponse(json_body=page1)
        assert params["page_token"] == "tok"
        return FakeResponse(json_body=page2)

    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_news(["AAPL"], date(2024, 1, 1), date(2024, 1, 6))
    assert len(calls) == 2
    assert len(out) == 2
    by_id = out.set_index("news_id")
    assert by_id.loc["1", "pit_status"] == "PIT"  # updated_at == created_at
    assert by_id.loc["2", "pit_status"] == "PIT_CONSERVATIVE"  # updated_at is created_at + 5min > 60s tolerance
    assert (out["available_at"] == out["created_at"]).all()


def test_article_fans_out_to_one_row_per_requested_symbol(fake_clock, monkeypatch):
    def handler(url, params, headers):
        return FakeResponse(json_body={
            "news": [{"id": 3, "headline": "H3", "summary": "", "source": "benzinga", "url": "",
                     "created_at": "2024-01-05T10:00:00Z", "updated_at": "2024-01-05T10:00:00Z",
                     "symbols": ["AAPL", "MSFT", "TSLA"]}],
            "next_page_token": None,
        })

    provider = make_provider(handler, fake_clock, monkeypatch)
    out = provider.get_news(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 6))
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]
    assert (out["news_id"] == "3").all()


def test_missing_required_field_raises(fake_clock, monkeypatch):
    def handler(url, params, headers):
        return FakeResponse(json_body={
            "news": [{"id": 4, "headline": "H4", "created_at": "2024-01-05T10:00:00Z", "symbols": ["AAPL"]}],
            "next_page_token": None,
        })

    provider = make_provider(handler, fake_clock, monkeypatch)
    import pytest
    from quantlab.data.providers.http import ProviderResponseError
    with pytest.raises(ProviderResponseError):
        provider.get_news(["AAPL"], date(2024, 1, 1), date(2024, 1, 6))
