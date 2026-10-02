"""Options data client against a scripted fake HTTP layer (never the network): pagination,
parsing (greeks/IV present or absent, null OI), credentials, 403 OPRA, bars, paper-host guard."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantlab.data.providers.base import ProviderNotConfigured
from quantlab.data.providers.http import ProviderForbidden, ProviderResponseError
from quantlab.execution.broker import LiveTradingForbidden
from quantlab.options.data import OptionsDataClient
from quantlab.options.settings import OptionsConfigError, OptionsSettings


class FakeHttp:
    def __init__(self, pages):
        self.pages = pages          # path-suffix -> list of page dicts (served in order of page_token)
        self.calls = []

    def get_json(self, url, params=None, headers=None, not_found_ok=False):
        assert headers and "APCA-API-KEY-ID" in headers
        self.calls.append((url, dict(params or {})))
        for suffix, pages in self.pages.items():
            if url.endswith(suffix):
                if isinstance(pages, Exception):
                    raise pages
                idx = int((params or {}).get("page_token", "0") or 0)
                return pages[idx]
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "PKTEST000")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "secret-test-value")


def _contract(strike, oi=None, sym=None):
    k = f"{int(strike * 1000):08d}"
    return {"symbol": sym or f"TGT261016C{k}", "expiration_date": "2026-10-16", "strike_price": str(strike),
            "type": "call", "style": "american", "multiplier": "100", "size": "100", "open_interest": oi,
            "open_interest_date": None, "close_price": None, "tradable": True, "status": "active",
            "underlying_symbol": "TGT", "root_symbol": "TGT"}


def test_contracts_paginate_and_parse(config, creds):
    trading = FakeHttp({"/v2/options/contracts": [
        {"option_contracts": [_contract(150, oi="1234")], "next_page_token": "1"},
        {"option_contracts": [_contract(155)], "next_page_token": None}]})
    cl = OptionsDataClient(config, http=FakeHttp({}), trading_http=trading)
    out = cl.contracts("tgt", expiration_gte=date(2026, 10, 1))
    assert [c.strike for c in out] == [150.0, 155.0]
    assert out[0].open_interest == 1234.0 and out[1].open_interest is None and out[0].multiplier == 100.0
    assert trading.calls[0][0].startswith("https://paper-api.alpaca.markets/v2/options/contracts")
    assert trading.calls[0][1]["underlying_symbols"] == "TGT" and trading.calls[1][1]["page_token"] == "1"


def test_contract_disagreeing_with_its_occ_symbol_is_refused(config, creds):
    bad = _contract(150, sym="TGT261016C00160000")
    cl = OptionsDataClient(config, http=FakeHttp({}),
                           trading_http=FakeHttp({"/v2/options/contracts": [{"option_contracts": [bad],
                                                                             "next_page_token": None}]}))
    with pytest.raises(ProviderResponseError):
        cl.contracts("TGT")


def test_snapshots_indicative_with_and_without_greeks(config, creds):
    snaps = {"TGT261016C00150000": {"latestQuote": {"bp": 2.9, "ap": 3.0, "bs": 5, "as": 7,
                                                    "t": "2026-10-01T14:59:00Z"},
                                    "greeks": {"delta": 0.52, "gamma": 0.03}, "impliedVolatility": 0.31,
                                    "dailyBar": {"v": 120}},
             "TGT261016C00200000": {"latestQuote": {"bp": 0, "ap": 0.05, "t": "2026-10-01T14:58:00Z"}}}
    data = FakeHttp({"/v1beta1/options/snapshots/TGT": [{"snapshots": snaps, "next_page_token": None}]})
    cl = OptionsDataClient(config, http=data, trading_http=FakeHttp({}))
    out = cl.snapshots("TGT", expiration_date=date(2026, 10, 16))
    assert data.calls[0][1]["feed"] == "indicative"
    a, b = out["TGT261016C00150000"], out["TGT261016C00200000"]
    assert (a.bid, a.ask, a.vendor_iv, a.vendor_delta, a.feed) == (2.9, 3.0, 0.31, 0.52, "indicative")
    assert a.mid == pytest.approx(2.95) and a.quote_time == pd.Timestamp("2026-10-01T14:59:00Z")
    assert b.vendor_iv is None and b.vendor_greeks is None and not b.two_sided and b.mid is None


def test_opra_403_propagates_and_opra_feed_is_refused_by_config(config, creds):
    err = ProviderForbidden("alpaca: HTTP 403: OPRA agreement is not signed", 403)
    cl = OptionsDataClient(config, http=FakeHttp({"/v1beta1/options/snapshots/TGT": err}), trading_http=FakeHttp({}))
    with pytest.raises(ProviderForbidden):
        cl.snapshots("TGT")
    with pytest.raises(OptionsConfigError):
        OptionsSettings.from_config(config.with_overrides({"options": {"feed": "opra"}}))


def test_missing_credentials(config, monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    cl = OptionsDataClient(config, http=FakeHttp({}), trading_http=FakeHttp({}))
    with pytest.raises(ProviderNotConfigured):
        cl.contracts("TGT")


def test_non_paper_trading_host_is_refused(config):
    with pytest.raises(LiveTradingForbidden):
        OptionsDataClient(config.with_overrides({"providers": {"alpaca": {"trading_base_url": "https://example.com"}}}),
                          http=FakeHttp({}), trading_http=FakeHttp({}))


def test_option_bars_map_to_ny_sessions(config, creds):
    bars = {"TGT250117C00150000": [{"t": "2024-12-02T05:00:00Z", "o": 0.68, "h": 0.87, "l": 0.45, "c": 0.62,
                                    "v": 1509, "n": 363, "vw": 0.54},
                                   {"t": "2024-12-04T05:00:00Z", "o": 0.59, "h": 0.59, "l": 0.43, "c": 0.49,
                                    "v": 1042, "n": 129, "vw": 0.47}]}
    cl = OptionsDataClient(config, http=FakeHttp({"/v1beta1/options/bars": [{"bars": bars, "next_page_token": None}]}),
                           trading_http=FakeHttp({}))
    df = cl.option_bars(["TGT250117C00150000", "NOT-AN-OPTION"], date(2024, 12, 1), date(2024, 12, 5))
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-12-02", "2024-12-04"]   # no 12-03 invented
    assert df["close"].tolist() == [0.62, 0.49]
