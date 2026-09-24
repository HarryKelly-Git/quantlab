"""AlpacaDataProvider.get_corporate_actions: split/dividend mapping, windowing, unmapped types."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantlab.data.providers.alpaca_data import AlpacaDataProvider
from quantlab.data.providers.http import HttpClient
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession

BASE_CONFIG = {"providers": {"alpaca": {"key_id_env": "TEST_ALPACA_KEY", "secret_env": "TEST_ALPACA_SECRET",
                                        "data_base_url": "https://data.alpaca.markets", "feed": "iex",
                                        "feed_fallback": None, "asof": "-"}}}

# Windows from `_windows` are contiguous and non-overlapping, so keying each record by its own
# ex_date and slicing per requested [start,end] window mirrors a real server and avoids double
# counting the same record across multiple padded windows.
ALL_RECORDS = {
    "forward_splits": [{"id": "fs1", "symbol": "AAPL", "ex_date": "2024-01-15", "old_rate": "1", "new_rate": "4"}],
    "reverse_splits": [{"id": "rs1", "symbol": "AAPL", "ex_date": "2024-01-20", "old_rate": "10", "new_rate": "1"}],
    "cash_dividends": [
        {"id": "cd1", "symbol": "AAPL", "ex_date": "2024-01-10", "rate": "0.24", "currency": "USD"},
        {"id": "cd2", "symbol": "AAPL", "ex_date": "2024-01-12"},  # malformed: no rate -> rejected
    ],
    "spin_offs": [{"id": "so1", "source_symbol": "AAPL", "new_symbol": "XYZ", "ex_date": "2024-01-25"}],
}


def _handler(url, params, headers):
    assert "/v1/corporate-actions" in url
    ws, we = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
    out = {}
    for key, records in ALL_RECORDS.items():
        keep = [r for r in records if ws <= pd.Timestamp(r["ex_date"]) <= we]
        if keep:
            out[key] = keep
    return FakeResponse(json_body={"corporate_actions": out, "next_page_token": None})


def make_provider(handler, fake_clock, monkeypatch):
    monkeypatch.setenv("TEST_ALPACA_KEY", "AKITEST")
    monkeypatch.setenv("TEST_ALPACA_SECRET", "SEKRIT")
    config = FakeConfig(BASE_CONFIG)
    http = HttpClient(session=FakeSession(handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    return AlpacaDataProvider(config, http=http, clock=lambda: pd.Timestamp("2024-02-01T12:00:00Z"))


def test_split_and_dividend_mapping(fake_clock, monkeypatch):
    provider = make_provider(_handler, fake_clock, monkeypatch)
    out = provider.get_corporate_actions(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))

    fwd = out[out["action_type"] == "split"]
    assert set(fwd["ratio"].round(4)) == {4.0, 0.1}  # forward 4-for-1, reverse 1-for-10

    div = out[out["action_type"] == "cash_dividend"].iloc[0]
    assert div["amount"] == pytest.approx(0.24)
    assert pd.isna(div["ratio"])

    # available_at = ex_date 09:30 ET in UTC; January -> EST (UTC-5) -> 14:30Z
    assert div["available_at"] == pd.Timestamp("2024-01-10T14:30:00Z")
    assert (out["pit_status"] == "PIT_CONSERVATIVE").all()
    assert (out["provider"] == "alpaca").all()


def test_unmapped_types_are_counted_and_logged_not_dropped(fake_clock, monkeypatch):
    provider = make_provider(_handler, fake_clock, monkeypatch)
    out = provider.get_corporate_actions(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
    assert out.attrs["unmapped_counts"].get("spin_offs") == 1
    assert any(r["symbol"] == "AAPL" for r in out.attrs["unmapped_records"])
    assert "so1" not in set(out["source_id"])  # never silently mapped in as if it were a split/dividend


def test_malformed_dividend_record_is_rejected_not_crashed(fake_clock, monkeypatch):
    provider = make_provider(_handler, fake_clock, monkeypatch)
    out = provider.get_corporate_actions(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
    assert any(r["id"] == "cd2" for r in out.attrs["rejected"])
    assert "cd2" not in set(out["source_id"])


def test_windows_are_at_most_ca_max_days_and_contiguous(fake_clock, monkeypatch):
    provider = make_provider(_handler, fake_clock, monkeypatch)
    windows = provider._windows(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01"))
    assert len(windows) > 1
    for s, e in windows:
        assert (e - s).days <= 89
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert s2 == e1 + pd.Timedelta(days=1)  # contiguous, no gap, no overlap
    assert windows[0][0] == pd.Timestamp("2024-01-01")
    assert windows[-1][1] == pd.Timestamp("2024-04-01")


def test_out_of_range_ex_date_filtered_out(fake_clock, monkeypatch):
    provider = make_provider(_handler, fake_clock, monkeypatch)
    out = provider.get_corporate_actions(["AAPL"], date(2024, 1, 16), date(2024, 1, 18))
    assert len(out) == 0  # no mapped action's ex_date falls in this narrow window
