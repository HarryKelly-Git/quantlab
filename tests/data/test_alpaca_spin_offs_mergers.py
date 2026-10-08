"""AlpacaDataProvider.get_corporate_actions with providers.alpaca.map_spin_offs_and_mergers: spin-offs
and mergers are mapped under their OWN action types (never as a split/dividend), other types stay
counted as unmapped, and the frame feeds build_panel."""
from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import build_panel
from quantlab.data.providers.alpaca_data import AlpacaDataProvider
from quantlab.data.providers.http import HttpClient
from tests.data.fakes import FakeConfig, FakeResponse, FakeSession
from tests.data.test_alpaca_corporate_actions import BASE_CONFIG

# record fields as documented (docs/EXTERNAL-SERVICES.md, Alpaca corporate-action date fields)
EVENT_RECORDS = {
    "spin_offs": [{"id": "so1", "source_symbol": "PARENT", "source_rate": 5, "new_symbol": "SPINCO",
                   "new_rate": 1, "ex_date": "2024-01-09"}],
    "cash_mergers": [
        {"id": "cm1", "acquirer_symbol": "BUYER", "acquiree_symbol": "TARGET", "rate": 25.5,
         "effective_date": "2024-01-17"},
        {"id": "cm2", "acquirer_symbol": "BUYER", "rate": 3.0, "effective_date": "2024-01-18"},  # no acquiree
    ],
    "stock_mergers": [{"id": "sm1", "acquirer_symbol": "BUYER", "acquirer_rate": 0.5, "acquiree_symbol": "TGT2",
                       "acquiree_rate": 1, "effective_date": "2024-01-22"}],
    "stock_and_cash_mergers": [{"id": "sc1", "acquirer_symbol": "BUYER", "acquirer_rate": 0.25,
                                "acquiree_symbol": "TGT3", "acquiree_rate": 1, "rate": 10.0,
                                "effective_date": "2024-01-23"}],
    "stock_dividends": [{"id": "sd1", "symbol": "PARENT", "ex_date": "2024-01-11"}],
}


def _handler(url, params, headers):
    assert "/v1/corporate-actions" in url
    ws, we = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
    out = {}
    for key, records in EVENT_RECORDS.items():
        keep = [r for r in records if ws <= pd.Timestamp(r.get("ex_date") or r["effective_date"]) <= we]
        if keep:
            out[key] = keep
    return FakeResponse(json_body={"corporate_actions": out, "next_page_token": None})


def _provider(fake_clock, monkeypatch, flag: bool | None):
    monkeypatch.setenv("TEST_ALPACA_KEY", "AKITEST")
    monkeypatch.setenv("TEST_ALPACA_SECRET", "SEKRIT")
    cfg = copy.deepcopy(BASE_CONFIG)
    if flag is not None:
        cfg["providers"]["alpaca"]["map_spin_offs_and_mergers"] = flag
    http = HttpClient(session=FakeSession(_handler), sleep=fake_clock.sleep, clock=fake_clock.time, max_retries=1)
    return AlpacaDataProvider(FakeConfig(cfg), http=http, clock=lambda: pd.Timestamp("2024-02-01T12:00:00Z"))


def test_spin_offs_and_mergers_mapped_under_their_own_types(fake_clock, monkeypatch):
    out = _provider(fake_clock, monkeypatch, True).get_corporate_actions(None, date(2024, 1, 1), date(2024, 1, 31))
    by_id = out.set_index("source_id")
    assert set(out["action_type"]) == {"spin_off", "cash_merger", "stock_merger", "stock_and_cash_merger"}
    so = by_id.loc["so1"]
    assert so["symbol"] == "PARENT" and so["action_type"] == "spin_off"            # the parent, not SPINCO
    assert so["ex_date"] == pd.Timestamp("2024-01-09") and so["ratio"] == pytest.approx(0.2)
    cm = by_id.loc["cm1"]
    assert cm["symbol"] == "TARGET" and cm["ex_date"] == pd.Timestamp("2024-01-17")  # acquiree, effective_date
    assert cm["amount"] == pytest.approx(25.5) and np.isnan(cm["ratio"])
    assert by_id.loc["sm1", "symbol"] == "TGT2" and by_id.loc["sm1", "ratio"] == pytest.approx(0.5)
    assert by_id.loc["sc1", "symbol"] == "TGT3" and by_id.loc["sc1", "amount"] == pytest.approx(10.0)
    # same PIT convention as splits/dividends: available from the ex/effective date's open
    assert cm["available_at"] == pd.Timestamp("2024-01-17T14:30:00Z")
    assert (out["pit_status"] == "PIT_CONSERVATIVE").all()
    # a malformed record is rejected with a reason; other types stay counted, never dropped silently
    assert any(r["id"] == "cm2" for r in out.attrs["rejected"]) and "cm2" not in set(out["source_id"])
    assert out.attrs["unmapped_counts"] == {"stock_dividends": 1}


@pytest.mark.parametrize("flag", [None, False])
def test_without_the_flag_they_stay_unmapped(flag, fake_clock, monkeypatch):
    out = _provider(fake_clock, monkeypatch, flag).get_corporate_actions(None, date(2024, 1, 1), date(2024, 1, 31))
    assert len(out) == 0
    assert out.attrs["unmapped_counts"] == {"spin_offs": 1, "cash_mergers": 2, "stock_mergers": 1,
                                            "stock_and_cash_mergers": 1, "stock_dividends": 1}


def test_provider_frame_feeds_build_panel(fake_clock, monkeypatch):
    acts = _provider(fake_clock, monkeypatch, True).get_corporate_actions(None, date(2024, 1, 1), date(2024, 1, 31))
    dates = pd.bdate_range("2024-01-02", "2024-01-31")
    rows = []
    for sym, last in (("PARENT", dates[-1]), ("TARGET", pd.Timestamp("2024-01-16")), ("SPY", dates[-1])):
        for d in dates[dates <= last]:
            c = 88.0 if (sym == "PARENT" and d >= pd.Timestamp("2024-01-09")) else 100.0     # -12% on the ex-date
            rows.append({"symbol": sym, "date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1e6,
                         "vwap": c, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    p = build_panel(pd.DataFrame(rows), acts, calendar=TradingCalendar.from_dates(dates))
    assert p.ret.at[pd.Timestamp("2024-01-09"), "PARENT"] == 0.0
    assert list(p.merger.index[p.merger["TARGET"]]) == [pd.Timestamp("2024-01-17")]
