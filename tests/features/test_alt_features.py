"""Congress / insider features (features/alt.py): hand-computed counts, point in time from the
DISCLOSURE (never the trade date), UNKNOWN before coverage (never zero), truncation invariance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.data import schemas
from quantlab.data.panel import DataBundle, build_panel
from quantlab.data.providers.sec_edgar import conservative_available_at
from quantlab.features.alt import ALT_FEATURES
from quantlab.features.base import FeatureSet, load_all_features
from quantlab.testing.pit import assert_truncation_invariant

DATES = pd.bdate_range("2024-01-01", periods=120)
CAL = TradingCalendar.from_dates(DATES)


def trade(source, symbol, side, disclosed, traded=None, value=1000.0):
    d = pd.Timestamp(disclosed)
    return {"record_id": f"{source}-{symbol}-{side}-{disclosed}-{value}", "source": source, "symbol": symbol,
            "actor": "x", "actor_detail": "x", "side": side, "amount_low_usd": value, "amount_high_usd": value,
            "transaction_date": pd.Timestamp(traded) if traded else d - pd.Timedelta(days=10), "disclosed_date": d,
            "available_at": conservative_available_at(d, CAL), "pit_status": "PIT_CONSERVATIVE", "raw_json": "{}",
            "provider": "test", "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")}


ROWS = [
    trade("congress", "AAA", "BUY", "2024-01-02"),                       # source's first record: usable 01-03
    trade("congress", "AAA", "BUY", "2024-02-20", traded="2023-12-01"),   # traded long before; usable 02-21
    trade("congress", "AAA", "SELL", "2024-02-22"),
    trade("congress", "AAA", "OTHER", "2024-02-22"),                      # disclosed, neither buy nor sale
    trade("congress", "CCC", "SELL", "2024-03-01"),                       # Fri -> usable Mon 03-04
    trade("insider", "AAA", "BUY", "2024-01-05", value=50_000.0),         # insider source starts 01-08
    trade("insider", "AAA", "BUY", "2024-03-05", value=20_000.0),
    trade("insider", "AAA", "SELL", "2024-03-06", value=5_000.0),
    trade("insider", "AAA", "SELL", "2024-04-02", value=np.nan),          # value UNKNOWN
    trade("insider", "BBB", "OTHER", "2024-03-05"),                       # a grant: covers BBB, counts nothing
]


def make_bundle(rows=ROWS) -> DataBundle:
    bars = []
    for sym in ("AAA", "BBB", "CCC", "SPY"):
        for d in DATES:
            bars.append({"symbol": sym, "date": d, "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1e6,
                         "vwap": 10.0, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    alt = schemas.conform("alt_trades", pd.DataFrame(rows)) if rows else schemas.empty("alt_trades")
    return DataBundle(build_panel(pd.DataFrame(bars), calendar=CAL), CAL, benchmarks={"market": "SPY", "sectors": {}},
                      alt_trades=alt)


def at(df, d, s):
    return df.at[pd.Timestamp(d), s]


def test_registered_in_group_alt():
    reg = load_all_features()
    for n in ALT_FEATURES:
        spec = reg.spec(n)
        assert spec.group == "alt" and spec.pit_status.value == "PIT_CONSERVATIVE"


def test_counts_from_the_disclosure_with_a_30_day_window():
    fs = FeatureSet(make_bundle())
    buys, sells, net = fs.get("congress_buys_30d"), fs.get("congress_sells_30d"), fs.get("congress_net_30d")
    assert at(buys, "2024-02-20", "AAA") == 0                              # disclosed today: usable tomorrow
    assert at(buys, "2024-02-21", "AAA") == 1                              # traded 2023-12-01: irrelevant
    assert at(sells, "2024-02-23", "AAA") == 1 and at(net, "2024-02-23", "AAA") == 0
    assert at(buys, "2024-03-21", "AAA") == 1 and at(buys, "2024-03-22", "AAA") == 0   # (D - 30 days, D]
    assert at(buys, "2024-02-15", "AAA") == 0                              # a KNOWN zero: source and symbol covered


def test_unknown_before_coverage_never_zero():
    fs = FeatureSet(make_bundle())
    buys = fs.get("congress_buys_30d")
    # the congress source starts 01-03: no 30-day count is complete before 02-02
    assert np.isnan(at(buys, "2024-01-25", "AAA")) and np.isnan(at(buys, "2024-02-01", "AAA"))
    assert at(buys, "2024-02-02", "AAA") == 0
    assert buys["BBB"].isna().all()                                        # congress never showed BBB: UNKNOWN
    assert np.isnan(at(buys, "2024-03-01", "CCC")) and at(buys, "2024-03-04", "CCC") == 0
    assert at(fs.get("congress_sells_30d"), "2024-03-04", "CCC") == 1
    ib = fs.get("insider_buys_30d")
    assert ib["CCC"].isna().all() and at(ib, "2024-03-06", "BBB") == 0 and np.isnan(at(ib, "2024-03-05", "BBB"))


def test_insider_net_value_is_unknown_when_any_value_is_unknown():
    fs = FeatureSet(make_bundle())
    nv = fs.get("insider_net_value_30d")
    assert at(nv, "2024-03-07", "AAA") == pytest.approx(20_000.0 - 5_000.0)
    assert at(fs.get("insider_buys_30d"), "2024-03-07", "AAA") == 1 and at(fs.get("insider_sells_30d"), "2024-03-07", "AAA") == 1
    assert np.isnan(at(nv, "2024-04-03", "AAA"))                           # an unknown sale value in the window
    assert at(fs.get("insider_sells_30d"), "2024-04-03", "AAA") == 2       # the count is still known


def test_no_source_means_unknown_everywhere():
    fs = FeatureSet(make_bundle(rows=[]))
    for n in ALT_FEATURES:
        assert fs.get(n).isna().all().all(), n


@pytest.mark.parametrize("name", ALT_FEATURES)
def test_truncation_invariant_on_hand_built_disclosures(name):
    b = make_bundle()
    dates = [pd.Timestamp(d) for d in ("2024-01-03", "2024-02-02", "2024-02-20", "2024-02-21", "2024-03-04",
                                       "2024-03-07", "2024-04-03", "2024-05-01")]
    assert_truncation_invariant(lambda x: FeatureSet(x).get(name), b, check_dates=dates, name=name)


@pytest.mark.parametrize("name", ALT_FEATURES)
def test_truncation_invariant_on_the_synthetic_world(name, bundle):
    assert len(bundle.alt_trades)
    assert_truncation_invariant(lambda x: FeatureSet(x).get(name), bundle, n_dates=5, min_history=60, name=name)


def test_a_late_disclosure_does_not_leak_into_earlier_sessions():
    """Corrupt-the-future check: adding a disclosure dated after D changes nothing at or before D."""
    base = FeatureSet(make_bundle()).get("congress_buys_30d")
    later = FeatureSet(make_bundle(ROWS + [trade("congress", "AAA", "BUY", "2024-04-10", traded="2024-02-01")]))
    after = later.get("congress_buys_30d")
    d = pd.Timestamp("2024-04-10")
    pd.testing.assert_frame_equal(base.loc[:d], after.loc[:d])
    assert at(after, "2024-04-11", "AAA") == at(base, "2024-04-11", "AAA") + 1
