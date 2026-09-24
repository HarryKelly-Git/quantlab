"""Feature catalog: point-in-time safety (truncation invariance) for EVERY registered feature, plus
hand-computed correctness checks per group."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import DataBundle, build_panel
from quantlab.features.base import FeatureSet, load_all_features
from quantlab.testing.pit import assert_truncation_invariant
from quantlab.universe import UniverseEngine

from ..conftest import ROOT

REGISTRY = load_all_features()
# Catalog groups not implemented yet (tracked in README "Status"); everything else must exist.
PENDING_GROUPS: set[str] = set()
PENDING_NAMES: set[str] = set()


def _catalog_names() -> dict[str, str]:
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    section = text.split("## 4.")[1].split("\n## ")[0]
    names: dict[str, str] = {}
    for line in section.splitlines():
        m = re.match(r"^\|\s*([a-z0-9_ ,]+?)\s*\|\s*([a-z]+)\s*\|", line)
        if m and m.group(1) != "name":
            for n in m.group(1).split(","):
                names[n.strip()] = m.group(2)
    return names


def test_catalog_names_registered():
    catalog = _catalog_names()
    assert len(catalog) > 40
    missing = [n for n, g in catalog.items() if g not in PENDING_GROUPS and n not in PENDING_NAMES and n not in REGISTRY]
    assert not missing, f"catalog features not registered: {missing}"
    extra = [n for n in REGISTRY.names() if n not in catalog]
    assert not extra, f"registered features missing from ARCHITECTURE.md catalog: {extra}"


@pytest.fixture(scope="module")
def universe_fn(request):
    from quantlab.config import load_config
    cfg = load_config(root=ROOT)
    eng = UniverseEngine(cfg)
    return lambda b: eng.membership(b)


@pytest.mark.parametrize("name", REGISTRY.names())
def test_feature_is_point_in_time(name, bundle, universe_fn):
    def compute(b):
        return FeatureSet(b, universe=universe_fn(b)).get(name)
    assert_truncation_invariant(compute, bundle, n_dates=4, min_history=300, name=name)


@pytest.mark.parametrize("name", REGISTRY.names())
def test_feature_has_values_and_no_infinities(name, bundle, universe_fn):
    out = FeatureSet(bundle, universe=universe_fn(bundle)).get(name)
    vals = out.to_numpy(dtype="float64")
    assert np.isfinite(vals).any(), f"{name} produced no values on synthetic data"
    assert not np.isinf(vals).any()


# --- hand-computed checks on a tiny panel ------------------------------------------------------
def _tiny_bundle(closes: dict[str, list[float]], volumes: float = 1e6) -> DataBundle:
    dates = pd.bdate_range("2024-01-01", periods=len(next(iter(closes.values()))))
    rows = []
    for sym, cl in closes.items():
        for d, c in zip(dates, cl):
            rows.append({"symbol": sym, "date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": volumes, "vwap": c, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    cal = TradingCalendar.from_dates(dates)
    return DataBundle(build_panel(pd.DataFrame(rows), calendar=cal), cal, benchmarks={"market": "SPY", "sectors": {}})


def test_returns_and_ma_distance_hand_computed():
    closes = [100 + i for i in range(30)]
    b = _tiny_bundle({"AAA": closes, "SPY": [200.0] * 30})
    fs = FeatureSet(b)
    d = b.panel.dates[25]
    assert fs.get("ret_5d").at[d, "AAA"] == pytest.approx(closes[25] / closes[20] - 1)
    assert fs.get("dist_ma20").at[d, "AAA"] == pytest.approx(closes[25] / np.mean(closes[6:26]) - 1)
    assert np.isnan(fs.get("dist_ma20").at[b.panel.dates[18], "AAA"])   # not a full window yet
    assert fs.get("rs_spy_20").at[d, "AAA"] == pytest.approx(closes[25] / closes[5] - 1)


def test_relative_volume_is_split_invariant():
    closes = [100.0] * 25 + [50.0] * 5       # raw price halves (2:1 split) ...
    b = _tiny_bundle({"AAA": closes, "SPY": [200.0] * 30})
    # ... and share volume doubles: dollar volume unchanged -> rel volume stays 1
    p = b.panel
    p.fields["volume"].loc[p.dates[25]:, "AAA"] = 2e6
    p.fields["dollar_volume"] = p.close * p.volume
    fs = FeatureSet(b)
    assert fs.get("rel_volume_1d").at[p.dates[27], "AAA"] == pytest.approx(1.0)


def test_xs_rank_respects_universe():
    b = _tiny_bundle({"AAA": [100 + i for i in range(80)], "BBB": [100 - 0.1 * i for i in range(80)],
                      "CCC": [100.0 + 2 * i for i in range(80)], "SPY": [200.0] * 80})
    mask = pd.DataFrame(True, index=b.panel.dates, columns=b.panel.symbols)
    mask["CCC"] = False
    mask["SPY"] = False
    r = FeatureSet(b, universe=mask).get("xs_rank_ret_63")
    d = b.panel.dates[70]
    assert np.isnan(r.at[d, "CCC"])
    assert r.at[d, "AAA"] == pytest.approx(1.0) and r.at[d, "BBB"] == pytest.approx(0.5)


def test_market_features_single_column(bundle):
    fs = FeatureSet(bundle)
    s = fs.market("market_trend_200")
    assert s.notna().sum() > 100
    br = fs.market("breadth_50").dropna()
    assert ((br >= 0) & (br <= 1)).all()


def test_ear_known_only_after_reaction_plus_one(bundle):
    """ear_3d must be NaN on the reaction session and appear at reaction+1."""
    fs = FeatureSet(bundle)
    ear = fs.get("ear_3d")
    ev = bundle.events.dropna(subset=["reaction_date"])
    checked = 0
    for sym, r in zip(ev["symbol"], ev["reaction_date"]):
        if sym not in ear.columns or r not in ear.index:
            continue
        i = ear.index.get_loc(r)
        if i + 1 >= len(ear.index) or i < 70:
            continue
        assert np.isnan(ear.iat[i, ear.columns.get_loc(sym)])
        if bundle.panel.close[sym].iloc[i - 1:i + 2].notna().all():
            assert np.isfinite(ear.iat[i + 1, ear.columns.get_loc(sym)])
            checked += 1
        if checked > 20:
            break
    assert checked > 5


def test_days_since_earnings_counts_from_reaction(bundle):
    fs = FeatureSet(bundle)
    ds = fs.get("days_since_earnings")
    r = bundle.events.dropna(subset=["reaction_date"]).iloc[10]
    i = ds.index.get_loc(r["reaction_date"])
    assert ds.iat[i, ds.columns.get_loc(r["symbol"])] == 0
    assert ds.iat[i + 1, ds.columns.get_loc(r["symbol"])] == 1


def test_restated_fundamental_visible_only_after_its_filing(bundle):
    """As-of rule: a restatement changes the value only from the session it became usable."""
    f = bundle.fundamentals
    rs = f[f["accession"].str.contains("restate") & (f["concept"] == "Revenues")]
    assert len(rs), "synthetic data should contain restated revenue"
    r = rs.iloc[0]
    sym = r["symbol"]
    usable = bundle.calendar.first_usable_session(r["available_at"])
    i = bundle.panel.dates.get_loc(usable)
    before = FeatureSet(bundle.truncate(bundle.panel.dates[i - 1])).get("rev_growth_yoy")[sym].iloc[-1]
    full = FeatureSet(bundle).get("rev_growth_yoy")[sym]
    assert full.iloc[i - 1] == before or (np.isnan(full.iloc[i - 1]) and np.isnan(before))


def test_fundamental_features_have_coverage(bundle):
    fs = FeatureSet(bundle)
    for n in ("rev_growth_yoy", "ni_margin", "roe", "leverage", "ep_ttm", "sue"):
        assert fs.get(n).notna().to_numpy().sum() > 1000, n
