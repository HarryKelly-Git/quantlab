"""Momentum-breakout Stage 1 scanner: pre-registered values, hand-computed rules, point-in-time safety."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quantlab.config import load_config
from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import DataBundle, build_panel
from quantlab.momentum_breakout.scanner import (
    FEATURE_ORDER, ScannerParams, days_since_high, fires, impulse_gain, regime, scan,
)
from quantlab.testing.fixtures import make_synthetic_bundle
from quantlab.testing.pit import assert_truncation_invariant
from quantlab.universe import UniverseEngine

from ..conftest import ROOT, SMALL_SPEC


def test_config_matches_preregistered_defaults():
    """docs/MOMENTUM-BREAKOUT-PREREG.md values live in both places; they must agree."""
    cfg = load_config(root=ROOT)
    assert ScannerParams.from_config(cfg) == ScannerParams()


def test_unknown_config_key_rejected():
    class C:
        def get(self, k, d=None):
            return {"momentum_min_pctt": 0.9}
    with pytest.raises(ValueError, match="unknown"):
        ScannerParams.from_config(C())


# --- hand-computed rules --------------------------------------------------------------------------
def test_regime_rule():
    up = pd.Series(np.arange(1.0, 31.0))
    r = regime(up, 10, 20)
    assert not r["regime_on"].iloc[:19].any()          # SMA20 not defined yet = off
    assert r["regime_on"].iloc[19:].all()              # rising: close > SMA10 > SMA20
    down = pd.Series(np.arange(30.0, 0.0, -1.0))
    assert not regime(down, 10, 20)["regime_on"].any()
    # close dips below SMA10 while SMA10 > SMA20 -> off
    s = pd.Series(np.r_[np.arange(1.0, 30.0), 20.0])
    assert not regime(s, 10, 20)["regime_on"].iloc[-1]


def test_impulse_uses_low_before_high_only():
    idx = pd.RangeIndex(5)
    hi = pd.DataFrame({"A": [10, 9, 10, 12.4, 11], "B": [20, 5, 5, 5, 5]}, index=idx, dtype=float)
    lo = pd.DataFrame({"A": [10, 8, 9, 12, 11], "B": [19, 4, 4, 4, 4]}, index=idx, dtype=float)
    g = impulse_gain(hi, lo, 5)
    assert g["A"].iloc[:4].isna().all()
    assert g["A"].iloc[4] == pytest.approx(12.4 / 8 - 1)
    # B: the 20 high came BEFORE the 4 low -> not an impulse; best leg is 5/4 (or 20/19 same bar)
    assert g["B"].iloc[4] == pytest.approx(0.25)
    lo.loc[2, "A"] = np.nan
    assert np.isnan(impulse_gain(hi, lo, 5)["A"].iloc[4])     # gap -> unknown, never guessed


def test_days_since_high_most_recent():
    hi = pd.DataFrame({"A": [1, 3, 2, 3, 1.0]})
    top, since = days_since_high(hi, 5)
    assert top["A"].iloc[4] == 3 and since["A"].iloc[4] == 1
    assert np.isnan(since["A"].iloc[3])


def _bundle(closes: dict[str, list[float]], dollar_volume: float = 5e7) -> DataBundle:
    dates = pd.bdate_range("2020-01-01", periods=len(next(iter(closes.values()))))
    rows = []
    for sym, cl in closes.items():
        for d, c in zip(dates, cl):
            rows.append({"symbol": sym, "date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": dollar_volume / c, "vwap": c, "trade_count": np.nan, "provider": "test",
                         "retrieved_at": pd.Timestamp("2025-01-01", tz="UTC")})
    cal = TradingCalendar.from_dates(dates)
    return DataBundle(build_panel(pd.DataFrame(rows), calendar=cal), cal,
                      benchmarks={"market": "SPY", "sectors": {"XLK": "Information Technology"}})


def test_scan_end_to_end_tiny():
    n = 120
    t = np.arange(n, dtype=float)
    closes = {
        "SPY": list(100 + t),                       # uptrend -> regime on
        "XLK": list(100 + 2 * t),                   # sector beats SPY
        "FAST": list(10 * 1.01 ** t),               # strongest stock
        "SLOW": list(10 + 0.0001 * t),           # weaker than every S_i with i >= 1
        **{f"S{i}": list(10 + 0.02 * i * t / n) for i in range(20)},
    }
    b = _bundle(closes)
    stocks = [s for s in b.panel.symbols if s not in ("SPY", "XLK")]
    u = pd.DataFrame(False, index=b.panel.dates, columns=b.panel.symbols)
    u[stocks] = True
    sectors = {s: None for s in b.panel.symbols} | {"FAST": "XLK", "SLOW": "XLK"}
    out = scan(b, u, ScannerParams(), sectors=sectors)
    last = out.xs(b.panel.dates[-1], level="date")
    assert set(last.index) == set(stocks)                          # only universe members
    assert last.at["FAST", "regime_on"] and last.at["FAST", "momentum"]
    assert last.at["FAST", "mom_ret"] == pytest.approx(1.01 ** 63 - 1, rel=1e-6)
    assert not last.at["SLOW", "momentum"]
    assert last.at["FAST", "sector_rs"] and last.at["FAST", "sector_etf"] == "XLK"
    assert not last.at["S1", "sector_rs"]                          # unknown sector -> rule fails
    assert last.at["FAST", "impulse"]                              # 1.01^62 - 1 ~ 85%
    assert last.at["FAST", "days_since_high"] == 0 and not last.at["FAST", "consolidation"]
    f = fires(out)
    assert f.xs(b.panel.dates[-1], level="date")["FAST"]
    assert not fires(out, ("consolidation",)).xs(b.panel.dates[-1], level="date")["FAST"]
    with pytest.raises(ValueError):
        fires(out, ("not_a_feature",))
    # liquidity floor: below $20M median dollar volume nobody is in
    thin = scan(_bundle(closes, dollar_volume=1e7), u, ScannerParams(), sectors=sectors)
    assert thin.empty


def test_scan_refuses_without_market():
    b = _bundle({"AAA": list(np.linspace(10, 20, 80))})
    u = pd.DataFrame(True, index=b.panel.dates, columns=b.panel.symbols)
    with pytest.raises(ValueError, match="UNKNOWN"):
        scan(b, u)


# --- point-in-time ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth():
    return make_synthetic_bundle(SMALL_SPEC)


def test_scan_truncation_invariant(synth, tmp_path_factory):
    cfg = load_config(root=ROOT)
    eng = UniverseEngine(cfg)
    # loosened so the small synthetic market produces candidates; the PIT property does not depend on values
    p = dataclasses.replace(ScannerParams(), min_median_dollar_volume=0.0, momentum_min_pct=0.5)

    def compute(b: DataBundle) -> pd.DataFrame:
        out = scan(b, eng.membership(b), p)
        return out.drop(columns=["sector_etf"]).astype("float64")

    full = compute(synth)
    assert len(full) > 1000 and full["momentum"].any() and full["regime_on"].any()
    for f in FEATURE_ORDER:
        assert full[f].any(), f"{f} never true on synthetic data: test would be vacuous"
    assert_truncation_invariant(compute, synth, n_dates=8, min_history=130, name="momentum_breakout.scan")
