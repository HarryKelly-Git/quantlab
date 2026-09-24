"""Universe engine: exclusions, raw-price rules, PIT safety."""
from __future__ import annotations

import pandas as pd

from quantlab.testing.pit import assert_truncation_invariant
from quantlab.universe import UniverseEngine


def test_non_common_and_benchmarks_excluded(bundle, config):
    eng = UniverseEngine(config)
    m = eng.membership(bundle)
    for sym in ["SYNPA", "SYNWS", "SYNU", "SYNRT", "SYNTT", "SYNETF", "SPY", "XLK"]:
        assert not m[sym].any(), sym
    assert m.any().sum() > 10          # plenty of synthetic common stocks qualify


def test_explain_gives_reasons(bundle, config):
    eng = UniverseEngine(config)
    d = bundle.panel.dates[-1]
    ex = eng.explain(bundle, d).set_index("symbol")
    assert ex.at["SYNPA", "reason"].startswith("security type PREFERRED")
    assert ex.at["SYNTT", "reason"] == "test issue"
    assert ex.at["SPY", "reason"] == "benchmark"
    inc = ex[ex["included"]]
    assert (inc["reason"] == "included").all()
    assert ex.loc[~ex["included"], "reason"].ne("included").all()


def test_min_history_and_quarantine(bundle, config):
    eng = UniverseEngine(config)
    m = eng.membership(bundle)
    assert not m.iloc[:259].any().any()   # nobody has 260 sessions of history yet
    sym = m.iloc[-1][m.iloc[-1]].index[0]
    assert not eng.membership(bundle, exclude={sym})[sym].any()


def test_raw_price_rule_uses_unadjusted_close(bundle, config):
    eng = UniverseEngine(config)
    m = eng.membership(bundle)
    close = bundle.panel.close
    assert not (m & (close < config.get("universe.min_price"))).any().any()


def test_universe_is_point_in_time(bundle, config):
    eng = UniverseEngine(config)
    assert_truncation_invariant(lambda b: eng.membership(b).astype(float), bundle, min_history=300, name="universe")


def test_max_symbols_cap(bundle, config):
    eng = UniverseEngine(config.with_overrides({"universe": {"max_symbols": 5}}))
    assert eng.membership(bundle).sum(axis=1).max() <= 5


def test_survivorship_status_reports_partial(bundle, config):
    s = UniverseEngine(config).survivorship_status(bundle)
    assert s["status"] in {"PARTIAL", "UNKNOWN"} and s["symbols_ending_before_panel_end"] >= 1
