"""smart_money context family (congress / insider disclosures): recorded only where the source covers
the symbol, UNKNOWN otherwise (never 0), never scored, and with zero effect on the discovery score,
the selection score or which families fire. All data is SYNTHETIC."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.config import load_config
from quantlab.data import schemas
from quantlab.data.providers.sec_edgar import conservative_available_at
from quantlab.discovery.engine import DiscoveryEngine
from quantlab.discovery.families import (CONTEXT, CONTEXT_FEATURES, SCORED, SCORED_FEATURES, SCORED_POINTS,
                                         SELECTION_FEATURES, fire_context)

from .world import BENCH, crafted_bundle

ROOT = Path(__file__).resolve().parents[2]
SM = CONTEXT_FEATURES["smart_money"]


def _row(source, symbol, side, disclosed, cal, value=10_000.0, k=0):
    d = pd.Timestamp(disclosed)
    return {"record_id": f"{source}-{symbol}-{side}-{d.date()}-{k}", "source": source, "symbol": symbol, "actor": "a",
            "actor_detail": "House / D" if source == "congress" else "CEO", "side": side, "amount_low_usd": value,
            "amount_high_usd": value, "transaction_date": d - pd.Timedelta(days=20), "disclosed_date": d,
            "available_at": conservative_available_at(d, cal), "pit_status": "PIT_CONSERVATIVE", "raw_json": "{}",
            "provider": "synthetic", "retrieved_at": pd.Timestamp("2024-06-01", tz="UTC")}


def with_alt(b, extra=()):
    dates = b.panel.dates
    cal = b.calendar
    rows = [_row("congress", "MOMO", "BUY", dates[10], cal, k=1),        # sources start early: whole window covered
            _row("insider", "VOLX", "SELL", dates[12], cal, k=2),
            _row("congress", "MOMO", "BUY", dates[-8], cal, k=3),        # inside D's 30-day window
            _row("insider", "VOLX", "BUY", dates[-5], cal, k=4),
            _row("insider", "VOLX", "SELL", dates[-4], cal, value=np.nan, k=5),
            _row("congress", "DROP", "SELL", dates[-200], cal, k=6),     # covered, quiet in the window -> 0
            *extra]
    return replace(b, alt_trades=schemas.conform("alt_trades", pd.DataFrame(rows)))


@pytest.fixture(scope="module")
def cfg():
    return load_config(root=ROOT, overrides={"benchmarks": BENCH})


@pytest.fixture(scope="module")
def cb():
    return crafted_bundle(news_for=("MOMO",))


def test_family_is_context_only_and_never_scored():
    assert "smart_money" in CONTEXT and "smart_money" not in SCORED and "smart_money" not in SCORED_POINTS
    scored_feats = {f for fam in SCORED for f, _ in SCORED_FEATURES[fam]} | {f for f, _ in SELECTION_FEATURES}
    assert not scored_feats & set(SM)


def test_scores_selection_and_firing_are_identical_with_and_without_disclosures(cfg, cb):
    eng = DiscoveryEngine(cfg)
    d = cb.panel.dates[-1]
    a = eng.scan(cb, d).table
    b = eng.scan(with_alt(cb), d).table
    for col in ("score", "selection_score", "coverage", *[f"c_{f}" for f in SCORED]):
        pd.testing.assert_series_equal(a[col], b[col])
    assert a["fired"].equals(b["fired"]) and a["catalyst_fired"].equals(b["catalyst_fired"])
    assert all(a.at[s, "context"]["smart_money"]["state"] == "UNKNOWN" for s in a.index)   # no source: UNKNOWN


def test_known_only_where_the_source_covers_the_symbol(cfg, cb):
    scan = DiscoveryEngine(cfg).scan(with_alt(cb), cb.panel.dates[-1])
    t = scan.table
    momo = t.at["MOMO", "context"]["smart_money"]
    assert momo["state"] == "KNOWN" and momo["values"]["congress_buys_30d"] == 1
    assert momo["values"]["insider_buys_30d"] is None                     # insiders never showed MOMO: UNKNOWN
    assert momo["fired"] and "never scored" in momo["reasons"][0]
    volx = t.at["VOLX", "context"]["smart_money"]
    assert volx["values"]["insider_buys_30d"] == 1 and volx["values"]["insider_sells_30d"] == 1
    assert volx["values"]["insider_net_value_30d"] is None               # an unknown value in the window
    drop = t.at["DROP", "context"]["smart_money"]
    assert drop["state"] == "KNOWN" and drop["values"]["congress_buys_30d"] == 0 and not drop["fired"]
    for s in t.index.drop(["MOMO", "VOLX", "DROP"]):
        c = t.at[s, "context"]["smart_money"]
        assert c["state"] == "UNKNOWN" and "values" not in c and c["why"]   # never a fabricated zero
    row = next(r for r in scan.coverage if r["key"] == "smart_money")
    assert row["role"] == "context only (never scored)" and row["known"] == 3


def test_a_disclosure_after_the_cutoff_changes_nothing_at_d(cfg, cb):
    d = cb.panel.dates[-1]
    late = _row("congress", "N001", "BUY", d, cb.calendar, k=9)           # disclosed on D: usable only after D
    base = DiscoveryEngine(cfg).scan(with_alt(cb), d).table
    more = DiscoveryEngine(cfg).scan(with_alt(cb, extra=[late]), d).table
    assert base.at["N001", "context"] == more.at["N001", "context"]
    assert more.at["N001", "context"]["smart_money"]["state"] == "UNKNOWN"
    # truncation: the scan at an earlier session sees the same context on the full and the truncated bundle
    full = with_alt(cb, extra=[late])
    e = cb.panel.dates[-6]
    s_full = DiscoveryEngine(cfg).scan(full, e).table
    s_trunc = DiscoveryEngine(cfg).scan(full.truncate(e), e).table
    assert s_full["context"].map(lambda c: c["smart_money"]).equals(s_trunc["context"].map(lambda c: c["smart_money"]))


def test_fire_context_handles_partial_unknowns():
    fired, why = fire_context("smart_money", pd.Series({f: np.nan for f in SM}), None)
    assert (fired, why) == (False, [])
    r = pd.Series({**{f: np.nan for f in SM}, "insider_buys_30d": 2.0, "insider_sells_30d": 0.0})
    fired, why = fire_context("smart_money", r, None)
    assert fired and "insiders 2 open-market buy / 0 sell, net value UNKNOWN" in why[0]
