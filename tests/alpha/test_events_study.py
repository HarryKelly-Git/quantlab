"""Event study (docs/EVENT-STUDY-PREREG.md) on SYNTHETIC data: events are point-in-time (truncation-invariant),
the cooldown keeps one event per stock per 20 sessions, outcomes start at the NEXT open, and headlines are
classified by the fixed rules (price recaps are not news)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha import events_study as es
from quantlab.alpha import research_data
from quantlab.alpha.synthetic import make_panel, truncate


def _setup(p):
    d = research_data.from_panel(p, manifest={"synthetic": True})
    u = p["close"].notna() & (p["n_hist"] >= 60)
    return d, u


def _jumpy():
    p = make_panel(n_stocks=30, n_days=400, seed=8)
    f = p.f
    for k in ("open", "close", "adj_open", "adj_close", "volume", "dollar_volume"):
        f[k] = f[k].copy()
    for s, i in (("S001", 200), ("S001", 210), ("S002", 250), ("S003", 300)):
        for k in ("open", "close", "adj_open", "adj_close"):
            f[k].loc[p.dates[i]:, s] *= 1.12                      # +12% from day i on
        f["volume"].iat[i, p.symbols.get_loc(s)] *= 5
    f["dollar_volume"] = f["close"] * f["volume"]
    ac = f["adj_close"]
    f["ret_cc"] = ac / ac.shift(1) - 1
    return p


def test_events_truncation_invariant_and_cooldown():
    p = _jumpy()
    d, u = _setup(p)
    ev = es.detect(p, u, "up")
    got = set(zip(ev["entity"], ev["date"]))
    assert ("S001", p.dates[200]) in got and ("S002", p.dates[250]) in got
    assert ("S001", p.dates[210]) not in got                       # within 20 sessions of the first S001 event
    for D in p.dates[[260, 320]]:
        pt = truncate(p, D)
        _, ut = _setup(pt)
        et = es.detect(pt, ut, "up")
        a = ev[ev["date"] <= D][["date", "entity"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, et[["date", "entity"]].reset_index(drop=True))


def test_forward_starts_at_next_open():
    p = _jumpy()
    d, u = _setup(p)
    rows = pd.DataFrame({"date": [p.dates[250]], "entity": ["S002"]})
    out = es.forward(p, d.ret_cc_pnl, rows, horizons=(1, 5))
    o, c = p["adj_open"]["S002"], p["adj_close"]["S002"]
    assert abs(out["ret_1"].iloc[0] - (c.iloc[251] / o.iloc[251] - 1)) < 1e-12
    assert abs(out["ret_5"].iloc[0] - (c.iloc[255] / o.iloc[251] - 1)) < 1e-12
    spy = p["adj_close"]["SPY"].iloc[255] / p["adj_open"]["SPY"].iloc[251] - 1
    assert abs(out["excess_5"].iloc[0] - (out["ret_5"].iloc[0] - spy)) < 1e-12


def test_headline_classification():
    dates = pd.bdate_range("2021-03-01", periods=5)
    ev = pd.DataFrame({"date": [dates[2], dates[3]], "entity": ["AAA", "BBB"], "ticker": ["AAA", "BBB"]})
    t = lambda d, h: (pd.Timestamp(d.date()).tz_localize(es.ET) + pd.Timedelta(hours=h)).tz_convert("UTC").isoformat()  # noqa: E731
    news = pd.DataFrame([
        {"id": 1, "created_at": t(dates[2], 9), "headline": "AAA wins $2B Pentagon contract", "symbols": ["AAA"]},
        {"id": 2, "created_at": t(dates[2], 11), "headline": "Why Is AAA Stock Soaring Today?", "symbols": ["AAA"]},
        {"id": 3, "created_at": t(dates[3], 10), "headline": "BBB Shares Are Trading Higher", "symbols": ["BBB"]},
        {"id": 4, "created_at": t(dates[3], 17), "headline": "BBB to be acquired by CCC", "symbols": ["BBB"]},  # after the close
    ])
    out = es.classify(ev, news, dates).set_index("entity")
    assert out.at["AAA", "news"] and out.at["AAA", "category"] == "deal" and out.at["AAA", "n_articles"] == 2
    assert not out.at["BBB", "news"] and out.at["BBB", "category"] == "none"         # recap only; the deal came later
