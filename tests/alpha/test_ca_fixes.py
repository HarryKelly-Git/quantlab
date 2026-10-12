"""Sprint-c corporate-action patch on a SYNTHETIC panel. A fake adjusted crash (the adjustment amplified the
move; raw flat) is flagged UNKNOWN and re-chained away. A real split (raw move larger than the adjusted one)
and a real crash (raw = adjusted) are left alone. The replay bundle books the flagged day as neutral."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha import ca_fixes
from quantlab.alpha import master_replay as mr
from quantlab.alpha.synthetic import make_panel


def _panel():
    p = make_panel(n_stocks=20, n_days=500, seed=2)   # S015-S019 delist mid-sample; S001-S003 stay listed
    f, d = p.f, p.dates
    for k in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"):
        f[k] = f[k].copy()                       # the synthetic panel shares raw and adjusted frames: decouple
    for k in ("adj_open", "adj_high", "adj_low", "adj_close"):          # S001: fake -80% adjusted, raw flat
        f[k].loc[d[300]:, "S001"] = f[k].loc[d[300]:, "S001"] * 0.2
    for k in ("open", "high", "low", "close"):                          # S002: real 2:1 split (raw halves)
        f[k].loc[d[350]:, "S002"] = f[k].loc[d[350]:, "S002"] * 0.5
    for k in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"):  # S003: real crash
        f[k].loc[d[400]:, "S003"] = f[k].loc[d[400]:, "S003"] * 0.4
    ac = f["adj_close"]
    f["ret_cc"] = ac / ac.shift(1) - 1
    return p


def test_flags_only_amplified_adjustments():
    p = _panel()
    empty = pd.DataFrame(columns=["ticker", "date", "entity"])
    fl, tr = ca_fixes.flags(p, empty, pd.DataFrame(columns=["kind"]))
    got = set(zip(fl["entity"], pd.to_datetime(fl["date"])))
    assert ("S001", p.dates[300]) in got
    assert not any(e in ("S002", "S003") for e, _ in got)
    assert tr.empty


def test_apply_rechains_and_marks_unknown():
    p = _panel()
    before = p["adj_close"]["S001"].copy()
    empty = pd.DataFrame(columns=["ticker", "date", "entity"])
    fl, tr = ca_fixes.flags(p, empty, pd.DataFrame(columns=["kind"]))
    ca_fixes.apply(p, fl, tr)
    d = p.dates
    assert np.isnan(p["ret_cc"].at[d[300], "S001"])                                 # UNKNOWN, not invented
    ac = p["adj_close"]["S001"]
    assert abs(ac[d[300]] / ac[d[299]] - 1) < 1e-12                                 # no fake jump in the chain
    np.testing.assert_allclose((ac / ac.shift(1)).loc[d[301]:], (before / before.shift(1)).loc[d[301]:], rtol=1e-12)
    assert p.meta["ca_patch"]["flagged_days"] == len(fl)


def test_bundle_books_flagged_day_neutral_or_raw():
    p = _panel()
    empty = pd.DataFrame(columns=["ticker", "date", "entity"])
    fl, tr = ca_fixes.flags(p, empty, pd.DataFrame(columns=["kind"]))
    ca_fixes.apply(p, fl, tr)
    p.meta["ca_flags"] = fl[["entity", "date"]]
    # make the raw close drop too, as on a real mis-adjusted spin-off day (raw -12%)
    p.f["close"].loc[p.dates[300]:, "S001"] *= 0.88
    d = p.dates
    b = mr.bundle_from_research(p, p.symbols, p.master, None, {}, unknown_days="neutral")
    assert abs(b.panel.ret.at[d[300], "S001"]) < 1e-12 and abs(b.panel.aclose["S001"][d[300]] / b.panel.aclose["S001"][d[299]] - 1) < 1e-12
    r = (b.panel.close * b.panel.split_ratio + b.panel.dividend) / b.panel.close.shift(1) - 1
    assert abs(r.at[d[300], "S001"]) < 1e-12                                        # holder value unchanged
    braw = mr.bundle_from_research(p, p.symbols, p.master, None, {}, unknown_days="raw")
    assert abs(braw.panel.ret.at[d[300], "S001"] - (p["close"].at[d[300], "S001"] / p["close"].at[d[299], "S001"] - 1)) < 1e-9
