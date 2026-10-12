"""Intraday news reaction (docs/INTRADAY-NEWS-PREREG.md) on SYNTHETIC minute bars. The pre-headline price is the
last bar CLOSED by the headline minute (bars are stamped at their start), the reaction uses the bar closed in
[tau+10, tau+15], the exit is the last regular-session bar, and costs use doubled half-spreads."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha import intraday_news as inn


def _bars(day="2021-03-03", jump_at="10:31", jump=0.03):
    t = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="1min", tz=inn.ET)
    px = pd.Series(100.0, index=t)
    px[t >= pd.Timestamp(f"{day} {jump_at}", tz=inn.ET)] = 100.0 * (1 + jump)
    px[t >= pd.Timestamp(f"{day} 14:00", tz=inn.ET)] = 100.0 * (1 + jump) * 1.01
    spy = pd.Series(400.0, index=t)
    return pd.concat([pd.DataFrame({"symbol": "AAA", "t": t, "c": px.to_numpy()}),
                      pd.DataFrame({"symbol": "SPY", "t": t, "c": spy.to_numpy()})], ignore_index=True)


def test_measure_uses_only_closed_bars_and_returns_net_excess():
    h = pd.DataFrame({"ticker": ["AAA"], "created": [pd.Timestamp("2021-03-03 10:30:20", tz=inn.ET)], "mdv20": [50e6]})
    m = inn.measure(h, _bars())
    assert len(m) == 1
    r = m.iloc[0]
    assert r["p0"] == 100.0                     # the 10:31 bar (jump) is not closed until 10:32 > tau = 10:31
    assert abs(r["r0"] - 0.03) < 1e-12 and r["p15"] == 103.0
    x = inn.returns(m)
    cost = 2 * (2 * 5e-4 + 5e-4)                 # 5 bps half-spread tier (MDV20 $50M), doubled, + 5 bps slippage
    assert abs(x["net_close"].iloc[0] - (0.01 - cost)) < 1e-12


def test_stale_or_late_headlines_are_skipped():
    late = pd.DataFrame({"ticker": ["AAA"], "created": [pd.Timestamp("2021-03-03 15:50:00", tz=inn.ET)], "mdv20": [50e6]})
    assert inn.measure(late, _bars()).empty      # no bar after tau+15 before the close... and no entry possible
    missing = pd.DataFrame({"ticker": ["ZZZ"], "created": [pd.Timestamp("2021-03-03 10:30:00", tz=inn.ET)], "mdv20": [50e6]})
    assert inn.measure(missing, _bars()).empty


def test_sample_days_is_seeded_stratified_and_pre_holdout():
    d = pd.bdate_range("2017-01-01", "2024-12-31")
    a, b = inn.sample_days(d), inn.sample_days(d)
    assert a.equals(b) and a.max() < inn.HOLDOUT
    per_year = pd.Series(a.year).value_counts()
    assert per_year.min() >= 60 and per_year.max() <= 70
