"""Event features around earnings releases (ARCHITECTURE.md section 4, event group).

Timing (the whole point of this module):
  * An event counts from its REACTION session r = the first session whose regular trading can
    react (pre-market / intraday release -> that session; after-close release -> next session),
    and never before the first session whose cutoff is >= its ``available_at``.
  * ``ear_3d`` sums abnormal returns (stock - SPY) over r-1, r, r+1, so it is only KNOWN at the
    close of r+1. From r (a new event happened, its reaction not yet known) until r+1 the value is
    NaN; from r+1 it is carried forward until the next event.
  * Earnings surprises vs analyst consensus are NOT available point-in-time for free, so they are
    not used. The price reaction (EAR) is the PIT-safe surprise proxy (Brandt et al. 2008).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import active_from, full_like_nan, market_series, memo, rolling_std, safe_div

_SRC_EV = "bundle.events (earnings_release reaction_date/available_at) + panel ret/dollar_volume"
_RESET = -1.0e300   # sentinel: "new event, value not known yet"


def _reaction_index(fs: FeatureSet) -> pd.DataFrame:
    """Boolean frame: True at (reaction session, symbol) of each usable earnings event."""
    def build() -> pd.DataFrame:
        p = fs.panel
        out = pd.DataFrame(False, index=p.dates, columns=p.symbols)
        ev = fs.bundle.events
        if ev.empty:
            return out
        ev = ev[(ev["event_type"] == "earnings_release") & ev["symbol"].isin(p.symbols)]
        # an 8-K/A refiles an earlier release: it is not a new event (no new reaction session)
        ev = ev[~ev["payload_json"].astype(str).str.contains('"form": "8-K/A"', regex=False)]
        if ev.empty:
            return out
        usable = fs.bundle.calendar.first_usable_sessions(ev["available_at"])
        react = pd.to_datetime(ev["reaction_date"])
        sess = pd.concat([usable, react], axis=1).max(axis=1)          # never before it is usable
        for sym, d in zip(ev["symbol"], sess):
            if pd.notna(d) and d in out.index:
                out.at[d, sym] = True
        return out
    return memo(fs, "event_reaction_index", build)  # type: ignore[return-value]


def _row_positions(fs: FeatureSet) -> pd.DataFrame:
    n = len(fs.panel.dates)
    return pd.DataFrame(np.repeat(np.arange(n, dtype="float64")[:, None], len(fs.panel.symbols), axis=1),
                        index=fs.panel.dates, columns=fs.panel.symbols)


def _abnormal(fs: FeatureSet) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        m = market_series(fs, "ret")
        if m is None:
            return full_like_nan(fs)
        return fs.panel.ret.sub(m, axis=0)
    return memo(fs, "abnormal_ret", build)  # type: ignore[return-value]


def _carry_known_at_next(values_at_r: pd.DataFrame, react: pd.DataFrame) -> pd.DataFrame:
    """Place each event's value at r+1 (when known), NaN from r until then, carry forward."""
    known = values_at_r.where(react).shift(1)                # value lands on r+1
    marks = pd.DataFrame(np.where(react.to_numpy(), _RESET, np.nan), index=react.index, columns=react.columns)
    marks = marks.where(known.isna(), known)                 # r+1 overrides a reset on the same row
    carried = marks.ffill()
    return carried.where(carried != _RESET)


@FEATURES.feature("days_since_earnings", "event",
                  "sessions since the latest earnings reaction session <= D (usable by cutoff(D))",
                  _SRC_EV, PitStatus.PIT, lookback=0)
def days_since_earnings(fs: FeatureSet) -> pd.DataFrame:
    react = _reaction_index(fs)
    pos = _row_positions(fs)
    last = pos.where(react).ffill()
    return pos - last


@FEATURES.feature("ear_3d", "event",
                  "sum of (ret - SPY ret) over reaction-1..reaction+1; known from the close of reaction+1",
                  _SRC_EV, PitStatus.PIT, lookback=2)
def ear_3d(fs: FeatureSet) -> pd.DataFrame:
    ar = _abnormal(fs)
    window = ar.shift(1) + ar + ar.shift(-1)                 # centred at r; only READ at r+1 below
    react = _reaction_index(fs)
    # shift(-1) reads r+1, which is exactly when the value is placed (known at r+1's close).
    return _carry_known_at_next(window.where(react), react)


@FEATURES.feature("ear_z", "event",
                  "ear_3d / (std(ret - SPY ret, 60) measured through reaction-2, * sqrt(3))",
                  _SRC_EV, PitStatus.PIT, lookback=62)
def ear_z(fs: FeatureSet) -> pd.DataFrame:
    ar = _abnormal(fs)
    react = _reaction_index(fs)
    pre_vol = rolling_std(ar, 60).shift(2)                   # uses nothing on/after r-1
    window = ar.shift(1) + ar + ar.shift(-1)
    z = safe_div(window, pre_vol * np.sqrt(3.0))
    return _carry_known_at_next(z.where(react), react)


@FEATURES.feature("event_rel_volume", "event",
                  "dollar volume on the reaction session / mean dollar volume of the 20 sessions before it; carried forward",
                  _SRC_EV, PitStatus.PIT, lookback=20)
def event_rel_volume(fs: FeatureSet) -> pd.DataFrame:
    dv = fs.panel.dollar_volume
    react = _reaction_index(fs)
    rel = safe_div(dv, dv.shift(1).rolling(20, min_periods=20).mean())
    return rel.where(react).ffill()


@FEATURES.feature("est_sessions_to_earnings", "event",
                  "63 - days_since_earnings clipped to [0, 63] (MODEL estimate: no PIT earnings calendar)",
                  _SRC_EV, PitStatus.PIT, lookback=0)
def est_sessions_to_earnings(fs: FeatureSet) -> pd.DataFrame:
    return (63.0 - fs.get("days_since_earnings")).clip(lower=0.0, upper=63.0)


@FEATURES.feature("reaction_ret_1d", "event",
                  "abnormal return (ret - SPY ret) on the latest earnings reaction session r; known from the close "
                  "of r, carried until the next event", _SRC_EV, PitStatus.PIT, lookback=1)
def reaction_ret_1d(fs: FeatureSet) -> pd.DataFrame:
    return _abnormal(fs).where(_reaction_index(fs)).ffill()


@FEATURES.feature("reaction_z_1d", "event",
                  "reaction_ret_1d / std(ret - SPY ret, 60) measured through r-1; carried until the next event",
                  _SRC_EV, PitStatus.PIT, lookback=61)
def reaction_z_1d(fs: FeatureSet) -> pd.DataFrame:
    ar = _abnormal(fs)
    z = safe_div(ar, rolling_std(ar, 60).shift(1))
    return z.where(_reaction_index(fs)).ffill()


@FEATURES.feature("abn_ret_since_reaction", "event",
                  "cumulative abnormal return from the close of the latest reaction session r to D (0 at r); "
                  "missing bars count as 0", _SRC_EV, PitStatus.PIT, lookback=1)
def abn_ret_since_reaction(fs: FeatureSet) -> pd.DataFrame:
    cum = _abnormal(fs).fillna(0.0).cumsum()
    at_r = cum.where(_reaction_index(fs)).ffill()
    return cum - at_r


SEC_SOURCE_TYPES = ("sec_8k", "earnings_release", "periodic_report", "foreign_report")
_SRC_8K = "bundle.events sec_8k (8-K items other than 2.02/7.01/9.01; available_at = SEC acceptance)"


@FEATURES.feature("sec_material_1d", "event",
                  "material 8-K filings (items other than 2.02/7.01/9.01) usable in (cutoff(D-1), cutoff(D)]",
                  _SRC_8K, PitStatus.PIT, lookback=0)
def sec_material_1d(fs: FeatureSet) -> pd.DataFrame:
    p = fs.panel
    ev_all = fs.bundle.events
    sec = ev_all[ev_all["event_type"].isin(SEC_SOURCE_TYPES)] if len(ev_all) else ev_all
    if sec.empty:
        return full_like_nan(fs)            # no SEC filings source => UNKNOWN, not zero
    ev = sec[(sec["event_type"] == "sec_8k") & sec["symbol"].isin(p.symbols)]
    out = pd.DataFrame(0.0, index=p.dates, columns=p.symbols)
    if ev.empty:
        return active_from(fs, sec["available_at"], out)
    sess = fs.bundle.calendar.first_usable_sessions(ev["available_at"])
    c = pd.DataFrame({"s": sess.to_numpy(), "sym": ev["symbol"].to_numpy()}).dropna()
    c = c.groupby(["s", "sym"]).size().unstack(fill_value=0).reindex(index=p.dates, columns=p.symbols, fill_value=0)
    return active_from(fs, sec["available_at"], out.add(c.astype("float64"), fill_value=0.0))
