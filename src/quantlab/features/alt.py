"""Congress / insider disclosure features (group ``alt``). CONTEXT ONLY: recorded with discovery
candidates and exploration decisions, never scored, never a selection or sizing input.

Source: ``bundle.alt_trades`` (data/alt_trades.py): SEC Form 4 (insider; available_at = EDGAR acceptance
time, PIT) and House PTRs (congress; available_at = cutoff of the session after the filing date,
PIT_CONSERVATIVE). A disclosure counts from the first session whose cutoff is >= its ``available_at``,
never from its transaction date.

Counted rows: record_status PARSED, side BUY / SELL. Insider rows count only reporting owners who are
directors or officers (the population of the SEC dataset import and of the pre-registered study; 10%
owners that are funds are not "insiders" in that sense).

Window: disclosures whose first usable session s satisfies D - 30 calendar days < s <= D.

UNKNOWN (NaN), never 0, until the source (congress or insider, separately) has delivered for the whole
window: D must be at least 30 days after the first usable session of the source's earliest stored
record (any record, including filings without a ticker). After that a symbol without disclosures is
0: both sources cover every issuer / every House member, so silence is information. Coverage only looks
at records usable by D, so the features are truncation-invariant. (A refresh gap longer than the
refresh window would undercount; data/alt_trades.py reports every run's fetched / remaining counts.)

insider_net_value_30d is UNKNOWN when any counted buy or sale in the window has no known value
(shares x price): a sum over partly unknown values would be fabricated.

Evidence status: a pre-registered study found insider buying FLAT at realistic timing and its one
positive variant failed the locked 2025+ holdout. These are context for the learning loop, not a signal.
"""
from __future__ import annotations

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import full_like_nan, memo

WINDOW_DAYS = 30
_SRC = ("bundle.alt_trades (SEC Form 4: available_at = acceptance time; House PTR: cutoff of the session after the "
        f"filing date); window = sessions in (D - {WINDOW_DAYS} calendar days, D]")


def _is_insider_person(detail: pd.Series) -> pd.Series:
    d = detail.fillna("").astype(str)
    return d.str.contains("Director", regex=False) | d.str.contains("Officer", regex=False)


def _grid(fs: FeatureSet) -> pd.DatetimeIndex:
    """Calendar sessions from one window before the first panel date to the last panel date, so a
    window at the start of a tail bundle still sees the sessions before it."""
    p = fs.panel.dates
    sess = fs.bundle.calendar.sessions
    lo = p[0] - pd.Timedelta(days=WINDOW_DAYS + 1)
    return sess[(sess >= lo) & (sess <= p[-1])].union(p)


def _build(fs: FeatureSet, source: str) -> dict[str, pd.DataFrame] | None:
    """Windowed buys / sells / net value and the coverage mask for one source, on panel dates."""
    a = fs.bundle.alt_trades
    if a is None or a.empty:
        return None
    a = a[a["source"] == source]
    if a.empty:
        return None
    cal = fs.bundle.calendar
    sess = cal.first_usable_sessions(a["available_at"])
    a = a.assign(_s=sess.to_numpy()).dropna(subset=["_s"])
    if a.empty:
        return None
    p = fs.panel
    grid = _grid(fs)
    src_first = pd.Timestamp(a["_s"].min())                  # coverage starts with the earliest record of any kind
    counted = a[(a["record_status"] == "PARSED") & a["symbol"].isin(p.symbols) & a["side"].isin(["BUY", "SELL"])]
    if source == "insider":
        counted = counted[_is_insider_person(counted["actor_detail"])]
    zero = pd.DataFrame(0.0, index=grid, columns=p.symbols)

    def daily(rows: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        if rows.empty:
            return zero.copy()
        k = pd.DataFrame({"s": rows["_s"].to_numpy(), "sym": rows["symbol"].to_numpy(),
                          "w": 1.0 if weights is None else weights.to_numpy(dtype="float64")})
        k = k.groupby(["s", "sym"])["w"].sum().unstack(fill_value=0.0)
        k = k.reindex(index=grid, columns=p.symbols, fill_value=0.0).astype("float64")
        return zero.add(k, fill_value=0.0)

    def window(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(f"{WINDOW_DAYS}D").sum().reindex(p.dates)

    buys = counted[counted["side"] == "BUY"]
    sells = counted[counted["side"] == "SELL"]
    val = pd.to_numeric(counted["amount_low_usd"], errors="coerce")
    signed = val.where(counted["side"] == "BUY", -val)
    out = {"buys": window(daily(buys)), "sells": window(daily(sells)),
           "value": window(daily(counted, signed.fillna(0.0))),
           "unknown_value": window(daily(counted, val.isna().astype("float64")))}
    # coverage: the whole window lies after the source's first usable record
    dates = pd.Series(p.dates, index=p.dates)
    src_ok = ((dates - pd.Timedelta(days=WINDOW_DAYS)) >= src_first).to_numpy()
    out["mask"] = pd.DataFrame(src_ok[:, None].repeat(len(p.symbols), axis=1), index=p.dates, columns=p.symbols)
    return out


def _parts(fs: FeatureSet, source: str) -> dict[str, pd.DataFrame] | None:
    return memo(fs, f"alt_trades_{source}", lambda: _build(fs, source))  # type: ignore[return-value]


def _feature(fs: FeatureSet, source: str, what: str) -> pd.DataFrame:
    b = _parts(fs, source)
    if b is None:
        return full_like_nan(fs)          # no source => UNKNOWN, not zero
    if what == "net":
        v = b["buys"] - b["sells"]
    elif what == "net_value":
        v = b["value"].where(b["unknown_value"] == 0)
    else:
        v = b[what]
    return v.where(b["mask"])


@FEATURES.feature("congress_buys_30d", "alt",
                  "congressional purchase disclosures usable in the last 30 days (UNKNOWN before coverage)",
                  _SRC, PitStatus.PIT_CONSERVATIVE)
def congress_buys_30d(fs: FeatureSet) -> pd.DataFrame:
    return _feature(fs, "congress", "buys")


@FEATURES.feature("congress_sells_30d", "alt",
                  "congressional sale disclosures usable in the last 30 days (UNKNOWN before coverage)",
                  _SRC, PitStatus.PIT_CONSERVATIVE)
def congress_sells_30d(fs: FeatureSet) -> pd.DataFrame:
    return _feature(fs, "congress", "sells")


@FEATURES.feature("congress_net_30d", "alt", "congress_buys_30d - congress_sells_30d (disclosure counts)",
                  _SRC, PitStatus.PIT_CONSERVATIVE)
def congress_net_30d(fs: FeatureSet) -> pd.DataFrame:
    return _feature(fs, "congress", "net")


@FEATURES.feature("insider_buys_30d", "alt",
                  "insider open-market purchases (Form 4 code P) filed and usable in the last 30 days",
                  _SRC, PitStatus.PIT_CONSERVATIVE)
def insider_buys_30d(fs: FeatureSet) -> pd.DataFrame:
    return _feature(fs, "insider", "buys")


@FEATURES.feature("insider_sells_30d", "alt",
                  "insider open-market sales (Form 4 code S) filed and usable in the last 30 days",
                  _SRC, PitStatus.PIT_CONSERVATIVE)
def insider_sells_30d(fs: FeatureSet) -> pd.DataFrame:
    return _feature(fs, "insider", "sells")


@FEATURES.feature("insider_net_value_30d", "alt",
                  "USD value (shares x price) of insider open-market purchases minus sales usable in the last "
                  "30 days; UNKNOWN if any of them has no known value",
                  _SRC, PitStatus.PIT_CONSERVATIVE)
def insider_net_value_30d(fs: FeatureSet) -> pd.DataFrame:
    return _feature(fs, "insider", "net_value")


ALT_FEATURES = ("congress_buys_30d", "congress_sells_30d", "congress_net_30d", "insider_buys_30d", "insider_sells_30d",
                "insider_net_value_30d")

__all__ = ["ALT_FEATURES", "WINDOW_DAYS"]
