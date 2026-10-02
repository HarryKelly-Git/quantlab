"""Congress / insider disclosure features (group ``alt``). CONTEXT ONLY: recorded with discovery
candidates and exploration decisions, never scored, never a selection or sizing input.

Source: ``bundle.alt_trades`` (Quiver, data/providers/quiver.py). A disclosure counts from the first
session whose cutoff is >= its ``available_at`` (= the 16:00 ET cutoff of the session AFTER the
disclosure date), never from its transaction date.

Window: disclosures whose first usable session s satisfies D - 30 calendar days < s <= D.

UNKNOWN (NaN), never 0, unless BOTH hold at D:
  * the source (congress or insider, separately) has delivered for the whole window: D is at least
    30 days after the first usable session of the source's earliest stored record (a window that
    started before the source did would be a partial count);
  * the source covers the symbol: the symbol has at least one record of that source usable by D
    (a ticker the feed has never shown may simply not be mapped by it).
Both conditions only look at records usable by D, so the features are truncation-invariant.

insider_net_value_30d is UNKNOWN when any open-market buy or sale in the window has no known value
(shares x price): a sum over partly unknown values would be fabricated.

Evidence status: a separate study found insider buying FLAT at realistic timing and its one positive
variant failed the locked 2025+ holdout. These are context for the learning loop, not a signal.
"""
from __future__ import annotations

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import full_like_nan, memo

WINDOW_DAYS = 30
_SRC = ("bundle.alt_trades (Quiver; available_at = cutoff of the session after the disclosure date); "
        f"window = sessions in (D - {WINDOW_DAYS} calendar days, D]")


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
    src_first = pd.Timestamp(a["_s"].min())
    sym_first = a.groupby("symbol")["_s"].min()
    in_panel = a[a["symbol"].isin(p.symbols)]
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

    buys = in_panel[in_panel["side"] == "BUY"]
    sells = in_panel[in_panel["side"] == "SELL"]
    trades = pd.concat([buys, sells])
    val = pd.to_numeric(trades["amount_low_usd"], errors="coerce")
    signed = val.where(trades["side"] == "BUY", -val)
    out = {"buys": window(daily(buys)), "sells": window(daily(sells)),
           "value": window(daily(trades, signed.fillna(0.0))),
           "unknown_value": window(daily(trades, val.isna().astype("float64")))}
    # coverage: whole window after the source started AND the symbol already seen in the source
    dates = pd.Series(p.dates, index=p.dates)
    src_ok = (dates - pd.Timedelta(days=WINDOW_DAYS)) >= src_first
    first = sym_first.reindex(p.symbols)
    first_ns = pd.to_datetime(first).to_numpy(dtype="datetime64[ns]")
    sym_ok = pd.DataFrame(p.dates.to_numpy(dtype="datetime64[ns]")[:, None] >= first_ns[None, :],
                          index=p.dates, columns=p.symbols)          # NaT (never seen) compares False
    out["mask"] = sym_ok & src_ok.to_numpy()[:, None]
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
