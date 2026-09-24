"""Trading calendar and the information-cutoff rule.

INFORMATION CUTOFF RULE (the single most important point-in-time convention in QuantLab):
    A decision "as of session D" may use only information whose ``available_at`` timestamp is
    <= cutoff(D) = D at 16:00 America/New_York. Orders generated from that decision execute at
    the NEXT session's open. The same rule is used in backtests and in the live paper pipeline.
"""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def to_session(d: date | datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Normalize anything date-like to a tz-naive midnight Timestamp (the session label)."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(MARKET_TZ).tz_localize(None)
    return ts.normalize()


def to_utc(ts: datetime | pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"naive timestamp {t!r}: availability timestamps must be timezone-aware")
    return t.tz_convert("UTC")


class TradingCalendar:
    """Ordered set of session dates plus cutoff / availability helpers."""

    def __init__(self, sessions: pd.DatetimeIndex | list, cutoff_time: time = MARKET_CLOSE, tz: ZoneInfo = MARKET_TZ):
        idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize()
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        self.sessions = idx.unique().sort_values()
        if len(self.sessions) == 0:
            raise ValueError("TradingCalendar needs at least one session")
        self.cutoff_time = cutoff_time
        self.tz = tz
        # vectorized cutoffs (UTC) for fast searchsorted
        local = pd.DatetimeIndex(
            [datetime.combine(s.date(), cutoff_time) for s in self.sessions]
        ).tz_localize(tz)
        self._cutoffs_utc = local.tz_convert("UTC")

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_dates(cls, dates, **kw) -> "TradingCalendar":
        return cls(pd.DatetimeIndex(pd.to_datetime(dates)), **kw)

    @classmethod
    def business_days(cls, start: str | date, end: str | date, **kw) -> "TradingCalendar":
        """Weekday calendar — ONLY for synthetic data and tests (ignores exchange holidays)."""
        return cls(pd.bdate_range(start, end), **kw)

    # -- queries --------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sessions)

    def __contains__(self, d) -> bool:
        return to_session(d) in self.sessions

    @property
    def first(self) -> pd.Timestamp:
        return self.sessions[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.sessions[-1]

    def cutoff(self, session: date | str | pd.Timestamp) -> pd.Timestamp:
        """UTC timestamp of the information cutoff for a session date."""
        s = to_session(session)
        local = pd.Timestamp(datetime.combine(s.date(), self.cutoff_time)).tz_localize(self.tz)
        return local.tz_convert("UTC")

    def first_usable_session(self, available_at: pd.Timestamp | datetime | str) -> pd.Timestamp | None:
        """Earliest session D with available_at <= cutoff(D). None if after the calendar end."""
        ts = to_utc(available_at)
        i = int(self._cutoffs_utc.searchsorted(ts, side="left"))
        return self.sessions[i] if i < len(self.sessions) else None

    def first_usable_sessions(self, available_at: pd.Series) -> pd.Series:
        """Vectorized :meth:`first_usable_session` (NaT when beyond the calendar)."""
        ts = pd.to_datetime(available_at, utc=True)
        idx = self._cutoffs_utc.searchsorted(ts.values, side="left")
        out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
        ok = idx < len(self.sessions)
        out[ok] = self.sessions.values[idx[ok]]
        return pd.Series(out, index=available_at.index)

    def reaction_session(self, event_time: pd.Timestamp | datetime | str) -> pd.Timestamp | None:
        """First session whose regular trading can react to an event at ``event_time``:
        before 16:00 ET on a session day -> that session (pre-market events react at the open);
        at/after the close -> the next session."""
        return self.first_usable_session(to_utc(event_time) + pd.Timedelta(microseconds=1))

    def next_session(self, d, n: int = 1) -> pd.Timestamp | None:
        s = to_session(d)
        i = int(self.sessions.searchsorted(s, side="right")) + n - 1
        return self.sessions[i] if 0 <= i < len(self.sessions) else None

    def prev_session(self, d, n: int = 1) -> pd.Timestamp | None:
        s = to_session(d)
        i = int(self.sessions.searchsorted(s, side="left")) - n
        return self.sessions[i] if 0 <= i < len(self.sessions) else None

    def offset(self, d, n: int) -> pd.Timestamp | None:
        """Session n sessions after (n>0) or before (n<0) session d (d must be a session)."""
        s = to_session(d)
        i = self.sessions.get_indexer([s])[0]
        if i < 0:
            raise KeyError(f"{s.date()} is not a session")
        j = i + n
        return self.sessions[j] if 0 <= j < len(self.sessions) else None

    def between(self, start, end) -> pd.DatetimeIndex:
        s, e = to_session(start), to_session(end)
        return self.sessions[(self.sessions >= s) & (self.sessions <= e)]

    def last_session_on_or_before(self, d) -> pd.Timestamp | None:
        s = to_session(d)
        i = int(self.sessions.searchsorted(s, side="right")) - 1
        return self.sessions[i] if i >= 0 else None
