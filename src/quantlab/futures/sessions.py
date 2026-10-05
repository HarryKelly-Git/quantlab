"""CME Globex equity-index futures sessions, in EXCHANGE-LOCAL time (America/Chicago).

Facts used (CME Group equity-index product specs; re-verify before relying on edge cases):
* Globex trades Sunday-Friday 17:00 - 16:00 CT with a daily 16:00 - 17:00 CT maintenance break.
* "RTH" here means the cash-equity session 08:30 - 15:00 CT (09:30 - 16:00 ET), the window opening-range,
  VWAP and gap studies are defined on.
* A bar at/after 17:00 CT belongs to the NEXT trading date's session.

Time zones are never hard-coded as offsets: all conversion goes through zoneinfo, so DST changes (which
differ between the US and other regions) are handled by the tz database. Holidays/half-days are NOT
modelled here: the data's own bars decide which sessions exist (a missing session is just absent).
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

EXCHANGE_TZ = "America/Chicago"
GLOBEX_OPEN = time(17, 0)
RTH_OPEN = time(8, 30)
RTH_CLOSE = time(15, 0)
GLOBEX_CLOSE = time(16, 0)


def label_bars(index_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-bar session labels. ``index_utc`` = bar START times (tz-aware UTC, 1-minute bars).

    Columns: local (exchange-local start time), session_date (trading date), is_rth, is_overnight,
    minute_of_rth (0 = the 08:30 bar; -1 outside RTH)."""
    if index_utc.tz is None:
        raise ValueError("bar index must be tz-aware (UTC)")
    local = index_utc.tz_convert(EXCHANGE_TZ)
    t = local.hour * 60 + local.minute
    after_open = t >= GLOBEX_OPEN.hour * 60
    sess = (local.normalize() + pd.to_timedelta(after_open.astype(int), unit="D")).tz_localize(None).date
    rth = (t >= RTH_OPEN.hour * 60 + RTH_OPEN.minute) & (t < RTH_CLOSE.hour * 60)
    in_break = (t >= GLOBEX_CLOSE.hour * 60) & (t < GLOBEX_OPEN.hour * 60)
    minute = np.where(rth, t - (RTH_OPEN.hour * 60 + RTH_OPEN.minute), -1)
    return pd.DataFrame({"local": local, "session_date": sess, "is_rth": rth, "is_overnight": ~rth & ~in_break,
                         "in_break": in_break, "minute_of_rth": minute}, index=index_utc)
