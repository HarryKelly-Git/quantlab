"""Data splits: in-sample development period, walk-forward windows, locked holdout.

Walk-forward windows (config ``validation.walk_forward``):
  * rolling (default) or anchored (expanding) training windows of ``train_years``;
  * ``embargo_sessions`` sessions are skipped between the last training session and the first
    test session, so trades/labels formed at the end of training cannot overlap the test period;
  * each test window spans ``test_months`` calendar months; windows advance by ``step_months``;
  * EVERY window (train and test) ends strictly before ``validation.holdout.start``. A final test
    window that would reach the holdout is cut at the last pre-holdout session and dropped if it
    has fewer than ``min_test_sessions`` sessions.
Windows are expressed in sessions of the supplied calendar, so they are exact and reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from quantlab.core.calendar import TradingCalendar, to_session


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_sessions: int

    @property
    def segment(self) -> str:
        return f"oos:{self.index}"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("train_start", "train_end", "test_start", "test_end"):
            d[k] = str(pd.Timestamp(d[k]).date())
        return d


def holdout_start(config) -> pd.Timestamp:
    return to_session(config.get("validation.holdout.start"))


def in_sample_range(config) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of the in-sample development period. Must end before the holdout."""
    s = to_session(config.get("validation.in_sample.start"))
    e = to_session(config.get("validation.in_sample.end"))
    if e >= holdout_start(config):
        raise ValueError("validation.in_sample.end must be before validation.holdout.start")
    if s > e:
        raise ValueError("validation.in_sample.start is after its end")
    return s, e


def holdout_range(config, calendar: TradingCalendar | None = None) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """(holdout start, last available session or None). Access requires validation.holdout unlock."""
    start = holdout_start(config)
    end = calendar.last if calendar is not None and calendar.last >= start else None
    return start, end


def walk_forward_windows(calendar: TradingCalendar, config) -> list[WalkForwardWindow]:
    wf = config.section("validation").get("walk_forward", {})
    train_years = int(wf.get("train_years", 3))
    test_months = int(wf.get("test_months", 6))
    step_months = int(wf.get("step_months", test_months))
    embargo = int(wf.get("embargo_sessions", 10))
    mode = str(wf.get("mode", "rolling"))
    min_test = int(wf.get("min_test_sessions", 20))
    if train_years <= 0 or test_months <= 0 or step_months <= 0 or embargo < 0:
        raise ValueError("invalid walk_forward configuration")
    if step_months < test_months:
        # Overlapping test windows would chain capital from a later date into an earlier decision
        # and count the same sessions/trades twice in the aggregate OOS statistics.
        raise ValueError(f"walk_forward.step_months ({step_months}) must be >= test_months ({test_months}): "
                         "overlapping out-of-sample windows are not allowed")
    if mode not in ("rolling", "anchored"):
        raise ValueError(f"walk_forward.mode must be rolling|anchored, got {mode!r}")
    hold = holdout_start(config)
    sessions = calendar.sessions[calendar.sessions < hold]
    if len(sessions) == 0:
        return []
    anchor_cfg = wf.get("start") or config.get("validation.in_sample.start", None)
    anchor = max(to_session(anchor_cfg), sessions[0]) if anchor_cfg else sessions[0]

    windows: list[WalkForwardWindow] = []
    k = 0
    while True:
        roll_start = anchor + pd.DateOffset(months=step_months * k)
        train_start_raw = anchor if mode == "anchored" else roll_start
        train_end_excl = roll_start + pd.DateOffset(years=train_years) if mode == "rolling" else \
            anchor + pd.DateOffset(years=train_years) + pd.DateOffset(months=step_months * k)
        test_end_excl = train_end_excl + pd.DateOffset(months=test_months)
        train = sessions[(sessions >= train_start_raw) & (sessions < train_end_excl)]
        if len(train) == 0 or train_end_excl > sessions[-1]:
            break
        after = sessions[sessions > train[-1]]
        if len(after) <= embargo:
            break
        test_start = after[embargo]
        test = sessions[(sessions >= test_start) & (sessions < test_end_excl)]
        if len(test) < min_test:
            break
        windows.append(WalkForwardWindow(len(windows), train[0], train[-1], test[0], test[-1], embargo))
        if test_end_excl > sessions[-1]:
            break          # the last test window was cut at the holdout boundary
        k += 1
    return windows
