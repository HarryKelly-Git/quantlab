"""The runner keeps the PC awake while it runs, and the display on in the windows that matter."""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from quantlab.pipeline.runner import (_ES_CONTINUOUS, _ES_DISPLAY_REQUIRED, _ES_SYSTEM_REQUIRED,
                                      keep_awake_flags)

ET = ZoneInfo("America/New_York")
ARGS = (time(8, 30), time(10, 0), time(19, 5))


def flags_at(y, mo, d, hh, mm, **kw):
    return keep_awake_flags(datetime(y, mo, d, hh, mm, tzinfo=ET), *ARGS, **kw)


def test_always_system_awake_while_running():
    for hh in (0, 3, 12, 22):
        f = flags_at(2026, 10, 3, hh, 0)                     # a Saturday
        assert f & _ES_SYSTEM_REQUIRED and f & _ES_CONTINUOUS and not f & _ES_DISPLAY_REQUIRED


def test_display_held_through_preopen_and_open_on_weekdays():
    assert not flags_at(2026, 10, 2, 8, 14) & _ES_DISPLAY_REQUIRED       # before 08:15 ET
    for hh, mm in ((8, 15), (8, 30), (9, 25), (9, 30), (9, 59)):
        assert flags_at(2026, 10, 2, hh, mm) & _ES_DISPLAY_REQUIRED, (hh, mm)
    assert not flags_at(2026, 10, 2, 10, 0) & _ES_DISPLAY_REQUIRED


def test_display_held_through_evening_processing():
    assert not flags_at(2026, 10, 1, 18, 59) & _ES_DISPLAY_REQUIRED
    for hh, mm in ((19, 0), (19, 5), (20, 34)):
        assert flags_at(2026, 10, 1, hh, mm) & _ES_DISPLAY_REQUIRED, (hh, mm)
    assert not flags_at(2026, 10, 1, 20, 35) & _ES_DISPLAY_REQUIRED


def test_display_hold_can_be_switched_off():
    assert not flags_at(2026, 10, 2, 9, 30, display=False) & _ES_DISPLAY_REQUIRED
    assert flags_at(2026, 10, 2, 9, 30, display=False) & _ES_SYSTEM_REQUIRED


def test_runner_applies_changes_only_and_releases_on_shutdown(monkeypatch):
    from quantlab.pipeline import runner as rn
    calls = []
    r = rn.PaperRunner.__new__(rn.PaperRunner)
    r.keep_awake, r.keep_display_on, r._awake_flags = True, True, None
    r.awake_submit_after, r.process_after = time(8, 30), time(19, 5)
    r.event = lambda *a, **k: None
    r._set_execution_state = lambda f: calls.append(f) or True
    monkeypatch.setattr(rn.os, "name", "nt")
    r._keep_awake(datetime(2026, 10, 2, 9, 0, tzinfo=ET))
    r._keep_awake(datetime(2026, 10, 2, 9, 1, tzinfo=ET))              # unchanged -> no second call
    r._keep_awake(datetime(2026, 10, 2, 11, 0, tzinfo=ET))             # display released
    assert calls == [_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED, _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED]
