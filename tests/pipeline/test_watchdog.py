"""The watchdog's decisions: leave a healthy runner alone, kill only a hung one, restart through Task
Scheduler, give the supervisor time, and never override a deliberate operator stop."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("watchdog", ROOT / "scripts" / "watchdog.py")
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


def d(**kw):
    base = dict(session={"status": "RUNNING", "stop_reason": None}, pid_is_python=True, hb_age=10.0,
                task_running=True, no_runner_since=None)
    base.update(kw)
    return wd.decide(**base)[0]


def test_healthy_runner_is_left_alone():
    assert d() == "OK"


def test_hung_runner_is_killed_so_the_supervisor_restarts_it():
    assert d(hb_age=wd.HUNG_SECONDS + 1) == "KILL_HUNG"
    assert d(hb_age=None) == "KILL_HUNG"


def test_dead_runner_is_restarted_through_task_scheduler():
    assert d(pid_is_python=False, task_running=False) == "START_TASK"
    assert d(session={"status": "CRASHED"}, pid_is_python=False, task_running=False) == "START_TASK"
    assert d(session=None, pid_is_python=False, task_running=False) == "START_TASK"


def test_supervisor_gets_time_then_a_stuck_task_is_restarted():
    assert d(pid_is_python=False, task_running=True, no_runner_since=60.0) == "WAIT"
    assert d(pid_is_python=False, task_running=True, no_runner_since=wd.STUCK_SECONDS + 1) == "RESTART_TASK"


def test_a_deliberate_operator_stop_is_respected():
    stopped = {"status": "STOPPED", "stop_reason": "operator stop (quantlab paper stop)"}
    assert d(session=stopped, pid_is_python=False, task_running=False) == "RESPECT_STOP"
    # a crash or refusal is not an operator stop
    assert d(session={"status": "STOPPED", "stop_reason": "startup failed"}, pid_is_python=False,
             task_running=False) == "START_TASK"
