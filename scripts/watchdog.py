r"""QuantLab watchdog: keep the PAPER runner and the dashboard alive. Runs every 5 minutes from the
Windows Task Scheduler task "QuantLab watchdog" (with pythonw.exe, so no window), independent of
any interactive or Claude session.

What it does, and only this:
  * runner alive (PID is a python process AND heartbeat fresh)          -> nothing
  * runner process alive but heartbeat stale > HUNG_SECONDS (hung)      -> kill that process; the
                                                                           supervisor restarts it
  * no live runner, and the operator did not stop it (`paper stop`)    -> ensure the Task Scheduler
                                                                           task "QuantLab paper runner"
                                                                           is running (schtasks /Run);
                                                                           if it has been "running"
                                                                           with no live runner for
                                                                           > STUCK_SECONDS, end + rerun
  * dashboard not answering twice in a row                             -> end + rerun its task
It never starts the bot by spawning processes itself, never writes the database, never touches orders.
Every action is one line in var/logs/watchdog.log (plus an hourly "ok" line).

    .venv\Scripts\python scripts\watchdog.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "var" / "quantlab.db"
LOG = ROOT / "var" / "logs" / "watchdog.log"
STATE = ROOT / "var" / "logs" / "watchdog.state.json"
RUNNER_TASK, DASH_TASK = "QuantLab paper runner", "QuantLab dashboard"
DASH_URL = "http://127.0.0.1:8765/"
HUNG_SECONDS, STUCK_SECONDS = 300.0, 600.0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_alive import _process_image  # noqa: E402  (PID -> image path of a RUNNING process, or None)


def decide(*, session: dict | None, pid_is_python: bool, hb_age: float | None, task_running: bool,
           no_runner_since: float | None) -> tuple[str, str]:
    """Pure decision. Returns (action, why); action in OK, KILL_HUNG, START_TASK, RESTART_TASK, RESPECT_STOP, WAIT."""
    status = (session or {}).get("status")
    if status == "RUNNING" and pid_is_python and hb_age is not None and hb_age < HUNG_SECONDS:
        return "OK", f"runner alive, heartbeat {hb_age:.0f}s"
    if status == "RUNNING" and pid_is_python:
        return "KILL_HUNG", f"runner process alive but heartbeat {('%.0fs' % hb_age) if hb_age is not None else 'missing'} old"
    if status == "STOPPED" and str((session or {}).get("stop_reason") or "").startswith("operator stop") and not task_running:
        return "RESPECT_STOP", "stopped by the operator (quantlab paper stop): not restarting"
    if not task_running:
        return "START_TASK", f"no live runner (last session {status or 'none'}) and the task is not running"
    if no_runner_since is not None and no_runner_since > STUCK_SECONDS:
        return "RESTART_TASK", f"task running but no live runner for {no_runner_since:.0f}s"
    return "WAIT", "task running; supervisor restart in progress"


def _schtasks(*args: str) -> str:
    r = subprocess.run(["schtasks", *args], capture_output=True, text=True, timeout=60)
    return (r.stdout or "") + (r.stderr or "")


def _task_running(name: str) -> bool:
    return '"Running"' in _schtasks("/Query", "/TN", name, "/FO", "CSV", "/NH")


def _log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {line}\n")


def _latest_session() -> dict | None:
    if not DB.exists():
        return None
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT session_id, pid, status, last_heartbeat_at, stop_reason FROM paper_runner_sessions "
                        "ORDER BY started_at DESC LIMIT 1").fetchone()
        return dict(r) if r else None
    finally:
        con.close()


def _dashboard_up() -> bool:
    try:
        with urllib.request.urlopen(DASH_URL, timeout=15) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def main(dry_run: bool = False) -> int:
    now = datetime.now(timezone.utc)
    st = {}
    try:
        st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    except Exception:
        st = {}
    s = _latest_session()
    hb = (now - datetime.fromisoformat(s["last_heartbeat_at"])).total_seconds() if s and s.get("last_heartbeat_at") else None
    img = _process_image(int(s["pid"])) if s and s.get("pid") else None
    is_py = bool(img) and "python" in Path(img).name.lower()
    task_running = _task_running(RUNNER_TASK)
    alive = s is not None and s.get("status") == "RUNNING" and is_py and hb is not None and hb < HUNG_SECONDS
    if alive:
        st.pop("no_runner_since", None)
    elif "no_runner_since" not in st:
        st["no_runner_since"] = now.isoformat()
    since = (now - datetime.fromisoformat(st["no_runner_since"])).total_seconds() if "no_runner_since" in st else None
    action, why = decide(session=s, pid_is_python=is_py, hb_age=hb, task_running=task_running, no_runner_since=since)
    acted = "(dry run)" if dry_run else ""
    if not dry_run:
        if action == "KILL_HUNG":
            subprocess.run(["taskkill", "/PID", str(int(s["pid"])), "/F"], capture_output=True, timeout=30)
        elif action == "START_TASK":
            acted = _schtasks("/Run", "/TN", RUNNER_TASK).strip()[:120]
        elif action == "RESTART_TASK":
            _schtasks("/End", "/TN", RUNNER_TASK)
            acted = _schtasks("/Run", "/TN", RUNNER_TASK).strip()[:120]
            st["no_runner_since"] = now.isoformat()
    # dashboard: two consecutive failures before acting
    dash_ok = _dashboard_up()
    st["dash_fail"] = 0 if dash_ok else int(st.get("dash_fail", 0)) + 1
    if st["dash_fail"] >= 2 and not dry_run:
        _schtasks("/End", "/TN", DASH_TASK)
        _schtasks("/Run", "/TN", DASH_TASK)
        _log(f"DASHBOARD restarted after {st['dash_fail']} failed checks")
        st["dash_fail"] = 0
    if action != "OK" or now.minute < 5 or dry_run:              # actions always, "ok" about hourly
        _log(f"{action}: {why} {acted}".rstrip() + f" | dashboard {'up' if dash_ok else 'DOWN'}")
    STATE.write_text(json.dumps(st), encoding="utf-8")
    print(f"{action}: {why} | dashboard {'up' if dash_ok else 'DOWN'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(main(ap.parse_args().dry_run))
