r"""Exit 0 if a QuantLab paper runner is genuinely alive on this machine, 1 otherwise.

Used by scripts\run_paper.cmd so that a second supervisor (the Windows Startup entry plus a manual
double-click, say) exits instead of retrying a refused start every minute.

"Alive" needs all three, because each alone can mislead:
  * a paper_runner_sessions row with status RUNNING on this host,
  * a heartbeat younger than paper.runner.stale_heartbeat_seconds,
  * and its recorded PID is a running python process.
The PID check matters most. After a crash the dead runner's heartbeat still looks fresh for up to
two minutes, and a check on the database alone would make the supervisor give up on a dead bot.
After a reboot the old PID is gone (or reused by something that is not python).

Read-only: never writes the database, never signals a process. Standard library only.

    .venv\Scripts\python scripts\runner_alive.py        # prints one line, exit code is the answer
"""
from __future__ import annotations

import ctypes
import socket
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "var" / "quantlab.db"
STALE_SECONDS = 120.0            # mirrors paper.runner.stale_heartbeat_seconds (config/default.yaml)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _process_image(pid: int) -> str | None:
    """Full image path of a RUNNING process, or None if it does not exist / has exited."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code)) or code.value != _STILL_ACTIVE:
            return None
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(len(buf))
        if not k32.QueryFullProcessImageNameW(ctypes.c_void_p(h), 0, buf, ctypes.byref(size)):
            return ""
        return buf.value
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))


def alive() -> tuple[bool, str]:
    if not DB.exists():
        return False, "no database yet"
    try:
        con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=10)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT session_id, pid, host, last_heartbeat_at FROM paper_runner_sessions "
                           "WHERE status='RUNNING' ORDER BY started_at DESC").fetchall()
        con.close()
    except sqlite3.Error as exc:
        # unreadable -> do not claim a live runner; the runner's own single-instance guard still applies
        return False, f"database not readable ({exc})"
    now = datetime.now(timezone.utc)
    host = socket.gethostname()
    for r in rows:
        if r["host"] and r["host"].lower() != host.lower():
            continue
        hb = r["last_heartbeat_at"]
        age = (now - datetime.fromisoformat(hb)).total_seconds() if hb else None
        if age is None or age >= STALE_SECONDS:
            continue
        img = _process_image(r["pid"]) if r["pid"] else None
        if img and "python" in Path(img).name.lower():
            return True, f"session {r['session_id']} pid {r['pid']} heartbeat {age:.0f}s ago"
    return False, "no live runner process"


if __name__ == "__main__":
    ok, why = alive()
    print(("ALIVE: " if ok else "NOT ALIVE: ") + why)
    sys.exit(0 if ok else 1)
