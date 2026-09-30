r"""Record what happened at the open, independently of any tool session.

Polls the paper database from before the pre-open submit window until after the opening
auction and appends a timestamped report to var/logs/open-report.log. Read-only: it never
submits, cancels or changes anything.

    .venv\Scripts\python scripts\watch_open.py [--until-utc HH:MM]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "var" / "logs" / "open-report.log"


def q(db: sqlite3.Connection, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    try:
        return db.execute(sql, args).fetchall()
    except sqlite3.Error as exc:                       # a concurrent writer must never kill the watch
        return [{"error": str(exc)}]                   # type: ignore[list-item]


def snapshot(day: str) -> str:
    db = sqlite3.connect(f"file:{(ROOT / 'var' / 'quantlab.db').as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    out = [f"=== {datetime.now(timezone.utc).isoformat(timespec='seconds')} ==="]
    orders = q(db, "SELECT symbol, qty, purpose, order_type, time_in_force, limit_price, status, "
                   "filled_qty, filled_avg_price, created_at FROM orders WHERE created_at >= ? "
                   "ORDER BY created_at", (day,))
    out.append(f"orders today: {len(orders)}")
    for r in orders:
        d = dict(r)
        out.append("  " + " ".join(f"{k}={d[k]}" for k in d if d[k] is not None))
    trades = q(db, "SELECT symbol, qty, status, entry_date, entry_price, stop_price FROM trades "
                   "ORDER BY rowid DESC LIMIT 10")
    out.append(f"trades: {len(trades)}")
    for r in trades:
        out.append("  " + str(dict(r)))
    for r in q(db, "SELECT event, COUNT(*) n FROM exploration_events WHERE at >= ? GROUP BY event", (day,)):
        out.append(f"  exploration_event {dict(r)}")
    for r in q(db, "SELECT at, level, kind, message FROM paper_runner_events WHERE at >= ? "
                   "AND (kind IN ('order','reconcile','stream','error') OR level != 'INFO') "
                   "ORDER BY at DESC LIMIT 12", (day,)):
        out.append(f"  event {r['at'][11:19]} {r['level']} {r['kind']}: {str(r['message'])[:160]}")
    sess = q(db, "SELECT status, phase, stream_status, last_heartbeat_at FROM paper_runner_sessions "
                 "ORDER BY rowid DESC LIMIT 1")
    out.append(f"  runner {dict(sess[0]) if sess else 'none'}")
    db.close()
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until-utc", default="13:55", help="stop after this UTC time (HH:MM)")
    ap.add_argument("--every", type=int, default=120, help="seconds between snapshots")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).date().isoformat()
    stop_h, stop_m = (int(x) for x in args.until_utc.split(":"))
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n##### watch started {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                 f"(until {args.until_utc}Z)\n")
        fh.flush()
        while True:
            now = datetime.now(timezone.utc)
            try:
                fh.write(snapshot(day))
            except Exception as exc:                   # never die silently
                fh.write(f"!!! snapshot failed: {exc!r}\n")
            fh.flush()
            if (now.hour, now.minute) >= (stop_h, stop_m) or now.date().isoformat() != day:
                fh.write("##### watch finished\n")
                fh.flush()
                return 0
            time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
