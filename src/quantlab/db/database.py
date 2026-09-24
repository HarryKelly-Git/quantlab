"""SQLite persistence: connection management, migrations and small typed helpers.

SQLite (WAL mode) is deliberate: one file, zero ops, concurrent readers (dashboard) with a single
writer (pipeline). Bulk market data lives in Parquet (see :mod:`quantlab.data.store`); this database
holds the relational audit trail: runs, candidates, decisions, orders, trades, experiments, ledger.

Migrations are plain ``NNN_name.sql`` files in ``db/migrations`` applied in order exactly once.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (datetime, date, pd.Timestamp)):
        return o.isoformat()
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    return str(o)


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, sort_keys=True)


def from_json(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


def _adapt(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return to_json(value)
    return value


class Database:
    """Thin wrapper around a sqlite3 connection in autocommit mode with explicit transactions."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._tx_depth = 0

    # -- lifecycle ------------------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        """Nestable transaction (inner levels use SAVEPOINTs)."""
        if self._tx_depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        else:
            self.conn.execute(f"SAVEPOINT sp{self._tx_depth}")
        self._tx_depth += 1
        try:
            yield self
        except BaseException:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO sp{self._tx_depth}")
                self.conn.execute(f"RELEASE sp{self._tx_depth}")
            raise
        else:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute(f"RELEASE sp{self._tx_depth}")

    # -- migrations -----------------------------------------------------------------------------
    def migrate(self, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {r["version"] for r in self.conn.execute("SELECT version FROM schema_migrations")}
        newly: list[str] = []
        for f in sorted(migrations_dir.glob("*.sql")):
            version = f.stem
            if version in applied:
                continue
            sql = f.read_text(encoding="utf-8")
            with self.transaction():
                # executescript would COMMIT implicitly; run statement-by-statement instead.
                for stmt in _split_sql(sql):
                    self.conn.execute(stmt)
                self.conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, utcnow_iso())
                )
            newly.append(version)
        return newly

    # -- helpers --------------------------------------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def insert(self, table: str, row: dict[str, Any], or_ignore: bool = False) -> None:
        cols = list(row.keys())
        verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
        sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        self.conn.execute(sql, [_adapt(row[c]) for c in cols])

    def insert_many(self, table: str, rows: list[dict[str, Any]], or_ignore: bool = False) -> int:
        if not rows:
            return 0
        cols = list(rows[0].keys())
        verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
        sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        with self.transaction():
            self.conn.executemany(sql, [[_adapt(r.get(c)) for c in cols] for r in rows])
        return len(rows)

    def upsert(self, table: str, row: dict[str, Any], key_cols: list[str]) -> None:
        cols = list(row.keys())
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in key_cols)
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT ({', '.join(key_cols)}) DO UPDATE SET {updates}"
        )
        self.conn.execute(sql, [_adapt(row[c]) for c in cols])

    def fetchone(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> dict[str, Any] | None:
        r = self.conn.execute(sql, params).fetchone()
        return dict(r) if r is not None else None

    def fetchall(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def query_df(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> pd.DataFrame:
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=cols)

    def tables(self) -> list[str]:
        return [r["name"] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def _split_sql(sql: str) -> list[str]:
    """Split a migration into statements, keeping CREATE TRIGGER ... BEGIN ... END; intact."""
    statements: list[str] = []
    buf: list[str] = []
    in_trigger = False
    for line in sql.splitlines():
        stripped = line.strip()
        if not buf and (not stripped or stripped.startswith("--")):
            continue
        buf.append(line)
        upper = stripped.upper()
        if upper.startswith("CREATE TRIGGER"):
            in_trigger = True
        if in_trigger:
            if upper.endswith("END;"):
                statements.append("\n".join(buf))
                buf, in_trigger = [], False
        elif stripped.endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    if buf and "".join(buf).strip():
        statements.append("\n".join(buf))
    return statements


def open_db(path: str | Path, migrate: bool = True) -> Database:
    db = Database(path)
    if migrate:
        db.migrate()
    return db


def git_info(root: str | Path) -> tuple[str | None, bool | None]:
    """(commit sha, dirty flag) of the repository at root; (None, None) if unavailable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if sha.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=10
        )
        return sha.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None
