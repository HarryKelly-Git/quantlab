"""HOLDOUT_LOCK (Phase 3, Part 3): the 2025+ historical market data is SEALED.

Every code path that could touch holdout dates calls ``refuse(context)`` (or ``guard_dates``):
- the research-store downloads and loads;
- the panel build;
- the options export and features;
- ``splits.slice_split("HOLDOUT")``.

Each attempt is appended to ``research/alpha/holdout_access.jsonl``, a git-tracked, append-only log of
time, context, caller and git commit, and then refused.

The only way in is ``unlock(prereg_commit, spec_path, reason)``. It is a context manager that needs:
- a pre-registration file committed BEFORE the unlock (``spec_path`` must exist at ``prereg_commit``
  and contain the line ``HOLDOUT EVALUATION``);
- remaining budget in ``research/alpha/HOLDOUT_LOCK.json`` (``max_evaluations``, default 1).

Unlock, every access inside it and the relock are all logged. Exploratory code, notebooks, selection,
ranking and tuning have no route to the holdout.

Out of scope: the paper bot's own forward records (``alpha.botdb``). They are decisions the bot already
made live, read from its database, and they never load holdout-period market data from the research
store.
"""
from __future__ import annotations

import json
import subprocess
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NoReturn

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "research" / "alpha" / "HOLDOUT_LOCK.json"
LOG = ROOT / "research" / "alpha" / "holdout_access.jsonl"
HOLDOUT_START = "2025-01-01"
_state = {"unlocked": False, "token": None}


class HoldoutAccessError(RuntimeError):
    pass


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:          # pragma: no cover
        return ""


def _log(event: str, context: str, **extra) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    caller = [f"{f.filename.split('/src/')[-1]}:{f.lineno}:{f.name}" for f in traceback.extract_stack()[-6:-2]]
    row = {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "event": event, "context": context, "caller": caller,
           "git": _git("rev-parse", "--short", "HEAD"), **extra}
    with LOG.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def refuse(context: str) -> NoReturn | None:
    """Log the attempt; refuse unless inside a registered ``unlock`` block."""
    if _state["unlocked"]:
        _log("ACCESS_UNDER_UNLOCK", context, token=_state["token"])
        return None
    _log("REFUSED", context)
    raise HoldoutAccessError(f"HOLDOUT_LOCK: {context} would touch {HOLDOUT_START}+ data, which is sealed "
                             "(see src/quantlab/alpha/holdout.py; attempt logged)")


def guard_dates(max_date, context: str) -> None:
    if max_date is not None and pd.notna(max_date) and pd.Timestamp(max_date) >= pd.Timestamp(HOLDOUT_START):
        refuse(context)


def lock_state() -> dict:
    return json.loads(LOCK.read_text()) if LOCK.exists() else {"sealed": True, "max_evaluations": 1, "evaluations_used": 0}


@contextmanager
def unlock(prereg_commit: str, spec_path: str, reason: str) -> Iterator[str]:
    st = lock_state()
    ok_commit = bool(_git("cat-file", "-e", f"{prereg_commit}:{spec_path}") == "" and
                     "HOLDOUT EVALUATION" in _git("show", f"{prereg_commit}:{spec_path}"))
    if not ok_commit:
        _log("UNLOCK_REFUSED", f"no committed pre-registration {spec_path}@{prereg_commit}", reason=reason)
        raise HoldoutAccessError("unlock needs a committed pre-registration containing 'HOLDOUT EVALUATION'")
    if st.get("evaluations_used", 0) >= st.get("max_evaluations", 1):
        _log("UNLOCK_REFUSED", "evaluation budget spent", reason=reason)
        raise HoldoutAccessError("the holdout evaluation budget is spent")
    token = f"{prereg_commit}:{spec_path}"
    _log("UNLOCK", reason, token=token)
    _state.update(unlocked=True, token=token)
    try:
        yield token
    finally:
        _state.update(unlocked=False, token=None)
        st["evaluations_used"] = st.get("evaluations_used", 0) + 1
        st.setdefault("evaluations", []).append({"token": token, "reason": reason, "ts": pd.Timestamp.now(tz="UTC").isoformat()})
        LOCK.write_text(json.dumps(st, indent=1))
        _log("RELOCK", reason, token=token)
