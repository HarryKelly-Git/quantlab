"""Phase 3: HOLDOUT_LOCK refuses and logs every attempt, and the bot-database importer validates the
schema and only reports (no verdict on a handful of dates)."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import botdb, holdout, splits


@pytest.fixture()
def lock(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "LOG", tmp_path / "holdout_access.jsonl")
    monkeypatch.setattr(holdout, "LOCK", tmp_path / "HOLDOUT_LOCK.json")
    return tmp_path


def test_holdout_lock_refuses_and_logs(lock):
    s = pd.Series(1.0, index=pd.bdate_range("2024-12-20", "2025-01-10"))
    with pytest.raises(holdout.HoldoutAccessError):
        splits.slice_split(s, "equity", "HOLDOUT")
    with pytest.raises(holdout.HoldoutAccessError):
        holdout.guard_dates(s.index.max(), "test.guard")
    holdout.guard_dates(pd.Timestamp("2024-12-31"), "fine")            # pre-holdout: allowed, not logged
    rows = [json.loads(x) for x in (lock / "holdout_access.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["REFUSED", "REFUSED"]
    assert "slice_split" in rows[0]["context"]
    with pytest.raises(holdout.HoldoutAccessError):                     # no committed pre-registration
        with holdout.unlock("HEAD", "docs/NO-SUCH-PREREG.md", "test"):
            pass
    assert json.loads((lock / "holdout_access.jsonl").read_text().splitlines()[-1])["event"] == "UNLOCK_REFUSED"


def _db(path, n_dates=3, seed=0):
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("create table shadow_opportunities (opportunity_id text primary key, candidate_id text, as_of_date text, symbol text, "
                "strategy_id text, strategy_version text, score real, bot_decision text, reject_stage text, reject_reason text, "
                "created_at text, is_synthetic integer)")
    con.execute("create table shadow_outcomes (opportunity_id text, horizon_sessions integer, measured_at text, ret real, ret_hold real, "
                "mfe real, mae real, hit_stop integer, hit_target integer, benchmark_ret real, excess_ret real, status text)")
    dates = pd.bdate_range("2026-09-24", periods=n_dates).strftime("%Y-%m-%d")
    k = 0
    for d in dates:
        for i in range(30):
            k += 1
            dec = "TRADE" if i < 3 else "NO_TRADE"
            con.execute("insert into shadow_opportunities values (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"o{k}", f"c{k}", d, f"S{i}", "s0", "v1", float(rng.normal()), dec, "NONE" if dec == "TRADE" else "RISK",
                         None if dec == "TRADE" else "score_below_threshold", d, 0))
            for h in (5, 20):
                r = float(rng.normal(0, 0.05))
                con.execute("insert into shadow_outcomes values (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (f"o{k}", h, d, r, r, abs(r) + 0.01, -abs(r) - 0.01, 0, 0, 0.0, r, "complete"))
    con.commit()
    con.close()


def test_botdb_import_validates_and_reports_only(tmp_path):
    db = tmp_path / "export.db"
    _db(db)
    res = botdb.run(db, tmp_path / "out")
    assert res["quality"]["status"] == "PASS" and res["quality"]["distinct_dates"] == 3
    h5 = res["analysis"]["by_horizon"]["5"]
    assert h5["traded"]["n"] == 9 and h5["rejected"]["n"] == 81
    assert h5["too_conservative_verdict"].startswith("INSUFFICIENT EVIDENCE")
    bad = tmp_path / "bad.db"
    sqlite3.connect(bad).execute("create table orders (id text)").connection.commit()
    with pytest.raises(botdb.SchemaError):
        botdb.load(bad)
