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


def test_botdb_filter_ablation_readmits_only_single_filter_rejections(tmp_path):
    """P1: 'remove filter X' re-admits rejected candidates whose ONLY blocking failures are X, re-checks EV
    for candidates the risk chain stopped earlier, and labels the result exploratory under 20 dates."""
    db = tmp_path / "abl.db"
    _db(db, n_dates=3)
    con = sqlite3.connect(db)
    con.execute("create table risk_checks (id integer primary key autoincrement, candidate_id text, decision_id text, "
                "check_name text, passed integer, severity text, reason text, details_json text, created_at text)")
    con.execute("create table decisions (decision_id text, candidate_id text, decision text, reject_stage text, ev_json text)")
    # candidate c{k}: per date, i=0..2 traded; i=3 liquidity only (EV 20bp); i=4 liquidity+volatility; i=5 volatility only
    # (EV 5bp: fails the 10bp EV re-check, passes the relaxed 0bp one); i=6 portfolio capacity; the rest score_below.
    rows = con.execute("select candidate_id, symbol from shadow_opportunities").fetchall()
    for cid, sym in rows:
        i = int(sym[1:])
        fails = {3: [("no_trade.liquidity", {"adv20": 3e6, "min": 5e6})],
                 4: [("no_trade.liquidity", {"adv20": 1e6, "min": 5e6}), ("no_trade.volatility", {"vol_20d": 2.0, "max": 1.2})],
                 5: [("no_trade.volatility", {"vol_20d": 1.4, "max": 1.2})],
                 6: [("risk.portfolio", {"rejection_reason": "max_positions"})]}.get(i, [] if i < 3 else [("risk.ev", {"ev_bps": -5.0, "min_ev_bps": 10.0})])
        for name, det in fails:
            con.execute("insert into risk_checks (candidate_id, check_name, passed, severity, details_json) values (?,?,?,?,?)",
                        (cid, name, 0, "CRITICAL", json.dumps(det)))
            con.execute("insert into risk_checks (candidate_id, check_name, passed, severity, details_json) values (?,?,?,?,?)",
                        (cid, name, 0, "CRITICAL", json.dumps(det)))           # the pipeline records no-trade checks twice
        ev = {3: 0.0020, 4: 0.0020, 5: 0.0005, 6: 0.0030}.get(i, 0.0030 if i < 3 else -0.0005)
        con.execute("insert into decisions values (?,?,?,?,?)", (f"d{cid}", cid, "x", "x", json.dumps({"ev": ev})))
    con.commit(); con.close()
    res = botdb.ablation(botdb.load(db))
    sc = res["scenarios"]
    assert res["status"].startswith("EXPLORATORY") and res["blocking_source"] == "risk_checks"
    assert sc["A_current"]["n_added"] == 0
    assert sc["B_remove_liquidity"]["n_added"] == 3                    # i=3 on each date; i=4 still fails volatility
    assert sc["C_remove_volatility"]["n_added"] == 0                   # i=5 fails the EV re-check (5bp <= 10bp)
    assert sc["E_remove_risk.portfolio"]["n_added"] == 3
    assert sc["F_relaxed_thresholds"]["n_added"] == 6                  # i=3 and i=5 per date; i=4 too far; EV < 0 never
    assert "NOT IDENTIFIABLE" in res["D_remove_score_threshold"]
    h5 = sc["B_remove_liquidity"]["by_horizon"]["5"]
    assert h5["added"]["n"] == 3 and h5["traded_now"]["n"] == 9 and h5["book_after"]["n"] == 12
