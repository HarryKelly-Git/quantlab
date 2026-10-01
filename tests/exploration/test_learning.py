"""Upside profiles on every decision, and the learning report that scores them against outcomes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from quantlab.db.database import utcnow_iso
from quantlab.discovery import run_discovery
from quantlab.exploration import plan_exploration
from quantlab.exploration.engine import learning_report
from quantlab.exploration.upside import _table, move_distribution, upside_profile

from .test_sanity import PRE_OPEN, SESSION, _ctx, _decision, _store
from tests.discovery.world import crafted_bundle


def test_profile_reads_the_table_and_unknown_stays_unknown():
    meta = _table()["meta"]
    assert meta["holds"] == [5, 10, 20] and meta["n_obs"] > 100_000
    p = upside_profile(0.03, 0.6, 20)
    assert p["state"] == "KNOWN" and p["hold"] == 20 and 0 < p["p_touch_10"] < 1
    assert p["p_clean_10"] <= p["p_touch_10"]                      # before-the-stop can only be rarer
    for bad in ((None, 0.6, 20), (0.03, None, 20), (0.03, 0.6, 7), (float("nan"), 0.6, 10), (-0.01, 0.5, 5)):
        assert upside_profile(*bad)["state"] == "UNKNOWN", bad
    assert upside_profile({"value": 0.03}, {"value": 0.6}, 10)["state"] == "KNOWN"    # factor-record form


def test_more_volatile_means_more_big_moves_both_ways():
    lo, hi = upside_profile(0.012, 0.6, 20), upside_profile(0.08, 0.6, 20)
    assert hi["p_touch_10"] > lo["p_touch_10"]
    assert hi["p_down_first_10"] > lo["p_down_first_10"]            # size, not direction
    assert upside_profile(0.03, 0.6, 20)["p_touch_10"] > upside_profile(0.03, 0.6, 5)["p_touch_10"]


def test_move_distribution_for_the_options_comparator():
    d = move_distribution(0.03, 0.6, 10)
    assert d is not None and len(d) == 21 and np.all(np.diff(d) >= 0)
    assert move_distribution(None, 0.6, 10) is None


def test_every_decision_carries_an_upside_profile_for_its_own_hold(tmp_path):
    ctx = _ctx(tmp_path)
    cb = crafted_bundle()
    _store(ctx)
    run_discovery(ctx, cb, cb.panel.dates[-1], links={})
    plan_exploration(ctx, equity=100_000.0, now=PRE_OPEN)
    rows = ctx.db.fetchall("SELECT holding_sessions, pre_trade_json FROM exploration_decisions WHERE session_date=?",
                           (SESSION,))
    assert rows
    for r in rows:
        up = json.loads(r["pre_trade_json"])["upside"]
        assert up["state"] in ("KNOWN", "UNKNOWN")
        if up["state"] == "KNOWN":
            assert up["hold"] == r["holding_sessions"]
    assert any(json.loads(r["pre_trade_json"])["upside"]["state"] == "KNOWN" for r in rows)
    ctx.close()


def _outcome(ctx, did, hold, net, mfe, mae, stopped):
    ctx.db.insert("exploration_outcomes", {
        "decision_id": did, "horizon_sessions": hold, "entry_date": "2024-03-25", "end_date": "2024-04-20",
        "entry_price": 50.0, "gross_ret": net + 0.002, "cost_ret": 0.002, "net_ret": net, "spy_ret": 0.01, "mfe": mfe,
        "mae": mae, "stop_breached": int(stopped), "thesis_valid": 1, "catalyst_persisted": None,
        "computed_at": utcnow_iso()})


def test_learning_report_finds_missed_winners_failed_picks_and_calibrates(tmp_path):
    ctx = _ctx(tmp_path)
    up = upside_profile(0.03, 0.6, 10)
    sel = _decision(ctx, symbol="PICK", pre_trade={"upside": up, "candidate": {"selection_score": 0.9}})
    _outcome(ctx, sel, 10, -0.06, 0.01, -0.07, True)                    # traded, stopped out
    ctx.db.insert("exploration_decisions", {**dict(ctx.db.fetchone("SELECT * FROM exploration_decisions WHERE decision_id=?",
                                                                     (sel,))),
                                            "decision_id": "expl_MISS", "symbol": "MISS", "selection": "WATCHED_NOT_TRADED",
                                            "reason": "eligible but beyond the session budget", "qty": 0})
    _outcome(ctx, "expl_MISS", 10, 0.11, 0.14, -0.01, False)            # passed over, then +14%
    rep = learning_report(ctx.db, synthetic=True)
    assert rep["matured"] == 2
    assert [m["symbol"] for m in rep["missed_big_winners"]] == ["MISS"]
    assert [f["symbol"] for f in rep["failed_picks"]] == ["PICK"]
    groups = {(g["selection"], g["hold"]): g for g in rep["by_selection_and_hold"]}
    assert groups[("SELECTED", 10)]["stop_rate"] == 1.0 and groups[("WATCHED_NOT_TRADED", 10)]["p_up_10"] == 1.0
    assert rep["calibration"] and all(0 <= c["realized"] <= 1 for c in rep["calibration"])
    json.dumps(rep)                                                     # printable by the CLI
    ctx.close()


def test_learning_report_with_nothing_matured(tmp_path):
    ctx = _ctx(tmp_path)
    assert learning_report(ctx.db, synthetic=True)["matured"] == 0
    ctx.close()
