"""Congress / insider context reaches every exploration decision's pre-trade record, and the layer has
ZERO effect on which trades are made or how: the same world with and without disclosures produces
identical strategy decisions, discovery scores, exploration selections, sizes, stops and orders.
All data is SYNTHETIC."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.data import schemas
from quantlab.data.ingest import IngestionService
from quantlab.data.providers.synthetic import SyntheticMarket, SyntheticSpec
from quantlab.exploration.engine import smart_money_record
from quantlab.pipeline.daily import DailyPipeline

ROOT = Path(__file__).resolve().parents[2]
SPEC = SyntheticSpec(n_stocks=40, start="2018-01-02", end="2020-06-30", seed=7)
N_SESSIONS = 3


def _world(tmp, name, with_alt: bool):
    var = tmp / name
    cfg = load_config(root=ROOT, overrides={"paper": {"mode": "EXPLORATION"}, "project": {
        "var_dir": str(var), "db_path": str(var / "q.db"), "data_dir": str(var / "data"), "log_dir": str(var / "logs"),
        "report_dir": str(var / "r"), "model_dir": str(var / "m")}})
    ctx = AppContext.create(cfg, init_logging=False)
    with pytest.MonkeyPatch.context() as mp:
        if not with_alt:
            mp.setattr(SyntheticMarket, "_alt_trades", lambda self, *a: schemas.empty("alt_trades"))
        IngestionService(cfg, ctx.store, ctx.db).ingest_synthetic(SPEC)
    pipe = DailyPipeline(ctx, synthetic=True)
    dates = pipe.full_bundle().panel.dates
    for d in dates[-40:-40 + N_SESSIONS]:
        r = pipe.run(d)
        assert not r.errors, r.errors
    return ctx


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smart_money")
    a, b = _world(tmp, "with", True), _world(tmp, "without", False)
    yield a, b
    a.close()
    b.close()


def test_every_exploration_decision_records_the_context(pair):
    ctx, _ = pair
    assert ctx.store.dataset_ids("alt_trades", synthetic=True)
    rows = ctx.db.fetchall("SELECT d.symbol, d.selection, d.pre_trade_json, c.catalyst_json FROM exploration_decisions d "
                           "JOIN discovery_candidates c ON c.discovery_id = d.discovery_id")
    assert rows
    states = []
    for r in rows:
        sm = json.loads(r["pre_trade_json"])["smart_money"]
        assert sm["state"] in ("KNOWN", "UNKNOWN") and "never scored" in sm["role"]
        disc = (json.loads(r["catalyst_json"]) or {}).get("smart_money") or {}
        if sm["state"] == "KNOWN":
            assert disc["state"] == "KNOWN" and sm["values"] == disc["values"]       # copied from the scan, not recomputed
            assert set(sm["values"]) == {"congress_buys_30d", "congress_sells_30d", "congress_net_30d",
                                         "insider_buys_30d", "insider_sells_30d", "insider_net_value_30d"}
        else:
            assert "values" not in sm and sm["why"]                                  # UNKNOWN, never zero
        states.append(sm["state"])
    assert "KNOWN" in states                                                         # the synthetic feed covers some


def test_without_a_source_every_record_is_unknown(pair):
    _, ctx = pair
    rows = ctx.db.fetchall("SELECT pre_trade_json FROM exploration_decisions")
    assert rows and all(json.loads(r["pre_trade_json"])["smart_money"]["state"] == "UNKNOWN" for r in rows)


def test_zero_effect_on_decisions_selection_sizing_and_orders(pair):
    a, b = pair
    q = {
        "decisions": "SELECT c.symbol, c.as_of_date, c.strategy_id, d.decision, d.reject_stage FROM decisions d "
                     "JOIN candidates c ON c.candidate_id = d.candidate_id ORDER BY 1, 2, 3",
        "discovery": "SELECT as_of_date, symbol, discovery_score, rank, status, on_watchlist, setup_json "
                     "FROM discovery_candidates ORDER BY 1, 2",
        "exploration": "SELECT session_date, symbol, selection, rank, qty, ref_price, stop_price, holding_sessions "
                       "FROM exploration_decisions ORDER BY 1, 2",
        "orders": "SELECT symbol, side, qty, purpose, status FROM orders ORDER BY created_at, symbol",
        "trades": "SELECT symbol, entry_date, qty, entry_price, status FROM trades ORDER BY 1, 2",
    }
    for name, sql in q.items():
        ra = [tuple(r) for r in a.db.fetchall(sql)]
        rb = [tuple(r) for r in b.db.fetchall(sql)]
        if name == "discovery":                          # the setup record lists UNKNOWN context: compare scores only
            ra, rb = [r[:-1] for r in ra], [r[:-1] for r in rb]
        assert ra == rb, name
    assert a.db.fetchall("SELECT 1 FROM exploration_decisions WHERE selection='SELECTED'")


def test_record_helper_never_invents_values():
    assert smart_money_record({"catalyst_json": None})["state"] == "UNKNOWN"
    assert smart_money_record({})["state"] == "UNKNOWN"
    rec = smart_money_record({"catalyst_json": json.dumps({"smart_money": {"state": "UNKNOWN", "why": "no feed"}})})
    assert rec == {"state": "UNKNOWN", "why": "no feed", "role": rec["role"]}
    rec = smart_money_record({"catalyst_json": json.dumps({"smart_money": {
        "state": "KNOWN", "fired": True, "reasons": ["x"], "values": {"congress_buys_30d": 1.0}}})})
    assert rec["state"] == "KNOWN" and rec["any_purchase_disclosed"] and rec["values"] == {"congress_buys_30d": 1.0}
