"""Marking from historical option bars (UNKNOWN when a leg has no bar, never interpolated), and
the `options` CLI / service path end to end with a fake data client (no network)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.cli import build_parser
from quantlab.options import service
from quantlab.options.book import record_evaluation
from quantlab.options.marking import mark_evaluations, outcomes
from quantlab.options.distribution import MoveDistribution

from .helpers import NOW, contract, quote
from .test_compare import CALLS, EXP, run

LATER = date(2026, 10, 16)       # >= 5 + 5 (default buffer) sessions after 2026-10-01


class FakeClient:
    """Stands in for OptionsDataClient. Bars are hand-made (labelled test data, not market data)."""

    def __init__(self, opt_bars=None, stock_bars=None):
        self.opt_bars = opt_bars or []
        self.stock_bars = stock_bars

    def option_bars(self, symbols, start, end):
        df = pd.DataFrame(self.opt_bars, columns=["symbol", "date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        return df[df["symbol"].isin(symbols)]

    def underlying_daily_bars(self, symbol, start, end, adjustment="all"):
        return self.stock_bars

    def underlying_spot(self, symbol):
        from quantlab.options.data import UnderlyingSpot
        return UnderlyingSpot(symbol, 100.0, "fake", NOW)

    def contracts(self, symbol, expiration_gte=None, expiration_lte=None):
        return [contract(k, exp=LATER) for k, *_ in CALLS]

    def snapshots(self, symbol, expiration_date=None, type_=None):
        assert expiration_date == LATER and type_ == "call"
        return {contract(k, exp=LATER).symbol: quote(contract(k, exp=LATER), b, a, delta=d) for k, b, a, d in CALLS}


def _stock(n=300, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    dates = pd.bdate_range(end="2026-09-30", periods=n)
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1e6})


def test_marks_are_known_only_with_a_bar_and_never_interpolated(db):
    cmp = run(MoveDistribution(np.array([0.20, -0.08])))
    eid = record_evaluation(db, cmp)
    c100, c105 = contract(100, exp=EXP).symbol, contract(105, exp=EXP).symbol
    bars = [(c100, "2026-10-02", 3.5), (c105, "2026-10-02", 1.4),
            (c100, "2026-10-05", 4.0),                                  # c105 did not trade on 10-05
            (c100, "2026-10-06", 2.0), (c105, "2026-10-06", 0.5)]
    stock = pd.DataFrame({"symbol": "TGT", "date": pd.to_datetime(["2026-10-02", "2026-10-05"]), "close": [101.0, 99.0]})
    res = mark_evaluations(db, FakeClient(bars, stock), date(2026, 10, 6))
    assert res["evaluations"] == 1
    spread_id = next(r.expression_id for r in cmp.results if r.kind == "CALL_DEBIT_SPREAD")
    rows = {r["session_date"]: r for r in db.fetchall(
        "SELECT * FROM options_marks WHERE ref_id=? AND expression_id=?", (eid, spread_id))}
    assert rows["2026-10-02"]["status"] == "KNOWN" and rows["2026-10-02"]["value_per_unit"] == pytest.approx(3.5 - 1.4)
    assert rows["2026-10-02"]["pnl_usd"] == pytest.approx((3.5 - 1.4) * 100 - 200.0)   # entry 3.00 - 1.00
    assert rows["2026-10-05"]["status"] == "UNKNOWN" and rows["2026-10-05"]["value_per_unit"] is None
    stock_rows = {r["session_date"]: r for r in db.fetchall(
        "SELECT * FROM options_marks WHERE ref_id=? AND expression_id='STOCK'", (eid,))}
    assert stock_rows["2026-10-05"]["pnl_usd"] == pytest.approx(-1.0) and stock_rows["2026-10-06"]["status"] == "UNKNOWN"
    assert mark_evaluations(db, FakeClient(bars, stock), date(2026, 10, 6))["known"] == 0   # idempotent
    with pytest.raises(Exception, match="append-only"):
        db.execute("UPDATE options_marks SET status='KNOWN'")
    assert any(o["expression_id"] == spread_id for o in outcomes(db))


def test_service_evaluate_records_and_paper_is_refused_by_default(ctx):
    client = FakeClient(stock_bars=_stock())
    res = service.evaluate(ctx, "tgt", 5, "LONG", client=client, now=NOW, paper=True)
    assert res["ok"] and res["expiration"] == "2026-10-16"
    assert res["choice"] in {r["expression"] for r in res["expressions"]} | {"NO_TRADE"}
    assert res["distribution"]["label"] == "EMPIRICAL_UNCONDITIONAL" and res["contracts_checked"] == len(CALLS)
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM options_eval_runs")["n"] == 1
    if res["choice"] not in ("STOCK", "NO_TRADE"):
        assert res["paper"]["ok"] is False and "gated OFF" in res["paper"]["refused"]
    st = service.status(ctx)
    assert st["paper_trading"] is False and st["evaluations_total"] == 1 and st["opt_book"]["positions"] == {}


def test_cli_parser_has_options_command():
    args = build_parser().parse_args(["options", "evaluate", "--symbol", "TGT", "--horizon", "5"])
    assert args.action == "evaluate" and args.direction == "LONG" and args.paper is False
    assert build_parser().parse_args(["options", "status"]).action == "status"
