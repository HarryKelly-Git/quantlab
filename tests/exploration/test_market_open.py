"""Market-open path of the PAPER runner (fake Alpaca, SYNTHETIC data): evening processing plans
exploratory entries without sending them; the pre-open step (overnight refresh -> pre-open recheck
-> exploration submit) runs once in its window; STRICT mode never submits exploration. Plus the
incremental catalyst refresh and the real-money review gate."""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quantlab.context import AppContext
from quantlab.data.ingest import IngestionService
from quantlab.data.providers.synthetic import SyntheticSpec
from quantlab.exploration import create_hypothesis, advance
from quantlab.exploration.review import review_queue
from quantlab.pipeline.runner import PaperRunner

from tests.pipeline.fake_alpaca import PAPER_ENV, FakeAlpacaBroker, FakeStream

ET = ZoneInfo("America/New_York")


def at(d, hh, mm=0) -> datetime:
    return datetime.combine(pd.Timestamp(d).date(), time(hh, mm), ET).astimezone(timezone.utc)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _world(config, mode):
    ctx = AppContext.create(config.with_overrides({"paper": {"mode": mode}}), init_logging=False)
    IngestionService(ctx.config, ctx.store, ctx.db).ingest_synthetic(
        SyntheticSpec(n_stocks=40, start="2017-01-03", end="2019-12-31", seed=5))
    return ctx


def _runner(ctx, broker, clock):
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    streams = []

    def factory(on_event, on_state):
        streams.append(FakeStream(on_event, on_state))
        return streams[-1]
    r = PaperRunner(ctx, broker=broker, stream_factory=factory, now=clock, ingest=lambda c, d: {"skipped": True},
                    bundle_loader=lambda c: b, env=PAPER_ENV, heartbeat_thread=False, allow_synthetic=True)
    r.reconcile_retry_seconds = 0.0
    return r, [x.date() for x in b.panel.dates]


@pytest.mark.parametrize("mode", ["EXPLORATION", "STRICT"])
def test_market_open_path(config, mode):
    ctx = _world(config, mode)
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    d, nxt = sessions[-30], sessions[-29]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(d, 19, 10))
    r, _ = _runner(ctx, broker, clock)
    r.start()
    r.tick()                                                                    # evening: process session D
    job = ctx.db.fetchone("SELECT * FROM paper_session_jobs")
    assert job["status"] == "succeeded"
    planned = ctx.db.fetchall("SELECT * FROM exploration_decisions WHERE session_date=?", (str(d),))
    assert planned                                                              # exploration planned (or SHADOW)
    assert broker.submits == []                                                 # nothing sent at EOD
    clock.t = at(nxt, 8, 0)
    r.tick()
    assert broker.submits == [] and not ctx.db.fetchall("SELECT 1 FROM preopen_checks")   # before the pre-open window
    clock.t = at(nxt, 8, 45)
    r.tick()
    assert ctx.db.fetchall("SELECT 1 FROM preopen_checks")                    # pre-open recheck ran (both modes)
    events = [e["message"] for e in ctx.db.fetchall("SELECT message FROM paper_runner_events")]
    assert any("pre-open for" in m for m in events)
    if mode == "EXPLORATION":
        sel = ctx.db.fetchall("SELECT decision_id FROM exploration_decisions WHERE selection='SELECTED'")
        assert sel and broker.submits                                          # submitted in the window
        sub = {e["decision_id"] for e in ctx.db.fetchall("SELECT decision_id FROM exploration_events WHERE event='SUBMITTED'")}
        assert sub and sub <= {s["decision_id"] for s in sel}
        n = len(broker.submits)
        clock.t = at(nxt, 8, 50)
        r.tick()
        assert len(broker.submits) == n                                        # once: never duplicated
    else:
        assert broker.submits == []                                             # STRICT: exploration never submits
        assert not ctx.db.fetchall("SELECT 1 FROM exploration_decisions WHERE selection='SELECTED'")
    clock.t = at(nxt, 9, 40)
    before = len(broker.submits)
    r.tick()
    assert len(broker.submits) == before                                       # after the cutoff: nothing new
    r.shutdown("test")
    ctx.close()


def test_catalyst_refresh_resumes_dedupes_and_never_raises(ctx, monkeypatch):
    from quantlab.data import catalyst_refresh as cr
    from quantlab.data.providers import alpaca_data, sec_edgar
    from quantlab.data import sec_catalysts
    calls = []
    t = pd.Timestamp("2024-03-01T15:00:00Z")

    def fake_news(self, start, end, keep_summary=False):
        calls.append(("news", pd.Timestamp(start)))
        return pd.DataFrame([{"news_id": "n1", "symbol": "AAA", "headline": "AAA Wins Contract", "summary": "",
                              "source": "benzinga", "url": "", "created_at": t, "updated_at": t, "available_at": t,
                              "pit_status": "PIT", "provider": "alpaca", "retrieved_at": pd.Timestamp.now(tz="UTC")}])

    def fake_run(self, symbols, workers=1):
        calls.append(("sec", len(list(symbols))))
        raise RuntimeError("SEC unreachable")                                   # a failure must be reported, not raised
    monkeypatch.setattr(alpaca_data.AlpacaDataProvider, "get_news_market", fake_news)
    monkeypatch.setattr(sec_catalysts.SecCatalystIngest, "run", fake_run)
    monkeypatch.setattr(sec_catalysts, "market_calendar", lambda store, m="SPY": None)
    now = pd.Timestamp("2024-03-02T12:00:00Z")
    r1 = cr.refresh_catalysts(ctx, pd.Timestamp("2024-03-01"), symbols=["AAA"], now=now)
    assert r1["news"]["ok"] and r1["news"]["rows"] == 1 and r1["sec"]["ok"] is False and "unreachable" in r1["sec"]["error"]
    r2 = cr.refresh_catalysts(ctx, pd.Timestamp("2024-03-01"), symbols=["AAA"], parts=("news",),
                              now=now + pd.Timedelta(hours=6))
    assert calls[-1][0] == "news" and calls[-1][1] == now - pd.Timedelta(days=1)   # resumes at last end - 1 day overlap
    assert len(ctx.store.load("news")) == 1                                     # the re-fetched article is not duplicated
    _ = (r2, sec_edgar)


def test_review_queue_is_empty_until_the_full_ladder_and_forward_record(ctx):
    rv = review_queue(ctx.db, ctx.config)
    assert rv["queue"] == [] and "full validation ladder" in rv["why_empty"]
    hid = create_hypothesis(ctx.db, "post-earnings + volume", {"families": ["post_earnings"]}, "human:harry", "n=40 outcomes")
    for stage in ("HISTORICAL_PIT_TEST", "WALK_FORWARD", "LOCKED_HOLDOUT", "PROSPECTIVE_PAPER", "STRICT_ELIGIBLE"):
        advance(ctx.db, hid, stage, "human:harry", f"evidence for {stage}", n_observations=40)
    rv = review_queue(ctx.db, ctx.config)
    assert rv["queue"] == [] and "closed paper trade" in rv["why_empty"]         # a validated pattern alone is not enough
    assert "not advice" in rv["note"] and "Upside Engine" in rv["note"]


def test_failing_catalyst_refresh_never_blocks_the_session_and_restart_never_duplicates(config):
    ctx = _world(config, "EXPLORATION")
    b = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    sessions = [x.date() for x in b.panel.dates]
    d, nxt = sessions[-30], sessions[-29]
    broker = FakeAlpacaBroker(sessions)
    clock = Clock(at(d, 19, 10))

    def boom(*a, **k):
        raise RuntimeError("SEC and Alpaca news unreachable")

    def runner():
        streams = []

        def factory(on_event, on_state):
            streams.append(FakeStream(on_event, on_state))
            return streams[-1]
        r = PaperRunner(ctx, broker=broker, stream_factory=factory, now=clock, ingest=lambda c, x: {"skipped": True},
                        bundle_loader=lambda c: b, env=PAPER_ENV, heartbeat_thread=False, allow_synthetic=True,
                        catalyst_refresh=boom)
        r.reconcile_retry_seconds = 0.0
        return r
    r = runner()
    r.start()
    r.tick()
    assert ctx.db.fetchone("SELECT status FROM paper_session_jobs")["status"] == "succeeded"   # refresh failure != crash
    msgs = [e["message"] for e in ctx.db.fetchall("SELECT message FROM paper_runner_events")]
    assert any("catalyst refresh" in m and "failed" in m for m in msgs)
    clock.t = at(nxt, 8, 45)
    r.tick()
    n_orders, n_sub = ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"], len(broker.submits)
    counts = {t: ctx.db.fetchone(f"SELECT COUNT(*) AS n FROM {t}")["n"]
              for t in ("decisions", "candidates", "exploration_decisions", "fills", "trades")}
    r.shutdown("restart test")
    r2 = runner()                                                               # restart inside the pre-open window
    r2.start()
    clock.t = at(nxt, 8, 55)
    r2.tick()
    r2.tick()
    assert len(broker.submits) == n_sub and ctx.db.fetchone("SELECT COUNT(*) AS n FROM orders")["n"] == n_orders
    assert {t: ctx.db.fetchone(f"SELECT COUNT(*) AS n FROM {t}")["n"] for t in counts} == counts
    r2.shutdown("test")
    ctx.close()
