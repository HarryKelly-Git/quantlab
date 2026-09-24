"""IngestionService with injected fake providers (offline)."""
from __future__ import annotations

import pandas as pd

from quantlab.data.ingest import IngestionService
from quantlab.data.providers.base import ProviderNotConfigured
from quantlab.data.providers.synthetic import SyntheticMarket, SyntheticSpec


class _NoKeys:
    name = "alpaca"
    def get_news(self, *a, **k):
        raise ProviderNotConfigured("ALPACA_PAPER_KEY_ID not set")


def test_ingest_all_with_synthetic_providers_and_a_missing_key(ctx):
    mkt = SyntheticMarket(SyntheticSpec(n_stocks=12, start="2019-01-02", end="2020-06-30", seed=9))
    providers = {"prices": mkt, "corporate_actions": mkt, "reference": mkt, "events": mkt,
                 "fundamentals": mkt, "news": _NoKeys()}
    svc = IngestionService(ctx.config, ctx.store, ctx.db, providers=providers)
    rep = svc.ingest_all("2019-01-02", "2020-06-30", run_id="r1", bar_chunk=5)
    out = rep.summary()
    assert out["bars"]["status"] == "ok" and out["bars"]["datasets"] >= 3      # chunked
    assert out["news"]["status"] == "skipped" and "not configured" in out["news"]["detail"]
    assert rep.ok
    issue = ctx.db.fetchone("SELECT * FROM data_quality_issues WHERE check_name='ingest_news'")
    assert issue is not None
    bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), synthetic=True)
    assert "SPY" in bundle.panel.symbols and bundle.is_synthetic
    ev = ctx.store.load("events")
    assert pd.to_datetime(ev["reaction_date"]).notna().all()


def test_select_symbols_keeps_common_and_benchmarks(ctx):
    mkt = SyntheticMarket(SyntheticSpec(n_stocks=5, start="2019-01-02", end="2020-06-30"))
    svc = IngestionService(ctx.config, ctx.store, ctx.db, providers={})
    syms = svc.select_symbols(mkt.get_securities())
    assert "SYN000" in syms and "SPY" in syms and "SYNPA" not in syms and "SYNTT" not in syms
