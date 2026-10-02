"""Reusable builders for tests (and offline demos). Everything here is SYNTHETIC."""
from __future__ import annotations

from quantlab.core.calendar import TradingCalendar
from quantlab.data.panel import DataBundle, build_panel
from quantlab.data.providers.synthetic import SECTOR_ETFS, SECTOR_NAMES, SyntheticMarket, SyntheticSpec

SYNTHETIC_BENCHMARKS = {"market": "SPY", "sectors": {e: SECTOR_NAMES[e] for e in SECTOR_ETFS}}


def make_synthetic_bundle(spec: SyntheticSpec | None = None, market: SyntheticMarket | None = None) -> DataBundle:
    """Build a full-history DataBundle straight from the synthetic generator (no database)."""
    mkt = market or SyntheticMarket(spec or SyntheticSpec(n_stocks=40, start="2018-01-02", end="2021-12-31"))
    w = mkt.world
    bars = w["bars"]
    calendar = TradingCalendar.from_dates(sorted(bars.loc[bars["symbol"] == "SPY", "date"].unique()))
    panel = build_panel(bars, w["corporate_actions"], calendar=calendar)
    return DataBundle(
        panel=panel,
        calendar=calendar,
        reference=w["reference"],
        actions=w["corporate_actions"],
        events=w["events"],
        fundamentals=w["fundamentals"],
        news=w["news"],
        alt_trades=w["alt_trades"],
        benchmarks=SYNTHETIC_BENCHMARKS,
        dataset_ids=["synthetic"],
        is_synthetic=True,
    )
