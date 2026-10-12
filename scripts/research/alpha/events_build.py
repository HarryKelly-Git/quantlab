"""Event study, step 1 (docs/EVENT-STUDY-PREREG.md): detect UP/DOWN events on the corrected 2016-24 panel and
compute sector flags, laggard peers, the ticker used on each date, next-open outcomes (net of costs) and the
news download plan. PAPER research. Writes var/alpha/cache/events.parquet, events_laggards.parquet and
events_news_plan.parquet."""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import ca_fixes, research_data  # noqa: E402
from quantlab.alpha import events_study as es  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402


def main() -> None:
    t0 = time.time()
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps"):
        setattr(d, k, None)
    gc.collect()
    p, u = d.p, d.u_liquid
    ev = pd.concat([es.detect(p, u, "up"), es.detect(p, u, "down")], ignore_index=True)
    print(f"events: {ev['side'].value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)
    ev, lag = es.sector_flags(ev, p, u, d.sectors)
    print(f"sector-wide up events {int(ev.loc[ev.side == 'up', 'sector_wide'].sum())}; laggard rows {len(lag)}", flush=True)
    ev["ticker"] = es.ticker_map(d.resolved(), ev)
    mdv = d.mdv20
    for x in (ev, lag):
        x["mdv20"] = mdv.to_numpy()[p.dates.get_indexer(pd.to_datetime(x["date"])), mdv.columns.get_indexer(x["entity"])]
        x["cost_rt"] = es.cost_round_trip(x["mdv20"].to_numpy())
    ret_pnl = d.ret_cc_pnl
    ev = es.forward(p, ret_pnl, ev)
    lag = es.forward(p, ret_pnl, lag, horizons=(1, 5, 20))
    hi = p["adj_close"].rolling(252, min_periods=200).max().shift(1)
    ev["new_52w_high"] = p["adj_close"].to_numpy()[ev["i"], p.symbols.get_indexer(ev["entity"])] >= \
        hi.to_numpy()[ev["i"], p.symbols.get_indexer(ev["entity"])]
    spy = p["ret_cc"]["SPY"]
    ev["spy_ret_t"] = spy.reindex(pd.to_datetime(ev["date"])).to_numpy()
    cache = store_dir() / "cache"
    ev.to_parquet(cache / "events.parquet", index=False)
    lag.to_parquet(cache / "events_laggards.parquet", index=False)
    plan = (ev.groupby("date")["ticker"].apply(lambda s: sorted(set(s))).rename("tickers").reset_index())
    pos = p.dates.get_indexer(pd.to_datetime(plan["date"]))
    plan["prev_date"] = p.dates[np.maximum(pos - 1, 0)]
    plan.to_parquet(cache / "events_news_plan.parquet", index=False)
    print(f"saved {len(ev)} events, {len(lag)} laggard rows, news plan {len(plan)} dates "
          f"({int(plan['tickers'].str.len().sum())} date-tickers); {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
