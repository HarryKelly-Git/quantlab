"""Incremental catalyst refresh for daily operation (paper runner / CLI).

The full historical backfills (`catalysts ingest-sec | ingest-news | ingest-facts`) take tens of
minutes to hours. Every day only the recent tail is needed:

  news   market-wide articles from the last stored window end (minus an overlap for revisions) to
         now -- one call per ~50 articles, seconds per day;
  sec    each symbol's recent filings (submissions ``recent`` block only; older history pages are
         skipped), SIC headers only when a new periodic report appears;
  facts  companyfacts only for symbols that filed a new 10-Q / 10-K in this refresh.

Every part is best effort and independent: a failure is reported, never raised into the caller's
trading path. Rows are written as new immutable datasets; the store de-duplicates on each kind's
key, so re-running a refresh never duplicates rows. Availability timestamps are exactly those of
the backfills (SEC acceptance time; news creation time).
"""
from __future__ import annotations

import json
import time as _time
from typing import Any, Iterable

import pandas as pd

from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)


def _news_resume_point(store, now: pd.Timestamp, max_days: int) -> pd.Timestamp:
    """Latest market-news window end stored (minus a 1-day overlap), capped at ``max_days`` back."""
    ends = []
    for r in store.db.fetchall("SELECT params_json FROM datasets WHERE kind='news' AND is_synthetic=0"):
        pj = json.loads(r["params_json"] or "{}")
        if pj.get("what") == "market_news" and pj.get("end"):
            ends.append(pd.Timestamp(pj["end"]))
    floor = now - pd.Timedelta(days=max_days)
    if not ends:
        return floor
    last = max(e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC") for e in ends)
    return max(floor, last - pd.Timedelta(days=1))


def refresh_catalysts(ctx, session, *, symbols: Iterable[str] | None = None, parts=("news", "sec", "facts"),
                      sec_days: int = 10, news_max_days: int = 30, workers: int = 6, now=None) -> dict[str, Any]:
    """Refresh recent catalyst data. ``symbols``: SEC scope (default: stored-bar COMMON stocks)."""
    from quantlab.secrets import load_dotenv
    load_dotenv(ctx.config.root / ".env")
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    session = pd.Timestamp(session)
    out: dict[str, Any] = {"session": str(session.date()), "now": now.isoformat()}
    new_periodic: set[str] = set()
    if "news" in parts:
        t0 = _time.time()
        try:
            from quantlab.data.providers.alpaca_data import AlpacaDataProvider
            start = _news_resume_point(ctx.store, now, news_max_days)
            df = AlpacaDataProvider(ctx.config).get_news_market(start, now)
            ds = ctx.store.write("news", df, "alpaca", params={"what": "market_news", "start": str(start), "end": str(now),
                                                              "refresh": True},
                                 pit_notes="market-wide; available_at=created_at; window selected by updated_at; "
                                           "summary not stored") if len(df) else None
            out["news"] = {"ok": True, "from": str(start), "rows": int(len(df)), "dataset": ds,
                           "secs": round(_time.time() - t0, 1)}
        except Exception as exc:                 # best effort: reported, never raised into trading
            out["news"] = {"ok": False, "error": repr(exc)[:300]}
            log_event(log, "catalyst refresh: news failed", level=30, error=repr(exc)[:300])
    if "sec" in parts:
        t0 = _time.time()
        try:
            from quantlab.data.providers.sec_edgar import SecEdgarProvider
            from quantlab.data.sec_catalysts import SecCatalystIngest, catalyst_symbols, market_calendar
            syms = sorted(set(symbols)) if symbols is not None else catalyst_symbols(ctx.store)
            cal = market_calendar(ctx.store, ctx.config.get("benchmarks.market", "SPY"))
            prov = SecEdgarProvider(ctx.config, calendar=cal)
            since = (session - pd.Timedelta(days=sec_days)).tz_localize("UTC")
            ing = SecCatalystIngest(prov, cal, since, fetch_sic=True)
            df = ing.run(syms, workers=workers)
            ds = ctx.store.write("events", df, "sec_edgar", params={"what": "ingest-sec", "since": str(since.date()),
                                                                    "refresh": True}) if len(df) else None
            if len(df):
                new_periodic = set(df.loc[df["event_type"] == "periodic_report", "symbol"])
            counts = df["event_type"].value_counts().to_dict() if len(df) else {}
            out["sec"] = {"ok": True, "symbols": len(syms), "rows": int(len(df)), "by_type": counts, "dataset": ds,
                          "requests": prov.http.request_count, "secs": round(_time.time() - t0, 1),
                          "unmapped": len(df.attrs.get("unmapped_symbols", [])) if len(df) else None}
        except Exception as exc:
            out["sec"] = {"ok": False, "error": repr(exc)[:300]}
            log_event(log, "catalyst refresh: SEC failed", level=30, error=repr(exc)[:300])
    if "facts" in parts and new_periodic:
        t0 = _time.time()
        try:
            from concurrent.futures import ThreadPoolExecutor
            from quantlab.data.providers.sec_edgar import SecEdgarProvider
            prov = SecEdgarProvider(ctx.config)
            since = (session - pd.Timedelta(days=sec_days)).tz_localize("UTC")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts_ = [p for p in pool.map(lambda s: prov.get_fundamentals([s], since=since), sorted(new_periodic))
                          if len(p)]
            df = pd.concat(parts_, ignore_index=True) if parts_ else None
            ds = ctx.store.write("fundamentals", df, "sec_edgar", params={"what": "ingest-facts", "since": str(since.date()),
                                                                          "refresh": True}) if df is not None else None
            out["facts"] = {"ok": True, "symbols": len(new_periodic), "rows": 0 if df is None else int(len(df)),
                            "dataset": ds, "secs": round(_time.time() - t0, 1)}
        except Exception as exc:
            out["facts"] = {"ok": False, "error": repr(exc)[:300]}
            log_event(log, "catalyst refresh: facts failed", level=30, error=repr(exc)[:300])
    elif "facts" in parts:
        out["facts"] = {"ok": True, "symbols": 0, "rows": 0, "note": "no new 10-Q/10-K in this refresh"}
    log_event(log, "catalyst refresh", **{k: (v.get("rows") if isinstance(v, dict) else v) for k, v in out.items()})
    return out


__all__ = ["refresh_catalysts"]
