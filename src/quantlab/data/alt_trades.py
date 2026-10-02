"""Congress / insider disclosures (Quiver): ingestion into the store, the daily refresh, and read
helpers for the dashboard and the CLI. CONTEXT ONLY (docs/QUIVER.md).

Status vocabulary (per source, ``congress`` and ``insider``):
  OK        fetched; ``rows`` rows written (0 = the provider returned nothing new in the window)
  SKIPPED   no key configured (``providers.quiver.key_env``) or ``providers.quiver.enabled: false``:
            nothing fetched, the data is UNKNOWN (never zero)
  FAILED    the request failed (network, 401/403, 5xx, malformed body): UNKNOWN, reported, never raised

Nothing here raises into a caller's trading path: every source is wrapped and reported.
Rows are written as new immutable ``alt_trades`` datasets; the store de-duplicates on
(source, record_id), so re-running an ingest never duplicates a disclosure.
"""
from __future__ import annotations

import json
import time as _time
from typing import Any, Iterable

import pandas as pd

from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

SOURCES = ("congress", "insider")
PIT_NOTES = ("Quiver disclosures. available_at = 16:00 ET cutoff of the session AFTER the disclosure date "
             "(congress report date / Form 4 filing date; date-only). The transaction date is NOT availability. "
             "Look-ahead return fields are dropped. CONTEXT ONLY: never scored.")


def _utc(now: Any = None) -> pd.Timestamp:
    t = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def quiver_settings(config) -> dict[str, Any]:
    g = lambda k, d: config.get(f"providers.quiver.{k}", d)   # noqa: E731
    return {"enabled": bool(g("enabled", True)), "key_env": str(g("key_env", "QUIVER_API_KEY")),
            "lookback_days": int(g("lookback_days", 365)), "refresh_days": int(g("refresh_days", 7)),
            "congress_resync_days": float(g("congress_resync_days", 7)),
            "congress_resync_window_days": int(g("congress_resync_window_days", 120))}


def key_status(config, root=None) -> dict[str, Any]:
    """Whether a key is configured (presence only; the value is never read out or shown)."""
    from quantlab.secrets import get_secret, load_dotenv
    s = quiver_settings(config)
    if root is not None:
        load_dotenv(root / ".env")
    present = get_secret(s["key_env"]) is not None
    return {"enabled": s["enabled"], "key_env": s["key_env"], "key_present": present,
            "configured": bool(s["enabled"] and present)}


def _error_status(exc: Exception) -> str:
    from quantlab.data.providers.base import ProviderNotConfigured
    from quantlab.data.providers.http import ProviderAuthError, ProviderForbidden
    if isinstance(exc, ProviderNotConfigured):
        return "SKIPPED"
    if isinstance(exc, ProviderForbidden):
        return "FAILED (403: the key's plan may not include this endpoint; insiders are 'Tier 2')"
    if isinstance(exc, ProviderAuthError):
        return "FAILED (401: key not accepted)"
    return "FAILED"


def _last_pull(store, what: str) -> pd.Timestamp | None:
    ends = []
    for r in store.db.fetchall("SELECT params_json FROM datasets WHERE kind='alt_trades' AND is_synthetic=0"):
        pj = json.loads(r["params_json"] or "{}")
        if pj.get("what") == what and pj.get("end"):
            ends.append(_utc(pj["end"]))
    return max(ends) if ends else None


def _write(store, df: pd.DataFrame, what: str, params: dict[str, Any]) -> str | None:
    if df is None or not len(df):
        return None
    meta = {k: v for k, v in df.attrs.items() if isinstance(v, (str, int, float, bool, dict, type(None)))}
    return store.write("alt_trades", df, "quiver", params={"what": what, **params, **meta}, pit_notes=PIT_NOTES)


def ingest_alt_trades(ctx, *, days: int | None = None, mode: str = "history", sources: Iterable[str] = SOURCES,
                      now: Any = None, provider=None) -> dict[str, Any]:
    """Fetch congress and/or insider disclosures and write them to the store.

    ``mode="history"`` (``quantlab alt ingest --days N``): congress from the bulk endpoint (rows
    disclosed in the last N days), insiders day by day over the last N days.
    ``mode="refresh"`` (daily, from the catalyst refresh): congress from the live endpoint, plus a
    bulk re-pull (last ``congress_resync_window_days``) when the previous one is older than
    ``congress_resync_days`` (late disclosures arrive up to 45 days after the trade); insiders day by
    day over the last ``refresh_days``.
    Never raises: each source reports OK / SKIPPED / FAILED (see module docstring)."""
    from quantlab.secrets import load_dotenv
    cfg = ctx.config
    s = quiver_settings(cfg)
    load_dotenv(cfg.root / ".env")
    now = _utc(now)
    out: dict[str, Any] = {"mode": mode, "now": now.isoformat()}
    if provider is None:
        try:
            from quantlab.data.providers.quiver import QuiverProvider
            from quantlab.data.sec_catalysts import market_calendar
            cal = market_calendar(ctx.store, cfg.get("benchmarks.market", "SPY"))
            provider = QuiverProvider(cfg, calendar=cal)
        except Exception as exc:              # a broken config is reported like any other failure
            for src in sources:
                out[src] = {"status": _error_status(exc), "rows": 0, "error": repr(exc)[:300]}
            return _summarise(out)
    if not provider.configured():
        why = ("providers.quiver.enabled is false" if not s["enabled"]
               else f"{s['key_env']} is not set: congress/insider activity is UNKNOWN (not zero)")
        for src in sources:
            out[src] = {"status": "SKIPPED", "rows": 0, "reason": why}
        log_event(log, "quiver skipped", reason=why)
        return _summarise(out)
    window = int(days if days is not None else (s["lookback_days"] if mode == "history" else s["refresh_days"]))
    start = (now - pd.Timedelta(days=window)).tz_localize(None).normalize()
    end = now.tz_localize(None).normalize()
    for src in sources:
        t0 = _time.time()
        try:
            written, rows = [], 0
            if src == "congress":
                if mode == "history":
                    df = provider.congress_history(since=start)
                    written.append(_write(ctx.store, df, "quiver_congress_bulk",
                                          {"start": str(start.date()), "end": now.isoformat(), "mode": mode}))
                    rows += len(df)
                else:
                    df = provider.congress_recent()
                    written.append(_write(ctx.store, df, "quiver_congress_live", {"end": now.isoformat(), "mode": mode}))
                    rows += len(df)
                    last = _last_pull(ctx.store, "quiver_congress_bulk")
                    if last is None or now - last >= pd.Timedelta(days=s["congress_resync_days"]):
                        rs = (now - pd.Timedelta(days=s["congress_resync_window_days"])).tz_localize(None).normalize()
                        bulk = provider.congress_history(since=rs)
                        written.append(_write(ctx.store, bulk, "quiver_congress_bulk",
                                              {"start": str(rs.date()), "end": now.isoformat(), "mode": mode}))
                        rows += len(bulk)
                        out.setdefault("notes", []).append(f"congress bulk re-pull since {rs.date()}: {len(bulk)} rows")
            else:
                df = provider.insiders_history(start, end)
                written.append(_write(ctx.store, df, "quiver_insider_daily",
                                      {"start": str(start.date()), "end": now.isoformat(), "mode": mode}))
                rows += len(df)
            out[src] = {"status": "OK", "rows": int(rows), "datasets": [w for w in written if w],
                        "secs": round(_time.time() - t0, 1), "requests": getattr(provider.http, "request_count", None)}
        except Exception as exc:              # an outage is reported, never raised into the pipeline
            out[src] = {"status": _error_status(exc), "rows": 0, "error": repr(exc)[:300]}
            log_event(log, f"quiver {src} failed", level=30, error=repr(exc)[:300])
    return _summarise(out)


def _summarise(out: dict[str, Any]) -> dict[str, Any]:
    st = [out[s]["status"] for s in SOURCES if s in out]
    ok = [x for x in st if x == "OK"]
    skipped = [x for x in st if x == "SKIPPED"]
    if st and len(skipped) == len(st):
        status = "SKIPPED"
    elif st and len(ok) == len(st):
        status = "OK"
    elif ok:
        status = "PARTIAL"
    else:
        status = "FAILED" if st else "UNKNOWN"
    out["status"] = status
    out["ok"] = status in ("OK", "SKIPPED")
    n = sum(int(out[s].get("rows") or 0) for s in SOURCES if s in out)
    # ``rows`` is what the paper runner prints for each refresh part: a count when something was
    # fetched, otherwise the status itself (SKIPPED / FAILED is never shown as "0 rows")
    out["rows"] = n if ok else status
    if status != "OK":
        out["error"] = "; ".join(f"{s}: {out[s].get('reason') or out[s].get('error') or out[s]['status']}"
                                 for s in SOURCES if s in out and out[s]["status"] != "OK")[:500]
    log_event(log, "quiver ingest", status=status, rows=n, mode=out.get("mode"))
    return out


def refresh_quiver(ctx, now: Any = None, provider=None) -> dict[str, Any]:
    """Daily refresh entry point used by :func:`quantlab.data.catalyst_refresh.refresh_catalysts`."""
    try:
        return ingest_alt_trades(ctx, mode="refresh", now=now, provider=provider)
    except Exception as exc:                  # pragma: no cover - ingest_alt_trades already never raises
        return {"status": "FAILED", "ok": False, "rows": "FAILED", "error": repr(exc)[:300]}


# ------------------------------------------------------------------------------------------------
# read helpers (dashboard / CLI)
# ------------------------------------------------------------------------------------------------
DISPLAY_COLUMNS = ["source", "symbol", "actor", "actor_detail", "side", "amount_low_usd", "amount_high_usd",
                   "transaction_date", "disclosed_date", "available_at", "pit_status", "retrieved_at", "record_id"]


def load_recent(store, *, since_days: int = 120, synthetic: bool = False, now: Any = None) -> pd.DataFrame:
    """Disclosures with disclosed_date in the last ``since_days`` (column projection + row filter per
    dataset, so the dashboard never loads the whole history). De-duplicated like ``store.load``."""
    import pyarrow.parquet as pq
    now = _utc(now)
    lo = (now - pd.Timedelta(days=since_days)).tz_localize(None).normalize()
    frames = []
    for ds in store.dataset_ids("alt_trades", synthetic=synthetic):
        row = store.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (ds,))
        if row is None:
            continue
        t = pq.read_table(store.data_dir / row["path"], columns=DISPLAY_COLUMNS).to_pandas()
        t = t[pd.to_datetime(t["disclosed_date"]) >= lo]
        if len(t):
            frames.append(t)
    if not frames:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("retrieved_at", kind="mergesort").drop_duplicates(["source", "record_id"], keep="last")
    return df.sort_values(["disclosed_date", "available_at", "symbol"], ascending=[False, False, True]).reset_index(drop=True)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def records(df: pd.DataFrame, now: Any = None) -> list[dict[str, Any]]:
    """Display rows: amounts as text (UNKNOWN when missing), dates as ISO, and whether the record is
    already usable by the decision chain (available_at <= now) or only from the next cutoff."""
    now = _utc(now)
    out = []
    for r in df.itertuples(index=False):
        lo, hi = _num(r.amount_low_usd), _num(r.amount_high_usd)
        if lo is None and hi is None:
            amount = "UNKNOWN"
        elif r.source == "insider" or (lo is not None and hi is not None and lo == hi):
            amount = f"${(lo if lo is not None else hi):,.0f}"
        elif hi is None:
            amount = f"${lo:,.0f}+ (upper bound UNKNOWN)"
        elif lo is None:
            amount = f"up to ${hi:,.0f}"
        else:
            amount = f"${lo:,.0f} - ${hi:,.0f}"
        av = pd.Timestamp(r.available_at)
        av = av.tz_localize("UTC") if av.tzinfo is None else av.tz_convert("UTC")
        tx = pd.Timestamp(r.transaction_date) if pd.notna(r.transaction_date) else None
        out.append({"source": r.source, "symbol": r.symbol, "actor": r.actor, "detail": r.actor_detail, "side": r.side,
                    "amount": amount, "traded": str(tx.date()) if tx is not None else "UNKNOWN",
                    "disclosed": str(pd.Timestamp(r.disclosed_date).date()),
                    "usable_from": av.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET"),
                    "usable_now": bool(av <= now)})
    return out


def recent_disclosures(store, *, symbols: Iterable[str] | None = None, limit: int = 20, since_days: int = 120,
                       synthetic: bool = False, now: Any = None, frame: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    df = frame if frame is not None else load_recent(store, since_days=since_days, synthetic=synthetic, now=now)
    if symbols is not None:
        want = {str(s).upper() for s in symbols}
        df = df[df["symbol"].isin(want)]
    return records(df.head(limit), now=now)


def source_status(store, config, synthetic: bool = False) -> list[dict[str, Any]]:
    """Per source: key configured?, rows stored, latest disclosure, last pull. Counts only."""
    ks = key_status(config, getattr(config, "root", None))
    rows = []
    for src, what in (("congress", ("quiver_congress_live", "quiver_congress_bulk")), ("insider", ("quiver_insider_daily",))):
        pulls = [p for p in (_last_pull(store, w) for w in what) if p is not None] if not synthetic else []
        rows.append({"source": src, "key": "configured" if ks["configured"] else
                     ("disabled" if not ks["enabled"] else f"{ks['key_env']} not set (SKIPPED: UNKNOWN, not zero)"),
                     "last_pull": str(max(pulls))[:16] if pulls else None})
    agg = {}
    for ds in store.dataset_ids("alt_trades", synthetic=synthetic):
        r = store.db.fetchone("SELECT row_count, end_date, params_json FROM datasets WHERE dataset_id=?", (ds,))
        pj = json.loads(r["params_json"] or "{}")
        src = "insider" if "insider" in str(pj.get("what", "")) else ("congress" if "congress" in str(pj.get("what", ""))
                                                                     else "mixed")
        a = agg.setdefault(src, {"datasets": 0, "latest_disclosure": None})
        a["datasets"] += 1
        if r["end_date"] and (a["latest_disclosure"] is None or r["end_date"] > a["latest_disclosure"]):
            a["latest_disclosure"] = r["end_date"]
    for row in rows:
        a = agg.get(row["source"]) or agg.get("mixed") or {}
        row.update(datasets=a.get("datasets", 0), latest_disclosure=a.get("latest_disclosure"))
    return rows


__all__ = ["DISPLAY_COLUMNS", "SOURCES", "ingest_alt_trades", "key_status", "load_recent", "quiver_settings",
           "recent_disclosures", "records", "refresh_quiver", "source_status"]
