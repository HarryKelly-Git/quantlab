"""Congress (House PTR) + insider (SEC Form 4) disclosures: ingestion into the store, the bounded
daily refresh, the research-parquet import, and read helpers for the dashboard and CLI.
CONTEXT ONLY: never scored, never a selection, sizing or order input (docs/ALT-DATA.md).

Status vocabulary (per source, ``insider`` and ``congress``):
  OK        every listed filing in the window is stored (``rows`` rows written this run, 0 = nothing new)
  PARTIAL   the time / filing budget ran out: ``remaining`` filings are fetched by the next run
  SKIPPED   disabled in config, or the SEC user agent is not set: nothing fetched, data UNKNOWN (never zero)
  FAILED    an outage (network, 403, 5xx, malformed index) or a missing PDF library: UNKNOWN, reported,
            never raised. Whatever was fetched before the failure is still written.

Incremental: a filing whose accession (Form 4) / DocID (PTR) is already stored is never fetched again.
Rows are written as new immutable ``alt_trades`` datasets; the store de-duplicates on
(source, record_id), so re-running an ingest never duplicates a disclosure.
"""
from __future__ import annotations

import json
import time as _time
from typing import Any, Callable, Iterable

import pandas as pd

from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

SOURCES = ("insider", "congress")
PIT_NOTES = {
    "insider": "SEC Form 4. available_at = EDGAR acceptance datetime of the filing (PIT). The transaction "
               "date is NOT availability. CONTEXT ONLY: never scored.",
    "congress": "House Clerk PTRs. available_at = cutoff of the session AFTER the index FilingDate (date-only, "
                "PIT_CONSERVATIVE). The transaction date is NOT availability. CONTEXT ONLY: never scored.",
}
COVERAGE_GAPS = ("Senate: no free automated source (the official eFD site refuses automated access; the "
                 "community dataset stopped in 2020). Senate trades are UNKNOWN, not zero.")


def _utc(now: Any = None) -> pd.Timestamp:
    t = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def alt_settings(config) -> dict[str, Any]:
    g = lambda k, d: config.get(f"alt_data.{k}", d)          # noqa: E731
    return {
        "insider": {"enabled": bool(g("insider.enabled", True)), "refresh_days": int(g("insider.refresh_days", 5)),
                    "current_pages": int(g("insider.current_feed_pages", 10)),
                    "max_filings": int(g("insider.max_filings_per_run", 3000)),
                    "max_seconds": float(g("insider.max_seconds_per_run", 300))},
        "congress": {"enabled": bool(g("congress.enabled", True)), "refresh_days": int(g("congress.refresh_days", 60)),
                     "max_filings": int(g("congress.max_filings_per_run", 60)),
                     "max_seconds": float(g("congress.max_seconds_per_run", 180))},
        "refresh_http": {"timeout": float(g("refresh_http_timeout_seconds", 20)),
                         "retries": int(g("refresh_http_max_retries", 1))},
        "max_consecutive_errors": int(g("max_consecutive_errors", 5)),
    }


def _calendar(ctx):
    try:
        from quantlab.data.sec_catalysts import market_calendar
        return market_calendar(ctx.store, ctx.config.get("benchmarks.market", "SPY"))
    except Exception:                         # no bars yet: next-business-day fallback (documented)
        return None


def stored_filing_ids(store, source: str) -> set[str]:
    """Accessions (insider) / DocIDs (congress) already stored (real data only)."""
    if not store.dataset_ids("alt_trades", synthetic=False):
        return set()
    ids = store.load("alt_trades", synthetic=False, columns=["record_id"])
    ids = ids.loc[ids["source"] == source, "record_id"].dropna().astype(str)
    return {r.split(":")[1] for r in ids if r.count(":") >= 1}


def _write(store, rows: list[dict[str, Any]], source: str, params: dict[str, Any]) -> str | None:
    if not rows:
        return None
    from quantlab.data import schemas
    df = schemas.conform("alt_trades", pd.DataFrame(rows))
    provider = "sec_form4" if source == "insider" else "house_clerk"
    return store.write("alt_trades", df, provider, params={"what": f"alt_{source}", **params},
                       pit_notes=PIT_NOTES[source])


def _status_for(exc: Exception) -> str:
    from quantlab.data.providers.base import ProviderNotConfigured
    return "SKIPPED" if isinstance(exc, ProviderNotConfigured) else "FAILED"


def _fetch_loop(todo: list[dict[str, Any]], fetch: Callable[[dict[str, Any]], list[dict[str, Any]]], *,
                max_filings: int | None, deadline: float | None, max_errors: int) -> dict[str, Any]:
    """Fetch filings oldest first within the budget. Per-filing errors are counted; ``max_errors`` in a
    row (or any 403 / rate-limit / missing PDF library) stops the loop as an outage."""
    from quantlab.data.providers.house_ptr import PdfLibraryMissing
    from quantlab.data.providers.http import ProviderForbidden, ProviderRateLimited
    rows: list[dict[str, Any]] = []
    fetched = failed = consecutive = 0
    stop: str | None = None
    errors: list[str] = []
    for item in todo:
        if max_filings is not None and fetched + failed >= max_filings:
            stop = "budget"
            break
        if deadline is not None and _time.time() >= deadline:
            stop = "budget"
            break
        try:
            rows.extend(fetch(item))
            fetched += 1
            consecutive = 0
        except (ProviderForbidden, ProviderRateLimited, PdfLibraryMissing) as exc:
            errors.append(repr(exc)[:200])
            stop = "outage"
            break
        except Exception as exc:              # one bad filing never stops the rest
            failed += 1
            consecutive += 1
            errors.append(f"{item.get('accession') or item.get('doc_id')}: {exc!r}"[:200])
            if consecutive >= max_errors:
                stop = "outage"
                break
    return {"rows": rows, "fetched": fetched, "failed": failed, "remaining": len(todo) - fetched - failed,
            "stop": stop, "errors": errors[:5]}


def ingest_insider(ctx, *, days: int, now: Any = None, provider=None, include_current: bool = True,
                   max_filings: int | None = None, max_seconds: float | None = None) -> dict[str, Any]:
    """Form 4 filings listed in the daily indexes of the last ``days`` calendar days (plus the current
    feed, for filings not yet in a published index), minus those already stored."""
    s = alt_settings(ctx.config)
    now = _utc(now)
    t0 = _time.time()
    if not s["insider"]["enabled"]:
        return {"status": "SKIPPED", "rows": 0, "reason": "alt_data.insider.enabled is false"}
    if provider is None:
        from quantlab.data.providers.sec_form4 import SecForm4Provider
        provider = SecForm4Provider(ctx.config, calendar=_calendar(ctx))
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    listed: dict[str, dict[str, Any]] = {}
    unpublished: list[str] = []
    try:
        for d in pd.date_range(today - pd.Timedelta(days=max(int(days), 0)), today):
            if d.dayofweek >= 5:
                continue
            got = provider.daily_index(d)
            if got is None:
                unpublished.append(str(d.date()))
                continue
            for f in got:
                listed.setdefault(f["accession"], f)
        if include_current:
            for f in provider.current_filings(pages=s["insider"]["current_pages"]):
                listed.setdefault(f["accession"], f)
    except Exception as exc:                  # the listing itself failed: nothing to fetch this run
        log_event(log, "alt insider listing failed", level=30, error=repr(exc)[:300])
        return {"status": _status_for(exc), "rows": 0, "error": repr(exc)[:300]}
    have = stored_filing_ids(ctx.store, "insider")
    todo = sorted((f for a, f in listed.items() if a not in have), key=lambda f: (f.get("date_filed") or "", f["accession"]))
    deadline = t0 + max_seconds if max_seconds else None
    res = _fetch_loop(todo, lambda f: provider.fetch_filing(f["cik"], f["accession"], f.get("date_filed")),
                      max_filings=max_filings, deadline=deadline, max_errors=s["max_consecutive_errors"])
    params = {"start": str((today - pd.Timedelta(days=days)).date()), "end": now.isoformat(), "listed": len(listed),
              "already_stored": len(listed) - len(todo), "fetched": res["fetched"], "failed": res["failed"],
              "remaining": res["remaining"], "unpublished_days": unpublished}
    ds = _write(ctx.store, res["rows"], "insider", params)
    status = "FAILED" if res["stop"] == "outage" else ("PARTIAL" if res["remaining"] > 0 else "OK")
    out = {"status": status, "rows": len(res["rows"]), "dataset": ds, **{k: v for k, v in params.items() if k != "end"},
           "secs": round(_time.time() - t0, 1), "requests": getattr(provider.http, "request_count", None)}
    if res["errors"]:
        out["errors"] = res["errors"]
    return out


def ingest_congress(ctx, *, days: int, now: Any = None, provider=None, max_filings: int | None = None,
                    max_seconds: float | None = None) -> dict[str, Any]:
    """House PTRs (FilingType P) filed in the last ``days`` calendar days, minus those already stored."""
    s = alt_settings(ctx.config)
    now = _utc(now)
    t0 = _time.time()
    if not s["congress"]["enabled"]:
        return {"status": "SKIPPED", "rows": 0, "reason": "alt_data.congress.enabled is false"}
    if provider is None:
        from quantlab.data.providers.house_ptr import HousePtrProvider
        provider = HousePtrProvider(ctx.config, calendar=_calendar(ctx))
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    since = today - pd.Timedelta(days=max(int(days), 0))
    filings: list[dict[str, Any]] = []
    missing_years = []
    try:
        for year in range(since.year, today.year + 1):
            idx = provider.index(year)
            if idx is None:
                missing_years.append(year)
                continue
            filings += [f for f in idx if f["filing_type"] == "P" and f["doc_id"] and f["filing_date"]
                        and since <= pd.Timestamp(f["filing_date"]) <= today]
    except Exception as exc:
        log_event(log, "alt congress index failed", level=30, error=repr(exc)[:300])
        return {"status": _status_for(exc), "rows": 0, "error": repr(exc)[:300]}
    have = stored_filing_ids(ctx.store, "congress")
    uniq = {f["doc_id"]: f for f in filings}
    todo = sorted((f for d, f in uniq.items() if d not in have), key=lambda f: (f["filing_date"], f["doc_id"]))
    deadline = t0 + max_seconds if max_seconds else None
    res = _fetch_loop(todo, provider.ptr, max_filings=max_filings, deadline=deadline,
                      max_errors=s["max_consecutive_errors"])
    unparseable = sum(1 for r in res["rows"] if r["record_status"] == "UNPARSEABLE")
    params = {"start": str(since.date()), "end": now.isoformat(), "listed": len(uniq),
              "already_stored": len(uniq) - len(todo), "fetched": res["fetched"], "failed": res["failed"],
              "remaining": res["remaining"], "unparseable_filings": unparseable, "missing_index_years": missing_years}
    ds = _write(ctx.store, res["rows"], "congress", params)
    status = "FAILED" if res["stop"] == "outage" else ("PARTIAL" if res["remaining"] > 0 else "OK")
    out = {"status": status, "rows": len(res["rows"]), "dataset": ds, **{k: v for k, v in params.items() if k != "end"},
           "secs": round(_time.time() - t0, 1), "requests": getattr(provider.http, "request_count", None)}
    if res["errors"]:
        out["errors"] = res["errors"]
    return out


def _bounded_http(ctx, s: dict[str, Any], name: str, rate_key: str, rps: float, **kw):
    from quantlab.data.providers.http import HttpClient, shared_rate_limiter
    http = HttpClient.from_config(ctx.config, rate_limiter=shared_rate_limiter(rate_key, rps), name=name, **kw)
    http.timeout = min(http.timeout, s["refresh_http"]["timeout"])
    http.max_retries = min(http.max_retries, s["refresh_http"]["retries"])
    return http


def ingest_alt_trades(ctx, *, days: int | None = None, sources: Iterable[str] = SOURCES, mode: str = "history",
                      now: Any = None, providers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ingest the chosen sources. ``mode="refresh"`` (evening pipeline): each source's ``refresh_days``
    window, time/filing budgets and short HTTP timeouts. ``mode="history"`` (CLI): ``days`` window, no
    time budget (the CLI user waits). Never raises: each source reports its status."""
    from quantlab.secrets import load_dotenv
    load_dotenv(ctx.config.root / ".env")
    s = alt_settings(ctx.config)
    now = _utc(now)
    providers = dict(providers or {})
    out: dict[str, Any] = {"mode": mode, "now": now.isoformat()}
    for src in sources:
        cfg = s[src]
        try:
            if mode == "refresh" and src not in providers:
                cal = _calendar(ctx)
                if src == "insider":
                    from quantlab.data.providers.sec_form4 import SecForm4Provider
                    rps = float(ctx.config.get("providers.sec_edgar.max_requests_per_second", 8.0))
                    providers[src] = SecForm4Provider(ctx.config, calendar=cal,
                                                      http=_bounded_http(ctx, s, "sec_form4", "sec-edgar", rps))
                else:
                    from quantlab.data.providers.house_ptr import HousePtrProvider
                    rps = float(ctx.config.get("providers.house_clerk.max_requests_per_second", 1.0))
                    ua = ctx.config.get("providers.house_clerk.user_agent", "QuantLab research (personal, non-commercial)")
                    providers[src] = HousePtrProvider(ctx.config, calendar=cal, http=_bounded_http(
                        ctx, s, "house_clerk", "house-clerk", rps, user_agent=ua))
            window = int(days if days is not None else cfg["refresh_days"])
            budget = {"max_filings": cfg["max_filings"], "max_seconds": cfg["max_seconds"]} if mode == "refresh" else {}
            fn = ingest_insider if src == "insider" else ingest_congress
            out[src] = fn(ctx, days=window, now=now, provider=providers.get(src), **budget)
        except Exception as exc:              # an outage is reported, never raised into the pipeline
            out[src] = {"status": _status_for(exc), "rows": 0, "error": repr(exc)[:300]}
            log_event(log, f"alt {src} failed", level=30, error=repr(exc)[:300])
    return _summarise(out)


def _summarise(out: dict[str, Any]) -> dict[str, Any]:
    st = [out[s]["status"] for s in SOURCES if s in out]
    if st and all(x == "SKIPPED" for x in st):
        status = "SKIPPED"
    elif st and all(x == "OK" for x in st):
        status = "OK"
    elif any(x in ("OK", "PARTIAL") for x in st):
        status = "PARTIAL"
    else:
        status = "FAILED" if st else "UNKNOWN"
    out["status"] = status
    out["ok"] = status in ("OK", "PARTIAL", "SKIPPED")
    n = sum(int(out[s].get("rows") or 0) for s in SOURCES if s in out)
    # ``rows`` is what the paper runner prints per refresh part: a count when something was fetched,
    # otherwise the status itself (SKIPPED / FAILED is never shown as "0 rows")
    out["rows"] = n if status in ("OK", "PARTIAL") else status
    bad = [f"{s}: {out[s]['status']} {out[s].get('reason') or out[s].get('error') or ''}".strip()
           for s in SOURCES if s in out and out[s]["status"] != "OK"]
    if bad:
        out["error"] = "; ".join(bad)[:500]
    log_event(log, "alt ingest", status=status, rows=n, mode=out.get("mode"))
    return out


def refresh_alt(ctx, now: Any = None, providers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Daily refresh entry point used by :func:`quantlab.data.catalyst_refresh.refresh_catalysts`."""
    try:
        return ingest_alt_trades(ctx, mode="refresh", now=now, providers=providers)
    except Exception as exc:                  # pragma: no cover - ingest_alt_trades already never raises
        return {"status": "FAILED", "ok": False, "rows": "FAILED", "error": repr(exc)[:300]}


# ------------------------------------------------------------------------------------------------
# historical insider import (research parquet built from the SEC Insider Transactions Data Sets)
# ------------------------------------------------------------------------------------------------
def insider_parquet_rows(df: pd.DataFrame, calendar=None, retrieved_at: Any = None) -> pd.DataFrame:
    """research/2026-10-01-new-data-tests/data/insider_tx*.parquet -> alt_trades rows.

    That file holds only open-market P (acquired) / S (disposed) lines of Form 4s by directors/officers
    with price > 0, one reporting owner per accession, and a date-only FILING_DATE. So availability is
    the cutoff of the session after the filing date (PIT_CONSERVATIVE), and the record ids are the live
    parser's (accession + content hash), so a later live fetch of the same filing never double-counts."""
    from quantlab.core.types import PitStatus
    from quantlab.data import schemas
    from quantlab.data.providers.sec_edgar import conservative_available_at
    from quantlab.data.providers.sec_form4 import form4_record_id
    need = {"accession", "filing_date", "trans_date", "issuer_symbol", "owner_cik", "relationship", "title", "code",
            "shares", "price", "value"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"insider parquet: missing columns {sorted(miss)}")
    d = df[df["code"].isin(["P", "S"])].copy()
    filed = pd.to_datetime(d["filing_date"]).dt.normalize()
    days = sorted(set(filed.dropna()))
    avail = {f: conservative_available_at(f, calendar) for f in days}
    seen: dict[str, int] = {}
    ad = d["code"].map({"P": "A", "S": "D"})
    rid = [form4_record_id(a, "nd", c, t, sh, pr, x, seen)
           for a, c, t, sh, pr, x in zip(d["accession"], d["code"], d["trans_date"], d["shares"], d["price"], ad)]

    def detail(rel: Any, title: Any) -> str:
        parts = [p.strip() for p in str(rel or "").split(",") if p.strip()]
        t = str(title or "").strip()
        return "; ".join(f"Officer: {t}" if p == "Officer" and t else ("10% owner" if p == "TenPercentOwner" else p)
                         for p in parts) or "UNKNOWN"

    sym = d["issuer_symbol"].astype("object").where(d["issuer_symbol"].astype(str).str.match(r"^[A-Z][A-Z0-9.\-]{0,9}$"))
    out = pd.DataFrame({
        "record_id": rid, "source": "insider", "symbol": sym.to_numpy(),
        "actor": ("CIK " + d["owner_cik"].astype(str).str.lstrip("0")).to_numpy(),
        "actor_detail": [detail(r, t) for r, t in zip(d["relationship"], d["title"])],
        "side": d["code"].map({"P": "BUY", "S": "SELL"}).to_numpy(),
        "amount_low_usd": d["value"].to_numpy(dtype="float64"), "amount_high_usd": d["value"].to_numpy(dtype="float64"),
        "shares": d["shares"].to_numpy(dtype="float64"), "price": d["price"].to_numpy(dtype="float64"),
        "transaction_date": pd.to_datetime(d["trans_date"]).to_numpy(),
        "disclosed_at": [f.tz_localize("America/New_York").tz_convert("UTC") for f in filed],
        "available_at": [avail[f] for f in filed],
        "pit_status": PitStatus.PIT_CONSERVATIVE.value,
        "record_status": ["PARSED" if isinstance(x, str) else "NO_SYMBOL" for x in sym],
        "raw_json": [json.dumps({"accession": a, "code": c, "import": "sec_insider_dataset"}, separators=(",", ":"))
                     for a, c in zip(d["accession"], d["code"])],
        "provider": "sec_insider_dataset", "retrieved_at": _utc(retrieved_at)})
    return schemas.conform("alt_trades", out)


def import_insider_parquet(ctx, path, *, since: Any = None) -> dict[str, Any]:
    """Write the research parquet into the store (one dataset; idempotent through record ids)."""
    df = pd.read_parquet(path)
    if since is not None:
        df = df[pd.to_datetime(df["filing_date"]) >= pd.Timestamp(since)]
    rows = insider_parquet_rows(df, calendar=_calendar(ctx))
    if not len(rows):
        return {"status": "OK", "rows": 0, "dataset": None}
    ds = ctx.store.write("alt_trades", rows, "sec_insider_dataset",
                         params={"what": "alt_insider_import", "path": str(path), "rows": int(len(rows))},
                         pit_notes="SEC Insider Transactions Data Sets via research parquet: directors/officers, "
                                   "open-market P/S only; available_at = cutoff of the session after the filing "
                                   "date (PIT_CONSERVATIVE). CONTEXT ONLY.")
    return {"status": "OK", "rows": int(len(rows)), "dataset": ds,
            "first_filing": str(rows["disclosed_at"].min())[:10], "last_filing": str(rows["disclosed_at"].max())[:10]}


# ------------------------------------------------------------------------------------------------
# read helpers (dashboard / CLI)
# ------------------------------------------------------------------------------------------------
DISPLAY_COLUMNS = ["source", "symbol", "actor", "actor_detail", "side", "amount_low_usd", "amount_high_usd",
                   "shares", "price", "transaction_date", "disclosed_at", "available_at", "pit_status",
                   "record_status", "retrieved_at", "record_id"]


def load_recent(store, *, since_days: int = 120, synthetic: bool = False, now: Any = None) -> pd.DataFrame:
    """Disclosures with disclosed_at in the last ``since_days`` (column projection + row filter per
    dataset, so the dashboard never loads the whole history). De-duplicated like ``store.load``."""
    import pyarrow.parquet as pq
    now = _utc(now)
    lo = now - pd.Timedelta(days=since_days)
    frames = []
    for ds in store.dataset_ids("alt_trades", synthetic=synthetic):
        row = store.db.fetchone("SELECT path, end_date FROM datasets WHERE dataset_id=?", (ds,))
        if row is None or (row["end_date"] and pd.Timestamp(row["end_date"]) < lo.tz_localize(None).normalize()):
            continue
        t = pq.read_table(store.data_dir / row["path"], columns=DISPLAY_COLUMNS).to_pandas()
        t = t[pd.to_datetime(t["disclosed_at"], utc=True) >= lo]
        if len(t):
            frames.append(t)
    if not frames:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("retrieved_at", kind="mergesort").drop_duplicates(["source", "record_id"], keep="last")
    return df.sort_values(["disclosed_at", "symbol"], ascending=[False, True]).reset_index(drop=True)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def records(df: pd.DataFrame, now: Any = None) -> list[dict[str, Any]]:
    """Display rows: amounts as text (UNKNOWN when missing), dates as ISO, and whether the record is
    already usable by the decision chain (available_at <= now) or only from a later cutoff."""
    now = _utc(now)
    out = []
    for r in df.itertuples(index=False):
        lo, hi = _num(r.amount_low_usd), _num(r.amount_high_usd)
        if lo is None and hi is None:
            amount = "UNKNOWN"
        elif r.source == "insider" or (lo is not None and hi is not None and lo == hi):
            amount = f"${(lo if lo is not None else hi):,.0f}"
        elif hi is None:
            amount = f"over ${lo:,.0f}"
        elif lo is None:
            amount = f"up to ${hi:,.0f}"
        else:
            amount = f"${lo:,.0f} - ${hi:,.0f}"
        av = _utc(r.available_at)
        disc = _utc(r.disclosed_at)
        tx = pd.Timestamp(r.transaction_date) if pd.notna(r.transaction_date) else None
        out.append({"source": r.source, "symbol": r.symbol, "actor": r.actor, "detail": r.actor_detail, "side": r.side,
                    "amount": amount, "traded": str(tx.date()) if tx is not None else "UNKNOWN",
                    "disclosed": (disc.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET")
                                  if r.source == "insider" else str(disc.tz_convert("America/New_York").date())),
                    "usable_from": av.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET"),
                    "usable_now": bool(av <= now), "status": r.record_status})
    return out


def recent_disclosures(store, *, symbols: Iterable[str] | None = None, limit: int = 20, since_days: int = 120,
                       synthetic: bool = False, now: Any = None, frame: pd.DataFrame | None = None,
                       source: str | None = None) -> list[dict[str, Any]]:
    """Newest first; only rows naming a ticker (filings without one are counted in ``source_status``)."""
    df = frame if frame is not None else load_recent(store, since_days=since_days, synthetic=synthetic, now=now)
    df = df[df["symbol"].notna()]
    if source:
        df = df[df["source"] == source]
    if symbols is not None:
        want = {str(s).upper() for s in symbols}
        df = df[df["symbol"].isin(want)]
    return records(df.head(limit), now=now)


def source_status(store, config, synthetic: bool = False) -> list[dict[str, Any]]:
    """Per source: enabled?, datasets, latest disclosure, last pull and its status counts."""
    s = alt_settings(config)
    rows = []
    for src in SOURCES:
        last, n, latest = None, 0, None
        for ds in store.dataset_ids("alt_trades", synthetic=synthetic):
            r = store.db.fetchone("SELECT end_date, params_json, created_at FROM datasets WHERE dataset_id=?", (ds,))
            pj = json.loads(r["params_json"] or "{}")
            what = str(pj.get("what", ""))
            if synthetic or src in what:
                n += 1
                if r["end_date"] and (latest is None or r["end_date"] > latest):
                    latest = r["end_date"]
                if not synthetic and what == f"alt_{src}" and (last is None or str(r["created_at"]) > str(last.get("at"))):
                    last = {"at": str(r["created_at"])[:16], "fetched": pj.get("fetched"), "remaining": pj.get("remaining"),
                            "failed": pj.get("failed"), "unparseable": pj.get("unparseable_filings")}
        rows.append({"source": src, "enabled": s[src]["enabled"], "datasets": n, "latest_disclosure": latest,
                     "last_pull": (last or {}).get("at"), "last_pull_detail": last,
                     "coverage": ("SEC Form 4, all issuers" if src == "insider" else "House PTRs only (Senate: not covered)")})
    return rows


__all__ = ["COVERAGE_GAPS", "DISPLAY_COLUMNS", "SOURCES", "alt_settings", "import_insider_parquet", "ingest_alt_trades",
           "ingest_congress", "ingest_insider", "insider_parquet_rows", "load_recent", "recent_disclosures", "records",
           "refresh_alt", "source_status", "stored_filing_ids"]
