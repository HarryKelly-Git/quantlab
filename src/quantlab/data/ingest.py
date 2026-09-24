"""Ingestion: pull each dataset kind from its configured provider into the versioned store.

One provider failing never corrupts another kind; failures are recorded (``data_quality_issues``
+ the returned report) and the caller decides (health checks pause trading on stale data).
Missing credentials surface as SKIPPED with the reason — never as silently empty data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import TradingCalendar
from quantlab.data import schemas
from quantlab.data.providers.base import ProviderError, ProviderNotConfigured
from quantlab.data.store import MarketDataStore
from quantlab.db.database import Database, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

KINDS = ("reference", "bars", "corporate_actions", "events", "fundamentals", "news")


@dataclass
class KindOutcome:
    kind: str
    status: str                       # ok | skipped | failed | empty
    dataset_ids: list[str] = field(default_factory=list)
    rows: int = 0
    detail: str = ""


@dataclass
class IngestReport:
    outcomes: dict[str, KindOutcome] = field(default_factory=dict)
    is_synthetic: bool = False

    @property
    def ok(self) -> bool:
        return all(o.status in ("ok", "skipped", "empty") for o in self.outcomes.values()) and \
            self.outcomes.get("bars", KindOutcome("bars", "failed")).status == "ok"

    def summary(self) -> dict[str, Any]:
        return {k: {"status": o.status, "rows": o.rows, "datasets": len(o.dataset_ids), "detail": o.detail}
                for k, o in self.outcomes.items()}


class IngestionService:
    def __init__(self, config: Config, store: MarketDataStore, db: Database, providers: dict[str, Any] | None = None):
        self.config = config
        self.store = store
        self.db = db
        self._providers = providers

    @property
    def providers(self) -> dict[str, Any]:
        if self._providers is None:
            from quantlab.data.providers.registry import build_providers   # built by the providers module
            self._providers = build_providers(self.config)
        return self._providers

    # -- synthetic --------------------------------------------------------------------------------
    def ingest_synthetic(self, spec=None) -> IngestReport:
        """Write the whole SyntheticMarket world into the store, flagged is_synthetic=1."""
        from quantlab.data.providers.synthetic import SyntheticMarket, SyntheticSpec
        mkt = SyntheticMarket(spec or SyntheticSpec())
        rep = IngestReport(is_synthetic=True)
        params = {"spec": {k: getattr(mkt.spec, k) for k in mkt.spec.__dataclass_fields__}}
        for kind in KINDS:
            df = mkt.world[kind]
            ds = self.store.write(kind, df, "synthetic", params=params, is_synthetic=True,
                                  pit_notes="SYNTHETIC test data — not market evidence")
            rep.outcomes[kind] = KindOutcome(kind, "ok", [ds], len(df))
        log_event(log, "synthetic world ingested", **{k: o.rows for k, o in rep.outcomes.items()})
        return rep

    # -- real providers ---------------------------------------------------------------------------
    def _record_issue(self, kind: str, status: str, detail: str, run_id: str | None) -> None:
        self.db.insert("data_quality_issues", {
            "run_id": run_id, "dataset_id": None, "check_name": f"ingest_{kind}",
            "severity": "CRITICAL" if kind == "bars" else "WARNING", "symbol": None, "session_date": None,
            "detail": f"{status}: {detail}"[:2000], "created_at": utcnow_iso()})

    def _run_kind(self, rep: IngestReport, kind: str, fetch, run_id: str | None, provider_name: str,
                  params: dict[str, Any], chunk_writer=None) -> pd.DataFrame | None:
        prov = self.providers.get(kind)
        if prov is None:
            rep.outcomes[kind] = KindOutcome(kind, "skipped", detail="no provider configured (providers.%s)" % kind)
            return None
        try:
            df = fetch(prov)
        except ProviderNotConfigured as exc:
            rep.outcomes[kind] = KindOutcome(kind, "skipped", detail=f"not configured: {exc}")
            self._record_issue(kind, "skipped", str(exc), run_id)
            return None
        except (ProviderError, OSError, ValueError) as exc:
            rep.outcomes[kind] = KindOutcome(kind, "failed", detail=repr(exc)[:500])
            self._record_issue(kind, "failed", repr(exc), run_id)
            log_event(log, "ingest failed", level=40, kind=kind, error=repr(exc)[:500])
            return None
        if df is None or df.empty:
            rep.outcomes[kind] = KindOutcome(kind, "empty", detail="provider returned no rows")
            return df
        name = getattr(prov, "name", provider_name)
        extra = {"feed": getattr(prov, "last_feed_used", None)} if kind == "bars" else {}
        ds = self.store.write(kind, df, name, params={**params, **extra}, is_synthetic=bool(getattr(prov, "is_synthetic", False)))
        rep.outcomes[kind] = KindOutcome(kind, "ok", [ds], len(df))
        return df

    def select_symbols(self, reference: pd.DataFrame) -> list[str]:
        u = self.config.section("universe")
        allowed = set(u.get("allowed_exchanges", []))
        ref = reference
        keep = (ref["security_type"].astype(str).str.upper() == "COMMON") & ~ref["is_test_issue"].fillna(False).astype(bool)
        if allowed:
            keep &= ref["exchange"].isin(allowed)
        bench = [self.config.get("benchmarks.market")] + list(self.config.section("benchmarks").get("sectors", {}).keys())
        return sorted(set(ref.loc[keep, "symbol"]) | set(bench))

    def ingest_all(self, start: date | str | None = None, end: date | str | None = None,
                   symbols: list[str] | None = None, run_id: str | None = None,
                   bar_chunk: int = 200) -> IngestReport:
        start = pd.Timestamp(start or self.config.get("history.start_date")).date()
        end = pd.Timestamp(end or (datetime.now(timezone.utc) - timedelta(days=1)).date()).date()
        rep = IngestReport()
        params = {"start": str(start), "end": str(end)}
        ref = self._run_kind(rep, "reference", lambda p: p.get_securities(), run_id, "reference", params)
        if symbols is None:
            if ref is None or ref.empty:
                rep.outcomes.setdefault("bars", KindOutcome("bars", "failed", detail="no reference data to select symbols"))
                return rep
            symbols = self.select_symbols(ref)

        # bars in chunks: each chunk is its own immutable dataset
        prov = self.providers.get("bars")
        bar_frames: list[pd.DataFrame] = []
        chunks = [symbols[i:i + bar_chunk] for i in range(0, len(symbols), bar_chunk)]
        for n, chunk in enumerate(chunks):
            sub = IngestReport()
            df = self._run_kind(sub, "bars", lambda p, c=chunk: p.get_daily_bars(c, start, end), run_id, "bars",
                                {**params, "chunk": n, "n_chunks": len(chunks)})
            o = sub.outcomes["bars"]
            agg = rep.outcomes.setdefault("bars", KindOutcome("bars", o.status))
            agg.dataset_ids += o.dataset_ids
            agg.rows += o.rows
            if o.status != "ok":
                agg.status, agg.detail = o.status, o.detail
                if o.status in ("failed", "skipped"):
                    break
            if df is not None and not df.empty:
                bar_frames.append(df)
        del prov

        self._run_kind(rep, "corporate_actions", lambda p: p.get_corporate_actions(symbols, start, end), run_id,
                       "corporate_actions", params)
        calendar = None
        mkt = self.config.get("benchmarks.market")
        if bar_frames:
            allb = pd.concat(bar_frames)
            d = allb.loc[allb["symbol"] == mkt, "date"]
            calendar = TradingCalendar.from_dates(sorted((d if len(d) else allb["date"]).unique()))

        def _events(p):
            ev = p.get_earnings_events(symbols, start, end)
            if calendar is not None and len(ev):
                ev = ev.copy()
                ev["reaction_date"] = [calendar.reaction_session(t) for t in ev["event_time"]]
            return ev

        self._run_kind(rep, "events", _events, run_id, "events", params)
        self._run_kind(rep, "fundamentals", lambda p: p.get_fundamentals(symbols), run_id, "fundamentals", params)
        self._run_kind(rep, "news", lambda p: p.get_news(symbols, start, end), run_id, "news", params)
        log_event(log, "ingest finished", **{k: v["status"] for k, v in rep.summary().items()})
        return rep


__all__ = ["IngestionService", "IngestReport", "KindOutcome", "KINDS", "schemas"]
