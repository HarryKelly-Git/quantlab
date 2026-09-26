"""Alpaca Market Data (free Basic plan) -> RAW daily bars, corporate actions and news.

Point-in-time and completeness decisions (facts: docs/EXTERNAL-SERVICES.md, Alpaca section):
  * Bars are requested with ``adjustment=raw``. Adjusted bars are recomputed from Alpaca's CURRENT
    corporate-action database, which is corrected after the fact, so they are not PIT. QuantLab
    applies its own versioned actions in :mod:`quantlab.data.panel`.
  * ``feed`` is always explicit. The default is ambiguous in the docs, and SIP vs IEX changes
    volume by ~40x. If SIP is refused with 403 (e.g. Paper-Only accounts are entitled to IEX
    only), we fall back to ``feed_fallback`` and RECORD the feed actually used
    (``last_feed_used``, frame ``attrs['feed']`` and ``provider='alpaca:<feed>'`` on every row).
  * ``asof`` is pinned from config (default "-" = no rename mapping). Otherwise a re-run the day
    after a rename silently returns a different series.
  * ``start``/``end`` are explicit RFC-3339 timestamps with a New York offset. ``end`` is at most
    now-16min (the free plan rejects SIP data newer than 15 minutes) and never includes a session
    whose daily bar may still change: daily volume includes extended-hours trades until 20:00 ET,
    so a session is treated as final only after ``daily_bar_final_time`` ET.
  * Pagination follows ``next_page_token`` until it is null. A short page does NOT mean the end:
    pages are capped across all symbols and may be short while more data exists.
  * A daily bar's ``t`` is midnight America/New_York of the session, expressed in UTC (04:00Z in
    summer, 05:00Z in winter). It is converted to the NY calendar date, never to the UTC date and
    never used as an availability time (the close is known only at 16:00 ET).
  * Corporate actions carry no declaration timestamp and records are corrected after publication.
    So ``available_at`` = ex-date 09:30 ET with ``pit_status=PIT_CONSERVATIVE``: we never claim
    to have known a split before the market could see it in prices. Types we do not map
    (spin-offs, mergers, stock dividends...) are counted and returned in ``attrs`` so ingestion
    can log them. They are never dropped without a trace.
  * News: ``created_at`` is the availability time. If ``updated_at`` is later than
    ``created_at`` + tolerance, the REST payload may be a revised version (headline/summary/
    symbols), so the row is PIT_CONSERVATIVE, otherwise PIT.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.core.calendar import to_session
from quantlab.core.types import PitStatus
from quantlab.data import schemas
from quantlab.data.providers.base import (
    CorporateActionProvider,
    NewsProvider,
    PriceProvider,
    ProviderError,
    ProviderNotConfigured,
)
from quantlab.data.providers.http import (
    HttpClient,
    ProviderForbidden,
    ProviderResponseError,
    chunked,
    shared_rate_limiter,
)
from quantlab.logging_setup import get_logger, log_event
from quantlab.secrets import get_secret

log = get_logger("data.providers.alpaca")
NY = ZoneInfo("America/New_York")

# Corporate-action response keys we map into the canonical schema.
_SPLIT_KEYS = ("forward_splits", "reverse_splits")
_DIVIDEND_KEYS = ("cash_dividends",)
DEFAULT_CA_TYPES = [
    "forward_split", "reverse_split", "cash_dividend",
    # requested only so that price-relevant actions we cannot map are logged, not invisible:
    "unit_split", "stock_dividend", "spin_off", "cash_merger", "stock_merger", "stock_and_cash_merger",
    "name_change", "worthless_removal",
]


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def ny_midnight(d: date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(pd.Timestamp(d).date(), time(0, 0))).tz_localize(NY)


def rfc3339(ts: pd.Timestamp) -> str:
    """RFC-3339 with an explicit offset (e.g. 2024-01-03T00:00:00-05:00)."""
    return pd.Timestamp(ts).isoformat(timespec="seconds")


def bar_time_to_session(t: pd.Series) -> pd.Series:
    """Alpaca daily-bar ``t`` (midnight New York, written in UTC) -> tz-naive NY session date.

    DST-safe: 2024-03-08T05:00:00Z -> 2024-03-08 and 2024-03-11T04:00:00Z -> 2024-03-11. Raises if a
    timestamp is not NY midnight, because then the mapping is ambiguous and we refuse to guess.
    """
    ts = pd.to_datetime(t, utc=True)
    local = ts.dt.tz_convert(NY)
    bad = (local.dt.hour != 0) | (local.dt.minute != 0) | (local.dt.second != 0)
    if bool(bad.any()):
        sample = t[bad].head(3).tolist()
        raise ProviderError(f"alpaca: daily bar timestamps are not New York midnight: {sample}")
    return local.dt.tz_localize(None).dt.normalize()


class AlpacaDataProvider(PriceProvider, CorporateActionProvider, NewsProvider):
    name = "alpaca"
    is_synthetic = False

    def __init__(self, config: Any, http: HttpClient | None = None, clock: Callable[[], pd.Timestamp] | None = None):
        self.config = config
        self.key_env = config.get("providers.alpaca.key_id_env", "ALPACA_PAPER_KEY_ID")
        self.secret_env = config.get("providers.alpaca.secret_env", "ALPACA_PAPER_SECRET_KEY")
        # Credentials are looked up lazily in ``_auth()`` (called by every fetch), NEVER here, so
        # that ``AlpacaDataProvider(config)`` always succeeds -- e.g. the provider registry builds
        # one instance per configured kind up front, before any fetch is attempted, and must not
        # require keys to exist yet. ProviderNotConfigured is raised only when a fetch runs.
        self.base_url = str(config.get("providers.alpaca.data_base_url", "https://data.alpaca.markets")).rstrip("/")
        if not self.base_url.startswith("https://"):
            raise ProviderNotConfigured("providers.alpaca.data_base_url must be an https URL")
        self.feed = str(config.get("providers.alpaca.feed", "iex")).lower()
        fb = config.get("providers.alpaca.feed_fallback", None)
        self.feed_fallback = str(fb).lower() if fb else None
        asof = config.get("providers.alpaca.asof", "-")
        self.asof = str(asof) if asof not in (None, "") else "-"
        self.symbols_per_request = int(config.get("providers.alpaca.symbols_per_request", 100))
        self.page_limit = int(config.get("providers.alpaca.page_limit", 10000))
        self.max_pages = int(config.get("providers.alpaca.max_pages", 100000))
        self.final_time = time.fromisoformat(str(config.get("providers.alpaca.daily_bar_final_time", "20:15")))
        self.sip_min_delay = pd.Timedelta(minutes=float(config.get("providers.alpaca.sip_min_delay_minutes", 16)))
        self.ca_max_days = int(config.get("providers.alpaca.corporate_actions_max_days", 90))
        self.ca_pad_days = int(config.get("providers.alpaca.corporate_actions_pad_days", 30))
        self.ca_types = list(config.get("providers.alpaca.corporate_action_types", DEFAULT_CA_TYPES))
        self.news_per_request = int(config.get("providers.alpaca.news_symbols_per_request", 50))
        self.news_revision_tolerance = pd.Timedelta(
            seconds=float(config.get("providers.alpaca.news_revision_tolerance_seconds", 60)))
        self._clock = clock or _utcnow
        if http is None:
            rpm = float(config.get("providers.alpaca.max_requests_per_minute", 180))
            http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("alpaca-data", rpm / 60.0),
                                          name="alpaca")
        self.http = http
        self._active_feed = self.feed
        self.last_feed_used: str | None = None

    # ------------------------------------------------------------------------------------------
    def _auth(self) -> dict[str, str]:
        # Re-read on every call (never cached on self) so credentials set after construction are
        # picked up, and so the ProviderNotConfigured check happens at fetch time, not at
        # construction time. Built per request and handed straight to the HTTP client; never
        # logged or stored elsewhere.
        key, secret = get_secret(self.key_env), get_secret(self.secret_env)
        if not key or not secret:
            raise ProviderNotConfigured(
                f"Alpaca market data needs environment variables {self.key_env} and {self.secret_env}"
            )
        return {"APCA-API-KEY-ID": key.reveal(), "APCA-API-SECRET-KEY": secret.reveal(),
                "Accept": "application/json"}

    def last_final_session_date(self, now: pd.Timestamp | None = None) -> pd.Timestamp:
        """Latest calendar date whose daily bar is final (after ``daily_bar_final_time`` ET)."""
        now_ny = pd.Timestamp(now if now is not None else self._clock()).tz_convert(NY)
        today = pd.Timestamp(now_ny.date())
        return today if now_ny.time() >= self.final_time else today - pd.Timedelta(days=1)

    # ------------------------------------------------------------------------------------------
    # Bars
    # ------------------------------------------------------------------------------------------
    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        start_d, end_d = to_session(start), to_session(end)
        now = pd.Timestamp(self._clock())
        eff_end = min(end_d, self.last_final_session_date(now))
        attrs: dict[str, Any] = {"asof": self.asof, "adjustment": "raw", "requested_start": str(start_d.date()),
                                 "requested_end": str(end_d.date()), "effective_end": str(eff_end.date())}
        if not syms or eff_end < start_d:
            out = schemas.empty("bars")
            out.attrs.update(attrs, feed=self._active_feed, feeds_used=[], symbols_without_bars=syms)
            return out
        start_ts = ny_midnight(start_d)
        end_ts = min(ny_midnight(eff_end + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1),
                     (now - self.sip_min_delay).tz_convert(NY))
        frames, feeds_used = [], []
        for chunk in chunked(syms, self.symbols_per_request):
            df, feed = self._bars_chunk(chunk, start_ts, end_ts)
            frames.append(df)
            feeds_used.append(feed)
        out = pd.concat(frames, ignore_index=True) if frames else schemas.empty("bars")
        if len(out):
            out = out[(out["date"] >= start_d) & (out["date"] <= eff_end)]
            out = self._dedupe_bars(out)
        out = schemas.conform("bars", out) if len(out) else schemas.empty("bars")
        present = set(out["symbol"]) if len(out) else set()
        out.attrs.update(attrs, feed=self.last_feed_used, feeds_used=sorted(set(feeds_used)),
                         symbols_without_bars=[s for s in syms if s not in present])
        return out

    def _bars_chunk(self, chunk: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[pd.DataFrame, str]:
        feed = self._active_feed
        try:
            df = self._paginate_bars(chunk, feed, start_ts, end_ts)
        except ProviderForbidden as exc:
            if feed == "sip" and self.feed_fallback and self.feed_fallback != feed and self._is_entitlement_error(exc):
                log_event(log, "alpaca SIP refused (403); falling back", fallback=self.feed_fallback,
                          reason=str(exc)[:200])
                self._active_feed = feed = self.feed_fallback
                df = self._paginate_bars(chunk, feed, start_ts, end_ts)
            else:
                raise
        self.last_feed_used = feed
        return df, feed

    @staticmethod
    def _is_entitlement_error(exc: ProviderForbidden) -> bool:
        text = str(exc).lower()
        return exc.code in (42210000, "42210000") or "sip" in text or "subscription" in text

    def _paginate_bars(self, chunk: list[str], feed: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        url = f"{self.base_url}/v2/stocks/bars"
        base = {"symbols": ",".join(chunk), "timeframe": "1Day", "adjustment": "raw", "feed": feed,
                "asof": self.asof, "start": rfc3339(start_ts), "end": rfc3339(end_ts),
                "limit": self.page_limit, "sort": "asc"}
        rows: list[dict[str, Any]] = []
        for page in self._pages(url, base, "bars"):
            bars = page["bars"] or {}
            if not isinstance(bars, dict):
                raise ProviderResponseError("alpaca: 'bars' is not a symbol->list mapping")
            for sym, lst in bars.items():
                for b in lst or []:
                    missing = [k for k in ("t", "o", "h", "l", "c", "v") if k not in b]
                    if missing:
                        raise ProviderResponseError(f"alpaca: bar for {sym} missing fields {missing}")
                    rows.append({"symbol": str(sym).upper(), "t": b["t"], "open": b["o"], "high": b["h"],
                                 "low": b["l"], "close": b["c"], "volume": b["v"], "vwap": b.get("vw", np.nan),
                                 "trade_count": b.get("n", np.nan)})
        if not rows:
            return schemas.empty("bars")
        df = pd.DataFrame(rows)
        df["date"] = bar_time_to_session(df["t"]).to_numpy()
        df["provider"] = f"alpaca:{feed}"
        df["retrieved_at"] = pd.Timestamp(self._clock()).tz_convert("UTC")
        return df.drop(columns="t")

    @staticmethod
    def _dedupe_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Exact duplicates are dropped. Conflicting duplicates are an error, never a silent pick."""
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        exact = df.drop_duplicates(cols)
        dup = exact.duplicated(["symbol", "date"], keep=False)
        if bool(dup.any()):
            sample = exact.loc[dup, ["symbol", "date"]].head(3).to_dict("records")
            raise ProviderError(f"alpaca: conflicting duplicate bars returned, e.g. {sample}")
        return exact

    def _pages(self, url: str, params: dict[str, Any], key: str):
        """Yield pages following next_page_token until it is null (never infer the end from size)."""
        token: str | None = None
        seen: set[str] = set()
        for _ in range(self.max_pages):
            p = dict(params)
            if token:
                p["page_token"] = token
            data = self.http.get_json(url, params=p, headers=self._auth())
            if not isinstance(data, dict) or key not in data or "next_page_token" not in data:
                raise ProviderResponseError(f"alpaca: response from {url.split('//')[-1]} lacks '{key}'/'next_page_token'")
            yield data
            token = data.get("next_page_token")
            if not token:
                return
            if token in seen:
                raise ProviderError("alpaca: pagination token repeated (server loop)")
            seen.add(token)
        raise ProviderError(f"alpaca: more than {self.max_pages} pages; refusing to return a partial result")

    # ------------------------------------------------------------------------------------------
    # Corporate actions
    # ------------------------------------------------------------------------------------------
    def _windows(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        out = []
        s = start
        step = pd.Timedelta(days=max(1, self.ca_max_days) - 1)
        while s <= end:
            e = min(end, s + step)
            out.append((s, e))
            s = e + pd.Timedelta(days=1)
        return out

    def get_corporate_actions(self, symbols: list[str] | None, start: date, end: date) -> pd.DataFrame:
        start_d, end_d = to_session(start), to_session(end)
        pad = pd.Timedelta(days=self.ca_pad_days)
        # The docs do not say which date field start/end filter on (process_date is likely), so
        # the query window is padded and results are filtered on ex_date afterwards.
        windows = self._windows(start_d - pad, end_d + pad)
        sym_chunks: list[list[str] | None]
        if symbols is None:
            sym_chunks = [None]
        else:
            syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
            if not syms:
                return self._empty_actions()
            sym_chunks = chunked(syms, self.symbols_per_request)
        url = f"{self.base_url}/v1/corporate-actions"
        retrieved = pd.Timestamp(self._clock()).tz_convert("UTC")
        mapped: list[dict[str, Any]] = []
        unmapped: Counter[str] = Counter()
        unmapped_records: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for chunk in sym_chunks:
            for ws, we in windows:
                params: dict[str, Any] = {"types": ",".join(self.ca_types), "start": str(ws.date()),
                                          "end": str(we.date()), "limit": 1000, "sort": "asc"}
                if chunk is not None:
                    params["symbols"] = ",".join(chunk)
                for page in self._pages(url, params, "corporate_actions"):
                    groups = page["corporate_actions"] or {}
                    if not isinstance(groups, dict):
                        raise ProviderResponseError("alpaca: 'corporate_actions' is not a mapping")
                    for key, records in groups.items():
                        for rec in records or []:
                            row, why = self._map_action(key, rec)
                            if row is not None:
                                mapped.append(row)
                            elif why == "unmapped_type":
                                unmapped[key] += 1
                                unmapped_records.append(_unmapped_summary(key, rec))
                            else:
                                rejected.append({"type": key, "id": rec.get("id"), "symbol": rec.get("symbol"),
                                                 "reason": why})
        if unmapped:
            log_event(log, "alpaca corporate actions of unmapped types (not applied to returns)",
                      counts=dict(unmapped))
        if rejected:
            log_event(log, "alpaca corporate actions rejected as malformed", count=len(rejected),
                      sample=rejected[:5])
        if mapped:
            df = pd.DataFrame(mapped).drop_duplicates("source_id", keep="last")
            df = df[(df["ex_date"] >= start_d) & (df["ex_date"] <= end_d)]
        else:
            df = pd.DataFrame(columns=["symbol", "ex_date", "action_type", "ratio", "amount", "source_id"])
        if len(df):
            df = df.assign(
                declared_date=pd.NaT,
                available_at=[ny_open_utc(d) for d in df["ex_date"]],
                pit_status=PitStatus.PIT_CONSERVATIVE.value,
                provider=self.name,
                retrieved_at=retrieved,
            )
            out = schemas.conform("corporate_actions", df)
        else:
            out = self._empty_actions()
        if symbols is not None and len(out):
            out = out[out["symbol"].isin({str(s).upper() for s in symbols})].reset_index(drop=True)
        out.attrs.update(unmapped_counts=dict(unmapped), unmapped_records=unmapped_records, rejected=rejected,
                         requested_start=str(start_d.date()), requested_end=str(end_d.date()))
        return out

    @staticmethod
    def _empty_actions() -> pd.DataFrame:
        out = schemas.empty("corporate_actions")
        out.attrs.update(unmapped_counts={}, unmapped_records=[], rejected=[])
        return out

    @staticmethod
    def _map_action(key: str, rec: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        if key not in _SPLIT_KEYS and key not in _DIVIDEND_KEYS:
            return None, "unmapped_type"
        sym, ex, rid = rec.get("symbol"), rec.get("ex_date"), rec.get("id")
        if not sym or not ex or not rid:
            return None, "missing symbol/ex_date/id"
        try:
            ex_date = pd.Timestamp(str(ex)).normalize()
        except (ValueError, TypeError):
            return None, f"unparseable ex_date {ex!r}"
        if key in _SPLIT_KEYS:
            try:
                new, old = float(rec.get("new_rate")), float(rec.get("old_rate"))
            except (TypeError, ValueError):
                return None, "non-numeric split rates"
            if not (np.isfinite(new) and np.isfinite(old) and new > 0 and old > 0):
                return None, "non-positive split rates"
            # ratio = new shares per old share: 4-for-1 forward -> 4.0; 1-for-10 reverse -> 0.1
            return {"symbol": str(sym).upper(), "ex_date": ex_date, "action_type": "split",
                    "ratio": new / old, "amount": np.nan, "source_id": str(rid)}, ""
        currency = rec.get("currency")
        if currency not in (None, "", "USD"):
            return None, f"non-USD dividend ({currency})"
        try:
            rate = float(rec.get("rate"))
        except (TypeError, ValueError):
            return None, "non-numeric dividend rate"
        if not np.isfinite(rate) or rate < 0:
            return None, "invalid dividend rate"
        return {"symbol": str(sym).upper(), "ex_date": ex_date, "action_type": "cash_dividend",
                "ratio": np.nan, "amount": rate, "source_id": str(rid)}, ""

    # ------------------------------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------------------------------
    def get_news(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        start_d, end_d = to_session(start), to_session(end)
        now = pd.Timestamp(self._clock()).tz_convert("UTC")
        start_ts = ny_midnight(start_d)
        end_ts = min(ny_midnight(end_d + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1), now.tz_convert(NY))
        if not syms or end_ts < start_ts:
            return schemas.empty("news")
        url = f"{self.base_url}/v1beta1/news"
        rows: list[dict[str, Any]] = []
        for chunk in chunked(syms, self.news_per_request):
            wanted = set(chunk)
            params = {"symbols": ",".join(chunk), "start": rfc3339(start_ts), "end": rfc3339(end_ts), "limit": 50,
                      "sort": "asc", "include_content": "false", "exclude_contentless": "false"}
            for page in self._pages(url, params, "news"):
                for art in page["news"] or []:
                    for k in ("id", "created_at", "updated_at"):
                        if art.get(k) in (None, ""):
                            raise ProviderResponseError(f"alpaca: news article missing '{k}'")
                    for s in art.get("symbols") or []:
                        s = str(s).upper()
                        if s in wanted:
                            rows.append({"news_id": str(art["id"]), "symbol": s, "headline": art.get("headline") or "",
                                         "summary": art.get("summary") or "", "source": art.get("source") or "",
                                         "url": art.get("url") or "", "created_at": art["created_at"],
                                         "updated_at": art["updated_at"]})
        if not rows:
            return schemas.empty("news")
        df = pd.DataFrame(rows).drop_duplicates(["news_id", "symbol"], keep="last")
        created = pd.to_datetime(df["created_at"], utc=True)
        updated = pd.to_datetime(df["updated_at"], utc=True)
        revised = updated > created + self.news_revision_tolerance
        df = df.assign(
            created_at=created, updated_at=updated, available_at=created,
            pit_status=np.where(revised, PitStatus.PIT_CONSERVATIVE.value, PitStatus.PIT.value),
            provider=self.name, retrieved_at=now,
        )
        return schemas.conform("news", df)


    def get_news_market(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp, keep_summary: bool = False) -> pd.DataFrame:
        """ALL news in [start_ts, end_ts] (no symbol filter), one row per (article, tagged symbol).

        Every tagged symbol is kept, so ``rows per news_id`` = the article's tag count (used to tell
        company-specific articles from multi-ticker roundups). The API selects the window by
        ``updated_at`` (sort/pagination fields, docs/EXTERNAL-SERVICES.md Alpaca items 31-33), so an
        old article edited inside the window is returned too; availability is always ``created_at``.
        Summaries are dropped by default to keep universe-wide history small in memory."""
        now = pd.Timestamp(self._clock()).tz_convert("UTC")
        end_ts = min(pd.Timestamp(end_ts), now)
        if end_ts <= pd.Timestamp(start_ts):
            return schemas.empty("news")
        params = {"start": rfc3339(pd.Timestamp(start_ts)), "end": rfc3339(end_ts), "limit": 50, "sort": "asc",
                  "include_content": "false", "exclude_contentless": "false"}
        rows: list[dict[str, Any]] = []
        for page in self._pages(f"{self.base_url}/v1beta1/news", params, "news"):
            for art in page["news"] or []:
                for k in ("id", "created_at", "updated_at"):
                    if art.get(k) in (None, ""):
                        raise ProviderResponseError(f"alpaca: news article missing '{k}'")
                for s in dict.fromkeys(str(x).upper() for x in (art.get("symbols") or [])):
                    rows.append({"news_id": str(art["id"]), "symbol": s, "headline": art.get("headline") or "",
                                 "summary": (art.get("summary") or "") if keep_summary else "",
                                 "source": art.get("source") or "", "url": art.get("url") or "",
                                 "created_at": art["created_at"], "updated_at": art["updated_at"]})
        if not rows:
            return schemas.empty("news")
        df = pd.DataFrame(rows).drop_duplicates(["news_id", "symbol"], keep="last")
        created = pd.to_datetime(df["created_at"], utc=True)
        updated = pd.to_datetime(df["updated_at"], utc=True)
        revised = updated > created + self.news_revision_tolerance
        df = df.assign(created_at=created, updated_at=updated, available_at=created,
                       pit_status=np.where(revised, PitStatus.PIT_CONSERVATIVE.value, PitStatus.PIT.value),
                       provider=self.name, retrieved_at=now)
        return schemas.conform("news", df)


def ny_open_utc(d: pd.Timestamp) -> pd.Timestamp:
    """09:30 America/New_York on date ``d``, in UTC (DST-aware)."""
    return pd.Timestamp(datetime.combine(pd.Timestamp(d).date(), time(9, 30))).tz_localize(NY).tz_convert("UTC")


def _unmapped_summary(key: str, rec: dict[str, Any]) -> dict[str, Any]:
    sym = rec.get("symbol") or rec.get("acquiree_symbol") or rec.get("source_symbol") or rec.get("old_symbol")
    when = rec.get("ex_date") or rec.get("effective_date") or rec.get("process_date")
    return {"type": key, "id": rec.get("id"), "symbol": sym, "date": when}
