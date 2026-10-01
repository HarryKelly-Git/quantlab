"""Alpaca options data: contracts, INDICATIVE snapshots, historical daily option bars, and the
underlying's spot / daily bars needed to describe a thesis. Read-only (GET) everywhere.

Verified on this account (2026-10-01, docs/OPTIONS.md):
  * ``GET {paper-api}/v2/options/contracts`` -> ``{"option_contracts": [...], "next_page_token"}``.
    ``open_interest`` is often null and is a CURRENT value only (no history).
  * ``GET {data}/v1beta1/options/snapshots/{underlying}?feed=indicative`` -> ``{"snapshots": {occ:
    {latestQuote, latestTrade, dailyBar, minuteBar, prevDailyBar, greeks?, impliedVolatility?}},
    "next_page_token"}``. ``greeks``/``impliedVolatility`` are present for MOST but not all contracts
    (TGT 2026-09-30: 916 of 1232); absent ones are UNKNOWN until computed as MODEL in compare.py.
  * ``feed=opra`` -> HTTP 403 "OPRA agreement is not signed". Indicative quotes are NOT the OPRA
    NBBO: every quote carries ``feed='indicative'`` and is treated as approximate.
  * ``GET {data}/v1beta1/options/bars`` (1Day) works for expired contracts from Feb 2024: real
    traded prices, used only for marking (marking.py). There are NO historical quotes (404).

The HTTP layer is injectable (``http``/``trading_http``: anything with ``get_json(url, params,
headers)``), so tests never touch the network. Credentials come from ``get_secret`` per request and
are never stored on the client or logged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from quantlab.config import PAPER_TRADING_BASE_URL
from quantlab.data.providers.alpaca_data import bar_time_to_session, rfc3339, ny_midnight
from quantlab.data.providers.base import ProviderError, ProviderNotConfigured
from quantlab.data.providers.http import (
    HttpClient, ProviderForbidden, ProviderResponseError, chunked, shared_rate_limiter,
)
from quantlab.execution.broker import LiveTradingForbidden
from quantlab.logging_setup import get_logger, log_event
from quantlab.options.occ import is_option_symbol, parse_occ
from quantlab.secrets import get_secret

log = get_logger("options.data")

INDICATIVE = "indicative"


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _ts(v: Any) -> pd.Timestamp | None:
    if not v:
        return None
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    expiration: date
    strike: float
    type: str                      # call | put
    style: str = "american"
    multiplier: float = 100.0
    size: float = 100.0
    open_interest: float | None = None
    open_interest_date: str | None = None
    close_price: float | None = None
    tradable: bool = True
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expiration"] = self.expiration.isoformat()
        return d


@dataclass(frozen=True)
class OptionQuote:
    """One indicative snapshot. Prices are per share (x multiplier per contract)."""

    symbol: str
    bid: float | None
    ask: float | None
    bid_size: float | None = None
    ask_size: float | None = None
    quote_time: pd.Timestamp | None = None
    feed: str = INDICATIVE
    last_trade_price: float | None = None
    last_trade_time: pd.Timestamp | None = None
    daily_volume: float | None = None
    vendor_iv: float | None = None
    vendor_delta: float | None = None
    vendor_greeks: dict[str, float] | None = None

    @property
    def two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0

    @property
    def mid(self) -> float | None:
        """Midpoint for DESCRIPTION only (spread %, model IV). Never a fill price."""
        return 0.5 * (self.bid + self.ask) if self.two_sided and self.ask >= self.bid else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("quote_time", "last_trade_time"):
            d[k] = d[k].isoformat() if d[k] is not None else None
        return d


@dataclass(frozen=True)
class UnderlyingSpot:
    symbol: str
    price: float
    source: str                    # e.g. "alpaca:iex latestTrade"
    at: pd.Timestamp | None


def parse_contract(raw: dict[str, Any]) -> OptionContract:
    sym = str(raw.get("symbol") or "")
    if not is_option_symbol(sym):
        raise ProviderResponseError(f"alpaca options: contract symbol is not OCC: {sym!r}")
    occ = parse_occ(sym)
    exp = raw.get("expiration_date")
    strike = _num(raw.get("strike_price"))
    typ = str(raw.get("type") or "").lower()
    if not exp or strike is None or typ not in ("call", "put"):
        raise ProviderResponseError(f"alpaca options: contract {sym} lacks expiration/strike/type")
    expiration = date.fromisoformat(str(exp)[:10])
    if expiration != occ.expiration or abs(strike - occ.strike) > 1e-6 or typ != occ.type:
        # the OCC symbol is the identity: a disagreeing record is refused, never "fixed"
        raise ProviderResponseError(f"alpaca options: contract {sym} fields disagree with its OCC symbol")
    return OptionContract(
        symbol=sym, underlying=str(raw.get("underlying_symbol") or raw.get("root_symbol") or occ.root).upper(),
        expiration=expiration, strike=strike, type=typ, style=str(raw.get("style") or "american"),
        multiplier=_num(raw.get("multiplier")) or 100.0, size=_num(raw.get("size")) or 100.0,
        open_interest=_num(raw.get("open_interest")), open_interest_date=raw.get("open_interest_date"),
        close_price=_num(raw.get("close_price")), tradable=bool(raw.get("tradable", False)),
        status=str(raw.get("status") or "unknown"))


def parse_snapshot(symbol: str, raw: dict[str, Any], feed: str = INDICATIVE) -> OptionQuote:
    q = raw.get("latestQuote") or {}
    t = raw.get("latestTrade") or {}
    bar = raw.get("dailyBar") or {}
    g = raw.get("greeks") or None
    greeks = {k: float(v) for k, v in g.items() if _num(v) is not None} if isinstance(g, dict) else None
    return OptionQuote(
        symbol=symbol, bid=_num(q.get("bp")), ask=_num(q.get("ap")), bid_size=_num(q.get("bs")),
        ask_size=_num(q.get("as")), quote_time=_ts(q.get("t")), feed=feed, last_trade_price=_num(t.get("p")),
        last_trade_time=_ts(t.get("t")), daily_volume=_num(bar.get("v")),
        vendor_iv=_num(raw.get("impliedVolatility")),
        vendor_delta=(greeks or {}).get("delta"), vendor_greeks=greeks or None)


class OptionsDataClient:
    """Read-only Alpaca options data. ``feed`` is always ``indicative`` on this account."""

    def __init__(self, config: Any, *, http: Any = None, trading_http: Any = None,
                 clock: Callable[[], pd.Timestamp] | None = None):
        self.config = config
        self.key_env = config.get("providers.alpaca.key_id_env", "ALPACA_PAPER_KEY_ID")
        self.secret_env = config.get("providers.alpaca.secret_env", "ALPACA_PAPER_SECRET_KEY")
        self.data_base = str(config.get("providers.alpaca.data_base_url", "https://data.alpaca.markets")).rstrip("/")
        self.trading_base = str(config.get("providers.alpaca.trading_base_url", PAPER_TRADING_BASE_URL)).rstrip("/")
        if self.trading_base != PAPER_TRADING_BASE_URL:
            raise LiveTradingForbidden(f"options data refuses non-paper trading host {self.trading_base!r}")
        if not self.data_base.startswith("https://"):
            raise ProviderNotConfigured("providers.alpaca.data_base_url must be an https URL")
        self.feed = str(config.get("options.feed", INDICATIVE)).lower()
        self.underlying_feed = str(config.get("options.underlying_feed", "iex")).lower()
        self.bars_feed = str(config.get("providers.alpaca.feed", "sip")).lower()
        fb = config.get("providers.alpaca.feed_fallback", "iex")
        self.bars_feed_fallback = str(fb).lower() if fb else None
        rpm = float(config.get("providers.alpaca.max_requests_per_minute", 180))
        if http is None:
            http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("alpaca-data", rpm / 60.0),
                                          name="alpaca-options")
        if trading_http is None:
            # Trading API: 200 requests/min per account, shared with the paper runner's own calls
            trading_http = HttpClient.from_config(config, rate_limiter=shared_rate_limiter("alpaca-trading", rpm / 60.0),
                                                  name="alpaca-options-contracts")
        self.http = http
        self.trading_http = trading_http
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.max_pages = int(config.get("providers.alpaca.max_pages", 100000))

    # -- plumbing -------------------------------------------------------------------------------
    def _auth(self) -> dict[str, str]:
        key, secret = get_secret(self.key_env), get_secret(self.secret_env)
        if not key or not secret:
            raise ProviderNotConfigured(f"Alpaca options data needs environment variables {self.key_env} and "
                                        f"{self.secret_env}")
        return {"APCA-API-KEY-ID": key.reveal(), "APCA-API-SECRET-KEY": secret.reveal(), "Accept": "application/json"}

    def _pages(self, http: Any, url: str, params: dict[str, Any], key: str):
        token: str | None = None
        seen: set[str] = set()
        for _ in range(self.max_pages):
            p = dict(params)
            if token:
                p["page_token"] = token
            if http is self.trading_http and not url.startswith(PAPER_TRADING_BASE_URL + "/"):
                raise LiveTradingForbidden(f"refusing trading-host request to {url.split('?')[0]!r}")
            data = http.get_json(url, params=p, headers=self._auth())
            if not isinstance(data, dict) or key not in data or "next_page_token" not in data:
                raise ProviderResponseError(f"alpaca options: response lacks '{key}'/'next_page_token'")
            yield data
            token = data.get("next_page_token")
            if not token:
                return
            if token in seen:
                raise ProviderError("alpaca options: pagination token repeated (server loop)")
            seen.add(token)
        raise ProviderError(f"alpaca options: more than {self.max_pages} pages; refusing a partial result")

    # -- contracts ------------------------------------------------------------------------------
    def contracts(self, underlying: str, *, expiration_gte: date | None = None, expiration_lte: date | None = None,
                  type_: str | None = None, status: str = "active") -> list[OptionContract]:
        params: dict[str, Any] = {"underlying_symbols": underlying.upper(), "status": status, "limit": 1000}
        if expiration_gte:
            params["expiration_date_gte"] = str(expiration_gte)
        if expiration_lte:
            params["expiration_date_lte"] = str(expiration_lte)
        if type_:
            params["type"] = type_
        out: list[OptionContract] = []
        for page in self._pages(self.trading_http, f"{self.trading_base}/v2/options/contracts", params,
                                "option_contracts"):
            for raw in page["option_contracts"] or []:
                out.append(parse_contract(raw))
        return out

    # -- indicative snapshots ---------------------------------------------------------------------
    def snapshots(self, underlying: str, *, expiration_date: date | None = None, expiration_gte: date | None = None,
                  expiration_lte: date | None = None, type_: str | None = None) -> dict[str, OptionQuote]:
        if self.feed != INDICATIVE:
            raise ProviderNotConfigured("only the indicative options feed is available (opra -> HTTP 403)")
        params: dict[str, Any] = {"feed": self.feed, "limit": 1000}
        if expiration_date:
            params["expiration_date"] = str(expiration_date)
        if expiration_gte:
            params["expiration_date_gte"] = str(expiration_gte)
        if expiration_lte:
            params["expiration_date_lte"] = str(expiration_lte)
        if type_:
            params["type"] = type_
        out: dict[str, OptionQuote] = {}
        try:
            for page in self._pages(self.http, f"{self.data_base}/v1beta1/options/snapshots/{underlying.upper()}",
                                    params, "snapshots"):
                snaps = page["snapshots"] or {}
                if not isinstance(snaps, dict):
                    raise ProviderResponseError("alpaca options: 'snapshots' is not a symbol->snapshot mapping")
                for sym, raw in snaps.items():
                    out[str(sym)] = parse_snapshot(str(sym), raw or {}, self.feed)
        except ProviderForbidden as exc:
            log_event(log, "options snapshot refused (403)", underlying=underlying, error=str(exc)[:200])
            raise
        return out

    # -- historical option bars (real traded prices; marking only) --------------------------------
    def option_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        cols = ["symbol", "date", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        syms = sorted({s for s in symbols if is_option_symbol(s)})
        if not syms:
            return pd.DataFrame(columns=cols)
        rows: list[dict[str, Any]] = []
        start_ts, end_ts = ny_midnight(start), ny_midnight(pd.Timestamp(end) + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1)
        for chunk in chunked(syms, 100):
            params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": rfc3339(start_ts),
                      "end": rfc3339(end_ts), "limit": 10000, "sort": "asc"}
            for page in self._pages(self.http, f"{self.data_base}/v1beta1/options/bars", params, "bars"):
                bars = page["bars"] or {}
                if not isinstance(bars, dict):
                    raise ProviderResponseError("alpaca options: 'bars' is not a symbol->list mapping")
                for sym, lst in bars.items():
                    for b in lst or []:
                        if any(k not in b for k in ("t", "c", "v")):
                            raise ProviderResponseError(f"alpaca options: bar for {sym} missing t/c/v")
                        rows.append({"symbol": str(sym), "t": b["t"], "open": _num(b.get("o")), "high": _num(b.get("h")),
                                     "low": _num(b.get("l")), "close": _num(b["c"]), "volume": _num(b["v"]),
                                     "trade_count": _num(b.get("n")), "vwap": _num(b.get("vw"))})
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        df["date"] = bar_time_to_session(df["t"]).to_numpy()
        return df.drop(columns="t")[cols].drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)

    # -- underlying -----------------------------------------------------------------------------
    def underlying_spot(self, symbol: str) -> UnderlyingSpot:
        url = f"{self.data_base}/v2/stocks/{symbol.upper()}/snapshot"
        data = self.http.get_json(url, params={"feed": self.underlying_feed}, headers=self._auth())
        if not isinstance(data, dict):
            raise ProviderResponseError("alpaca: stock snapshot is not an object")
        trade = data.get("latestTrade") or {}
        price = _num(trade.get("p"))
        if price is None or price <= 0:
            raise ProviderError(f"alpaca: no usable latest trade for {symbol} on feed {self.underlying_feed}")
        return UnderlyingSpot(symbol=symbol.upper(), price=price, source=f"alpaca:{self.underlying_feed} latestTrade",
                              at=_ts(trade.get("t")))

    def underlying_daily_bars(self, symbol: str, start: date, end: date, adjustment: str = "all") -> pd.DataFrame:
        """Daily bars of the underlying. ``adjustment='all'`` is used ONLY to describe the current
        distribution of past returns (a descriptive input known today), never for PIT research;
        marking uses ``raw``. Records the feed actually used in ``attrs['feed']``."""
        feed = self.bars_feed
        try:
            df = self._stock_bars(symbol, start, end, adjustment, feed)
        except ProviderForbidden:
            if not self.bars_feed_fallback or self.bars_feed_fallback == feed:
                raise
            feed = self.bars_feed_fallback
            df = self._stock_bars(symbol, start, end, adjustment, feed)
        df.attrs["feed"] = feed
        df.attrs["adjustment"] = adjustment
        return df

    def _stock_bars(self, symbol: str, start: date, end: date, adjustment: str, feed: str) -> pd.DataFrame:
        now = pd.Timestamp(self._clock()).tz_convert("America/New_York")
        end_ts = min(ny_midnight(pd.Timestamp(end) + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1),
                     now - pd.Timedelta(minutes=16))
        params = {"symbols": symbol.upper(), "timeframe": "1Day", "adjustment": adjustment, "feed": feed,
                  "start": rfc3339(ny_midnight(start)), "end": rfc3339(end_ts), "limit": 10000, "sort": "asc",
                  "asof": "-"}
        rows: list[dict[str, Any]] = []
        for page in self._pages(self.http, f"{self.data_base}/v2/stocks/bars", params, "bars"):
            for sym, lst in (page["bars"] or {}).items():
                for b in lst or []:
                    rows.append({"symbol": str(sym).upper(), "t": b["t"], "open": _num(b.get("o")),
                                 "high": _num(b.get("h")), "low": _num(b.get("l")), "close": _num(b.get("c")),
                                 "volume": _num(b.get("v"))})
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        df["date"] = bar_time_to_session(df["t"]).to_numpy()
        return df.drop(columns="t")[cols].sort_values("date").reset_index(drop=True)
