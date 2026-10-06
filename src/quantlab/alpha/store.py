"""Survivorship-free daily research store (Alpaca SIP, 2016-01-01 .. 2024-12-31). PAPER research only.

Why this exists: QuantLab's price store held only CURRENT listings, so every long backtest was
flattered (docs/REAL-MONEY-READINESS.md). This store downloads every symbol Alpaca knows, active AND
inactive (delisted, acquired, bankrupt, renamed), so the historical universe on each date is the one
that actually traded then.

Facts this relies on (probed 2026-10-06, recorded in docs/ALPHA-DISCOVERY-PLAN.md):
  * ``/v2/assets?status=inactive`` lists ~19k inactive symbols (~2.9k formerly on NYSE/Nasdaq/Arca/
    BATS/AMEX; the rest OTC, many of which were exchange-listed before delisting, e.g. BBBYQ).
  * Bars with the DEFAULT ``asof`` (entity mapping) return a company's whole history under the symbol
    it is known by today: META includes FB's history; BBBYQ includes Bed Bath & Beyond's exchange
    history down to $0.075; WFM (acquired 2017), TWTR, SIVB, FRC return their full histories. With
    ``asof='-'`` a recycled ticker splices different companies (META 2021-06 = an ETF), so the
    default mapping is used. Each symbol's series is one entity.
  * Alpaca corporate actions only start ~2020, so returns use ``adjustment=all`` bars (split +
    dividend adjusted). Adjusted RATIOS are point-in-time correct (a later adjustment rescales the
    whole past by a constant); adjusted LEVELS are not, so every level rule (price >= $5, dollar
    volume) uses the RAW download.

Holdout: the store stops at RESEARCH_END. Data from HOLDOUT_START on is never downloaded here, so
no research code can read it by accident (docs/ALPHA-DISCOVERY-PLAN.md, splits).

Secrets come only from the environment and are never logged.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

RESEARCH_START = "2016-01-01"
RESEARCH_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
FEED = "sip"
DATA_URL = "https://data.alpaca.markets"
TRADING_URL = "https://paper-api.alpaca.markets"      # paper host only; used for the asset list

BENCHMARK_ETFS = ("SPY", "QQQ", "IWM", "MDY", "DIA", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
                  "XLRE", "XLU", "XLV", "XLY", "TLT", "IEF", "SHY", "HYG", "LQD", "UUP", "GLD", "USO",
                  "DBC", "VIXY", "EFA", "EEM")


_VALID_TICKER = re.compile(r"[A-Z][A-Z0-9]{0,5}(\.[A-Z0-9]{1,3})?")


def valid_ticker(symbol: str) -> bool:
    """Exchange ticker shape. Alpaca's inactive list also holds placeholders (CUSIP-like codes,
    escrow/CVR/contra lines such as '0029900E0' or '866CNT017') that the bars API rejects."""
    return bool(_VALID_TICKER.fullmatch(symbol or ""))


def store_dir() -> Path:
    root = Path(os.environ.get("QUANTLAB_ALPHA_DIR", Path(__file__).resolve().parents[3] / "var" / "alpha"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _auth() -> dict[str, str]:
    kid, sec = os.environ.get("ALPACA_PAPER_KEY_ID"), os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if not kid or not sec:
        raise RuntimeError("ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY not set: data is UNKNOWN, refusing")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


class RateLimiter:
    """At most ``per_minute`` calls per rolling minute (shared Alpaca account: leave the bot headroom)."""

    def __init__(self, per_minute: int = 120):
        self.min_interval = 60.0 / per_minute
        self._last = 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()


def _get(session: requests.Session, url: str, params: dict[str, Any], limiter: RateLimiter,
         retries: int = 6) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(retries):
        limiter.wait()
        try:
            r = session.get(url, params=params, timeout=90)
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"alpaca {url} -> HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError(f"alpaca {url}: gave up after {retries} attempts")


# --- security master ------------------------------------------------------------------------------
_TYPE_RULES: tuple[tuple[str, str], ...] = (       # ORDERED: first match wins; never guess COMMON from silence
    ("TEST", r"\btest\b"),
    ("ETF", r"\b(etf|etn|etp)s?\b|exchange[- ]traded|\bishares\b|\bspdr\b|proshares|direxion|vaneck|"
            r"wisdomtree|global x\b|\bxtrackers\b|velocityshares|\bipath\b|microsectors|graniteshares|"
            r"ultrashort|ultrapro|\b[23]x\b|\bleveraged\b|\binverse\b|\bindex (fund|shares)\b"),
    ("ADR", r"american depositary|\badrs?\b|\bads\b|depositary receipt"),
    ("WARRANT", r"\bwarrants?\b|\bwts?\b"),
    ("RIGHT", r"\brights?\b"),
    ("LP", r"\bl\.\s?p\.?(\s|$)|\blp\b|limited partnership"),
    ("UNIT", r"\bunits?\b"),
    ("PREFERRED", r"preferred|\bpfd\b|\bprf\b|cumulative|perpetual|fixed[- ]to[- ]floating|"
                  r"depositary shares?,? each|\bseries [a-z]\b"),
    ("NOTE", r"\bnotes?\b|debentures?|\bbonds?\b|due \d{4}|% senior|subordinated"),
    ("FUND", r"\bfund\b|closed[- ]end|municipal|\bincome (trust|shares)\b|\bportfolio trust\b|"
             r"\b(equity|income|growth|value|dividend|opportunit(y|ies)|global|strategic|premium)\s+(trust|fund)\b"),
    ("ROYALTY_TRUST", r"royalty trust|\btrust units?\b"),
    ("SPAC", r"acquisition (corp|corporation|company|co\b|ltd|limited|inc)|blank check|\bspac\b"),
)
_TEST_SYMBOLS = re.compile(r"^(ZVZZ|ZXZZ|ZWZZ|ZJZZ|ZBZZ|ZAZZ|NTEST|ATEST|CTEST|PTEST|ZZK|ZZT)")
_RULES = tuple((t, re.compile(p, re.I)) for t, p in _TYPE_RULES)


def classify_security(symbol: str, name: str | None, etf_flag: str | None = None) -> str:
    """COMMON / ETF / FUND / PREFERRED / NOTE / WARRANT / RIGHT / UNIT / SPAC / ADR / LP / ROYALTY_TRUST /
    TEST / UNKNOWN. Name-based (ASSUMED: Alpaca gives no security type); a Nasdaq Trader ETF flag, when
    known for a currently listed symbol, overrides."""
    if _TEST_SYMBOLS.match(symbol or ""):
        return "TEST"
    if etf_flag == "Y":
        return "ETF"
    n = (name or "").strip()
    if not n:
        return "UNKNOWN"
    for t, rx in _RULES:
        if rx.search(n):
            return t
    return "COMMON"


def fetch_assets(limiter: RateLimiter | None = None) -> pd.DataFrame:
    limiter = limiter or RateLimiter()
    s = requests.Session()
    s.headers.update(_auth())
    rows = []
    for status in ("active", "inactive"):
        limiter.wait()
        r = s.get(f"{TRADING_URL}/v2/assets", params={"status": status, "asset_class": "us_equity"}, timeout=120)
        r.raise_for_status()
        for a in r.json():
            rows.append({"symbol": a["symbol"], "name": a.get("name") or "", "exchange": a.get("exchange"),
                         "status": status, "tradable": bool(a.get("tradable")), "asset_id": a.get("id")})
    df = pd.DataFrame(rows).drop_duplicates("symbol", keep="first")
    df["retrieved_at"] = pd.Timestamp.now(tz="UTC")
    return df


def nasdaq_etf_flags() -> dict[str, str]:
    """Current Nasdaq Trader ETF flags (Y/N) for listed symbols; empty if unreachable (then names decide)."""
    flags: dict[str, str] = {}
    try:
        for url, sym_col, etf_col in (("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", 0, 6),
                                      ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", 0, 4)):
            txt = requests.get(url, timeout=60).text.splitlines()
            for line in txt[1:]:
                parts = line.split("|")
                if len(parts) > etf_col and not line.startswith("File Creation"):
                    flags[parts[sym_col].replace(".", ".")] = parts[etf_col]
    except requests.RequestException:
        return {}
    return flags


# --- bar download ------------------------------------------------------------------------------------
def _batch_path(kind: str, i: int) -> Path:
    d = store_dir() / "download" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"batch_{i:05d}.parquet"


def download_bars(symbols: list[str], adjustment: str, *, start: str = RESEARCH_START, end: str = RESEARCH_END,
                  batch_size: int = 100, per_minute: int = 120, log_every: int = 25) -> dict[str, Any]:
    """Resumable: one parquet per symbol batch; batches already on disk are skipped."""
    if pd.Timestamp(end) >= pd.Timestamp(HOLDOUT_START):
        raise ValueError(f"end {end} reaches the locked holdout ({HOLDOUT_START}); refusing")
    if adjustment not in ("raw", "all"):
        raise ValueError(adjustment)
    limiter = RateLimiter(per_minute)
    s = requests.Session()
    s.headers.update(_auth())
    syms = sorted(set(symbols))
    batches = [syms[i:i + batch_size] for i in range(0, len(syms), batch_size)]
    n_req = 0
    t0 = time.time()
    bad: list[str] = []

    def fetch(chunk: list[str]) -> list[tuple]:
        nonlocal n_req
        params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": f"{start}T00:00:00Z",
                  "end": f"{end}T23:59:59Z", "adjustment": adjustment, "feed": FEED, "limit": 10000,
                  "sort": "asc"}
        rows: list[tuple] = []
        token = None
        while True:
            p = dict(params, page_token=token) if token else params
            try:
                j = _get(s, f"{DATA_URL}/v2/stocks/bars", p, limiter)
            except RuntimeError as exc:
                if ("HTTP 400" in str(exc) or "HTTP 422" in str(exc)) and token is None:
                    if len(chunk) == 1:           # an invalid symbol: record it, never guess
                        bad.append(chunk[0])
                        return []
                    mid = len(chunk) // 2
                    return fetch(chunk[:mid]) + fetch(chunk[mid:])
                raise
            n_req += 1
            for sym, lst in (j.get("bars") or {}).items():
                for b in lst or []:
                    rows.append((sym, b["t"], b.get("o"), b.get("h"), b.get("l"), b.get("c"), b.get("v"),
                                 b.get("n"), b.get("vw")))
            token = j.get("next_page_token")
            if not token:
                return rows

    for bi, chunk in enumerate(batches):
        path = _batch_path(adjustment, bi)
        if path.exists():
            continue
        df = pd.DataFrame(fetch(chunk), columns=["symbol", "t", "open", "high", "low", "close", "volume",
                                                 "trade_count", "vwap"])
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        if bi % log_every == 0:
            print(f"[{adjustment}] batch {bi + 1}/{len(batches)} requests={n_req} "
                  f"elapsed={time.time() - t0:.0f}s rows={len(df)} bad_symbols={len(bad)}", flush=True)
    if bad:
        (store_dir() / f"bad_symbols_{adjustment}.json").write_text(json.dumps(sorted(bad)))
    return {"adjustment": adjustment, "batches": len(batches), "requests": n_req, "bad_symbols": len(bad),
            "seconds": time.time() - t0}


def _load_kind(kind: str) -> pd.DataFrame:
    files = sorted((store_dir() / "download" / kind).glob("batch_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # daily bar t = midnight New York of the session, in UTC -> NY calendar date (never the UTC date)
    t = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = t.dt.tz_localize(None).dt.normalize()
    return df.drop(columns="t")


@dataclass
class StoreManifest:
    built_at: str
    feed: str
    asof: str
    start: str
    end: str
    holdout_start: str
    n_symbols_requested: int
    n_symbols_with_bars: int
    n_duplicates_dropped: int
    n_rows: int
    notes: list[str]


def build_store(assets: pd.DataFrame) -> StoreManifest:
    """Join raw + adjusted downloads into one long table (date, symbol) and a security master."""
    raw = _load_kind("raw")
    adj = _load_kind("all")
    if raw["date"].max() >= pd.Timestamp(HOLDOUT_START) or adj["date"].max() >= pd.Timestamp(HOLDOUT_START):
        raise RuntimeError("holdout data found in the research store download: refusing to build")
    adj = adj.rename(columns={c: f"adj_{c}" for c in ("open", "high", "low", "close", "volume", "vwap")})
    df = raw.merge(adj[["symbol", "date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]],
                   on=["symbol", "date"], how="left")
    # the same entity can come back under two query symbols (e.g. old + new ticker): keep one
    fp = df.groupby("symbol").agg(first=("date", "min"), last=("date", "max"), n=("date", "size"),
                                  c0=("close", "first"), c1=("close", "last"), vol=("volume", "sum"))
    fp["key"] = list(zip(fp["first"], fp["last"], fp["n"], fp["c0"].round(4), fp["c1"].round(4), fp["vol"].round(0)))
    status = assets.set_index("symbol")["status"].reindex(fp.index).fillna("unknown")
    fp["rank"] = (status != "active").astype(int)
    fp = fp.sort_values(["key", "rank"]).reset_index()
    keep = set(fp.drop_duplicates("key", keep="first")["symbol"])
    dropped = len(fp) - len(keep)
    df = df[df["symbol"].isin(keep)].sort_values(["symbol", "date"]).reset_index(drop=True)
    for c in ("open", "high", "low", "close", "vwap", "adj_open", "adj_high", "adj_low", "adj_close"):
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    root = store_dir()
    df.to_parquet(root / "bars_daily.parquet", index=False)
    master = assets.copy()
    master["has_bars"] = master["symbol"].isin(keep)
    span = df.groupby("symbol")["date"].agg(["min", "max", "size"]).rename(
        columns={"min": "first_bar", "max": "last_bar", "size": "n_bars"})
    master = master.merge(span, left_on="symbol", right_index=True, how="left")
    master.to_parquet(root / "security_master.parquet", index=False)
    m = StoreManifest(
        built_at=pd.Timestamp.now(tz="UTC").isoformat(), feed=FEED, asof="default (entity mapping)",
        start=RESEARCH_START, end=RESEARCH_END, holdout_start=HOLDOUT_START,
        n_symbols_requested=int(len(assets)), n_symbols_with_bars=int(len(keep)), n_duplicates_dropped=int(dropped),
        n_rows=int(len(df)),
        notes=["security type is name-based (ASSUMED); Alpaca exposes none",
               "returns from adjustment=all bars; level rules from raw bars",
               "pure-OTC symbols have no SIP bars and drop out naturally"])
    (root / "manifest.json").write_text(json.dumps(m.__dict__, indent=2, default=str))
    return m


def load_bars(columns: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(store_dir() / "bars_daily.parquet", columns=list(columns) if columns else None)
    if "date" in df and df["date"].max() >= pd.Timestamp(HOLDOUT_START):
        raise RuntimeError("research store contains holdout dates: refusing to load")
    return df


def load_master() -> pd.DataFrame:
    return pd.read_parquet(store_dir() / "security_master.parquet")


def main() -> None:   # pragma: no cover - network
    import argparse
    ap = argparse.ArgumentParser(description="Build the survivorship-free alpha research store (paper research).")
    ap.add_argument("--per-minute", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=100)
    a = ap.parse_args()
    root = store_dir()
    assets_path = root / "assets.parquet"
    if assets_path.exists():
        assets = pd.read_parquet(assets_path)
    else:
        assets = fetch_assets()
        flags = nasdaq_etf_flags()
        assets["nasdaq_etf_flag"] = assets["symbol"].map(flags)
        assets["sec_type"] = [classify_security(s, n, f) for s, n, f in
                              zip(assets["symbol"], assets["name"], assets["nasdaq_etf_flag"])]
        assets.to_parquet(assets_path, index=False)
    print(assets["sec_type"].value_counts().to_dict(), flush=True)
    syms = sorted({x for x in assets["symbol"] if valid_ticker(x)} | set(BENCHMARK_ETFS))
    print(f"symbols to request: {len(syms)} (of {assets['symbol'].nunique()} listed)", flush=True)
    for kind in ("raw", "all"):
        print(download_bars(syms, kind, batch_size=a.batch_size, per_minute=a.per_minute), flush=True)
    print(build_store(assets).__dict__, flush=True)


if __name__ == "__main__":   # pragma: no cover
    main()


def download_minutes(symbols: Iterable[str], *, start: str = RESEARCH_START, end: str = RESEARCH_END,
                     per_minute: int = 100) -> dict[str, int]:
    """1-minute SIP bars (raw) for a few symbols, e.g. SPY/QQQ for intraday studies. Resumable per year."""
    if pd.Timestamp(end) >= pd.Timestamp(HOLDOUT_START):
        raise ValueError("refusing holdout dates")
    limiter = RateLimiter(per_minute)
    s = requests.Session()
    s.headers.update(_auth())
    out = {}
    d = store_dir() / "minute"
    d.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            path = d / f"{sym}_{y}.parquet"
            if path.exists():
                continue
            params = {"symbols": sym, "timeframe": "1Min", "start": f"{y}-01-01T00:00:00Z",
                      "end": f"{min(pd.Timestamp(f'{y}-12-31'), pd.Timestamp(end)).date()}T23:59:59Z",
                      "adjustment": "raw", "feed": FEED, "limit": 10000, "sort": "asc"}
            rows, token = [], None
            while True:
                p = dict(params, page_token=token) if token else params
                j = _get(s, f"{DATA_URL}/v2/stocks/bars", p, limiter)
                for b in (j.get("bars") or {}).get(sym, []) or []:
                    rows.append((b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
                token = j.get("next_page_token")
                if not token:
                    break
            df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
            df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
            df.to_parquet(path, index=False)
            out[f"{sym}_{y}"] = len(df)
            print(sym, y, len(df), flush=True)
    return out
