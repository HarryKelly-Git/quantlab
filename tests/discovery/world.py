"""A small, hand-built SYNTHETIC world where each supported discovery family has one clear example.
Not a test module (no test_ prefix).

Special symbols (the last session is the decision session D):
  MOMO  steady strong uptrend for 120 sessions            -> momentum, relative strength
  VOLX  ordinary, then 8x volume on an up day at D         -> volume/activity (BULLISH)
  BRKO  flat, compressed 20-session range, then a 55-day high with 3x volume at D -> breakout
  DROP  ordinary, then three -9% days into D               -> mean reversion
  ILLQ  MOMO-like trend but ~$1.5M dollar volume           -> discovered, fails the research universe
  NEWB  listed 100 sessions before D                       -> ret_120d UNKNOWN (not INVALID)
  GAPD  ordinary with one missing bar 120 sessions before D -> ret_120d INVALID (data-quality issue)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.core.calendar import TradingCalendar
from quantlab.data import schemas
from quantlab.data.panel import DataBundle, build_panel

BENCH = {"market": "SPY", "sectors": {}}
SPECIALS = ("MOMO", "VOLX", "BRKO", "DROP", "ILLQ", "NEWB", "GAPD")


def _rows(sym, dates, closes, vols, rng, keep=None):
    out, prev = [], closes[0]
    for i, (d, c, v) in enumerate(zip(dates, closes, vols)):
        if keep is not None and not keep[i]:
            prev = c
            continue
        o = prev * (1 + rng.normal(0, 0.002))
        hi, lo = max(o, c) * 1.004, min(o, c) * 0.996
        out.append({"symbol": sym, "date": d, "open": o, "high": hi, "low": lo, "close": c, "volume": v})
        prev = c
    return out


def crafted_bundle(n_days: int = 320, n_normal: int = 60, seed: int = 5, news_for: tuple[str, ...] = (),
                   specials: bool = True, flat: bool = False) -> DataBundle:
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-01-02", periods=n_days)]
    n = n_days
    bars: list[dict] = []

    def walk(start, mu, sigma):
        r = rng.normal(mu, sigma, n)
        if flat:
            r = np.full(n, 0.0003)
        return start * np.cumprod(1 + r)

    spy = walk(400.0, 0.0003, 0.008)
    bars += _rows("SPY", dates, spy, np.full(n, 5e7), rng)
    for k in range(n_normal):
        c = walk(50.0, 0.0002, 0.015)
        v = np.full(n, 1e6) if flat else rng.uniform(0.8e6, 1.2e6, n)
        bars += _rows(f"N{k:03d}", dates, c, v, rng)
    if specials and not flat:
        c = walk(40.0, 0.0002, 0.012)
        c[-120:] = c[-121] * np.cumprod(np.full(120, 1.006) + rng.normal(0, 0.004, 120))
        bars += _rows("MOMO", dates, c, np.full(n, 1e6), rng)
        c = walk(50.0, 0.0002, 0.012)
        c[-1] = c[-2] * 1.02
        v = np.full(n, 1e6)
        v[-1] = 8e6
        bars += _rows("VOLX", dates, c, v, rng)
        c = 30.0 * (1 + 0.02 * np.sin(np.arange(n) / 3.0))
        c[-21:-1] = 30.0 * (1 + 0.002 * np.sin(np.arange(20)))
        c[-1] = 30.0 * 1.10
        v = np.full(n, 1e6)
        v[-1] = 3e6
        bars += _rows("BRKO", dates, c, v, rng)
        c = walk(50.0, 0.0002, 0.012)
        c[-3:] = c[-4] * np.cumprod([0.91, 0.91, 0.91])
        bars += _rows("DROP", dates, c, np.full(n, 1e6), rng)
        c = walk(40.0, 0.0002, 0.012)
        c[-120:] = c[-121] * np.cumprod(np.full(120, 1.006))
        bars += _rows("ILLQ", dates, c, np.full(n, 3.0e4), rng)
        c = walk(20.0, 0.004, 0.01)
        keep = np.arange(n) >= n - 100
        bars += _rows("NEWB", dates, c, np.full(n, 1e6), rng, keep=keep)
        c = walk(50.0, 0.0002, 0.012)
        keep = np.ones(n, bool)
        keep[n - 121] = False
        bars += _rows("GAPD", dates, c, np.full(n, 1e6), rng, keep=keep)
    bdf = pd.DataFrame(bars)
    bdf["vwap"], bdf["trade_count"], bdf["provider"], bdf["retrieved_at"] = None, None, "synthetic", "2023-01-01T00:00:00+00:00"
    cal = TradingCalendar.from_dates(dates)
    acts = schemas.empty("corporate_actions")
    panel = build_panel(bdf, acts, calendar=cal)
    syms = sorted(bdf["symbol"].unique())
    ref = pd.DataFrame({"symbol": syms, "name": syms, "exchange": "NASDAQ",
                        "security_type": ["ETF" if s == "SPY" else "COMMON" for s in syms],
                        "is_etf": [s == "SPY" for s in syms], "is_test_issue": False, "cik": None, "sic": None,
                        "sector": None, "industry": None, "status": "active", "source": "synthetic",
                        "retrieved_at": "2023-01-01T00:00:00+00:00", "pit_status": "ASSUMED_STATIC"})
    news = schemas.empty("news")
    if news_for:
        rows = []
        for s in news_for:
            for j, d in enumerate(dates[-80:]):
                t = pd.Timestamp(d + " 14:00", tz="America/New_York").tz_convert("UTC")
                k = 3 if j == 79 else (1 if j % 5 == 0 else 0)
                for m in range(k):
                    rows.append({"news_id": f"{s}-{j}-{m}", "symbol": s, "headline": f"{s} announces product {j}",
                                 "summary": "", "source": "synthetic", "url": None, "created_at": t.isoformat(),
                                 "updated_at": t.isoformat(), "available_at": t, "pit_status": "PIT",
                                 "provider": "synthetic", "retrieved_at": t.isoformat()})
        news = pd.DataFrame(rows, columns=schemas.NEWS)
    return DataBundle(panel=panel, calendar=cal, reference=ref, actions=acts, events=schemas.empty("events"),
                      fundamentals=schemas.empty("fundamentals"), news=news, benchmarks=BENCH,
                      dataset_ids=["synthetic-crafted"], is_synthetic=True)
