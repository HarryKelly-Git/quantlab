"""Intraday news reaction (docs/INTRADAY-NEWS-PREREG.md). PAPER research, 2017-2024 only; the holdout stays sealed.

When a substantive single-stock headline arrives in regular hours and the stock reacts >= 2% within 15
minutes, does the move continue to the close or the next close, after doubled post-news spreads?

- **Downloads** run against the shared Alpaca account. The rate is lower during the paper bot's hours
  (``BotAwareLimiter``).
- **Classification** reuses the event study's fixed regexes. No LLM is used: one would know what happened
  next.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.alpha import events_study as es

HOLDOUT = pd.Timestamp("2025-01-01")
REACTION = 0.02
WINDOW_START, WINDOW_END = "09:30", "15:00"
ET = es.ET


class BotAwareLimiter:
    """At most ``off_hours`` calls/min when US markets are closed; ``bot_hours`` calls/min from 08:30 to 16:30 ET
    on weekdays, leaving the paper bot most of the shared 200/min account budget."""

    def __init__(self, off_hours: int = 170, bot_hours: int = 100):
        self.off, self.on = off_hours, bot_hours
        self._last = 0.0

    def wait(self) -> None:
        now = pd.Timestamp.now(tz=ET)
        busy = now.weekday() < 5 and pd.Timestamp("08:30").time() <= now.time() <= pd.Timestamp("16:30").time()
        gap = 60.0 / (self.on if busy else self.off)
        dt = time.monotonic() - self._last
        if dt < gap:
            time.sleep(gap - dt)
        self._last = time.monotonic()


def sample_days(dates: pd.DatetimeIndex, frac: float = 0.25, seed: int = 7, start: str = "2017-01-01",
                end: str = "2024-12-31") -> pd.DatetimeIndex:
    d = dates[(dates >= start) & (dates <= end)]
    rng = np.random.default_rng(seed)
    picks = []
    for y in sorted(set(d.year)):
        yd = d[d.year == y]
        k = int(round(frac * len(yd)))
        picks += list(rng.choice(yd, size=k, replace=False))
    out = pd.DatetimeIndex(sorted(picks))
    assert out.max() < HOLDOUT
    return out


def _utc(day: pd.Timestamp, hhmm: str) -> str:
    return (pd.Timestamp(f"{day.date()} {hhmm}").tz_localize(ET).tz_convert("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_news(days: pd.DatetimeIndex, out: Path, limiter: BotAwareLimiter | None = None) -> None:
    """Every Benzinga article created on each sampled day (00:00-23:59 ET). Resumable (one jsonl line per day)."""
    import requests

    from quantlab.alpha.store import DATA_URL, _auth, _get
    lim = limiter or BotAwareLimiter()
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["_date"])
            except Exception:
                pass
    s = requests.Session()
    s.headers.update(_auth())
    with out.open("a") as fh:
        for day in days:
            key = str(day.date())
            if key in done:
                continue
            assert day < HOLDOUT
            arts, tok = [], None
            while True:
                params = {"start": _utc(day, "00:00"), "end": _utc(day, "23:59"), "limit": 50}
                if tok:
                    params["page_token"] = tok
                j = _get(s, f"{DATA_URL}/v1beta1/news", params, lim)
                arts += [{"id": n["id"], "created_at": n["created_at"], "headline": n.get("headline", ""),
                          "symbols": n.get("symbols", [])} for n in j.get("news") or []]
                tok = j.get("next_page_token")
                if not tok:
                    break
            fh.write(json.dumps({"_date": key, "articles": arts}) + "\n")
            fh.flush()


def load_news(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        rows += [{**a, "_date": rec["_date"]} for a in rec["articles"]]
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["created"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET)
    return df


def qualify(news: pd.DataFrame, liquid: dict[str, dict[str, tuple[str, float]]]) -> pd.DataFrame:
    """Substantive, non-M&A, 1-2 symbol headlines created 09:30-15:00 ET on their own day, for stocks liquid at
    the prior close. ``liquid[day] = {ticker: (entity, mdv20)}``. First headline per stock per day."""
    n = news.copy()
    n["day"] = n["created"].dt.strftime("%Y-%m-%d")
    t = n["created"].dt.time
    n = n[(n["day"] == n["_date"]) & (t >= pd.Timestamp(WINDOW_START).time()) & (t < pd.Timestamp(WINDOW_END).time())]
    n = n[n["symbols"].apply(lambda s: 1 <= len(s) <= 2)]
    head = n["headline"].fillna("")
    ma = dict(es.CATEGORIES)["ma_target"]
    n = n[~head.str.contains(es.RECAP) & ~head.str.contains(ma)]
    n = n.explode("symbols").rename(columns={"symbols": "ticker"})
    keep = [(d in liquid and tk in liquid[d]) for d, tk in zip(n["day"], n["ticker"])]
    n = n[keep].copy()
    n["entity"] = [liquid[d][tk][0] for d, tk in zip(n["day"], n["ticker"])]
    n["mdv20"] = [liquid[d][tk][1] for d, tk in zip(n["day"], n["ticker"])]
    n = n.sort_values("created").drop_duplicates(["day", "ticker"], keep="first")
    n["category"] = [next((name for name, rx in es.CATEGORIES if rx.search(h)), "other") for h in n["headline"].fillna("")]
    return n.reset_index(drop=True)


def download_minutes(day: pd.Timestamp, tickers: list[str], limiter: BotAwareLimiter, chunk: int = 100) -> pd.DataFrame:
    """SIP 1-minute bars (raw prices) 09:20-16:01 ET for ``tickers`` on ``day``."""
    import requests

    from quantlab.alpha.store import DATA_URL, _auth, _get
    assert day < HOLDOUT
    s = requests.Session()
    s.headers.update(_auth())
    rows = []
    for k in range(0, len(tickers), chunk):
        syms = ",".join(tickers[k:k + chunk])
        tok = None
        while True:
            params = {"symbols": syms, "timeframe": "1Min", "start": _utc(day, "09:20"), "end": _utc(day, "16:01"),
                      "feed": "sip", "adjustment": "raw", "limit": 10000}
            if tok:
                params["page_token"] = tok
            j = _get(s, f"{DATA_URL}/v2/stocks/bars", params, limiter)
            for sym, bars in (j.get("bars") or {}).items():
                rows += [(sym, b["t"], b["c"]) for b in bars]
            tok = j.get("next_page_token")
            if not tok:
                break
    df = pd.DataFrame(rows, columns=["symbol", "t", "c"])
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    return df


def measure(h: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """r0 (last bar close before tau within 10 min -> last bar close in [tau+10, tau+15]) and the close-exit
    return for each headline row (one day). ``bars`` holds that day's minutes incl. SPY. Bars are stamped at
    their START (Alpaca), so a bar's close is known one minute after its stamp."""
    out = []
    by = {s: g.set_index("t")["c"].sort_index() for s, g in bars.groupby("symbol")}
    spy = by.get("SPY")
    for r in h.itertuples(index=False):
        px = by.get(r.ticker)
        if px is None or spy is None or px.empty:
            continue
        tau = r.created.ceil("min")
        known = px.index + pd.Timedelta(minutes=1)                       # close known at bar end
        pre = px[(known <= tau) & (known > tau - pd.Timedelta(minutes=10))]
        post = px[(known > tau + pd.Timedelta(minutes=10)) & (known <= tau + pd.Timedelta(minutes=15))]
        if pre.empty or post.empty:
            continue
        p0, p15, t15 = float(pre.iloc[-1]), float(post.iloc[-1]), post.index[-1]
        last = px[px.index.time <= pd.Timestamp("15:59").time()]
        s_last = spy[spy.index.time <= pd.Timestamp("15:59").time()]
        s15 = spy[spy.index <= t15]
        if last.empty or s_last.empty or s15.empty or last.index[-1] <= t15:
            continue
        out.append({**r._asdict(), "p0": p0, "p15": p15, "t15": t15, "r0": p15 / p0 - 1,
                    "p_close": float(last.iloc[-1]), "spy15": float(s15.iloc[-1]), "spy_close": float(s_last.iloc[-1])})
    return pd.DataFrame(out)


def returns(m: pd.DataFrame, next_close: pd.Series | None = None, spy_next: pd.Series | None = None) -> pd.DataFrame:
    """Signed excess returns net of the doubled-spread round trip, for exits at the close (I1) and next close (I2)."""
    from quantlab.alpha.opt_exec import equity_cost_frac
    x = m.copy()
    x["side"] = np.sign(x["r0"]).astype(int)
    hs = equity_cost_frac(x["mdv20"].to_numpy()) - 5e-4                 # half-spread part of the tier (bps)
    x["cost_rt"] = 2.0 * (2.0 * hs + 5e-4)
    x["net_close"] = x["side"] * ((x["p_close"] / x["p15"] - 1) - (x["spy_close"] / x["spy15"] - 1)) - x["cost_rt"]
    if next_close is not None and spy_next is not None:
        nc = next_close.reindex(x.index)
        sn = spy_next.reindex(x.index)
        x["net_next"] = x["side"] * ((nc / x["p15"] - 1) - (sn / x["spy15"] - 1)) - x["cost_rt"]
    return x
