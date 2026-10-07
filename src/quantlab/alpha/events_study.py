"""Event study: big one-day moves, news vs no-news, sector-wide vs idiosyncratic, laggard peers, and the call-option
expression (docs/EVENT-STUDY-PREREG.md). PAPER research, 2016-2024 only (the 2025+ holdout stays sealed).

Point-in-time. An event at the close of t uses only prices up to t and news created up to 16:00 ET on t.
Outcomes start at the t+1 OPEN, the paper bot's convention.

News comes from Alpaca's Benzinga feed (``/v1beta1/news``), queried only for event dates. ``created_at`` is
the availability time. Headlines are classified by fixed regexes, never by an LLM: an LLM trained after
2016-24 would know what happened next.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

UP, DOWN = 0.08, -0.08
VOL_MULT = 2.0
COOLDOWN = 20
SECTOR_JUMP, SECTOR_MIN_OTHERS, LAGGARD_MAX = 0.05, 3, 0.01
HORIZONS = (1, 5, 20, 60)
RECAP = re.compile(r"shares (are )?(trading|moving)|stock is (trading|moving|soaring|falling)|why is .* (stock|shares)|"
                   r"what'?s going on with|movers|mid-?day|pre-?market|after-?hours|52-week high|52-week low|"
                   r"gap(ping)? (up|down)|session|stocks? (moving|to watch)|top (gainers|losers)|unusual options", re.I)
CATEGORIES = (
    ("ma_target", re.compile(r"acquire|to buy|buyout|takeover|merger agreement|to be acquired|go private|going private", re.I)),
    ("earnings", re.compile(r"earnings|\bEPS\b|revenue|guidance|quarter|\bQ[1-4]\b|results|outlook", re.I)),
    ("analyst", re.compile(r"upgrade|downgrade|price target|initiates|reiterates", re.I)),
    ("clinical", re.compile(r"\bFDA\b|approval|trial|phase|data readout", re.I)),
    ("deal", re.compile(r"contract|agreement|partnership|\bdeal\b|award|\bloan\b|\border\b|selects", re.I)),
)
ET = "America/New_York"


# ------------------------------------------------------------------------------------------------
# events
# ------------------------------------------------------------------------------------------------
def detect(p, u: pd.DataFrame, side: str = "up") -> pd.DataFrame:
    """UP (>= +8%) or DOWN (<= -8%) close-to-close total-return days with dollar volume >= 2x the trailing
    20-session median (prior sessions only), in the universe at t; at most one per stock per 20 sessions."""
    ret = p["ret_cc"]
    dv = p["dollar_volume"]
    med = dv.rolling(20, min_periods=20).median().shift(1)
    hit = (ret >= UP) if side == "up" else (ret <= DOWN)
    cond = (hit & (dv >= VOL_MULT * med) & u.reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)).to_numpy()
    rows = []
    cols = ret.columns
    R = ret.to_numpy()
    D = dv.to_numpy() / med.to_numpy()
    for j in range(cond.shape[1]):
        idx = np.flatnonzero(cond[:, j])
        last = -10 ** 9
        for i in idx:
            if i - last > COOLDOWN:
                rows.append((p.dates[i], cols[j], float(R[i, j]), float(D[i, j]), int(i)))
                last = i
    ev = pd.DataFrame(rows, columns=["date", "entity", "ret", "dv_ratio", "i"])
    ev["side"] = side
    return ev.sort_values(["date", "entity"]).reset_index(drop=True)


def sector_flags(ev: pd.DataFrame, p, u: pd.DataFrame, sectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SECTOR-WIDE flag per event (>= 3 OTHER same-sector liquid stocks up >= +5% on t) and the LAGGARD rows
    (same-sector liquid stocks <= +1% on a sector-wide day). The sector is the point-in-time statistical one."""
    ret = p["ret_cc"]
    out = ev.copy()
    out["sector"] = [sectors.at[d, e] if e in sectors.columns else None for d, e in zip(ev["date"], ev["entity"])]
    n_jump = []
    lag_rows = []
    seen = set()
    for d, g in out.groupby("date"):
        uu = u.loc[d]
        members = uu[uu].index
        sec_d = sectors.loc[d, members]
        r_d = ret.loc[d, members]
        jumpers = r_d[r_d >= SECTOR_JUMP]
        cnt = sec_d.loc[jumpers.index].value_counts()
        for k, row in g.iterrows():
            s = row["sector"]
            c = int(cnt.get(s, 0)) - (1 if row["ret"] >= SECTOR_JUMP and sec_d.get(row["entity"]) == s else 0)
            n_jump.append((k, c))
            if row["side"] == "up" and s is not None and c >= SECTOR_MIN_OTHERS and (d, s) not in seen:
                seen.add((d, s))
                lag = r_d[(sec_d == s) & (r_d <= LAGGARD_MAX)]
                lag_rows += [(d, e, float(v), s) for e, v in lag.items()]
    nj = dict(n_jump)
    out["sector_jumpers_others"] = [nj[k] for k in out.index]
    out["sector_wide"] = out["sector_jumpers_others"] >= SECTOR_MIN_OTHERS
    lag = pd.DataFrame(lag_rows, columns=["date", "entity", "ret", "sector"])
    return out, lag


# ------------------------------------------------------------------------------------------------
# outcomes (next-open entry, close exit, excess over SPY, net of master's cost tiers)
# ------------------------------------------------------------------------------------------------
def cost_round_trip(mdv20: np.ndarray) -> np.ndarray:
    from quantlab.alpha.opt_exec import equity_cost_frac
    return 2.0 * equity_cost_frac(mdv20)


def forward(p, ret_pnl: pd.DataFrame, rows: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Return from the t+1 OPEN to the close of t+h (total return, delisting returns booked), SPY likewise."""
    dates = p.dates
    rid = (p["adj_close"] / p["adj_open"] - 1.0)
    lr = np.log1p(ret_pnl.clip(lower=-0.9999)).fillna(0.0)
    L = lr.cumsum().to_numpy()
    lid = np.log1p(rid.clip(lower=-0.9999)).to_numpy()
    spy_l = L[:, ret_pnl.columns.get_loc("SPY")]
    spy_id = lid[:, ret_pnl.columns.get_loc("SPY")]
    ci = ret_pnl.columns.get_indexer(rows["entity"])
    ti = dates.get_indexer(pd.to_datetime(rows["date"]))
    out = rows.copy()
    T = len(dates)
    for h in horizons:
        a = ti + 1
        b = ti + h
        ok = (ci >= 0) & (b < T)
        r_s = np.full(len(rows), np.nan)
        r_m = np.full(len(rows), np.nan)
        aa, bb, cc = a[ok], b[ok], ci[ok]
        first = lid[aa, cc]
        has_open = np.isfinite(first)
        tail = L[bb, cc] - L[aa, cc]
        r_s[ok] = np.where(has_open, np.expm1(first + tail), np.nan)
        r_m[ok] = np.expm1(np.nan_to_num(spy_id[aa]) + spy_l[bb] - spy_l[aa])
        out[f"ret_{h}"] = r_s
        out[f"excess_{h}"] = r_s - r_m
    return out


def cluster_t(x: pd.Series, d: pd.Series) -> tuple[float, float, int]:
    z = pd.DataFrame({"x": x.to_numpy(), "d": d.to_numpy()}).dropna()
    if len(z) < 10:
        return float("nan"), float("nan"), int(len(z))
    m = float(z["x"].mean())
    g = (z["x"] - m).groupby(z["d"]).sum()
    se = float(np.sqrt((g ** 2).sum())) / len(z)
    return m, (m / se if se > 0 else float("nan")), int(z["d"].nunique())


# ------------------------------------------------------------------------------------------------
# news (Alpaca / Benzinga), queried per event DATE for all event tickers of that date
# ------------------------------------------------------------------------------------------------
def window_utc(day: pd.Timestamp, prev: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(prev.date()).tz_localize(ET) + pd.Timedelta(hours=16)
    b = pd.Timestamp(day.date()).tz_localize(ET) + pd.Timedelta(hours=16)
    return a.tz_convert("UTC"), b.tz_convert("UTC")


def download_news(plan: pd.DataFrame, out: Path, per_minute: int = 100, chunk: int = 40) -> pd.DataFrame:
    """``plan``: one row per (date, prev_date, tickers list). Resumable: dates already in ``out`` (jsonl) are skipped."""
    import requests

    from quantlab.alpha.store import DATA_URL, RateLimiter, _auth, _get
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["_date"])
            except Exception:
                pass
    lim = RateLimiter(per_minute)
    s = requests.Session()
    s.headers.update(_auth())
    with out.open("a") as fh:
        for r in plan.itertuples(index=False):
            key = str(pd.Timestamp(r.date).date())
            if key in done:
                continue
            a, b = window_utc(pd.Timestamp(r.date), pd.Timestamp(r.prev_date))
            arts = []
            for k in range(0, len(r.tickers), chunk):
                syms = ",".join(r.tickers[k:k + chunk])
                tok = None
                while True:
                    params = {"symbols": syms, "start": (a - pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "end": (b + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50}
                    if tok:
                        params["page_token"] = tok
                    j = _get(s, f"{DATA_URL}/v1beta1/news", params, lim)
                    for n in j.get("news") or []:
                        arts.append({"id": n["id"], "created_at": n["created_at"], "updated_at": n.get("updated_at"),
                                     "headline": n.get("headline", ""), "symbols": n.get("symbols", []), "source": n.get("source")})
                    tok = j.get("next_page_token")
                    if not tok:
                        break
            fh.write(json.dumps({"_date": key, "articles": arts}) + "\n")
            fh.flush()
    rows = []
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        for a_ in rec["articles"]:
            rows.append({**a_, "_date": rec["_date"]})
    return pd.DataFrame(rows)


def classify(ev: pd.DataFrame, news: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """NEWS / NO-NEWS and CATEGORY per event from the articles tagged with the event's ticker in its window."""
    out = ev.copy()
    if news.empty:
        out["n_articles"], out["n_substantive"], out["news"], out["category"] = 0, 0, False, "none"
        return out
    nn = news.drop_duplicates("id").copy()
    nn["created"] = pd.to_datetime(nn["created_at"], utc=True)
    nn["recap"] = nn["headline"].fillna("").str.contains(RECAP)
    ex = nn.explode("symbols")
    by_sym = {k: g for k, g in ex.groupby("symbols")}
    n_all, n_sub, cats, heads = [], [], [], []
    for r in out.itertuples(index=False):
        i = dates.get_loc(pd.Timestamp(r.date))
        a, b = window_utc(pd.Timestamp(r.date), dates[i - 1])
        g = by_sym.get(r.ticker)
        if g is None:
            n_all.append(0); n_sub.append(0); cats.append("none"); heads.append("")
            continue
        w = g[(g["created"] > a) & (g["created"] <= b)]
        sub = w[~w["recap"]]
        n_all.append(int(len(w))); n_sub.append(int(len(sub)))
        text = " | ".join(sub["headline"].fillna("").tolist())
        cat = next((name for name, rx in CATEGORIES if rx.search(text)), "other") if len(sub) else "none"
        cats.append(cat)
        heads.append(text[:300])
    out["n_articles"], out["n_substantive"], out["category"], out["headlines"] = n_all, n_sub, cats, heads
    out["news"] = out["n_substantive"] > 0
    return out


def ticker_map(resolved: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Ticker used by each event's entity on the event date (renames: FB on 2016 dates for entity META)."""
    want = set(zip(pd.to_datetime(rows["date"]), rows["entity"].astype(str)))
    r = resolved[["ticker", "date", "entity"]].copy()
    r["entity"] = r["entity"].astype(str)
    r = r[r["entity"].isin({e for _, e in want})]
    r["date"] = pd.to_datetime(r["date"])
    m = {(d, e): str(t) for t, d, e in zip(r["ticker"], r["date"], r["entity"]) if (d, e) in want}
    return pd.Series([m.get((pd.Timestamp(d), str(e)), str(e).split("@")[0]) for d, e in zip(rows["date"], rows["entity"])],
                     index=rows.index)
