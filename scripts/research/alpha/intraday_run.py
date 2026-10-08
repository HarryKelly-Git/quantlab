"""Intraday news reaction (docs/INTRADAY-NEWS-PREREG.md). PAPER research, 2017-2024.

Steps, each resumable:
  1. qualify    headlines -> var/alpha/cache/intraday_headlines.parquet  (needs the research panel)
  2. minutes    SIP 1-minute bars per sampled day -> var/alpha/minute_news/<day>.parquet  (API, bot-aware rate)
  3. analyze    measure, test IN1/IN2 (+ IN3 descriptive) -> research/alpha/results/intraday/intraday_news.json

Usage: .venv/bin/python scripts/research/alpha/intraday_run.py [qualify|minutes|analyze|all]
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import intraday_news as inn  # noqa: E402
from quantlab.alpha import events_study as es  # noqa: E402
from quantlab.alpha import registry  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "intraday"
CACHE = store_dir() / "cache"
MIN_DIR = store_dir() / "minute_news"
SPLITS = {"TRAIN": ("2017-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
T_SIG = 2.5


def qualify() -> None:
    from quantlab.alpha import ca_fixes, research_data
    t0 = time.time()
    days = pd.DatetimeIndex(pd.read_parquet(CACHE / "intraday_days.parquet")["day"])
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    p, u, mdv = d.p, d.u_liquid, d.mdv20
    res = d.resolved()
    pos = p.dates.get_indexer(days)
    prev = p.dates[np.maximum(pos - 1, 0)]
    nxt = p.dates[np.minimum(pos + 1, len(p.dates) - 1)]
    r = res[res["date"].isin(days)][["ticker", "date", "entity"]].copy()
    r["ticker"], r["entity"] = r["ticker"].astype(str), r["entity"].astype(str)
    liquid: dict[str, dict[str, tuple[str, float]]] = {}
    for day, pv in zip(days, prev):
        ul = u.loc[pv]
        ents = set(ul[ul].index)
        rr = r[(r["date"] == day) & r["entity"].isin(ents)]
        liquid[str(day.date())] = {tk: (e, float(mdv.at[pv, e])) for tk, e in zip(rr["ticker"], rr["entity"])}
    news = inn.load_news(store_dir() / "external" / "intraday_news.jsonl")
    h = inn.qualify(news, liquid)
    # next-session factors (total return; flagged corporate-action days are UNKNOWN -> NaN) for exit I2
    ret = p["ret_cc"]
    nmap = dict(zip([str(x.date()) for x in days], nxt))
    h["next_day"] = [nmap[x] for x in h["day"]]
    h["g_next"] = [1 + ret.at[nd, e] if e in ret.columns else np.nan for nd, e in zip(h["next_day"], h["entity"])]
    h["g_spy_next"] = [1 + ret.at[nd, "SPY"] for nd in h["next_day"]]
    h.to_parquet(CACHE / "intraday_headlines.parquet", index=False)
    print(f"qualified {len(h)} headlines on {h['day'].nunique()} days from {len(news)} articles ({time.time() - t0:.0f}s)", flush=True)
    del d, p, u, mdv, res
    gc.collect()


def minutes() -> None:
    t0 = time.time()
    h = pd.read_parquet(CACHE / "intraday_headlines.parquet")
    MIN_DIR.mkdir(parents=True, exist_ok=True)
    lim = inn.BotAwareLimiter()
    for k, (day, g) in enumerate(h.groupby("day")):
        f = MIN_DIR / f"{day}.parquet"
        if f.exists():
            continue
        tick = sorted(set(g["ticker"])) + ["SPY"]
        bars = inn.download_minutes(pd.Timestamp(day), tick, lim)
        bars.to_parquet(f, index=False)
        if k % 25 == 0:
            print(f"{k} days, {day}: {len(bars)} bars ({time.time() - t0:.0f}s)", flush=True)
    print(f"minutes done ({time.time() - t0:.0f}s)", flush=True)


def _split(days: pd.Series) -> np.ndarray:
    d = pd.to_datetime(days)
    out = np.full(len(d), "NONE", dtype=object)
    for k, (a, b) in SPLITS.items():
        out[((d >= a) & (d <= b)).to_numpy()] = k
    return out


def _by(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for sp in SPLITS:
        g = df[df["split"] == sp]
        m, t, nd = es.cluster_t(g[col], g["day"])
        out[sp] = {"n": int(g[col].notna().sum()), "n_days": nd, "mean": m, "t": t,
                   "hit": float((g[col] > 0).mean()) if len(g) else None}
    o = df[df["split"] == "OOS"]
    out["OOS_years"] = {str(y): {"n": int(len(x)), "mean": float(x[col].mean())} for y, x in o.groupby(pd.to_datetime(o["day"]).dt.year)}
    return out


def _verdict(b: dict) -> str:
    ok = lambda s: (s.get("mean") or -1) > 0 and (s.get("t") or 0) >= T_SIG  # noqa: E731
    if ok(b["TRAIN"]) and ok(b["VAL"]) and ok(b["OOS"]):
        return "SURVIVES"
    if all((b[s].get("mean") or -1) > 0 for s in SPLITS):
        return "PROMISING (forward paper-shadow only)"
    return "FAILS"


def analyze() -> None:
    t0 = time.time()
    h = pd.read_parquet(CACHE / "intraday_headlines.parquet")
    h["created"] = pd.to_datetime(h["created"])
    if h["created"].dt.tz is None:
        h["created"] = h["created"].dt.tz_localize("UTC").dt.tz_convert(inn.ET)
    parts = []
    for day, g in h.groupby("day"):
        f = MIN_DIR / f"{day}.parquet"
        if not f.exists():
            continue
        bars = pd.read_parquet(f)
        bars["t"] = pd.to_datetime(bars["t"])
        if bars["t"].dt.tz is None:
            bars["t"] = bars["t"].dt.tz_localize("UTC").dt.tz_convert(inn.ET)
        parts.append(inn.measure(g, bars))
    m = pd.concat([x for x in parts if len(x)], ignore_index=True)
    x = inn.returns(m)
    x["net_next"] = x["side"] * ((x["p_close"] / x["p15"]) * x["g_next"] - (x["spy_close"] / x["spy15"]) * x["g_spy_next"]) - x["cost_rt"]
    x["gross_cont_close"] = x["side"] * ((x["p_close"] / x["p15"] - 1) - (x["spy_close"] / x["spy15"] - 1))
    x["split"] = _split(x["day"])
    sig = x[x["r0"].abs() >= inn.REACTION]
    lo, sh = sig[sig["side"] > 0], sig[sig["side"] < 0]
    res = {"prereg": "docs/INTRADAY-NEWS-PREREG.md", "git": registry.git_commit(),
           "counts": {"headlines_measured": int(len(x)), "signals_long": int(len(lo)), "signals_short": int(len(sh)),
                      "days": int(x["day"].nunique())},
           "IN1_long_close": _by(lo, "net_close"), "IN2_long_next_close": _by(lo, "net_next"),
           "IN3_short_close": _by(sh, "net_close"), "IN3_short_next_close": _by(sh, "net_next")}
    res["IN1_verdict"], res["IN2_verdict"] = _verdict(res["IN1_long_close"]), _verdict(res["IN2_long_next_close"])
    x["r0_dec"] = pd.qcut(x["r0"].rank(method="first"), 10, labels=False)
    res["descriptive"] = {
        "r0_deciles_gross_continuation_to_close": x.groupby("r0_dec").agg(r0=("r0", "mean"), cont=("gross_cont_close", "mean"),
                                                                           n=("r0", "size")).round(5).to_dict("list"),
        "long_signals_by_hour": lo.groupby(pd.to_datetime(lo["t15"]).dt.hour)["net_close"].agg(["count", "mean"]).round(5).to_dict("index"),
        "long_signals_by_category": lo.groupby("category")["net_close"].agg(["count", "mean"]).round(5).to_dict("index"),
        "median_cost_rt": float(x["cost_rt"].median()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intraday_news.json").write_text(json.dumps(registry._clean(res), indent=1, default=str))
    x.to_parquet(CACHE / "intraday_measured.parquet", index=False)
    registry.append_run(hypothesis_id="IN_intraday_news", family="intraday_news", split="OOS",
                        spec={"prereg": "INTRADAY-NEWS-PREREG", "reaction": inn.REACTION, "window": [inn.WINDOW_START, inn.WINDOW_END]},
                        metrics={"IN1": res["IN1_verdict"], "IN2": res["IN2_verdict"], "counts": res["counts"]},
                        conclusion=f"IN1 {res['IN1_verdict']}; IN2 {res['IN2_verdict']}", seed=7)
    print(f"IN1 {res['IN1_verdict']}; IN2 {res['IN2_verdict']}; counts {res['counts']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("qualify", "all"):
        qualify()
    if step in ("minutes", "all"):
        minutes()
    if step in ("analyze", "all"):
        analyze()
