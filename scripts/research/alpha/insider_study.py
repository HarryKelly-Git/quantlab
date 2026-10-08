"""Insider buying (docs/INSIDER-PREREG.md). PAPER research, filings 2016-2024.

Input: parsed OpenInsider screener rows (JSON list; quarterly queries of officer/director open-market purchases
>= $50k at >= $5). Signal day t = last session on or before the filing date; entry next open; excess vs SPY over
5/20/60 sessions net of master's cost tiers (corrected delisting). Signals A any, B CEO/CFO, C cluster (>= 2 insiders
within 10 days), D (B or C) and >= 20% below the 52-week high; once per company per 30 days.
Writes research/alpha/results/insider/insider_study.json and ledger rows INS_A..INS_D.

Usage: .venv/bin/python scripts/research/alpha/insider_study.py <rows.json>
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import ca_fixes, registry, research_data  # noqa: E402
from quantlab.alpha import events_study as es  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "insider"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
HORIZONS, PRIMARY, T_SIG = (5, 20, 60), 60, 2.5
TOP = re.compile(r"\b(ceo|cfo|chief executive|chief financial)\b", re.I)
PREREG = "docs/INSIDER-PREREG.md"


def num(x) -> float | None:
    x = str(x or "").replace("$", "").replace(",", "").replace("+", "").replace("%", "").strip()
    try:
        return float(x)
    except ValueError:
        return None


def dedupe(ev: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    keep, last = [], {}
    for r in ev.sort_values("filed").itertuples():
        prev = last.get(r.entity)
        if prev is None or (r.filed - prev).days > days:
            keep.append(r.Index)
            last[r.entity] = r.filed
    return ev.loc[keep]


def clusters(rows: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """The row at which an entity first has >= 2 distinct insiders buying within `window` days (one per run)."""
    out = []
    for _, g in rows.sort_values("filed").groupby("entity"):
        filed, names = g["filed"].to_numpy(), g["insider"].to_numpy()
        for k in range(len(g)):
            lo = filed[k] - np.timedelta64(window, "D")
            mask = (filed >= lo) & (filed <= filed[k])
            if len(set(names[mask])) >= 2:
                out.append(g.index[k])
    return rows.loc[out]


def verdict(b: dict) -> str:
    ok = lambda s: (b[s].get("mean") or -1) > 0 and (b[s].get("t") or 0) >= T_SIG  # noqa: E731
    if all(ok(s) for s in SPLITS):
        return "SURVIVES"
    if all((b[s].get("mean") or -1) > 0 for s in SPLITS):
        return "PROMISING (forward paper-shadow only)"
    return "FAILS"


def main(path: str) -> None:
    t0 = time.time()
    raw = pd.DataFrame(json.load(open(path)))
    n_raw = len(raw)
    raw = raw[raw["Trade Type"].astype(str).str.startswith("P")].copy()
    raw["filed"] = pd.to_datetime(raw["Filing Date"].astype(str).str[:10], errors="coerce")
    raw["value"] = raw["Value"].map(num)
    raw = raw.dropna(subset=["filed"])
    raw = raw[(raw["filed"] >= "2016-01-01") & (raw["filed"] <= "2024-12-31")]
    raw["ticker"] = raw["Ticker"].astype(str).str.strip().str.upper()
    raw["insider"] = raw["Insider Name"].astype(str).str.strip()
    raw["top"] = raw["Title"].astype(str).map(lambda s: bool(TOP.search(s)))
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    p, u = d.p, d.u_liquid
    dates = p.dates
    pos = dates.searchsorted(raw["filed"].to_numpy(), side="right") - 1
    raw = raw[pos >= 0].copy()
    raw["date"] = dates[pos[pos >= 0]]
    res = d.resolved()[["ticker", "date", "entity"]].copy()
    res["ticker"], res["entity"] = res["ticker"].astype(str), res["entity"].astype(str)
    res["date"] = pd.to_datetime(res["date"])
    m = raw.merge(res, on=["ticker", "date"], how="left")
    n_unresolved = int(m["entity"].isna().sum())
    m = m.dropna(subset=["entity"]).copy()
    ui, di = u.columns.get_indexer(m["entity"]), u.index.get_indexer(m["date"])
    un = u.to_numpy()
    m["liquid"] = [bool(un[a, b]) if a >= 0 and b >= 0 else False for a, b in zip(di, ui)]
    n_illiquid = int((~m["liquid"]).sum())
    m = m[m["liquid"]].copy()
    c = p["adj_close"]
    hi = c.rolling(252, min_periods=120).max()
    ci, ti = c.columns.get_indexer(m["entity"]), dates.get_indexer(m["date"])
    ok = ci >= 0
    ddv = np.full(len(m), np.nan)
    ddv[ok] = c.to_numpy()[ti[ok], ci[ok]] / hi.to_numpy()[ti[ok], ci[ok]] - 1
    m["dd_52w"] = ddv
    mdv = d.mdv20
    mi = mdv.columns.get_indexer(m["entity"])
    mv = np.full(len(m), np.nan)
    okm = mi >= 0
    mv[okm] = mdv.to_numpy()[ti[okm], mi[okm]]
    m["mdv20"] = mv
    m["cost_rt"] = es.cost_round_trip(m["mdv20"].to_numpy())
    print(f"rows {n_raw} -> purchases in window {len(raw)}; unresolved {n_unresolved}; illiquid {n_illiquid}; usable {len(m)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    sig = {"A_any": dedupe(m), "B_ceo_cfo": dedupe(m[m["top"]])}
    cl = clusters(m)
    sig["C_cluster"] = dedupe(cl)
    bc = pd.concat([m[m["top"]], cl]).drop_duplicates()
    sig["D_top_or_cluster_after_fall"] = dedupe(bc[bc["dd_52w"] <= -0.20])
    ret_pnl = d.ret_cc_pnl
    out: dict = {"prereg": PREREG, "git": registry.git_commit(), "source": "OpenInsider screener (SEC Form 4), quarterly queries",
                 "counts": {"rows": n_raw, "in_window": int(len(raw)), "unresolved_ticker": n_unresolved, "illiquid": n_illiquid,
                            "usable": int(len(m))}, "signals": {}}
    for k, ev in sig.items():
        f = es.forward(p, ret_pnl, ev[["date", "entity", "cost_rt", "value", "mdv20", "dd_52w"]].reset_index(drop=True),
                       horizons=HORIZONS)
        for h in HORIZONS:
            f[f"net_{h}"] = f[f"excess_{h}"] - f["cost_rt"]
        f["split"] = "NONE"
        for sp, (a, b) in SPLITS.items():
            f.loc[(f["date"] >= a) & (f["date"] <= b), "split"] = sp
        res_k: dict = {"n": int(len(f))}
        for h in HORIZONS:
            by = {}
            for sp in SPLITS:
                g = f[f["split"] == sp]
                mean, t, nd = es.cluster_t(g[f"net_{h}"], g["date"])
                by[sp] = {"n": int(g[f"net_{h}"].notna().sum()), "n_days": nd, "mean": mean, "t": t,
                          "hit": float((g[f"net_{h}"] > 0).mean()) if len(g) else None}
            res_k[f"h{h}"] = by
        res_k["verdict"] = verdict(res_k[f"h{PRIMARY}"])
        oos = f[f["split"] == "OOS"]
        res_k["oos_by_year_h60"] = {str(y): {"n": int(len(g)), "mean": float(g["net_60"].mean())}
                                    for y, g in oos.groupby(pd.to_datetime(oos["date"]).dt.year)}
        f["size"] = pd.qcut(f["mdv20"].rank(method="first"), 3, labels=["small", "mid", "large"])
        res_k["by_liquidity_tercile_h60"] = {str(s): {sp: float(g[g["split"] == sp]["net_60"].mean()) for sp in SPLITS}
                                            for s, g in f.groupby("size", observed=True)}
        out["signals"][k] = res_k
        print(f"{k}: n={len(f)} | " + " | ".join(f"{sp} h60 {res_k['h60'][sp]['mean']*100:+.2f}% (t {res_k['h60'][sp]['t']:.1f}) "
                                                  f"h20 {res_k['h20'][sp]['mean']*100:+.2f}%" for sp in SPLITS)
              + f" | {res_k['verdict']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "insider_study.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    for k, v in out["signals"].items():
        registry.append_run(hypothesis_id=f"INS_{k.split('_')[0]}", family="insider_buying", split="OOS",
                            spec={"prereg": PREREG, "signal": k, "primary_h": PRIMARY},
                            metrics={"verdict": v["verdict"], "h60": v["h60"]}, conclusion=v["verdict"], seed=7)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
