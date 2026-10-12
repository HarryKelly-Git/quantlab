"""Dip buying, Part II: single-stock pullbacks in uptrends (docs/DIP-BUYING-PREREG.md). PAPER research, 2016-2024.

Signal at close t (liquid universe): above the 200-day average; within 10% of its 52-week high as of t-10; down >= 8%
over the last 5 sessions. Buy at the next open; exit at the first close >= the close 5 sessions before the signal, or
after 20 sessions. Portfolio: up to 10 positions of 10% of equity, deepest dips first, idle cash in T-bills. Costs:
master's tiers by liquidity. Marking uses the corrected-delisting total return (ret_cc_pnl).
S1 all liquid stocks, S2 top 200 by 60-day median dollar volume. Writes research/alpha/results/dips/S_stocks.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantlab.alpha import ca_fixes, registry, research_data  # noqa: E402
from quantlab.alpha import events_study as es  # noqa: E402

from new_areas_ac import ff_daily, sharpe_diff_ci, stats  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "dips"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
SLOTS, SIZE, MAX_DAYS, DIP, NEAR_HIGH, LARGE_N = 10, 0.10, 20, -0.08, 0.10, 200
PREREG = "docs/DIP-BUYING-PREREG.md Part II"


def simulate(sig: np.ndarray, depth: np.ndarray, target: np.ndarray, o: np.ndarray, c: np.ndarray, rp: np.ndarray,
             cost_rt: np.ndarray, rf: np.ndarray, dates: pd.DatetimeIndex, cols: pd.Index, spy_o: np.ndarray,
             spy_c: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    cash, pos, pending = 100000.0, {}, []
    eq, trades = np.zeros(len(dates)), []
    for i in range(len(dates)):
        cash *= 1 + rf[i]
        # (1) entries decided at the previous close, at this open
        for j, notional, tgt, crt, sdate in pending:
            px = o[i, j]
            if not np.isfinite(px) or px <= 0 or notional > cash:
                continue
            cash -= notional
            val = notional * (1 - crt / 2) * (c[i, j] / px if np.isfinite(c[i, j]) else 1.0)
            pos[j] = {"val": val, "cost_in": notional, "tgt": tgt, "days": 0, "i0": i, "crt": crt, "spy0": spy_o[i], "sig": sdate}
        pending = []
        # (2) mark held positions (entry day already marked open->close), exits at this close
        for j in list(pos):
            p = pos[j]
            if p["i0"] != i:
                r = rp[i, j]
                p["val"] *= 1 + (r if np.isfinite(r) else 0.0)
                p["days"] += 1
            else:
                p["days"] = 1
            hit = np.isfinite(c[i, j]) and c[i, j] >= p["tgt"]
            if hit or p["days"] >= MAX_DAYS or i == len(dates) - 1:
                proceeds = p["val"] * (1 - p["crt"] / 2)
                cash += proceeds
                trades.append({"symbol": cols[j], "signal": p["sig"], "entry": dates[p["i0"]], "exit": dates[i], "days": p["days"],
                               "net_ret": proceeds / p["cost_in"] - 1, "spy_ret": spy_c[i] / p["spy0"] - 1, "recovered": bool(hit)})
                del pos[j]
        equity = cash + sum(p["val"] for p in pos.values())
        eq[i] = equity
        # (3) new signals at this close for the next open
        free = SLOTS - len(pos)
        if free > 0 and i < len(dates) - 1:
            js = np.where(sig[i])[0]
            js = [j for j in js if j not in pos]
            js.sort(key=lambda j: depth[i, j])
            for j in js[:free]:
                pending.append((j, SIZE * equity, target[i, j], cost_rt[i, j], dates[i]))
    return pd.Series(eq, index=dates), pd.DataFrame(trades)


def main() -> None:
    t0 = time.time()
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    p, u = d.p, d.u_liquid
    ever = u.any(axis=0)
    cols = pd.Index([s for s in p.symbols if ever.get(s, False) or s == "SPY"])
    c = p["adj_close"][cols]
    o = p["adj_open"][cols]
    rp = d.ret_cc_pnl[cols] if hasattr(d, "ret_cc_pnl") else p["ret_cc"][cols]
    dv = (p["close"][cols] * p["volume"][cols])
    mdv20 = d.mdv20.reindex(columns=cols)
    uu = u.reindex(columns=cols).fillna(False)
    del d
    gc.collect()
    sma200 = c.rolling(200, min_periods=200).mean()
    hi = c.rolling(252, min_periods=200).max()
    near_high = (c.shift(10) >= (1 - NEAR_HIGH) * hi.shift(10))
    r5 = c / c.shift(5) - 1
    base = (uu & (c > sma200) & near_high & (r5 <= DIP)).fillna(False)
    base["SPY"] = False
    mdv60 = dv.rolling(60, min_periods=40).median().where(uu)
    large = mdv60.rank(axis=1, ascending=False) <= LARGE_N
    sig_variants = {"S1": base, "S2": (base & large).fillna(False)}
    target = c.shift(5).to_numpy()
    cost_rt = es.cost_round_trip(mdv20.to_numpy())
    dates = c.index
    rf = ff_daily()["rf"].reindex(dates).fillna(0.0).to_numpy()
    spy = c["SPY"]
    spy_o, spy_c = o["SPY"].to_numpy(), spy.to_numpy()
    print(f"setup {time.time() - t0:.0f}s; signals S1 {int(base.to_numpy().sum())}, S2 {int(sig_variants['S2'].to_numpy().sum())}", flush=True)
    rfs = ff_daily()["rf"]
    spy_r = spy.pct_change().fillna(0.0)
    out: dict = {"prereg": PREREG, "git": registry.git_commit(), "variants": {}}
    bh = {sp: stats(spy_r.loc[a:b], rfs) for sp, (a, b) in SPLITS.items()}
    out["SPY"] = bh
    for k, sg in sig_variants.items():
        eq, tr = simulate(sg.to_numpy(), r5.to_numpy(), target, o.to_numpy(), c.to_numpy(), rp.to_numpy(), cost_rt, rf,
                          dates, cols, spy_o, spy_c)
        r = eq.pct_change().fillna(0.0)
        res: dict = {"splits": {}, "trades_by_split": {}}
        for sp, (a, b) in SPLITS.items():
            res["splits"][sp] = stats(r.loc[a:b], rfs)
            res["splits"][sp]["sharpe_minus_SPY"] = sharpe_diff_ci(r.loc[a:b], spy_r.loc[a:b], rfs)
            g = tr[(tr["entry"] >= a) & (tr["entry"] <= b)] if len(tr) else tr
            if len(g):
                ex = g["net_ret"] - g["spy_ret"]
                m, t, nd = es.cluster_t(ex, g["entry"])
                res["trades_by_split"][sp] = {"n": int(len(g)), "mean_net": float(g["net_ret"].mean()), "mean_excess": m, "t": t,
                                              "hit_rate": float((g["net_ret"] > 0).mean()), "avg_days": float(g["days"].mean()),
                                              "net_per_day_held": float(g["net_ret"].sum() / g["days"].sum()),
                                              "recovered_share": float(g["recovered"].mean()), "worst": float(g["net_ret"].min())}
        s = res["splits"]
        sh_ok = all(s[sp]["sharpe_minus_SPY"]["diff"] > 0 for sp in SPLITS) and s["OOS"]["sharpe_minus_SPY"]["ci90"][0] > 0
        cagr_ok = all(s[sp]["cagr"] > bh[sp]["cagr"] for sp in SPLITS)
        res["verdict"] = "BEATS SPY" if (sh_ok and cagr_ok) else "BETTER RISK-ADJUSTED ONLY" if sh_ok else "FAILS"
        res["by_year"] = {str(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}
        res["equity_monthly"] = {str(x.date()): round(float(v), 2) for x, v in eq.resample("ME").last().items()}
        out["variants"][k] = res
        print(f"{k}: " + " | ".join(f"{sp} {s[sp]['cagr']*100:5.1f}% Sh {s[sp]['sharpe']:.2f} DD {s[sp]['max_drawdown']*100:4.0f}% "
                                  f"(SPY {bh[sp]['cagr']*100:.1f}%/{bh[sp]['sharpe']:.2f})" for sp in SPLITS)
              + f" | {res['verdict']} ({time.time() - t0:.0f}s)", flush=True)
        print("   trades:", {sp: {kk: round(v, 4) if isinstance(v, float) else v for kk, v in dd.items()} for sp, dd in res["trades_by_split"].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S_stocks.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    for k, v in out["variants"].items():
        registry.append_run(hypothesis_id=f"DIP_{k}", family="dip_buying", split="OOS", spec={"prereg": PREREG, "variant": k},
                            metrics={"verdict": v["verdict"], "oos": {m: v["splits"]["OOS"].get(m) for m in ("cagr", "sharpe", "max_drawdown")}},
                            conclusion=v["verdict"], seed=7)


if __name__ == "__main__":
    main()
