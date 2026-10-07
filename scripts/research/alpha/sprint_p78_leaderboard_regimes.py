"""Sprint P7 (model competition by OOS economic value) and P8 (regimes) - docs/ALPHA-SPRINT-PREREG.md sections 7-8.
PAPER research. Needs the outputs of P2, P3 and P4-P6.

Writes research/alpha/results/sprint/P7_leaderboard.json and P8_regimes.json.
"""
from __future__ import annotations

import gc
import json
import math
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QUANTLAB_SKIP_LOCAL_CONFIG", "1")

from quantlab.alpha import master_replay as mr  # noqa: E402
from quantlab.alpha import ca_fixes, registry, research_data  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402
from quantlab.backtest.engine import BacktestEngine  # noqa: E402
from quantlab.config import load_config  # noqa: E402
from quantlab.features.base import FeatureSet  # noqa: E402
from quantlab.strategies.arena import StrategyArena  # noqa: E402
from quantlab.strategies.registry import build_strategies  # noqa: E402
from quantlab.universe.engine import UniverseEngine  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "sprint"
CACHE = store_dir() / "cache"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
LB_MODELS = ("HV21", "HV63", "EWMA", "GARCH", "HAR", "QL_OLS")
LABELS = ("trend", "volatility", "breadth", "correlation", "momentum", "risk")


def mem() -> str:
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak"


def cluster_t(x: pd.Series, dates: pd.Series) -> tuple[float, float]:
    """Mean and date-clustered t (trades entered the same day share one market)."""
    d = pd.DataFrame({"x": x.to_numpy(), "d": dates.to_numpy()}).dropna()
    if len(d) < 3:
        return float("nan"), float("nan")
    m = d["x"].mean()
    g = (d["x"] - m).groupby(d["d"]).sum()
    se = math.sqrt(float((g ** 2).sum())) / len(d)
    return float(m), float(m / se) if se > 0 else float("nan")


def main() -> None:
    t0 = time.time()
    lab = pd.read_parquet(CACHE / "sprint_regime_labels.parquet")
    trades = pd.read_parquet(CACHE / "sprint_p3_trades.parquet")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    dates_all = lab.index

    # ---------------- P8: regime cells on standalone (S1) trades; the disable rule fixed on TRAIN+VAL -------------
    sig_day = lambda e: dates_all[np.clip(dates_all.get_indexer(e) - 1, 0, None)]  # noqa: E731  signal session
    p8: dict = {"rule": "disable (strategy, label value) when TRAIN+VAL mean net trade return < 0 with date-clustered "
                        "t <= -2 and >= 100 trades; applied once to OOS", "cells": [], "disabled": []}
    solo = trades[trades["rule"].str.startswith("solo_")].copy()
    solo["sig"] = sig_day(pd.DatetimeIndex(solo["entry_date"]))
    for l in LABELS:
        solo[l] = lab[l].reindex(solo["sig"]).to_numpy()
    solo["era"] = np.where(solo["entry_date"] <= SPLITS["VAL"][1], "DEV", "OOS")
    for (sid, era), g in solo.groupby([solo["strategy_id"], "era"]):
        for l in LABELS:
            for val, h in g.groupby(l):
                m, t = cluster_t(h["net_ret"], h["sig"])
                p8["cells"].append({"strategy": sid, "era": era, "label": l, "value": str(val), "n": int(len(h)),
                                    "mean_net": m, "t": t})
    cells = pd.DataFrame(p8["cells"])
    dev = cells[cells["era"] == "DEV"].copy()
    dev["p"] = 2 * sps.norm.sf(np.abs(dev["t"].fillna(0)))
    m_tests = int(dev["p"].notna().sum())
    dev = dev.sort_values("p")
    dev["holm_p"] = np.minimum(1.0, (dev["p"].to_numpy() * (m_tests - np.arange(len(dev)))))
    dev["holm_p"] = np.maximum.accumulate(dev["holm_p"].to_numpy())          # Holm step-down: monotone
    dis = dev[(dev["mean_net"] < 0) & (dev["t"] <= -2) & (dev["n"] >= 100)]
    p8["n_cells_tested_dev"] = m_tests
    p8["disabled"] = dis[["strategy", "label", "value", "n", "mean_net", "t", "p", "holm_p"]].to_dict("records")
    disabled = {(r["strategy"], r["label"], r["value"]) for r in p8["disabled"]}
    print(f"P8: {len(disabled)} cells disabled by the rule out of {m_tests}", flush=True)

    # ---------------- rebuild the replay once: P7 sizing runs and the P8 OOS disable test -------------------------
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps", "buckets"):
        setattr(d, k, None)
    gc.collect()
    cfg = load_config(overrides={"backtest": {"sizing": "equal_risk"}})
    cfg_ew = load_config(overrides={"backtest": {"sizing": "equal_weight"}})
    bench = cfg.section("benchmarks").get("sectors", {})
    p = d.p
    ever = d.u_liquid.any(axis=0)
    cols = pd.Index([s for s in p.symbols if ever.get(s, False) or s == "SPY" or s in bench])
    b = mr.bundle_from_research(p, cols, p.master, d.sectors, bench)
    del d, p
    gc.collect()
    uni = UniverseEngine(cfg).membership(b)
    fs = FeatureSet(b, universe=uni)
    strats = build_strategies(cfg, only=list(mr.REPLAYED))
    S = {s.strategy_id: s for s in strats}
    sig = StrategyArena(strats).scores(fs, uni)
    hold = {s.strategy_id: int(s.params.get("hold_sessions", 20)) for s in strats}
    horizon_of = lambda sid: 5 if hold.get(sid, 20) <= 7 else 20  # noqa: E731
    print(f"replay ready {time.time() - t0:.0f}s {mem()}", flush=True)

    base = BacktestEngine(cfg, b).run(sig, S, uni, fs=fs)              # S1, identical to P3's combined_S1_current
    tr = base.trades.copy()
    ent = pd.to_datetime(tr["entry_date"])
    trn = tr[ent <= SPLITS["TRAIN"][1]]
    di = b.panel.dates.get_indexer(pd.to_datetime(trn["entry_date"])) - 1
    ci = cols.get_indexer(trn["symbol"])
    daily = pd.read_parquet(CACHE / "sprint_p2_daily.parquet")
    p7: dict = {"sizing_runs": {}, "rule": "winner = best OOS economic value; ties within noise -> simpler"}
    for m in LB_MODELS:
        by_h = {h: daily.pivot(index="date", columns="symbol", values=f"{m}_h{h}").reindex(index=b.panel.dates, columns=cols)
                .to_numpy(dtype="float64") for h in (5, 20)}
        scale = float(np.nanmedian([by_h[horizon_of(s)][i, c] for s, i, c in zip(trn["strategy_id"], di, ci)]))
        eng = mr.SizedEngine(cfg_ew, b, mr.inverse_vol_weight(None, scale, horizon_of=horizon_of, by_h=by_h), cap=0.10)
        r = eng.run(sig, S, uni, fs=fs)
        p7["sizing_runs"][m] = {sp: mr.period_metrics(r.equity, r.trades, a, z) for sp, (a, z) in SPLITS.items()}
        p7["sizing_runs"][m]["sharpe_minus_S1_OOS"] = mr.sharpe_diff_ci(r.equity, base.equity, *SPLITS["OOS"])
        p7["sizing_runs"][m]["fallback_weights"] = eng.sizing_diag["fallback_weight"]
        print(f"P7 sizing {m}: OOS sharpe {p7['sizing_runs'][m]['OOS'].get('sharpe')} ({time.time() - t0:.0f}s)", flush=True)
        del by_h
        gc.collect()
    p7["S1_current"] = {sp: mr.period_metrics(base.equity, base.trades, a, z) for sp, (a, z) in SPLITS.items()}
    del daily
    gc.collect()

    # P8 OOS test: the combined S1 book with the disabled cells filtered out at the signal session
    def filt(day, objs):
        row = lab.loc[pd.Timestamp(day)] if pd.Timestamp(day) in lab.index else None
        if row is None:
            return objs
        return [o for o in objs if not any((o.strategy_id, l, str(row[l])) in disabled for l in LABELS)]
    if disabled:
        r_dis = BacktestEngine(cfg, b).run(sig, S, uni, fs=fs, candidate_filter=filt)
        p8["OOS_test"] = {"with_disable": mr.period_metrics(r_dis.equity, r_dis.trades, *SPLITS["OOS"]),
                          "without": p7["S1_current"]["OOS"],
                          "sharpe_diff": mr.sharpe_diff_ci(r_dis.equity, base.equity, *SPLITS["OOS"])}
        wd, wo = p8["OOS_test"]["with_disable"], p8["OOS_test"]["without"]
        p8["OOS_verdict"] = ("IMPROVES net expectancy" if (wd.get("total_return") or -9) > (wo.get("total_return") or -9)
                             and (wd.get("sharpe") or -9) > (wo.get("sharpe") or -9) else "DOES NOT IMPROVE")
    else:
        p8["OOS_test"] = "no cell met the disable rule on TRAIN+VAL: nothing to apply"
    # where each strategy works / fails (OOS, standalone), descriptive
    oos_cells = cells[cells["era"] == "OOS"]
    p8["OOS_cells"] = oos_cells.to_dict("records")

    # options trades by regime (OOS; descriptive)
    tp = CACHE / "sprint_p6_trades.parquet"
    if tp.exists():
        ot = pd.read_parquet(tp)
        ot["sig"] = pd.to_datetime(ot["session"])
        out_o = []
        for l in LABELS:
            ot[l] = lab[l].reindex(ot["sig"]).to_numpy()
            for (hyp, val), g in ot.groupby(["hypothesis", l]):
                for lvl in ("CONSERVATIVE", "PESSIMISTIC"):
                    m_, t_ = cluster_t(g[f"ret_{lvl}"], g["sig"])
                    out_o.append({"hypothesis": hyp, "label": l, "value": str(val), "level": lvl, "n": int(len(g)),
                                  "mean": m_, "t": t_})
        p8["options_OOS_by_regime"] = out_o

    # ---------------- P7 leaderboard assembly -------------------------------------------------------------------
    p2 = json.loads((OUT / "P2_vol_forecast.json").read_text())
    p46 = json.loads((OUT / "P4_P6_options.json").read_text()) if (OUT / "P4_P6_options.json").exists() else {}
    s1o = p7["S1_current"]["OOS"]
    rows = []
    for m in LB_MODELS + ("IV", "ENSEMBLE", "NO_FORECAST"):
        row: dict = {"model": m}
        if m in p2["horizons"]["5"]["splits"]["OOS"]["models"]:
            for h in ("5", "20"):
                mm = p2["horizons"][h]["splits"]["OOS"]["models"][m]
                row[f"QLIKE_h{h}"] = mm["QLIKE"]
                row[f"IC_h{h}"] = mm["IC"]["mean"]
                row[f"q95_exceed_h{h}"] = mm["tail_coverage"]["q95"]["exceed_rate"]
                row[f"levelfree_QLIKE_h{h}"] = p2["horizons"][h]["splits"]["OOS"]["level_free_QLIKE"][m]
        if m in p7["sizing_runs"]:
            sr = p7["sizing_runs"][m]
            row.update({"sizing_OOS_sharpe": sr["OOS"].get("sharpe"), "sizing_OOS_cagr": sr["OOS"].get("cagr"),
                        "sizing_OOS_sharpe_minus_S1": sr["sharpe_minus_S1_OOS"]["diff"],
                        "sizing_OOS_sharpe_minus_S1_ci90": sr["sharpe_minus_S1_OOS"]["ci90"],
                        "sizing_improvement": bool(sr["sharpe_minus_S1_OOS"]["diff"] >= 0.10 and sr["sharpe_minus_S1_OOS"]["ci90"][0] > 0
                                                   and (sr["OOS"].get("cagr") or -9) >= (s1o.get("cagr") or -9))})
        key = "QL" if m == "QL_OLS" else m
        o1m = p46.get("O1_by_model", {}).get(key)
        if o1m and o1m.get("OOS"):
            row.update({"O1_k": o1m["k"], "O1_OOS_n": o1m["OOS"]["CONSERVATIVE"]["n"],
                        "O1_OOS_mean_CONSERVATIVE": o1m["OOS"]["CONSERVATIVE"]["mean"],
                        "O1_OOS_mean_PESSIMISTIC": o1m["OOS"]["PESSIMISTIC"]["mean"], "O1_class": o1m.get("OOS_class")})
        if m == "IV":
            row["note"] = ("IV is the forecast the options market prices; P4 compares it directly. It cannot size the "
                           "2016-24 equity book (options data start 2019 and cover optionable names only), and O1 with "
                           "IV as the forecast compares IV with itself (no signal).")
        if m == "ENSEMBLE":
            row["note"] = p46.get("ensemble", {}).get("status", "see P4_P6_options.json")
        if m == "NO_FORECAST":
            row.update({"sizing_OOS_sharpe": s1o.get("sharpe"), "sizing_OOS_cagr": s1o.get("cagr"),
                        "note": "S1 current sizing (inverse-ATR via stops) = the bot today; no option trade"})
        rows.append(row)
    p7["leaderboard"] = rows
    winners = [r for r in rows if r.get("sizing_improvement")]
    robust = [r for r in rows if r.get("O1_class") == "ROBUST"]
    if winners:
        best = max(winners, key=lambda r: r["sizing_OOS_sharpe"])
        p7["winner"] = {"model": best["model"], "via": "sizing", "why": "best OOS Sharpe among IMPROVEMENT rows"}
    elif robust:
        best = max(robust, key=lambda r: r["O1_OOS_mean_PESSIMISTIC"])
        p7["winner"] = {"model": best["model"], "via": "options O1", "why": "ROBUST at PESSIMISTIC fills"}
    else:
        p7["winner"] = {"model": None, "via": None,
                        "why": "no forecast creates positive OOS economic value after costs: no sizing rule met the IMPROVEMENT "
                               "test against the bot's current sizing, and no option rule is ROBUST"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P7_leaderboard.json").write_text(json.dumps(registry._clean(p7), indent=1, default=str))
    (OUT / "P8_regimes.json").write_text(json.dumps(registry._clean(p8), indent=1, default=str))
    registry.append_run(hypothesis_id="S7_model_competition", family="sprint_leaderboard", split="OOS",
                        spec={"prereg": "ALPHA-SPRINT-PREREG section 7", "models": list(LB_MODELS)},
                        metrics={r["model"]: {k: v for k, v in r.items() if k != "model" and not isinstance(v, str)} for r in rows},
                        conclusion=json.dumps(p7["winner"]), seed=7)
    registry.append_run(hypothesis_id="S8_regimes", family="sprint_regimes", split="OOS",
                        spec={"prereg": "ALPHA-SPRINT-PREREG section 8", "labels": list(LABELS)},
                        metrics={"disabled": p8["disabled"], "oos": p8.get("OOS_verdict")},
                        conclusion=str(p8.get("OOS_verdict", p8["OOS_test"])), seed=7)
    print(f"winner: {p7['winner']}; P8: {p8.get('OOS_verdict', p8['OOS_test'])}; done {time.time() - t0:.0f}s {mem()}", flush=True)


if __name__ == "__main__":
    main()
