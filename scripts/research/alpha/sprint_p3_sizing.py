"""Sprint P3 (+ P1-D, P8 inputs): the paper bot's own strategies replayed on the survivorship-free store, under
the five pre-registered sizing rules (docs/ALPHA-SPRINT-PREREG.md sections 1, 3 and 8). PAPER research.

Writes research/alpha/results/sprint/P3_sizing.json and var/alpha/cache/sprint_p3_trades.parquet.
Needs var/alpha/cache/sprint_p2_daily.parquet (P2) for the forecast-vol rule.

Usage: .venv/bin/python scripts/research/alpha/sprint_p3_sizing.py [--only-combined]
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
RELAXED = {"momentum_trend": {"min_score_pct": 0.80}, "relative_strength": {"vs_market_min_pct": 0.70},
           "mean_reversion": {"shock_z": -1.5}, "breakout": {"volume_mult": 1.25},
           "extreme_reversal": {"extreme_return_z": -3.0}, "sector_rotation": {"stock_rank_pct": 0.70}}


def mem() -> str:
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak"


def metrics_all(res) -> dict:
    return {sp: mr.period_metrics(res.equity, res.trades, a, b) for sp, (a, b) in SPLITS.items()}


def main(only_combined: bool = False) -> None:
    t0 = time.time()
    d = research_data.get()
    ca_info = ca_fixes.patch_research_data(d)
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps", "buckets"):
        setattr(d, k, None)
    gc.collect()
    cfg = load_config(overrides={"backtest": {"sizing": "equal_risk"}})
    bench = cfg.section("benchmarks").get("sectors", {})
    p = d.p
    st = p.master.set_index("symbol")["sec_type"].reindex(p.symbols)
    ever = d.u_liquid.any(axis=0)
    keep = [s for s in p.symbols if ever.get(s, False) or s == "SPY" or s in bench]
    cols = pd.Index(keep)
    b = mr.bundle_from_research(p, cols, p.master, d.sectors, bench)
    b_raw = mr.bundle_from_research(p, cols, p.master, d.sectors, bench, unknown_days="raw")
    regimes = d.regimes
    u_liq = d.u_liquid
    del d, p
    gc.collect()
    print(f"bundle {len(cols)} symbols ({int(st.reindex(cols).eq('COMMON').sum())} COMMON); meta {b.panel.meta}; "
          f"{time.time() - t0:.0f}s {mem()}", flush=True)
    uni = UniverseEngine(cfg).membership(b)
    print(f"master universe: {uni.sum(axis=1).mean():.0f} names/day "
          f"(research liquid universe {u_liq.sum(axis=1).mean():.0f}); {time.time() - t0:.0f}s", flush=True)
    fs = FeatureSet(b, universe=uni)
    strats = build_strategies(cfg, only=list(mr.REPLAYED))
    S = {s.strategy_id: s for s in strats}
    sig = StrategyArena(strats).scores(fs, uni)
    print(f"signals {', '.join(f'{k}:{int(np.isfinite(v.to_numpy()).sum())}' for k, v in sig.items())}; "
          f"{time.time() - t0:.0f}s {mem()}", flush=True)

    out: dict = {"prereg": "docs/ALPHA-SPRINT-PREREG.md sections 1 and 3", "git": registry.git_commit(), "ca_patch": ca_info,
                 "replayed": list(mr.REPLAYED), "not_replayed": mr.NOT_REPLAYED, "bundle_meta": b.panel.meta,
                 "universe_names_per_day": float(uni.sum(axis=1).mean()), "runs": {}}
    hold = {s.strategy_id: int(s.params.get("hold_sessions", 20)) for s in strats}
    horizon_of = lambda sid: 5 if hold.get(sid, 20) <= 7 else 20  # noqa: E731

    # forecasts aligned to the bundle: HV21 (S2/S3) and the P2-selected model (S4); TRAIN-only scale m
    lr = np.log1p(b.panel.ret.clip(lower=-0.99))
    hv21 = (lr.rolling(21, min_periods=17).std() * np.sqrt(252)).to_numpy()
    daily = pd.read_parquet(CACHE / "sprint_p2_daily.parquet")
    import pickle
    sel = pickle.load((CACHE / "sprint_p2_models.pkl").open("rb"))["selected"]
    by_h = {}
    for h in (5, 20):
        w = daily.pivot(index="date", columns="symbol", values=f"{sel}_h{h}").reindex(index=b.panel.dates, columns=cols)
        by_h[h] = w.to_numpy(dtype="float64")
    del daily
    gc.collect()
    out["forecast_model_S4"] = sel

    def run(name: str, engine, strategies=S, signals=sig):
        t = time.time()
        r = engine.run(signals, strategies, uni, fs=fs)
        out["runs"][name] = {"metrics": metrics_all(r), "diagnostics": {k: v for k, v in r.diagnostics.items()
                                                                        if isinstance(v, (int, float))},
                             "sizing_diag": getattr(engine, "sizing_diag", None)}
        print(f"{name}: OOS {out['runs'][name]['metrics']['OOS'].get('sharpe')}, trades {len(r.trades)} "
              f"({time.time() - t:.0f}s) {mem()}", flush=True)
        return r

    # S1 current (equal_risk) first: its TRAIN entries define m for S2-S4
    s1 = run("combined_S1_current", BacktestEngine(cfg, b))
    tr = s1.trades.copy()
    ent = pd.to_datetime(tr["entry_date"])
    tr_train = tr[ent <= SPLITS["TRAIN"][1]]
    di = b.panel.dates.get_indexer(pd.to_datetime(tr_train["entry_date"])) - 1      # signal session = entry - 1
    ci = cols.get_indexer(tr_train["symbol"])
    m_hv = float(np.nanmedian(hv21[di, ci]))
    m_fc = float(np.nanmedian([by_h[horizon_of(s)][i, c] for s, i, c in zip(tr_train["strategy_id"], di, ci)]))
    out["scale_m"] = {"HV21_median_TRAIN_entries": m_hv, f"{sel}_median_TRAIN_entries": m_fc, "n_train_entries": int(len(tr_train))}
    cfg_ew = load_config(overrides={"backtest": {"sizing": "equal_weight"}})
    s0 = run("combined_S0_fixed", BacktestEngine(cfg_ew, b))
    s2 = run("combined_S2_inverse_vol", mr.SizedEngine(cfg_ew, b, mr.inverse_vol_weight(hv21, m_hv), cap=0.25))
    s3 = run("combined_S3_capped_inverse_vol", mr.SizedEngine(cfg_ew, b, mr.inverse_vol_weight(hv21, m_hv), cap=0.10))
    s4 = run("combined_S4_forecast_vol", mr.SizedEngine(cfg_ew, b, mr.inverse_vol_weight(
        None, m_fc, horizon_of=horizon_of, by_h=by_h), cap=0.10))
    eqs = {"S0": s0.equity, "S1": s1.equity, "S2": s2.equity, "S3": s3.equity, "S4": s4.equity}
    # sensitivities of the S1 book: flagged corporate-action days at the RAW move; master's -30% delisting rule at 0
    uni_raw = UniverseEngine(cfg).membership(b_raw)
    fs_raw = FeatureSet(b_raw, universe=uni_raw)
    sig_raw = StrategyArena(strats).scores(fs_raw, uni_raw)
    t_ = time.time()
    rr_ = BacktestEngine(cfg, b_raw).run(sig_raw, S, uni_raw, fs=fs_raw)
    out["runs"]["combined_S1_sens_unknown_days_raw"] = {"metrics": metrics_all(rr_)}
    print(f"combined_S1_sens_unknown_days_raw: OOS {out['runs']['combined_S1_sens_unknown_days_raw']['metrics']['OOS'].get('sharpe')} ({time.time() - t_:.0f}s)", flush=True)
    del b_raw, uni_raw, fs_raw, sig_raw, rr_
    gc.collect()
    run("combined_S1_sens_delisting_return_0", BacktestEngine(load_config(overrides={"backtest": {"sizing": "equal_risk"},
                                                                                     "costs": {"delisting_return": 0.0}}), b))
    cls = {}
    for k in ("S0", "S2", "S3", "S4"):
        row = {}
        for sp in ("VAL", "OOS"):
            a, z = SPLITS[sp]
            ci_ = mr.sharpe_diff_ci(eqs[k], eqs["S1"], a, z)
            ma, mb = out["runs"][[n for n in out["runs"] if n.startswith(f"combined_{k}_")][0]]["metrics"][sp], \
                out["runs"]["combined_S1_current"]["metrics"][sp]
            row[sp] = {"sharpe_minus_S1": ci_, "cagr": ma.get("cagr"), "cagr_S1": mb.get("cagr"),
                       "vol": ma.get("vol"), "vol_S1": mb.get("vol"), "maxdd": ma.get("max_drawdown"),
                       "maxdd_S1": mb.get("max_drawdown")}
        o = row["OOS"]
        improve = (o["sharpe_minus_S1"]["diff"] >= 0.10 and o["sharpe_minus_S1"]["ci90"][0] > 0
                   and (o["cagr"] or -9) >= (o["cagr_S1"] or -9))
        less_risk = (o["vol"] or 9) < (o["vol_S1"] or 9) or (o["maxdd"] or -9) > (o["maxdd_S1"] or -9)
        row["classification_OOS"] = "IMPROVEMENT" if improve else ("RISK REDUCTION ONLY" if less_risk else "WORSE")
        cls[k] = row
    out["classification_vs_S1"] = cls
    trades_all = {"S1": s1.trades.assign(rule="S1")}

    # per-strategy standalone (S1 sizing): does ANY replayed strategy have a net edge on survivorship-free data?
    if not only_combined:
        out["standalone_S1"] = {}
        for sid in mr.REPLAYED:
            r = BacktestEngine(cfg, b).run({sid: sig[sid]}, {sid: S[sid]}, uni, fs=fs)
            out["standalone_S1"][sid] = metrics_all(r)
            trades_all[f"solo_{sid}"] = r.trades.assign(rule=f"solo_{sid}")
            print(f"solo {sid}: OOS {out['standalone_S1'][sid]['OOS']}", flush=True)
        # P1-D: score thresholds relaxed one step (historical ablation; combined book, S1 sizing)
        cfg_rel = load_config(overrides={"backtest": {"sizing": "equal_risk"},
                                         "strategies": {k: {"params": v} for k, v in RELAXED.items()}})
        strats_rel = build_strategies(cfg_rel, only=list(mr.REPLAYED))
        sig_rel = StrategyArena(strats_rel).scores(fs, uni)
        rr = run("combined_S1_relaxed_thresholds", BacktestEngine(cfg_rel, b), {s.strategy_id: s for s in strats_rel}, sig_rel)
        trades_all["S1_relaxed"] = rr.trades.assign(rule="S1_relaxed")
        out["P1_D_threshold_ablation"] = {"relaxed": RELAXED, "current": out["runs"]["combined_S1_current"]["metrics"],
                                          "relaxed_metrics": out["runs"]["combined_S1_relaxed_thresholds"]["metrics"]}

    # regime labels at each entry's signal session (P8 input), PIT
    spy = b.panel.aclose["SPY"]
    lab = pd.DataFrame(index=b.panel.dates)
    lab["trend"] = np.where(spy > spy.rolling(200, min_periods=200).mean(), "up", "down")
    lab["volatility"] = regimes["spy_vol"].reindex(b.panel.dates)
    lab["momentum"] = np.where(spy / spy.shift(126) - 1 > 0, "pos", "neg")
    lab["risk"] = regimes.get("credit", pd.Series(index=b.panel.dates, dtype=object)).reindex(b.panel.dates)
    above = (b.panel.aclose > b.panel.aclose.rolling(50, min_periods=50).mean()).astype("float64").where(uni)
    breadth = above.sum(axis=1) / uni.sum(axis=1)
    lab["breadth"] = pd.cut(breadth.rolling(504, min_periods=126).rank(pct=True), [0, 1 / 3, 2 / 3, 1.0],
                            labels=["low", "mid", "high"], include_lowest=True).astype(object)
    r_u = b.panel.ret.where(uni)
    ew = r_u.mean(axis=1)
    corr = ew.rolling(63, min_periods=50).var() / r_u.rolling(63, min_periods=50).var().mean(axis=1)
    lab["correlation"] = pd.cut(corr.rolling(504, min_periods=126).rank(pct=True), [0, 1 / 3, 2 / 3, 1.0],
                                labels=["low", "mid", "high"], include_lowest=True).astype(object)
    lab.to_parquet(CACHE / "sprint_regime_labels.parquet")
    tdf = pd.concat(trades_all.values(), ignore_index=True)
    tdf.to_parquet(CACHE / "sprint_p3_trades.parquet", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P3_sizing.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    for sp in ("VAL", "OOS"):
        registry.append_run(oos_override="SPRINT-C: re-run on the corporate-action-patched panel (fake adjusted moves found by the master replay); spec unchanged" if sp == "OOS" else None, hypothesis_id="S3_vol_sizing", family="sprint_sizing", split=sp,
                            spec={"prereg": "ALPHA-SPRINT-PREREG section 3", "rules": ["S0", "S1", "S2", "S3", "S4"],
                                  "forecast": sel},
                            metrics={k: v["metrics"][sp] for k, v in out["runs"].items()},
                            conclusion=json.dumps({k: v["classification_OOS"] for k, v in cls.items()}) if sp == "OOS" else "",
                            seed=7, data={"replayed": list(mr.REPLAYED)})
    print(f"done {time.time() - t0:.0f}s {mem()}", flush=True)


if __name__ == "__main__":
    main(only_combined="--only-combined" in sys.argv)
