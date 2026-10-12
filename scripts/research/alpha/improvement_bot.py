"""Improvement program Batch 1, bot tests B1-B5 (docs/IMPROVEMENT-PROGRAM.md Part B). PAPER research, 2016-2024.

Master's own engine on the survivorship-free replay (master_replay), corrected delisting convention, current
sizing (S1). One data load, then:
  B1 young-stock filter, B2 wide-stop filter, B3 both  (E4 IMPROVEMENT rule vs the base book)
  B4 pruning to strategies with Sharpe >= 0 in TRAIN and VAL (OOS-only verdict)
  B5 cash overlays from the base book: a) T-bill interest, b) idle cash in SPY (verdict vs SPY buy-and-hold)
Writes research/alpha/results/improvement/B_bot.json and ledger rows IP_B1..IP_B5.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QUANTLAB_SKIP_LOCAL_CONFIG", "1")

from quantlab.alpha import ca_fixes, registry, research_data  # noqa: E402
from quantlab.alpha import master_replay as mr  # noqa: E402
from quantlab.backtest.engine import BacktestEngine  # noqa: E402
from quantlab.config import load_config  # noqa: E402
from quantlab.features.base import FeatureSet  # noqa: E402
from quantlab.strategies.arena import StrategyArena  # noqa: E402
from quantlab.strategies.registry import build_strategies  # noqa: E402
from quantlab.universe.engine import UniverseEngine  # noqa: E402

from new_areas_ac import ff_daily  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "improvement"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
MIN_AGE, AGE_FROM, MAX_STOP = 252, pd.Timestamp("2017-01-03"), 0.20
PREREG = "docs/IMPROVEMENT-PROGRAM.md Part B"


def metrics(eq: pd.DataFrame, trades: pd.DataFrame) -> dict:
    return {sp: mr.period_metrics(eq, trades, a, z) for sp, (a, z) in SPLITS.items()}


def diffs(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    return {sp: mr.sharpe_diff_ci(a, b, x, z) for sp, (x, z) in SPLITS.items()}


def e4_rule(m_new: dict, m_base: dict, d: dict) -> str:
    o = d["OOS"]
    ok = (o["diff"] >= 0.10 and o["ci90"][0] > 0 and (m_new["OOS"].get("cagr") or -9) >= (m_base["OOS"].get("cagr") or -9)
          and d["TRAIN"]["diff"] > 0 and d["VAL"]["diff"] > 0)
    return "IMPROVEMENT" if ok else "NO IMPROVEMENT"


def monthly(eq: pd.DataFrame) -> dict:
    v = eq.set_index("date")["equity"] if "date" in eq.columns else eq["equity"]
    v.index = pd.to_datetime(v.index)
    return {str(k.date()): round(float(x), 2) for k, x in v.resample("ME").last().dropna().items()}


def overlay(base_eq: pd.DataFrame, other: pd.Series, cost_bps: float = 0.0) -> pd.DataFrame:
    """Base book + yesterday's cash weight invested in ``other`` (daily returns), with a cost per unit of
    weight changed. Returns an equity frame (date, equity, positions_value) for the period metrics."""
    e = base_eq.set_index("date") if "date" in base_eq.columns else base_eq
    r_bot = e["equity"].pct_change().fillna(0.0)
    w = (e["cash"] / e["equity"]).clip(lower=0.0).shift(1).fillna(0.0)
    o = other.reindex(e.index).fillna(0.0)
    r = r_bot + w * o - cost_bps * 1e-4 * w.diff().abs().fillna(0.0)
    eq = float(e["equity"].iloc[0]) * (1 + r).cumprod()
    return pd.DataFrame({"date": e.index, "equity": eq.to_numpy(), "positions_value": (e["positions_value"] / e["equity"] * eq).to_numpy()})


def main() -> None:
    t0 = time.time()
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps", "buckets"):
        setattr(d, k, None)
    gc.collect()
    cfg = load_config(overrides={"backtest": {"sizing": "equal_risk"}, "costs": {"delisting_return": 0.0}})
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
    age = b.panel.close.notna().cumsum()                     # sessions with a bar up to and including each day
    print(f"setup {time.time() - t0:.0f}s", flush=True)

    counts: dict[str, dict[str, int]] = {}

    def make_filter(name: str, use_age: bool, use_stop: bool):
        c = counts.setdefault(name, {"candidates": 0, "dropped_age": 0, "dropped_stop": 0, "kept": 0})

        def filt(day, objs):
            day = pd.Timestamp(day)
            arow = age.loc[day] if (use_age and day >= AGE_FROM and day in age.index) else None
            kept = []
            for o in objs:
                c["candidates"] += 1
                if arow is not None and float(arow.get(o.symbol, 0)) < MIN_AGE:
                    c["dropped_age"] += 1
                    continue
                if use_stop:
                    sp, ref = getattr(o.plan, "stop_price", None), getattr(o.plan, "entry_ref_price", None)
                    if sp is None or ref is None or not np.isfinite(sp) or not np.isfinite(ref) or sp <= 0 or ref <= 0 \
                            or sp / ref < 1 - MAX_STOP:
                        c["dropped_stop"] += 1
                        continue
                c["kept"] += 1
                kept.append(o)
            return kept
        filt.__name__ = name
        return filt

    eng = lambda: BacktestEngine(cfg, b)  # noqa: E731
    base = eng().run(sig, S, uni, fs=fs)
    mb = metrics(base.equity, base.trades)
    out: dict = {"prereg": PREREG, "git": registry.git_commit(), "convention": "corrected_delisting (0.0), S1 equal_risk",
                 "base": mb, "tests": {}, "equity_monthly": {"base": monthly(base.equity)}}
    print(f"base OOS sharpe {mb['OOS'].get('sharpe'):.3f} ({time.time() - t0:.0f}s)", flush=True)

    for tid, ua, us in (("B1_young", True, False), ("B2_wide_stop", False, True), ("B3_both", True, True)):
        res = eng().run(sig, S, uni, fs=fs, candidate_filter=make_filter(tid, ua, us))
        m, dd = metrics(res.equity, res.trades), diffs(res.equity, base.equity)
        out["tests"][tid] = {"metrics": m, "sharpe_diff": dd, "filter_counts": counts[tid], "verdict": e4_rule(m, mb, dd)}
        if ua and counts[tid]["dropped_age"] == 0:
            out["tests"][tid]["note"] = (f"The age rule never fired: master's universe already requires "
                                         f"{cfg.get('universe.min_history_sessions')} sessions of history (>= {MIN_AGE}). "
                                         "De-SPACs pass because the SPAC shell's trading history counts.")
        out["equity_monthly"][tid] = monthly(res.equity)
        print(f"{tid}: OOS {mb['OOS'].get('sharpe'):.3f} -> {m['OOS'].get('sharpe'):.3f}; "
              f"diffs { {k: round(v['diff'], 3) for k, v in dd.items()} }; {counts[tid]}; {out['tests'][tid]['verdict']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # B4 pruning: each strategy alone; keep if Sharpe >= 0 in TRAIN and VAL
    alone = {}
    for sid in S:
        r1 = eng().run({sid: sig[sid]}, {sid: S[sid]}, uni, fs=fs)
        alone[sid] = {sp: mr.period_metrics(r1.equity, r1.trades, a, z).get("sharpe") for sp, (a, z) in SPLITS.items()}
        print(f"  alone {sid}: {alone[sid]} ({time.time() - t0:.0f}s)", flush=True)
    keep = [sid for sid, v in alone.items() if (v["TRAIN"] or -9) >= 0 and (v["VAL"] or -9) >= 0]
    b4 = {"alone_sharpe": alone, "kept": keep}
    if 0 < len(keep) < len(S):
        r4 = eng().run({k: sig[k] for k in keep}, {k: S[k] for k in keep}, uni, fs=fs)
        m4, d4 = metrics(r4.equity, r4.trades), diffs(r4.equity, base.equity)
        o = d4["OOS"]
        b4.update(metrics=m4, sharpe_diff=d4,
                  verdict="IMPROVEMENT" if (o["diff"] >= 0.10 and o["ci90"][0] > 0
                                            and (m4["OOS"].get("cagr") or -9) >= (mb["OOS"].get("cagr") or -9)) else "NO IMPROVEMENT")
        out["equity_monthly"]["B4_pruned"] = monthly(r4.equity)
    else:
        b4["verdict"] = "NO CHANGE"
    out["tests"]["B4_pruned"] = b4
    print(f"B4 kept {keep}: {b4['verdict']} ({time.time() - t0:.0f}s)", flush=True)

    # B5 cash overlays
    rf = ff_daily()["rf"]
    spy_r = b.panel.ret["SPY"].fillna(0.0)
    b5a = overlay(base.equity, rf)
    b5b = overlay(base.equity, spy_r, cost_bps=1.0)
    beq = base.equity.set_index("date") if "date" in base.equity.columns else base.equity
    spy_eq = pd.DataFrame({"date": pd.DatetimeIndex(beq.index)})
    spy_eq["equity"] = 100000.0 * (1 + spy_r.reindex(pd.to_datetime(spy_eq["date"])).fillna(0.0).to_numpy()).cumprod()
    spy_eq["positions_value"] = spy_eq["equity"]
    empty = base.trades.iloc[0:0]
    ma, mbb, ms = metrics(b5a, empty), metrics(b5b, empty), metrics(spy_eq, empty)
    dvs = diffs(b5b, spy_eq)
    beats = all(dvs[s]["diff"] > 0 for s in SPLITS) and dvs["OOS"]["ci90"][0] > 0
    out["tests"]["B5_cash"] = {"B5a_tbill_metrics": ma, "B5b_spy_metrics": mbb, "spy_buy_hold": ms, "B5b_vs_spy_sharpe_diff": dvs,
                               "avg_cash_weight": float((beq["cash"] / beq["equity"]).mean()),
                               "verdict_B5b": "BOT ADDS VALUE OVER SPY" if beats else "SPY ALONE IS BETTER",
                               "note": "B5a is arithmetic (no verdict). Sharpe here is mean/std of daily returns, as in master_replay."}
    out["equity_monthly"].update(B5a_tbill=monthly(b5a), B5b_cash_in_spy=monthly(b5b), SPY=monthly(spy_eq))
    print(f"B5: base OOS cagr {mb['OOS'].get('cagr')}, T-bill {ma['OOS'].get('cagr')}, cash-in-SPY {mbb['OOS'].get('cagr')} "
          f"vs SPY {ms['OOS'].get('cagr')}; {out['tests']['B5_cash']['verdict_B5b']} ({time.time() - t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "B_bot.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    for tid, v in out["tests"].items():
        verdict = v.get("verdict") or v.get("verdict_B5b")
        registry.append_run(hypothesis_id=f"IP_{tid}", family="improvement_bot", split="OOS",
                            spec={"prereg": PREREG, "min_age": MIN_AGE, "max_stop": MAX_STOP},
                            metrics={"verdict": verdict, "oos_sharpe_diff": (v.get("sharpe_diff") or {}).get("OOS", {}).get("diff")},
                            conclusion=verdict, seed=7)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
