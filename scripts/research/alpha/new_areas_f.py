"""Area F (docs/NEW-AREAS-PREREG.md): a low-volatility screen on the paper bot's own replayed book. A candidate is
dropped when QuantLab's 20-day volatility forecast for it (P2 model QL_OLS, daily) is above the 80th percentile
of that day's liquid universe. A missing forecast keeps the candidate (counted). PAPER research, 2016-2024.
Writes research/alpha/results/new_areas/F_lowvol_screen.json."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QUANTLAB_SKIP_LOCAL_CONFIG", "1")

from quantlab.alpha import ca_fixes, registry, research_data  # noqa: E402
from quantlab.alpha import master_replay as mr  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402
from quantlab.backtest.engine import BacktestEngine  # noqa: E402
from quantlab.config import load_config  # noqa: E402
from quantlab.features.base import FeatureSet  # noqa: E402
from quantlab.strategies.arena import StrategyArena  # noqa: E402
from quantlab.strategies.registry import build_strategies  # noqa: E402
from quantlab.universe.engine import UniverseEngine  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "new_areas"
CACHE = store_dir() / "cache"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
MODEL, PCT = "QL_OLS_h20", 0.80


def main() -> None:
    t0 = time.time()
    d = research_data.get()
    ca_fixes.patch_research_data(d)
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps", "buckets"):
        setattr(d, k, None)
    gc.collect()
    cfg = load_config(overrides={"backtest": {"sizing": "equal_risk"}})
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

    # forecasts made at each day's close on that day's liquid universe; cut-off = that day's 80th percentile
    daily = pd.read_parquet(CACHE / "sprint_p2_daily.parquet", columns=["date", "symbol", MODEL])
    fc = daily.pivot(index="date", columns="symbol", values=MODEL)
    del daily
    gc.collect()
    q80 = fc.quantile(PCT, axis=1)
    high = fc.gt(q80, axis=0)                        # NaN forecast -> False (kept)
    has = fc.notna()
    first_fc = str(fc.index.min().date())
    print(f"forecasts {fc.shape}, first {first_fc}; {time.time() - t0:.0f}s", flush=True)
    seen = {"candidates": 0, "missing_kept": 0, "dropped": 0}

    def lowvol(day, objs):
        day = pd.Timestamp(day)
        hrow = high.loc[day] if day in high.index else None
        arow = has.loc[day] if day in has.index else None
        kept = []
        for o in objs:
            seen["candidates"] += 1
            if hrow is None or not bool(arow.get(o.symbol, False)):
                seen["missing_kept"] += 1
                kept.append(o)
            elif bool(hrow.get(o.symbol, False)):
                seen["dropped"] += 1
            else:
                kept.append(o)
        return kept

    def monthly(eq: pd.DataFrame) -> dict:
        e = eq.set_index("date") if "date" in eq.columns else eq
        v = e["equity"]
        v.index = pd.to_datetime(v.index)
        return {str(k.date()): round(float(x), 2) for k, x in v.resample("ME").last().dropna().items()}

    def tag(trades: pd.DataFrame) -> pd.DataFrame:
        tr = trades.copy()
        sd = pd.to_datetime(tr["signal_date"])
        tr["fc_state"] = ["missing" if x not in has.index or s not in has.columns or not has.at[x, s]
                          else ("high" if high.at[x, s] else "low") for x, s in zip(sd, tr["symbol"])]
        return tr

    out: dict = {"prereg": "docs/NEW-AREAS-PREREG.md section F", "git": registry.git_commit(), "model": MODEL,
                 "percentile": PCT, "first_forecast_date": first_fc,
                 "note": "No forecasts exist before the first forecast date (P2 warm-up), so the filter keeps every 2016 "
                         "candidate; TRAIN is effectively 2017-2019 for the screen.", "conventions": {}}
    for name, dr in (("corrected_delisting", 0.0), ("master_minus30", -0.30)):
        c = load_config(overrides={"backtest": {"sizing": "equal_risk"}, "costs": {"delisting_return": dr}})
        base = BacktestEngine(c, b).run(sig, S, uni, fs=fs)
        for k in seen:
            seen[k] = 0
        filt = BacktestEngine(c, b).run(sig, S, uni, fs=fs, candidate_filter=lowvol)
        tr = tag(base.trades)
        by_state = tr.groupby("fc_state")["net_ret"].agg(["count", "mean"]).round(5).to_dict("index")
        res = {"base": {sp: mr.period_metrics(base.equity, base.trades, a, z) for sp, (a, z) in SPLITS.items()},
               "lowvol": {sp: mr.period_metrics(filt.equity, filt.trades, a, z) for sp, (a, z) in SPLITS.items()},
               "sharpe_diff": {sp: mr.sharpe_diff_ci(filt.equity, base.equity, a, z) for sp, (a, z) in SPLITS.items()},
               "filter_counts": dict(seen),
               "base_trades_by_forecast_state": by_state,
               "base_high_vol_share_by_strategy": tr.assign(h=tr["fc_state"].eq("high")).groupby("strategy_id")["h"]
               .mean().round(4).to_dict(),
               "equity_monthly": {"base": monthly(base.equity), "lowvol": monthly(filt.equity)}}
        o = res["sharpe_diff"]["OOS"]
        improve = (o["diff"] >= 0.10 and o["ci90"][0] > 0
                   and (res["lowvol"]["OOS"].get("cagr") or -9) >= (res["base"]["OOS"].get("cagr") or -9)
                   and res["sharpe_diff"]["TRAIN"]["diff"] > 0 and res["sharpe_diff"]["VAL"]["diff"] > 0)
        res["verdict"] = "IMPROVEMENT" if improve else "NO IMPROVEMENT"
        out["conventions"][name] = res
        print(f"{name}: OOS sharpe base {res['base']['OOS'].get('sharpe'):.3f} -> lowvol {res['lowvol']['OOS'].get('sharpe'):.3f}; "
              f"diffs { {k: (round(v['diff'], 3), [round(x, 3) for x in v['ci90']]) for k, v in res['sharpe_diff'].items()} }; "
              f"counts {seen}; by state {by_state}; verdict {res['verdict']} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "F_lowvol_screen.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    registry.append_run(hypothesis_id="NA_F_lowvol_screen", family="new_areas", split="OOS",
                        spec={"prereg": "NEW-AREAS-PREREG F", "model": MODEL, "percentile": PCT},
                        metrics={k: {"verdict": v["verdict"], "oos_sharpe_diff": v["sharpe_diff"]["OOS"]["diff"],
                                     "oos_ci90": v["sharpe_diff"]["OOS"]["ci90"]} for k, v in out["conventions"].items()},
                        conclusion=out["conventions"]["corrected_delisting"]["verdict"], seed=7)


if __name__ == "__main__":
    main()
