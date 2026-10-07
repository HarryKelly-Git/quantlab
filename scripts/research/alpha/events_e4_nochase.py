"""E4 (docs/EVENT-STUDY-PREREG.md amendment 2): a no-chase filter on the paper bot's own replayed book. It drops
candidates whose signal day was a jump (total return >= +8% on >= 2x the trailing 20-session median dollar
volume). PAPER research, 2016-2024. Writes research/alpha/results/events/E4_nochase.json."""
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
from quantlab.backtest.engine import BacktestEngine  # noqa: E402
from quantlab.config import load_config  # noqa: E402
from quantlab.features.base import FeatureSet  # noqa: E402
from quantlab.strategies.arena import StrategyArena  # noqa: E402
from quantlab.strategies.registry import build_strategies  # noqa: E402
from quantlab.universe.engine import UniverseEngine  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "events"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}


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
    ret, dv = b.panel.ret, b.panel.dollar_volume
    jump = ((ret >= 0.08) & (dv >= 2.0 * dv.rolling(20, min_periods=20).median().shift(1))).fillna(False)

    def nochase(day, objs):
        row = jump.loc[pd.Timestamp(day)]
        return [o for o in objs if not bool(row.get(o.symbol, False))]

    out: dict = {"prereg": "EVENT-STUDY-PREREG amendment 2 (E4)", "git": registry.git_commit(), "conventions": {}}
    for name, dr in (("corrected_delisting", 0.0), ("master_minus30", -0.30)):
        c = load_config(overrides={"backtest": {"sizing": "equal_risk"}, "costs": {"delisting_return": dr}})
        base = BacktestEngine(c, b).run(sig, S, uni, fs=fs)
        filt = BacktestEngine(c, b).run(sig, S, uni, fs=fs, candidate_filter=nochase)
        tr = base.trades.copy()
        sd = pd.to_datetime(tr["signal_date"])
        tr["jump_signal"] = [bool(jump.at[x, s]) if x in jump.index and s in jump.columns else False for x, s in zip(sd, tr["symbol"])]
        res = {"base": {sp: mr.period_metrics(base.equity, base.trades, a, z) for sp, (a, z) in SPLITS.items()},
               "nochase": {sp: mr.period_metrics(filt.equity, filt.trades, a, z) for sp, (a, z) in SPLITS.items()},
               "sharpe_diff": {sp: mr.sharpe_diff_ci(filt.equity, base.equity, a, z) for sp, (a, z) in SPLITS.items()},
               "base_entries_after_jump_share": float(tr["jump_signal"].mean()),
               "by_strategy_share_after_jump": tr.groupby("strategy_id")["jump_signal"].mean().round(4).to_dict(),
               "base_trades_after_jump_mean_net": float(tr.loc[tr["jump_signal"], "net_ret"].mean()) if tr["jump_signal"].any() else None,
               "base_trades_other_mean_net": float(tr.loc[~tr["jump_signal"], "net_ret"].mean())}
        o = res["sharpe_diff"]["OOS"]
        improve = (o["diff"] >= 0.10 and o["ci90"][0] > 0 and (res["nochase"]["OOS"].get("cagr") or -9) >= (res["base"]["OOS"].get("cagr") or -9)
                   and res["sharpe_diff"]["TRAIN"]["diff"] > 0 and res["sharpe_diff"]["VAL"]["diff"] > 0)
        res["verdict"] = "IMPROVEMENT" if improve else "NO IMPROVEMENT"
        out["conventions"][name] = res
        print(f"{name}: OOS sharpe base {res['base']['OOS'].get('sharpe'):.3f} -> nochase {res['nochase']['OOS'].get('sharpe'):.3f}; "
              f"diffs { {k: round(v['diff'], 3) for k, v in res['sharpe_diff'].items()} }; after-jump share "
              f"{res['base_entries_after_jump_share']:.3f}; verdict {res['verdict']} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E4_nochase.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    registry.append_run(hypothesis_id="EV_E4_nochase", family="event_study", split="OOS",
                        spec={"prereg": "EVENT-STUDY-PREREG amendment 2", "jump": 0.08, "vol_mult": 2.0},
                        metrics={k: {"verdict": v["verdict"], "oos_sharpe_diff": v["sharpe_diff"]["OOS"]["diff"]}
                                 for k, v in out["conventions"].items()},
                        conclusion=out["conventions"]["corrected_delisting"]["verdict"], seed=7)


if __name__ == "__main__":
    main()
