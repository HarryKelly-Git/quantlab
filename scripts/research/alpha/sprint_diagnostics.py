"""Sprint diagnostics for the final report: robustness of the option candidates (O1/O2 by year, P&L concentration,
O1's excess over buying every liquid straddle on the same dates) and the delisted trades of the master replay.
Descriptive only; nothing here selects or tunes. Writes research/alpha/results/sprint/diagnostics.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import registry  # noqa: E402
from quantlab.alpha.options_store import options_dir  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "sprint"


def main() -> None:
    cache = store_dir() / "cache"
    t = pd.read_parquet(cache / "sprint_p6_trades.parquet")
    out: dict = {"git": registry.git_commit(), "options": {}}
    for h in ("O1", "O2"):
        x = t[t["hypothesis"] == h].copy()
        x["yr"] = pd.to_datetime(x["session"]).dt.year
        r = x["ret_PESSIMISTIC"]
        top = r.sort_values(ascending=False)
        k = max(1, len(r) // 100)
        out["options"][h] = {
            "n": int(len(x)), "n_dates": int(x["session"].nunique()), "n_underlyings": int(x["act_symbol"].nunique()),
            "mean_pess": float(r.mean()), "median_pess": float(r.median()), "hit_rate": float((r > 0).mean()),
            "by_year_pess": x.groupby("yr")["ret_PESSIMISTIC"].agg(["count", "mean", "median"]).round(4).to_dict("index"),
            "by_quarter_pess": {str(q): round(float(v), 4) for q, v in
                                x.groupby(pd.to_datetime(x["session"]).dt.to_period("Q"))["ret_PESSIMISTIC"].mean().items()},
            "top10_share_of_pnl": float(top.head(10).sum() / r.sum()) if r.sum() != 0 else None,
            "mean_excluding_top1pct": float(top.iloc[k:].mean())}
    # O1 vs buying EVERY liquid ~30-DTE straddle on the same dates (ask fills, settle at intrinsic): selection vs timing
    f = pd.read_parquet(options_dir() / "features.parquet", columns=["date", "session", "act_symbol", "dte", "straddle_bid",
                                                                     "straddle_ask", "straddle_mid", "ret_hold_ask",
                                                                     "split_or_special_div"])
    f = f[(f["dte"] >= 21) & (f["dte"] <= 45) & ~f["split_or_special_div"].astype(bool)].copy()
    f["dd"] = (f["dte"] - 30).abs()
    f = f.loc[f.groupby(["date", "act_symbol"])["dd"].idxmin()]
    f["week"] = pd.to_datetime(f["session"]).dt.to_period("W-FRI")
    f = f.sort_values("date").groupby(["week", "act_symbol"]).head(1)
    f = f[(f["straddle_ask"] - f["straddle_bid"]) / f["straddle_mid"] <= 0.10]
    base = f.groupby("date")["ret_hold_ask"].mean().rename("base")
    o1 = t[t["hypothesis"] == "O1"].merge(base, left_on="date", right_index=True, how="left")
    o1["ex"] = o1["ret_hold_ask"] - o1["base"]
    by_d = o1.groupby("date")["ex"].mean()
    out["O1_excess_vs_same_date_baseline"] = {
        "definition": "O1 straddle return at the ask minus the mean return of every liquid (spread <= 10%) ~30-DTE straddle "
                      "on the same snapshot date; settle at intrinsic, no fees (same metric on both sides)",
        "mean_per_trade": float(o1["ex"].mean()), "by_year": o1.groupby(pd.to_datetime(o1["session"]).dt.year)["ex"].mean().round(4).to_dict(),
        "t_across_dates": float(by_d.mean() / by_d.std() * np.sqrt(len(by_d))), "n_dates": int(len(by_d)),
        "baseline_mean_on_O1_dates": float(o1["base"].mean()), "O1_mean_same_metric": float(o1["ret_hold_ask"].mean())}
    p3 = pd.read_parquet(cache / "sprint_p3_trades.parquet")
    dl = p3[(p3["rule"] == "S1") & (p3["exit_reason"] == "DELISTED")]
    out["S1_delisted_trades_master_convention"] = dl[["symbol", "strategy_id", "entry_date", "exit_date", "net_ret"]].astype(str).to_dict("records")
    out["S1_delisted_note"] = ("every one was a cash acquisition (Corium, American Railcar, Dova, Casper, Turning Point, Biohaven, "
                               "Twitter, ChemoCentryx, Akouos, Albireo, Provention, BELLUS, National Western); master's backtester "
                               "books -30% on each")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    print(json.dumps(out["O1_excess_vs_same_date_baseline"], indent=1, default=str))


if __name__ == "__main__":
    main()
