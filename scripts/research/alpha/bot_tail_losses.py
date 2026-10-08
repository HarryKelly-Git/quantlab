"""Descriptive: where the paper bot's replayed losses come from (no hypothesis, no verdict). PAPER research.

Input: the trades of the current book (S1 sizing, corrected delisting convention) from the Area F base replay,
saved at var/alpha/cache/f_base_trades.parquet. Writes research/alpha/results/new_areas/bot_tail_losses.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import registry  # noqa: E402

BIG = -0.25


def main(path: str = str(ROOT / "var" / "alpha" / "cache" / "f_base_trades.parquet")) -> None:
    t = pd.read_parquet(path)
    t["stop_dist"] = 1 - t["stop_price"] / t["entry_price"]
    big = t[t["net_ret"] < BIG]
    st = t[t["exit_reason"] == "STOP"]
    beyond = st["exit_price"] / st["stop_price"] - 1
    out = {
        "source": "Area F base replay (S1, corrected delisting), docs/NEW-AREAS-PREREG.md section F", "git": registry.git_commit(),
        "trades": int(len(t)), "total_pnl": round(float(t["pnl"].sum()), 2),
        "big_losers": {"threshold": BIG, "n": int(len(big)), "share": round(len(big) / len(t), 4),
                       "pnl": round(float(big["pnl"].sum()), 2), "exit_reasons": big["exit_reason"].value_counts().to_dict(),
                       "by_strategy": big.groupby("strategy_id").size().to_dict(),
                       "worst": big.nsmallest(10, "net_ret")[["strategy_id", "symbol", "entry_date", "exit_reason", "net_ret"]]
                       .astype(str).to_dict("records")},
        "stop_distance_median_by_strategy": t.groupby("strategy_id")["stop_dist"].median().round(4).to_dict(),
        "stops_at_or_below_zero": int((t["stop_dist"] >= 1).sum()),
        "stop_exits": int(len(st)), "stop_exits_gapped_10pct_beyond_stop": int((beyond < -0.10).sum()),
        "pnl_by_year": t.groupby(pd.to_datetime(t["entry_date"]).dt.year)["pnl"].sum().round(2).rename(str).to_dict(),
    }
    dst = ROOT / "research" / "alpha" / "results" / "new_areas" / "bot_tail_losses.json"
    dst.write_text(json.dumps(registry._clean(out), indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("trades", "total_pnl", "stops_at_or_below_zero", "stop_exits_gapped_10pct_beyond_stop")}),
          out["big_losers"]["n"], out["big_losers"]["pnl"], out["stop_distance_median_by_strategy"])


if __name__ == "__main__":
    main(*sys.argv[1:])
