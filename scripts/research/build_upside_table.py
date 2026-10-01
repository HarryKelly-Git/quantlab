"""Build src/quantlab/exploration/data/upside_table.json from upside_obs.parquet.

Pipeline (research only, real data, PIT, 2025+ holdout never read):
  feat_replay.py -> feat_obs.parquet -> path_outcomes.py -> path_obs.parquet
  -> upside_analysis.py -> upside_obs.parquet -> THIS script.
The research scripts were run from a scratch directory; edit their SPW path to re-run.

Cells: fixed ATR% bucket edges (2021-2024 quintile boundaries) x range_contraction_20_60 terciles,
for each live hold arm (5/10/20 sessions). Every probability is an empirical frequency over the
liquid research universe (adv20 >= $5M). Descriptive: a calibrated base rate for the candidate's
volatility profile, NOT an edge -- the 2026-10-01 study found these characteristics predict the
SIZE of moves in both directions, not their direction (docs/UPSIDE-EVIDENCE.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src = Path(sys.argv[1])
out = Path(sys.argv[2])
o = pd.read_parquet(src)
o = o[(o["adv20"] >= 5e6) & o["entry"].notna() & o["atr14_pct"].notna() & o["range_contraction_20_60"].notna()]
atr_edges = [float(x) for x in o["atr14_pct"].quantile([0.2, 0.4, 0.6, 0.8]).round(5)]
rc_edges = [float(x) for x in o["range_contraction_20_60"].quantile([1 / 3, 2 / 3]).round(4)]
o = o.assign(atr_b=np.searchsorted(atr_edges, o["atr14_pct"], side="right"),
             rc_b=np.searchsorted(rc_edges, o["range_contraction_20_60"], side="right"))
cells = {}
for (ab, rb), g in o.groupby(["atr_b", "rc_b"]):
    for h in (5, 10, 20):
        stopped = g["days_to_stop"] <= h
        rec = {"n": int(len(g)), "p_stop": round(float(stopped.mean()), 4)}
        for t in (5, 8, 10, 15):
            hit = g[f"days_to_{t}"] <= h
            clean = hit & ~(g[f"stop_first_{t}"] & stopped)
            dn = (g[f"days_to_dn{t}"] <= h) & ((g[f"days_to_dn{t}"] < g[f"days_to_{t}"]) | g[f"days_to_{t}"].isna())
            rec[f"p_touch_{t}"] = round(float(hit.mean()), 4)
            rec[f"p_clean_{t}"] = round(float(clean.mean()), 4)
            rec[f"p_down_first_{t}"] = round(float(dn.mean()), 4)
            rec[f"median_days_to_{t}"] = float(g.loc[hit, f"days_to_{t}"].median()) if hit.any() else None
            rec[f"median_dd_before_{t}"] = round(float(g.loc[hit, f"mae_before_{t}"].median()), 4) if hit.any() else None
        net = np.where(stopped, -g["stop_pct"] - g[f"cost_{h}"], g[f"net_{h}"])
        net = net[np.isfinite(net)]
        rec["net_quantiles"] = [round(float(x), 4) for x in np.quantile(net, np.linspace(0, 1, 21))]
        rec["mfe_quantiles"] = [round(float(x), 4) for x in np.nanquantile(g[f"mfe_{h}"], np.linspace(0, 1, 21))]
        rec["mean_net"] = round(float(net.mean()), 5)
        cells[f"{ab}|{rb}|{h}"] = rec
meta = {"version": 1, "built_from": "PIT replay 2021-03-12..2024-11-27, every 5th session, research universe, adv20 >= $5M",
        "n_obs": int(len(o)), "atr_pct_edges": atr_edges, "range_contraction_edges": rc_edges,
        "holds": [5, 10, 20], "targets_pct": [5, 8, 10, 15],
        "definitions": {"entry": "next session open", "stop": "entry x (1 - 2 x atr14_pct), touched by the low",
                        "p_clean_T": "+T% touched before the stop (same-day counts as stop)",
                        "net_quantiles": "21 quantiles (0..100%) of net return with stop + time exit at the hold, after costs"},
        "caveat": "descriptive base rates by volatility profile; NOT a directional edge (docs/UPSIDE-EVIDENCE.md)"}
out.write_text(json.dumps({"meta": meta, "cells": cells}, indent=1), encoding="utf-8")
print(f"{len(cells)} cells from {len(o):,} obs -> {out}  atr edges {atr_edges}  rc edges {rc_edges}")
