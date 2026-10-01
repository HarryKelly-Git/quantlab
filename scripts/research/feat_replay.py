"""Feature -> forward-outcome replay (REAL data, PIT, holdout untouched).

For every research-universe symbol on every sampled session: the raw values of existing PIT features,
the (corrected) discovery score and fired families, then next-open-entry outcomes from
research.attach_outcomes (costs from the backtester's CostModel). Research only; writes a parquet.

    python feat_replay.py START END EVERY OUT.parquet
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\harry\Downloads\Businesspilot\quantlab")
sys.path.insert(0, str(REPO))
from quantlab.config import load_config                                          # noqa: E402
from quantlab.context import AppContext                                          # noqa: E402
from quantlab.core.costs import CostModel                                        # noqa: E402
from quantlab.data.panel import Panel                                            # noqa: E402
from quantlab.data.validation import quarantine_map                              # noqa: E402
from quantlab.discovery.engine import DiscoveryEngine                            # noqa: E402
from quantlab.discovery.families import SCORED                                   # noqa: E402
from quantlab.discovery.research import HORIZONS, attach_outcomes               # noqa: E402
from quantlab.features.base import FeatureSet                                    # noqa: E402
from quantlab.universe import UniverseEngine                                     # noqa: E402

FEATS = ["ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d", "mom_12_1",
         "dist_ma20", "dist_ma50", "dist_ma200", "ma50_over_ma200", "atr14_pct",
         "rel_volume_1d", "rel_volume_5d", "ret_z_1d", "ret_z_3d", "breakout_55", "range_contraction_20_60",
         "adv20", "industry_rank_63", "rs_industry_20", "sector_rs_spy_63_pit",
         "days_since_earnings", "ear_z", "news_count_1d", "news_count_z"]

SPW = Path(r"C:/Users/harry/AppData/Local/Temp/claude/C--Users-harry-Downloads-Businesspilot/a815f6ae-90eb-4d6b-8043-36bbc3440be6/scratchpad/research")
start, end, every, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
t0 = time.time()
ctx = AppContext.create(load_config(SPW / "cfg.yaml", root=REPO), init_logging=False)
bundle = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=False),
                               synthetic=False)
holdout = pd.Timestamp(ctx.config.get("validation.holdout.start", "2025-01-01"))
eng = DiscoveryEngine(ctx.config)
lb = eng.s.lookback_sessions
dates = bundle.panel.dates
sel = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
last_ok = dates[dates < holdout]
sel = sel[sel <= last_ok[-(max(HORIZONS) + 2)]]            # every outcome bar stays before the holdout
sel = sel[sel >= dates[min(lb, len(dates) - 1)]]
sample = sel[::every]
umask = UniverseEngine(ctx.config).membership(bundle.truncate(sample[-1]), exclude=quarantine_map(ctx.db))
benches = {bundle.market_symbol, *bundle.sector_etfs}
pos = {d: i for i, d in enumerate(dates)}
reg_names = set()
rows, bad = [], set()
blk = 12
for k in range(0, len(sample), blk):
    block = list(sample[k:k + blk])
    i0, i1 = max(0, pos[block[0]] - lb), pos[block[-1]]
    p = Panel({f: v.iloc[i0:i1 + 1] for f, v in bundle.panel.fields.items()}, dict(bundle.panel.meta))
    tb = replace(bundle, panel=p)
    fs = FeatureSet(tb, dtype="float32")
    if not reg_names:
        reg_names = {f for f in FEATS if f in set(fs.registry.names())}
        print("features available:", sorted(reg_names), "missing:", sorted(set(FEATS) - reg_names), flush=True)
    for d in block:
        c = eng.core(tb.panel, fs, d, benches)
        syms = c["syms"]
        if len(syms) == 0:
            continue
        df = pd.DataFrame({"date": d, "symbol": syms, "score": c["score"].to_numpy(), "adv20": c["adv20"].reindex(syms).to_numpy()})
        for fam in SCORED:
            df[f"fired_{fam}"] = c["masks"][fam].to_numpy()
            df[f"pts_{fam}"] = c["comp"][fam].to_numpy()
        for f in sorted(reg_names - bad):
            try:
                v = fs.cross_section(d, [f])[f].reindex(syms)
                df[f] = pd.to_numeric(v, errors="coerce").to_numpy()
            except Exception as exc:                          # a context source may be absent: record, move on
                bad.add(f)
                print(f"feature {f} unavailable: {exc!r}"[:200], flush=True)
        df["in_universe"] = umask.loc[d].reindex(syms).fillna(False).to_numpy() if d in umask.index else True
        rows.append(df)
    print(f"{block[-1].date()} rows={sum(len(r) for r in rows)} t={time.time() - t0:.0f}s", flush=True)
obs = pd.concat(rows, ignore_index=True)
obs = obs[obs["in_universe"]].reset_index(drop=True)
obs = attach_outcomes(obs, bundle, CostModel.from_config(ctx.config), stop_before=holdout)
obs.to_parquet(out, index=False)
print(f"DONE {len(obs)} obs, {obs['date'].nunique()} dates -> {out} in {time.time() - t0:.0f}s", flush=True)
ctx.close()
