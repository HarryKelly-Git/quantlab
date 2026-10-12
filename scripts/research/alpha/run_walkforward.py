"""Rolling walk-forward selection for every equity family (Part 5; design pre-registered in
docs/ALPHA-DISCOVERY-PLAN.md section 10.9 before this was run).

Protocol: compute every pre-registered variant's daily NET return series (same engine, costs and
borrow as run_family, from each book's first active day). At the start of each calendar year Y in
2018..2024, select the variant with the highest annualised net Sharpe over ALL earlier years (expanding
window from 2016, at least 2 years of history), trade it for year Y, and record year Y's returns. The
concatenated 2018-2024 stream is fully out-of-sample with respect to selection. It re-uses calendar years
already seen in OOS, so it is counted as an OOS look (split 'WF_2018_2024').

Writes research/alpha/results/walk_forward_selection.json.
Usage: python scripts/research/alpha/run_walkforward.py [H01 H03 ...]
"""
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from quantlab.alpha import registry, research_data
from quantlab.alpha.engine import run_weights
from quantlab.alpha.metrics import core_metrics
from quantlab.validation.stats import newey_west_tstat

spec = importlib.util.spec_from_file_location("run_batch", pathlib.Path(__file__).with_name("run_batch.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

YEARS = range(2018, 2025)


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * math.sqrt(252)) if len(x) > 20 and x.std() > 0 else -np.inf


def walk_forward(nets: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    picks, parts = [], []
    for y in YEARS:
        hist = nets.loc[: f"{y - 1}-12-31"]
        if hist.index.min() is pd.NaT or (hist.index.max() - hist.index.min()).days < 2 * 365 - 30:
            continue
        sr = {c: sharpe(hist[c]) for c in nets.columns}
        best = max(sr, key=sr.get)
        yr = nets.loc[f"{y}-01-01": f"{y}-12-31", best]
        parts.append(yr)
        picks.append({"year": y, "selected": best, "trailing_sharpe": sr[best], "year_return": float((1 + yr.fillna(0)).prod() - 1),
                      "year_sharpe": sharpe(yr)})
    return (pd.concat(parts) if parts else pd.Series(dtype=float)), picks


def main(keys: list[str]) -> None:
    d = research_data.get()
    fams = rb.families(d)
    from quantlab.alpha.experiments import earnings
    out = {"protocol": "docs/ALPHA-DISCOVERY-PLAN.md section 10.9", "years": [YEARS[0], YEARS[-1]], "families": {}}
    spy = d.bench_oo
    for key in keys:
        t0 = time.time()
        if key == "H23":                       # same two pre-registered variants as earnings.run_h23
            from quantlab.alpha.engine import quantile_weights
            from quantlab.alpha.experiment import Variant
            from quantlab.alpha.experiments.reversal import context
            ctx = context(d)
            variants = [Variant(f"REV_{w}", {}, (lambda w=w: quantile_weights(earnings.revision_signal(d, w), d.u_liquid, q=0.1)),
                                holding=21) for w in (21, 63)]
            fam = "H23_revision_momentum"
        else:
            args = fams[key]
            ctx, variants, fam = args["ctx"], args["variants"], args["family"]
        nets = {}
        for v in variants:
            r = run_weights(v.build(), ctx.ret_oo, ctx.cost_bps, holding=v.holding, execution=v.execution, ret_cc=ctx.ret_cc,
                            roundtrip_each_period=ctx.intraday_only)
            nz = r.gross_exposure[r.gross_exposure > 0]
            nets[v.name] = r.net.loc[nz.index[0]:] if len(nz) else r.net
        N = pd.DataFrame(nets).loc["2016-01-01":"2024-12-31"]
        wf, picks = walk_forward(N)
        res = {"n_variants": len(variants), "picks": picks, "n_distinct_picks": len({p["selected"] for p in picks})}
        if len(wf):
            m = core_metrics(wf, spy.reindex(wf.index))
            res["wf_2018_2024"] = {k: m.get(k) for k in ("cagr", "sharpe", "t_mean_nw", "max_drawdown", "beta", "alpha_ann", "alpha_t_nw")}
            oos = wf.loc["2022-01-01":]
            m2 = core_metrics(oos, spy.reindex(oos.index))
            res["wf_2022_2024"] = {k: m2.get(k) for k in ("cagr", "sharpe", "t_mean_nw", "max_drawdown", "beta", "alpha_ann", "alpha_t_nw")}
            res["share_of_years_positive"] = float(np.mean([p["year_return"] > 0 for p in picks]))
        out["families"][fam] = res
        registry.append_run(hypothesis_id=fam.split("_")[0], family=f"{fam}__walk_forward",
                            spec={"protocol": "expanding-window yearly re-selection by trailing net Sharpe", "years": "2018-2024"},
                            split="WF_2018_2024", metrics=res, data=d.manifest)
        w = res.get("wf_2018_2024", {})
        print(f"== {key} {time.time() - t0:.0f}s WF 2018-24 Sharpe {w.get('sharpe')} t {w.get('t_mean_nw')} | picks "
              f"{[p['selected'] for p in picks]}", flush=True)
        (registry.DIR / "results" / "walk_forward_selection.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or ["H04", "H16", "H17", "H07", "H03", "H23", "H01"])
