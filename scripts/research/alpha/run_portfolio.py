"""Parts 29-30: correlations between the family-selected equity streams and portfolio construction.
Every method is reported (no selection among methods). Component OOS periods were already evaluated
once each, so the combined OOS is NOT a virgin test - it is reported for completeness."""
import json

import numpy as np
import pandas as pd

from quantlab.alpha import portfolio as pf, registry, research_data, splits
from quantlab.alpha.engine import run_weights
from quantlab.alpha.metrics import core_metrics

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("run_batch", pathlib.Path(__file__).with_name("run_batch.py"))
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)

d = research_data.get()
fams = rb.families(d)
streams = {}
for key in ("H01", "H03", "H04", "H16", "H17"):
    args = fams[key]
    res = json.loads((registry.DIR / "results" / f"{args['family']}.json").read_text())
    sel = res["family_stats"]["selected"]
    v = next(x for x in args["variants"] if x.name == sel)
    r = run_weights(v.build(), args["ctx"].ret_oo, args["ctx"].cost_bps, holding=v.holding, execution=v.execution,
                    ret_cc=args["ctx"].ret_cc, roundtrip_each_period=args["ctx"].intraday_only)
    streams[f"{key}:{sel}"] = r.net
R = pd.DataFrame(streams).loc["2016-01-01":"2024-12-31"].fillna(0.0)
R = R.loc[R.abs().sum(axis=1) > 0]
out = {"streams": list(R.columns), "corr_full": R.corr().round(3).to_dict(),
       "corr_train": splits.slice_split(R, "equity", "TRAIN").corr().round(3).to_dict()}
from quantlab.alpha.mht import effective_trials
out["effective_independent_streams"] = effective_trials(R.to_numpy())
train = splits.slice_split(R, "equity", "TRAIN")
spy = d.bench_oo.reindex(R.index)
res = {}
for m in pf.METHODS:
    w = pf.weights(train, m)
    port = R @ w
    row = {"weights_train": w.round(3).to_dict()}
    for sp in ("TRAIN", "VALIDATION", "OOS"):
        x = splits.slice_split(port, "equity", sp)
        c = core_metrics(x, splits.slice_split(spy, "equity", sp))
        row[sp] = {k: c.get(k) for k in ("sharpe", "t_mean_nw", "max_drawdown", "cagr", "beta", "alpha_t_nw")}
    wf = pf.walk_forward(R, m, "2019-12-31")
    c = core_metrics(splits.slice_split(wf, "equity", "OOS"), splits.slice_split(spy, "equity", "OOS"))
    row["walk_forward_OOS"] = {k: c.get(k) for k in ("sharpe", "t_mean_nw", "max_drawdown", "cagr", "alpha_t_nw")}
    res[m] = row
    print(m, {sp: (round(row[sp]["sharpe"] or 0, 2), round(row[sp]["t_mean_nw"] or 0, 2)) for sp in ("TRAIN", "VALIDATION", "OOS")},
          "WF-OOS", round(row["walk_forward_OOS"]["sharpe"] or 0, 2), flush=True)
out["portfolios"] = res
single = {c: {sp: core_metrics(splits.slice_split(R[c], "equity", sp)).get("sharpe") for sp in ("TRAIN", "VALIDATION", "OOS")} for c in R.columns}
out["single_stream_sharpe"] = single
(registry.DIR / "results" / "H36_portfolio_construction.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
registry.append_run(hypothesis_id="H36", family="H36_portfolio_construction", spec={"methods": list(pf.METHODS), "fit": "TRAIN", "streams": list(R.columns)},
                    split="ALL", metrics=out)
print(json.dumps(registry._clean({"corr_full": out["corr_full"], "effective": out["effective_independent_streams"]}), indent=1))
