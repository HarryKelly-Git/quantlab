"""Run alpha-discovery families. Usage:
    python scripts/research/alpha/run_batch.py build            # build + cache research data, quality report
    python scripts/research/alpha/run_batch.py dev H01 H04 ...   # development-only (no OOS) sanity run
    python scripts/research/alpha/run_batch.py oos H01 ...       # full run incl. the single OOS evaluation
    ... --override="reason"   one-time documented OOS re-evaluation after a bug fix (registry.append_run)
Results: research/alpha/results/<family>.json, ledger lines in research/alpha/ledger.jsonl.
"""
import json
import sys
import time

from quantlab.alpha import research_data
from quantlab.alpha.experiments import anomalies, momentum, reversal

FAMILIES = {
    "H01": (reversal, "run_family_args"),
}


def families(d):
    from quantlab.alpha.experiment import run_family
    ctx = reversal.context(d)
    return {
        "H01": dict(family="H01_short_term_reversal", hypothesis_id="H01", variants=reversal.variants(d), ctx=ctx, prior_trials=2,
                    notes="prior_trials=2: QuantLab mean_reversion + extreme_reversal"),
        "H04": dict(family="H04_residual_momentum", hypothesis_id="H04", variants=momentum.residual_variants(d), ctx=ctx, prior_trials=1),
        "H03": dict(family="H03_momentum_matrix", hypothesis_id="H03", variants=momentum.matrix_variants(d), ctx=ctx, prior_trials=3),
        "H16": dict(family="H16_max_lottery", hypothesis_id="H16", variants=anomalies.max_variants(d), ctx=ctx),
        "H17": dict(family="H17_idiosyncratic_vol", hypothesis_id="H17", variants=anomalies.ivol_variants(d), ctx=ctx, prior_trials=1),
        "H07": dict(family="H07_gap_reversal_intraday", hypothesis_id="H07", variants=anomalies.gap_variants(d), ctx=anomalies.gap_context(d)),
    }


def main():
    override = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--override=")), None)
    sys.argv = [a for a in sys.argv if not a.startswith("--override=")]
    mode = sys.argv[1]
    if mode == "build":
        d = research_data.get(refresh=True)
        print(json.dumps(d.manifest, indent=1, default=str))
        return
    d = research_data.get()
    from quantlab.alpha.experiment import run_family
    fams = families(d)
    from quantlab.alpha.experiment import run_oos_only
    for key in sys.argv[2:]:
        t0 = time.time()
        args = dict(fams[key])
        if mode == "dev":
            args["family"] = args["family"] + "__dev"
        if mode == "oosonly":
            res = run_oos_only(family=args["family"], hypothesis_id=args["hypothesis_id"], variants=args["variants"],
                               ctx=args["ctx"], dev_family=args["family"] + "__dev", oos_override=override)
        else:
            res = run_family(run_oos=(mode == "oos"), oos_override=override, **args)
        sel = res["family_stats"]["selected"]
        print(f"== {key} [{mode}] {time.time() - t0:.0f}s selected={sel} class={res['classification']} ({res['classification_reason']})")
        print("   selected:", json.dumps(res["variants"][sel], default=str)[:600])
        fs = {k: v for k, v in res["family_stats"].items() if k in ("spa", "white_rc", "pbo", "effective_trials", "bh_rejections")}
        print("   family:", json.dumps(fs, default=str)[:800])
        if res.get("oos"):
            o = res["oos"]
            print("   OOS:", {k: o.get(k) for k in ("total_return", "sharpe", "t_mean_nw", "max_drawdown", "beta", "alpha_ann", "years_positive_frac", "t_net_2x_costs")})


if __name__ == "__main__":
    main()
