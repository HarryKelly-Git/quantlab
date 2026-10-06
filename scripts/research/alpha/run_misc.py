"""Families first run from inline commands, committed here for the audit re-run:
  H02 survivorship bias, H15 pairs (distance method), H12 ML target separation, and
  H06 overnight vs intraday decomposition (descriptive, no trial) of each equity family's SELECTED
  variant (gross, before costs; needs the family results files).
Usage: python scripts/research/alpha/run_misc.py H02 H15 H12 H06"""
import json
import sys
import time

from quantlab.alpha import registry, research_data

d = research_data.get()
RES = registry.DIR / "results"
RES.mkdir(parents=True, exist_ok=True)
for key in sys.argv[1:]:
    t0 = time.time()
    if key == "H02":
        from quantlab.alpha.experiments import survivorship
        r = survivorship.run(d)
        r["delistings"] = {"n_delisted": d.manifest.get("n_delisted_in_panel"), "n_distressed": d.manifest.get("n_delisted_distressed"),
                           "n_renamed_not_delisted": d.manifest.get("n_renamed_not_delisted")}
        (RES / "H02_survivorship_bias.json").write_text(json.dumps(registry._clean(r), indent=1, default=str))
        registry.append_run(hypothesis_id="H02", family="H02_survivorship_bias",
                            spec={"books": ["universe_ew", "momentum_12_1", "reversal_losers_5d", "high_vol"], "universes": ["liquid", "survivors_only"]},
                            split="ALL_2016_2024", metrics=r, data=d.manifest)
    elif key == "H15":
        from quantlab.alpha.experiments import pairs
        r = {}
        for ws in (True, False):
            x = pairs.run(d, within_sector=ws)
            x.pop("daily")
            r["within_sector" if ws else "any_sector"] = x
        (RES / "H15_pairs_distance.json").write_text(json.dumps(registry._clean(r), indent=1, default=str))
        registry.append_run(hypothesis_id="H15", family="H15_pairs_distance",
                            spec={"method": "GGR distance", "pairs": 20, "formation": 252, "trading": 126, "z": 2.0,
                                  "variants": ["within_sector", "any_sector"]}, split="ALL", metrics=r, data=d.manifest)
    elif key == "H12":
        from quantlab.alpha.experiments import ml_separation as ml
        r = ml.run(d)
        (RES / "H12_ml_target_separation.json").write_text(json.dumps(registry._clean(r), indent=1, default=str))
        registry.append_run(hypothesis_id="H12", family="H12_ml_target_separation",
                            spec={"features": ml.FEATURES, "targets": ["DIRECTION", "MAGNITUDE", "VOLATILITY", "TAIL", "TIMING"],
                                  "models": ["linear", "random_forest", "gradient_boosting"]}, split="ALL", metrics=r, data=d.manifest)
    elif key == "H06":
        import importlib.util
        import pathlib
        from quantlab.alpha import splits
        from quantlab.alpha.experiments import anomalies
        from quantlab.validation.stats import newey_west_tstat
        spec = importlib.util.spec_from_file_location("run_batch", pathlib.Path(__file__).with_name("run_batch.py"))
        rb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rb)
        fams = rb.families(d)
        r = {"note": "gross P&L of the selected variant held close-to-close, split into the overnight (close->open) "
                     "and intraday (open->close) legs; descriptive, no selection, no trial"}
        for k2 in ("H01", "H03", "H04", "H16", "H17"):
            args = fams[k2]
            sel = json.loads((RES / f"{args['family']}.json").read_text())["family_stats"]["selected"]
            v = next(x for x in args["variants"] if x.name == sel)
            dec = anomalies.decompose(d, v.build(), v.holding)
            act = dec.index[(dec.abs().sum(axis=1) > 0).to_numpy()]
            dec = dec.loc[act[0]:] if len(act) else dec
            row = {}
            for sp in ("TRAIN", "VALIDATION", "OOS"):
                x = splits.slice_split(dec, "equity", sp)
                row[sp] = {c: {"mean_bps": float(x[c].mean() * 1e4), "t_nw": newey_west_tstat(x[c].to_numpy()).t} for c in ("overnight", "intraday")}
            r[f"{k2}:{sel}"] = row
        (RES / "H06_overnight_intraday_decomposition.json").write_text(json.dumps(registry._clean(r), indent=1, default=str))
        registry.append_run(hypothesis_id="H06", family="H06_decomposition", spec={"families": ["H01", "H03", "H04", "H16", "H17"],
                            "legs": ["overnight", "intraday"]}, split="ALL", metrics=r, data=d.manifest)
    else:
        raise SystemExit(f"unknown {key}")
    print(f"== {key} {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(registry._clean(r), default=str)[:2500], flush=True)
