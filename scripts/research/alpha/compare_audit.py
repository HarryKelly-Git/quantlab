"""First run (results/pre_audit/) vs audit re-run (results/): one table per area, written to
research/alpha/results/audit_comparison.json and printed. Descriptive only."""
import json

import pandas as pd

from quantlab.alpha import registry

RES = registry.DIR / "results"
PRE = RES / "pre_audit"
V2 = RES / "audit_v2_panel"


def _load(d, name):
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fam_row(r):
    if not r or "family_stats" not in r:
        return {}
    sel = r["family_stats"]["selected"]
    v = r["variants"][sel]
    o = r.get("oos") or {}
    return {"selected": sel, "train_bps": v.get("TRAIN", {}).get("mean_bps"), "train_t": v.get("TRAIN", {}).get("t_net"),
            "val_t": v.get("VALIDATION", {}).get("t_net"), "dev_t_gross": v.get("DEV", {}).get("t_gross"),
            "spa_p": (r["family_stats"].get("spa") or {}).get("p_value"), "pbo": (r["family_stats"].get("pbo") or {}).get("pbo"),
            "oos_sharpe": o.get("sharpe"), "oos_t": o.get("t_mean_nw"), "class": r.get("classification")}


out = {}
rows = []
for fam in ("H01_short_term_reversal", "H03_momentum_matrix", "H04_residual_momentum", "H16_max_lottery",
            "H17_idiosyncratic_vol", "H07_gap_reversal_intraday", "H23_revision_momentum"):
    a, m2, b = _fam_row(_load(PRE, fam)), _fam_row(_load(V2, fam)), _fam_row(_load(RES, fam))
    rows.append({"family": fam, **{f"before_{k}": v for k, v in a.items()}, **{f"v2_{k}": v for k, v in m2.items()},
                 **{f"after_{k}": v for k, v in b.items()},
                 "oos_looks_in_ledger": registry.oos_looks(fam.split("_")[0])})
out["equity_families"] = rows
for fam in ("H20_earnings_announcement_premium", "H21_eps_surprise_drift"):
    a, b = _load(PRE, fam), _load(RES, fam)
    out[fam] = {"before": {"selected": a and a.get("selected"), "oos": a and a.get("oos"), "class": a and a.get("classification")},
                "after": {"selected": b and b.get("selected"), "oos": b and b.get("oos"), "class": b and b.get("classification"),
                          "train": b and {k: v["summary"].get("TRAIN") for k, v in b["variants"].items()}}}
for fam in ("H26_pre_earnings_straddle", "H02_survivorship_bias", "H15_pairs_distance", "H35_index_vrp",
            "H33_forecast_accuracy_by_liquidity", "H33_realistic_economics"):
    out[fam] = {"before": _load(PRE, fam), "after": _load(RES, fam)}
ov_a, ov_b = _load(PRE, "options_vol"), _load(RES, "options_vol")
if ov_a and ov_b:
    out["options_baseline"] = {"before": ov_a.get("baseline"), "after": ov_b.get("baseline")}
    fams = {}
    for k in ov_b.get("families", {}):
        for ret in ("ret_hold_ask", "fly_ret"):
            for sp in ("TRAIN", "VALIDATION", "OOS"):
                aa = ((ov_a.get("families", {}).get(k) or {}).get(ret) or {}).get(sp) or {}
                bb = ((ov_b.get("families", {}).get(k) or {}).get(ret) or {}).get(sp) or {}
                fams[f"{k}|{ret}|{sp}"] = {"before_tb": aa.get("top_minus_bottom_mean"), "before_t": aa.get("top_minus_bottom_t_nw"),
                                           "after_tb": bb.get("top_minus_bottom_mean"), "after_t": bb.get("top_minus_bottom_t_nw")}
    out["options_deciles"] = fams
    out["H33_forecast_vs_iv"] = {"before": ov_a.get("H33_forecast_vs_iv"), "after": ov_b.get("H33_forecast_vs_iv")}
(RES / "audit_comparison.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(pd.DataFrame(out["equity_families"]).round(3).to_string(index=False))
if "options_deciles" in out:
    print(pd.DataFrame(out["options_deciles"]).T.round(3).to_string())
