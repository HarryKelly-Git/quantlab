"""One table across all cross-sectional families (research/alpha/results/*.json, non-dev files)."""
import json

import pandas as pd

from quantlab.alpha import registry

rows = []
for f in sorted((registry.DIR / "results").glob("*.json")):
    if f.stem.endswith("__dev"):
        continue
    r = json.loads(f.read_text())
    if "family_stats" not in r:
        continue
    fs, sel = r["family_stats"], r["family_stats"]["selected"]
    v = r["variants"][sel]
    o = r.get("oos") or {}
    dsr = fs.get("dsr_selected_dev", {})
    rows.append({
        "family": r["family"], "variants": fs.get("n_variants"), "eff_trials": round(fs.get("effective_trials", {}).get("effective_trials", float("nan")), 1),
        "selected": sel,
        "train_net_bps": round(v.get("TRAIN", {}).get("mean_bps", float("nan")), 2), "train_t": round(v.get("TRAIN", {}).get("t_net", float("nan")), 2),
        "val_net_bps": round(v.get("VALIDATION", {}).get("mean_bps", float("nan")), 2), "val_t": round(v.get("VALIDATION", {}).get("t_net", float("nan")), 2),
        "dev_t_gross": round(v.get("DEV", {}).get("t_gross", float("nan")), 2),
        "dsr": round(dsr.get("dsr", float("nan")) if dsr.get("dsr") is not None else float("nan"), 3),
        "spa_p": round(fs.get("spa", {}).get("p_value", float("nan")), 3), "pbo": round(fs.get("pbo", {}).get("pbo", float("nan")), 2),
        "oos_sharpe": round(o.get("sharpe") or float("nan"), 2), "oos_t": round(o.get("t_mean_nw") or float("nan"), 2),
        "oos_total": round(o.get("total_return") or float("nan"), 3), "oos_beta": round(o.get("beta") or float("nan"), 2),
        "class": r.get("classification"),
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
df.to_csv(registry.DIR / "results" / "summary_cross_sectional.csv", index=False)
