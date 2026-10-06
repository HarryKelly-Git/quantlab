"""Options volatility lab batch (H24, H33, H25, H27, H28, H29, H30). Reads the cached equity research data
and the options features; writes research/alpha/results/options_vol.json and ledger lines."""
import json
import time

import numpy as np
import pandas as pd

from quantlab.alpha import registry, research_data, splits
from quantlab.alpha.experiments import options_vol as ov
from quantlab.validation.stats import newey_west_tstat

t0 = time.time()
d = research_data.get()
m = ov.build_obs(d)
print(f"obs {len(m)} built in {time.time() - t0:.0f}s", flush=True)
m.to_parquet(research_data.store_dir() / "options" / "obs_monthly.parquet", index=False)
out = {"n_obs": int(len(m)), "data": "DoltHub chains (archive import < 2021-05, commit-verified after); one 1-month ATM straddle per underlying per week"}

def split_of(x):
    return splits.label_series(pd.DatetimeIndex(pd.to_datetime(x["session"])), "options").to_numpy()
m["split"] = split_of(m)
ok = m.dropna(subset=["ret_hold_ask", "iv_atm", "rv_to_exp"])
# --- baseline facts: the variance risk premium and straddle returns ----------------------------------
base = {}
for sp in ("TRAIN", "VALIDATION", "OOS"):
    x = ok[ok["split"] == sp]
    wk = x.groupby("week", observed=True)
    base[sp] = {"n": int(len(x)), "mean_log_rv_over_iv": float(np.log(x["rv_to_exp"] / x["iv_atm"]).mean()),
                "share_rv_below_iv": float((x["rv_to_exp"] < x["iv_atm"]).mean()),
                "long_straddle_ask_mean": float(x["ret_hold_ask"].mean()), "long_straddle_mid_mean": float(x["ret_hold_mid"].mean()),
                "long_straddle_ask_t_weekly": newey_west_tstat(wk["ret_hold_ask"].mean().to_numpy(), min_obs=20).t,
                "iron_fly_mean": float(x["fly_ret"].mean()), "iron_fly_t_weekly": newey_west_tstat(wk["fly_ret"].mean().dropna().to_numpy(), min_obs=20).t,
                "median_spread_frac": float((x["straddle_ask"] / x["straddle_mid"] - 1).median()),
                "earn_in_window_share": float(x["earn_in_window"].mean())}
out["baseline"] = base
print(json.dumps(base, indent=1), flush=True)
# --- H33: forecast vs IV ----------------------------------------------------------------------------
model = ov.fit_rv_model(ok)
model_iv = ov.fit_rv_model(ok, with_iv=True)
m["fc"] = ov.predict_rv(m, model)
m["fc_iv"] = ov.predict_rv(m, model_iv)
out["H33_model"] = {"no_iv": model, "with_iv": model_iv}
out["H33_forecast_vs_iv"] = ov.forecast_vs_iv(m, m["fc"])
out["H33_forecast_with_iv_vs_iv"] = ov.forecast_vs_iv(m, m["fc_iv"])
m["fc_minus_iv"] = np.log(m["fc"] / m["iv_atm"])
m["fc_iv_minus_iv"] = np.log(m["fc_iv"] / m["iv_atm"])
print(json.dumps({k: out[k] for k in ("H33_forecast_vs_iv", "H33_forecast_with_iv_vs_iv")}, indent=1), flush=True)
# --- decile families (direction pre-specified: top decile = 'buy straddles') --------------------------
SORTS = {
    "H24_hv_minus_iv": "hv_minus_iv",          # Goyal-Saretto: high HV-IV -> options cheap -> long straddle
    "H33_forecast_minus_iv": "fc_minus_iv",     # Part 49: forecast RV above IV -> long straddle
    "H33_forecast_with_iv_minus_iv": "fc_iv_minus_iv",
    "H30_iv_pctile_low": None,                  # computed below with sign flipped (low IV percentile -> long)
    "H30_iv_change_low": None,
    "H28_term_slope": "term_slope",             # Vasquez: upward slope -> long straddle (positive)
    "H29_skew_low": None,
    "H27_option_momentum_3m": "opt_mom_3m",
    "H27_option_momentum_12m": "opt_mom_12m",
}
m["neg_iv_pctile"] = -m["iv_pctile"]
m["neg_iv_chg"] = -m["iv_chg_1w"]
m["neg_skew"] = -m["skew"]
SORTS["H30_iv_pctile_low"] = "neg_iv_pctile"
SORTS["H30_iv_change_low"] = "neg_iv_chg"
SORTS["H29_skew_low"] = "neg_skew"
fam = {}
for name, col in SORTS.items():
    res = {}
    for ret_col in ("ret_hold_ask", "fly_ret"):
        x = ov.decile_table(ok.assign(**{col: m.loc[ok.index, col]}), col, ret_col)
        res[ret_col] = ov.summarize(x, ret_col)
    # H29 directional: skew vs the stock's next-21-session market-adjusted return
    if name == "H29_skew_low":
        x = ov.decile_table(m.dropna(subset=["fwd21_mkt_adj"]), col, "fwd21_mkt_adj")
        res["fwd21_mkt_adj"] = ov.summarize(x, "fwd21_mkt_adj")
    fam[name] = res
    registry.append_run(hypothesis_id=name.split("_")[0], family=f"options_{name}", spec={"sort": col, "unit": "1M ATM straddle, weekly, hold to expiry",
                        "long": "top decile", "costs": "pay ask; intrinsic at expiry; fly wings BS-priced + 10%"}, split="ALL", metrics=res, data={"options": out["data"]})
    tr = res["ret_hold_ask"].get("TRAIN", {}); oo = res["ret_hold_ask"].get("OOS", {})
    print(f"{name:34s} TRAIN T-B {tr.get('top_minus_bottom_mean')} t {tr.get('top_minus_bottom_t_nw')} | OOS T-B {oo.get('top_minus_bottom_mean')} t {oo.get('top_minus_bottom_t_nw')}", flush=True)
out["families"] = fam
# --- H25 expected vs realised move -------------------------------------------------------------------
x = ok.dropna(subset=["move_to_exp", "em_pct"]).copy()
x["ratio"] = x["move_to_exp"] / x["em_pct"]
x["iv_bucket"] = x.groupby("week", observed=True)["iv_atm"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
h25 = {}
for (sp, earn, ivb), g in x.groupby(["split", "earn_in_window", "iv_bucket"], observed=True):
    h25[f"{sp}|earn={earn}|iv={ivb}"] = {"n": int(len(g)), "mean_realised_over_expected": float(g["ratio"].mean()),
                                        "median": float(g["ratio"].median()), "share_realised_gt_expected": float((g["ratio"] > 1).mean()),
                                        "long_straddle_ask_mean": float(g["ret_hold_ask"].mean())}
out["H25_expected_vs_realised"] = h25
(registry.DIR / "results").mkdir(parents=True, exist_ok=True)
(registry.DIR / "results" / "options_vol.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
print(f"done in {time.time() - t0:.0f}s", flush=True)
