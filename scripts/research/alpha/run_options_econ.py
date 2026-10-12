"""Realistic economics of EVERY options decile family (first run: an inline command; committed here).

For each pre-registered sort (direction fixed in run_options.py: top decile = buy straddles), every week:
long the top decile's one-month ATM straddles at the ASK, short the bottom decile's at the BID, settle at
intrinsic, P&L per $ of mid premium. Reported for all names, the liquid subset (straddle spread <= 10% of
mid) and the strict-ATM subset, by split, with weekly Newey-West t (>= 5 lags). Also the per-decile long
P&L at the ask in the liquid subset (is there any ordering at all?).
Writes research/alpha/results/options_realistic_economics.json. Descriptive, reads obs_monthly.parquet.
"""
import json

import numpy as np
import pandas as pd

from quantlab.alpha import registry, research_data, splits
from quantlab.alpha.experiments import options_vol as ov

RES = registry.DIR / "results"
r = json.loads((RES / "options_vol.json").read_text())
m = pd.read_parquet(research_data.store_dir() / "options" / "obs_monthly.parquet")
m["split"] = splits.label_series(pd.DatetimeIndex(pd.to_datetime(m["session"])), "options").to_numpy()
for nm in ("no_iv", "with_iv"):
    m[f"fc_{nm}"] = ov.predict_rv(m, r["H33_model"][nm])
m["fc_minus_iv"] = np.log(m["fc_no_iv"] / m["iv_atm"])
m["fc_iv_minus_iv"] = np.log(m["fc_with_iv"] / m["iv_atm"])
m["neg_iv_pctile"], m["neg_iv_chg"], m["neg_skew"] = -m["iv_pctile"], -m["iv_chg_1w"], -m["skew"]
m["neg_iv_pctile_150w"] = -m["iv_pctile_150w"]
m = m.dropna(subset=["payoff", "straddle_bid", "straddle_ask", "straddle_mid"])
m["long_pnl"] = (m["payoff"] - m["straddle_ask"]) / m["straddle_mid"]
m["short_pnl"] = (m["straddle_bid"] - m["payoff"]) / m["straddle_mid"]
m["liquid"] = (m["straddle_ask"] - m["straddle_bid"]) / m["straddle_mid"] <= 0.10
SORTS = {"H24_hv_minus_iv": "hv_minus_iv", "H33_forecast_minus_iv": "fc_minus_iv", "H33_forecast_with_iv_minus_iv": "fc_iv_minus_iv",
         "H30_iv_pctile_low": "neg_iv_pctile", "H30_iv_pctile_low_150w_first_run_deviation": "neg_iv_pctile_150w",
         "H30_iv_change_low": "neg_iv_chg", "H28_term_slope": "term_slope", "H29_skew_low": "neg_skew",
         "H27_option_momentum_3m": "opt_mom_3m", "H27_option_momentum_12m": "opt_mom_12m"}


def g(s: pd.Series):
    s = s.dropna()
    return [round(float(s.mean()), 4), round(float(ov.nw_t(s.to_numpy(), min_obs=10).t), 2)] if len(s) > 10 else None


out = {}
for name, col in SORTS.items():
    res = {}
    for subset in ("all", "liquid", "strict_atm"):
        x = m.dropna(subset=[col])
        x = x[x["liquid"]] if subset == "liquid" else (x[x["atm_ok"].astype(bool)] if subset == "strict_atm" else x)
        x = x.assign(dec=x.groupby("week", observed=True)[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) if len(s) >= 20 else np.nan))
        for sp in ("TRAIN", "VALIDATION", "OOS"):
            xs = x[x["split"] == sp]
            top = xs[xs["dec"] == 9].groupby("week", observed=True)["long_pnl"].mean()
            bot = xs[xs["dec"] == 0].groupby("week", observed=True)["short_pnl"].mean()
            row = {"long_top_at_ask": g(top), "short_bottom_at_bid": g(bot), "long_short": g(((top + bot) / 2)),
                   "n_weeks": int(((top + bot) / 2).dropna().shape[0])}
            if subset == "liquid":
                row["long_at_ask_by_decile"] = {int(k): round(float(v), 4) for k, v in xs.groupby("dec")["long_pnl"].mean().items()}
            res[f"{subset}|{sp}"] = row
    out[name] = res
    lq = res.get("liquid|OOS", {}).get("long_short")
    al = res.get("all|OOS", {}).get("long_short")
    print(f"{name:44s} OOS long-top@ask/short-bottom@bid: all {al} | liquid {lq} | strict-ATM {res.get('strict_atm|OOS', {}).get('long_short')}", flush=True)
(RES / "options_realistic_economics.json").write_text(json.dumps(registry._clean(out), indent=1))
registry.append_run(hypothesis_id="H24", family="options_realistic_economics", spec={"sorts": SORTS, "trade": "long top decile at ask, short bottom at bid, hold to expiry"},
                    split="ALL", metrics=out, data=r.get("data_version") or {})
print("done")
