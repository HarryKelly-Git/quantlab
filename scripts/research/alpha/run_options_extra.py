"""Options-lab follow-ups on the observation file written by run_options.py (first run: inline commands,
committed here so the audit re-run is reproducible):

  H33 realistic economics   deciles of log(forecast / IV); long the top decile at the ASK, short the bottom
                            decile at the BID, settle at intrinsic; P&L per $ of mid premium; all names and
                            the liquid subset (straddle spread <= 10% of mid, QuantLab's options.max_spread_pct)
  H33 accuracy by liquidity forecast-vs-IV accuracy and encompassing regression by subset
  H35 index VRP             short straddle / iron fly / long straddle on SPY, DIA, XLK, XLF one-month options
Writes research/alpha/results/H33_realistic_economics.json, H33_forecast_accuracy_by_liquidity.json and
H35_index_vrp.json.
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
bias = r["H33_model"]["no_iv"].get("train_mean_log_rv_minus_log_iv")
for nm in ("no_iv", "with_iv"):
    m[f"fc_{nm}"] = ov.predict_rv(m, r["H33_model"][nm])
    m[f"sig_{nm}"] = np.log(m[f"fc_{nm}"] / m["iv_atm"])
m["liquid"] = (m["straddle_ask"] - m["straddle_bid"]) / m["straddle_mid"] <= 0.10

# --- H33 accuracy by liquidity -------------------------------------------------------------------------
acc = {}
for lab, x in (("all", m), ("liquid_spread_le_10pct", m[m["liquid"]]),
               ("liquid_no_earnings", m[m["liquid"] & ~m["earn_in_window"].astype(bool)]),
               ("liquid_with_earnings", m[m["liquid"] & m["earn_in_window"].astype(bool)])):
    acc[lab] = ov.forecast_vs_iv(x, x["fc_no_iv"], train_bias=bias)
    for sp, v in acc[lab].items():
        print(f"{lab:24s} {sp:10s} n={v['n']:6d} MSE iv={v['mse_log_iv']:.4f} iv_debiased(train)={v.get('mse_log_iv_debiased_train', float('nan')):.4f} "
              f"forecast={v['mse_log_forecast']:.4f} | b_iv={v['encompassing_coef_iv']:.2f} (DK t {v['t_iv']:.1f}) "
              f"b_fc={v['encompassing_coef_fc']:.2f} (DK t {v['t_fc']:.1f}; week-cluster t {v['t_fc_week_cluster']:.1f})", flush=True)
(RES / "H33_forecast_accuracy_by_liquidity.json").write_text(json.dumps(registry._clean(acc), indent=1))

# --- H33 realistic economics -----------------------------------------------------------------------------
e = m.dropna(subset=["payoff", "straddle_bid", "straddle_ask", "straddle_mid"]).copy()
e["long_pnl"] = (e["payoff"] - e["straddle_ask"]) / e["straddle_mid"]
e["short_pnl"] = (e["straddle_bid"] - e["payoff"]) / e["straddle_mid"]
econ = {}
for col in ("sig_no_iv", "sig_with_iv"):
    for lq in (False, True):
        x = e.dropna(subset=[col])
        x = x[x["liquid"]] if lq else x
        x = x.assign(dec=x.groupby("week", observed=True)[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) if len(s) >= 20 else np.nan))
        for sp in ("TRAIN", "VALIDATION", "OOS"):
            xs = x[x["split"] == sp]
            top = xs[xs["dec"] == 9].groupby("week", observed=True)["long_pnl"].mean()
            bot = xs[xs["dec"] == 0].groupby("week", observed=True)["short_pnl"].mean()
            ls = ((top + bot) / 2).dropna()

            def g(s):
                return [round(float(s.mean()), 4), round(float(ov.nw_t(s.to_numpy(), min_obs=10).t), 2)] if len(s) > 10 else None
            dec_mean = xs.groupby("dec")["long_pnl"].mean().round(4).to_dict()
            econ[f"{col}|liquid={lq}|{sp}"] = {"long_top": g(top), "short_bottom": g(bot), "ls": g(ls), "n_weeks": int(len(ls)),
                                                "long_at_ask_by_decile": {int(k): float(v) for k, v in dec_mean.items()}}
            print(f"{col} liquid={lq!s:5s} {sp:10s} long-top {g(top)} | short-bottom {g(bot)} | L/S {g(ls)}", flush=True)
(RES / "H33_realistic_economics.json").write_text(json.dumps(registry._clean(econ), indent=1))

# --- Part 49 two-way test (plan section 10.7) ---------------------------------------------------------------
p49 = {}
z = e.dropna(subset=["fc_no_iv", "sig_no_iv"]).copy()
z["fc_t"] = z.groupby("week", observed=True)["fc_no_iv"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 30 else np.nan)
hi = z[z["fc_t"] == 2].copy()
hi["gap_t"] = hi.groupby("week", observed=True)["sig_no_iv"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 15 else np.nan)
for lq in (False, True):
    x = hi[hi["liquid"]] if lq else hi
    for sp in ("TRAIN", "VALIDATION", "OOS"):
        xs = x[x["split"] == sp]
        lo_iv = xs[xs["gap_t"] == 2]          # forecast well above IV -> long straddle at the ask
        hi_iv = xs[xs["gap_t"] == 0]          # IV above even a high forecast -> capped short vol (iron fly at the bid)

        def wk(df, col):
            s = df.groupby("week", observed=True)[col].mean().dropna()
            return {"mean": float(s.mean()) if len(s) else None, "t_nw": ov.nw_t(s.to_numpy(), min_obs=10).t if len(s) > 10 else None,
                    "n_trades": int(df[col].notna().sum()), "n_weeks": int(len(s))}
        p49[f"liquid={lq}|{sp}"] = {
            "high_move_low_iv_long_straddle_at_ask": wk(lo_iv, "long_pnl"),
            "high_move_low_iv_long_straddle_at_mid": wk(lo_iv.assign(mid_pnl=lo_iv["payoff"] / lo_iv["straddle_mid"] - 1), "mid_pnl"),
            "high_move_high_iv_iron_fly_at_bid": wk(hi_iv, "fly_ret"),
            "high_move_high_iv_short_straddle_at_bid": wk(hi_iv, "short_pnl"),
            "realised_over_iv_low_iv_group": float((lo_iv["rv_to_exp"] / lo_iv["iv_atm"]).median()) if len(lo_iv) else None,
            "realised_over_iv_high_iv_group": float((hi_iv["rv_to_exp"] / hi_iv["iv_atm"]).median()) if len(hi_iv) else None}
        v = p49[f"liquid={lq}|{sp}"]
        print(f"P49 liquid={lq!s:5s} {sp:10s} HIGH move+LOW IV long@ask {v['high_move_low_iv_long_straddle_at_ask']['mean']} "
              f"(t {v['high_move_low_iv_long_straddle_at_ask']['t_nw']}) | HIGH move+HIGH IV fly@bid {v['high_move_high_iv_iron_fly_at_bid']['mean']} "
              f"(t {v['high_move_high_iv_iron_fly_at_bid']['t_nw']})", flush=True)
(RES / "H49_two_way_test.json").write_text(json.dumps(registry._clean(p49), indent=1))
registry.append_run(hypothesis_id="H33", family="H49_two_way_test", spec={"design": "plan section 10.7", "groups": "weekly forecast terciles x log(forecast/IV) terciles",
                    "long": "high move + low IV: straddle at ask", "short": "high move + high IV: iron fly at bid (capped)"},
                    split="ALL", metrics=p49)

# --- H35 index / sector ETF variance premium ---------------------------------------------------------------
e["short_mid"] = (e["straddle_mid"] - e["payoff"]) / e["straddle_mid"]
e["rel_spread"] = (e["straddle_ask"] - e["straddle_bid"]) / e["straddle_mid"]
idx = {}
for sym in ("SPY", "DIA", "XLK", "XLF"):
    x = e[e["act_symbol"] == sym]
    for sp in ("TRAIN", "VALIDATION", "OOS"):
        xs = x[x["split"] == sp]
        if len(xs) < 10:
            continue

        def t(c):
            return ov.nw_t(xs[c].to_numpy(), min_obs=10).t
        idx[f"{sym}|{sp}"] = {"n": int(len(xs)), "rel_spread_median": float(xs["rel_spread"].median()),
                              "share_rv_below_iv": float((xs["rv_to_exp"] < xs["iv_atm"]).mean()),
                              "mean_log_rv_iv": float(np.log(xs["rv_to_exp"] / xs["iv_atm"]).replace(-np.inf, np.nan).mean()),
                              "short_bid": float(xs["short_pnl"].mean()), "short_bid_t": t("short_pnl"),
                              "short_mid": float(xs["short_mid"].mean()), "fly": float(xs["fly_ret"].mean()), "fly_t": t("fly_ret"),
                              "long_ask": float(xs["long_pnl"].mean())}
        v = idx[f"{sym}|{sp}"]
        print(f"{sym} {sp:10s} n={v['n']:3d} spread {v['rel_spread_median']:.3f} RV<IV {v['share_rv_below_iv']:.2f} | short@bid "
              f"{v['short_bid']:+.3f} (t {v['short_bid_t']:+.2f}) @mid {v['short_mid']:+.3f} | fly {v['fly']:+.3f} (t {v['fly_t']:+.2f})", flush=True)
(RES / "H35_index_vrp.json").write_text(json.dumps(registry._clean(idx), indent=1))
print("done")
