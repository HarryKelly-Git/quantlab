"""What do big winners have in common?  PRE-REGISTERED (written before results):

Universe : research universe, adv20 >= $5M, entry known. Split A < 2023-01-24 <= B. Holdout untouched.
Outcome  : CLEAN WIN (T, h) = +T% touched within h sessions BEFORE the 2xATR stop (same-day = loss).
           Pairs (5,5), (8,10), (10,20) -- one per live hold arm.
Lift     : vol-neutral. residual = clean_win - mean(clean_win | same date, same ATR quintile). For each
           characteristic: per-date quintiles, Q5-Q1 of mean residual, date-clustered t (h=5 sampled every
           5 sessions -> non-overlapping; h=10/20 overlap -> t divided by sqrt(h/5)).
CONFIRMED: |t_A| >= 3 on A, same sign and |t_B| >= 2 on B. Combinations are the six listed below,
           fixed in advance. Nothing is tuned on B.
Also reported (descriptive): base rates, time to target, drawdown before target, by volatility and
regime; expected net per trade of target/stop/time exits by opportunity type and hold, A and B.
"""
import numpy as np
import pandas as pd

SPW = "C:/Users/harry/AppData/Local/Temp/claude/C--Users-harry-Downloads-Businesspilot/a815f6ae-90eb-4d6b-8043-36bbc3440be6/scratchpad/research"
o = pd.read_parquet(f"{SPW}/path_obs.parquet")
o = o[(o["adv20"] >= 5e6) & o["entry"].notna() & o["atr14_pct"].notna()].copy()
o["half"] = np.where(o["date"] < pd.Timestamp("2023-01-24"), "A", "B")
PAIRS = [(5, 5), (8, 10), (10, 20)]
for t, h in PAIRS:
    hit = o[f"days_to_{t}"] <= h
    o[f"win_{t}_{h}"] = (hit & ~(o[f"stop_first_{t}"] & (o["days_to_stop"] <= h))).astype(float)
    o[f"touch_{t}_{h}"] = hit.astype(float)
o["atr_q"] = o.groupby("date")["atr14_pct"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
pd.set_option("display.width", 250, "display.max_columns", 40)

print(f"obs {len(o):,}   dates {o['date'].nunique()}")
print("\n=== 1. BASE RATES (all liquid, A+B) ===")
rows = []
for t, h in [(5, 5), (5, 10), (5, 20), (8, 10), (8, 20), (10, 10), (10, 20), (15, 20)]:
    d = o[f"days_to_{t}"]
    hit = d <= h
    rows.append({"target": f"+{t}%", "within": h, "P(touch)": hit.mean(), "P(clean, before stop)":
                 (hit & ~(o[f"stop_first_{t}"] & (o["days_to_stop"] <= h))).mean(),
                 "P(-T% first)": ((o[f"days_to_dn{t}"] <= h) & ((o[f"days_to_dn{t}"] < d) | d.isna())).mean(),
                 "median days": d[hit].median(), "median dd before": o.loc[hit, f"mae_before_{t}"].median()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== 2. VOLATILITY dominates the raw odds: P(clean +10% within 20) by ATR quintile ===")
v = o.groupby("atr_q").agg(atr_med=("atr14_pct", "median"), touch10=("touch_10_20", "mean"),
                           clean10=("win_10_20", "mean"), stop_hit=("days_to_stop", lambda s: (s <= 20).mean()),
                           net20=("net_20", "mean"))
print(v.round(3).to_string())
print("regime: P(clean +10%/20) when SPY above 200d MA vs below:",
      o.groupby("spy_above_200")["win_10_20"].mean().round(3).to_dict())

FEATS = ["ret_20d", "ret_60d", "ret_120d", "mom_12_1", "ret_252d", "dist_52w_high", "dist_ma20", "dist_ma50",
         "dist_ma200", "ma50_over_ma200", "rel_volume_1d", "rel_volume_5d", "ret_z_1d", "ret_z_3d", "breakout_55",
         "range_contraction_20_60", "adv20", "industry_rank_63", "rs_industry_20", "sector_rs_spy_63_pit",
         "days_since_earnings", "ear_z", "news_count_1d", "news_count_z", "score", "pts_momentum",
         "pts_volume_activity", "pts_breakout_compression", "pts_mean_reversion", "spy_ret_63"]
COMBOS = {
    "near 52w high & volume>=1.5x": lambda d: (d["dist_52w_high"] >= -0.05) & (d["rel_volume_1d"] >= 1.5),
    "55d breakout & volume>=2x": lambda d: (d["breakout_55"] > 0) & (d["rel_volume_1d"] >= 2.0),
    "strong 12-1 & near 52w high": lambda d: (d.groupby("date")["mom_12_1"].rank(pct=True) >= 0.8) & (d["dist_52w_high"] >= -0.05),
    "post-earnings up (ear_z>=1.5, <=3d)": lambda d: (d["ear_z"] >= 1.5) & (d["days_since_earnings"] <= 3),
    "compressed then expanding vol": lambda d: (d["range_contraction_20_60"] <= 0.6) & (d["rel_volume_1d"] >= 1.5),
    "leading industry & stock leads it": lambda d: (d["industry_rank_63"] >= 0.7) & (d["rs_industry_20"] > 0),
}


def residual(df, col):
    return df[col] - df.groupby(["date", "atr_q"])[col].transform("mean")


def lift(df, f, ycol, h):
    d = df[["date", f, ycol, "atr_q"]].dropna().copy()
    d = d[d.groupby("date")[f].transform("count") >= 50]
    d["r"] = residual(d, ycol)
    d["q"] = d.groupby("date")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
    per = d.groupby(["date", "q"])["r"].mean().unstack()
    if per.shape[1] < 5:
        return np.nan, np.nan, np.nan
    sp = (per[5] - per[1]).dropna()
    t = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))) / np.sqrt(max(h / 5, 1))
    mono = pd.Series(per.mean().values).corr(pd.Series(range(1, 6)), method="spearman")
    return sp.mean(), t, mono


print("\n=== 3. VOL-NEUTRAL LIFT in P(clean win), Q5-Q1 in percentage points; t ===")
res = []
for f in FEATS:
    rec = {"feature": f}
    for t, h in PAIRS:
        y = f"win_{t}_{h}"
        for half in ("A", "B"):
            s, tt, m = lift(o[o["half"] == half], f, y, h)
            rec[f"{t}/{h}_{half}"] = s * 100
            rec[f"t{t}/{h}_{half}"] = tt
    rec["CONFIRMED"] = ",".join(f"{t}/{h}" for t, h in PAIRS if abs(rec[f"t{t}/{h}_A"]) >= 3 and abs(rec[f"t{t}/{h}_B"]) >= 2
                                and np.sign(rec[f"{t}/{h}_A"]) == np.sign(rec[f"{t}/{h}_B"]))
    res.append(rec)
R = pd.DataFrame(res).set_index("feature")
print(R.round(2).sort_values("t10/20_A", key=lambda s: -s.abs()).to_string())
R.to_csv(f"{SPW}/upside_lift.csv")

print("\n=== 4. PRE-SPECIFIED COMBINATIONS: vol-neutral lift vs everyone else (pp), t by half ===")
crow = []
for name, fn in COMBOS.items():
    rec = {"combo": name}
    for half in ("A", "B"):
        d = o[o["half"] == half].copy()
        m = fn(d).fillna(False)
        rec[f"share_{half}%"] = m.mean() * 100
        for t, h in PAIRS:
            y = f"win_{t}_{h}"
            d["r"] = residual(d, y)
            per = d.assign(m=m).groupby(["date", "m"])["r"].mean().unstack()
            if True in per and False in per:
                sp = (per[True] - per[False]).dropna()
                rec[f"{t}/{h}_{half}"] = sp.mean() * 100
                rec[f"t{t}/{h}_{half}"] = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))) / np.sqrt(max(h / 5, 1))
    crow.append(rec)
print(pd.DataFrame(crow).set_index("combo").round(2).to_string())

print("\n=== 5. EXPECTED NET PER TRADE by opportunity type x hold: stop + time exit (as traded live) ===")
cost = lambda d, h: d[f"cost_{h}"]                                          # noqa: E731
fams = {"momentum": "fired_momentum", "relative_strength": "fired_relative_strength", "volume": "fired_volume_activity",
        "breakout": "fired_breakout_compression", "mean_reversion": "fired_mean_reversion"}


def stop_time_net(d, h):
    stopped = d["days_to_stop"] <= h
    return np.where(stopped, -d["stop_pct"] - cost(d, h), d[f"net_{h}"])


rows = []
for name, fcol in list(fams.items()) + [("ALL liquid", None)]:
    for half in ("A", "B"):
        d = o[o["half"] == half] if fcol is None else o[(o["half"] == half) & o[fcol].fillna(False)]
        rec = {"type": name, "half": half, "n": len(d)}
        for h in (5, 10, 20):
            rec[f"net_h{h}_bps"] = np.nanmean(stop_time_net(d, h)) * 1e4
            rec[f"clean_{ {5: 5, 10: 8, 20: 10}[h] }%/h{h}"] = d[f"win_{ {5: 5, 10: 8, 20: 10}[h] }_{h}"].mean()
        rows.append(rec)
print(pd.DataFrame(rows).round(3).to_string(index=False))
o.to_parquet(f"{SPW}/upside_obs.parquet", index=False)
