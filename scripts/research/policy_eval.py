"""(1) Is the liquidity effect gross alpha or just cost tiers?  (2) What would the bot EARN per trade
under each selection rule?  Composite C is fixed from LITERATURE priors, stated before this run:
    C = mean( pct(mom_12_1), pct(-atr14_pct), pct(adv20) )     equal weight, per date
    (12-1 momentum: Jegadeesh-Titman 1993; low volatility: Ang et al. 2006 / Frazzini-Pedersen 2014;
     liquidity: lower round-trip cost). Not fitted. B is NOT a clean OOS test for C (per-feature B
     results were seen) -- the clean tests are forward paper trading and the 2025+ holdout.
"""
import numpy as np
import pandas as pd

SPW = "C:/Users/harry/AppData/Local/Temp/claude/C--Users-harry-Downloads-Businesspilot/a815f6ae-90eb-4d6b-8043-36bbc3440be6/scratchpad/research"
o = pd.read_parquet(f"{SPW}/feat_obs.parquet")
o = o[o["adv20"] >= 5e6].copy()
o["half"] = np.where(o["date"] < pd.Timestamp("2023-01-24"), "A", "B")
fams = [c for c in o.columns if c.startswith("fired_")]
o["discovered"] = o[fams].any(axis=1)
for h in (5, 10, 20):
    o[f"ex_{h}"] = o[f"net_{h}"] - o.groupby("date")[f"net_{h}"].transform("mean")
    o[f"gex_{h}"] = o[f"gross_{h}"] - o.groupby("date")[f"gross_{h}"].transform("mean")

print("=== (1) liquidity quintiles: gross vs cost (5d, bps) ===")
o["adv_q"] = o.groupby("date")["adv20"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
t = o.groupby(["half", "adv_q"]).agg(adv_med=("adv20", "median"), gross_ex=("gex_5", "mean"), net_ex=("ex_5", "mean"),
                                        cost=("cost_5", "mean")).reset_index()
t["adv_med"] = (t["adv_med"] / 1e6).round(1)
for c in ("gross_ex", "net_ex", "cost"):
    t[c] = (t[c] * 1e4).round(1)
print(t.to_string(index=False))

rk = lambda s: s.rank(pct=True)                                                # noqa: E731
o["C"] = (o.groupby("date")["mom_12_1"].transform(rk) + o.groupby("date")["atr14_pct"].transform(lambda s: rk(-s))
          + o.groupby("date")["adv20"].transform(rk)) / 3.0


def top_k(df, key, k=5, pool=None):
    d = df if pool is None else df[pool]
    d = d.dropna(subset=[key])
    return d.sort_values(["date", key], ascending=[True, False]).groupby("date").head(k)


def summ(picks, h):
    per = picks.groupby("date")[[f"net_{h}", f"ex_{h}", f"spy_{h}"]].mean()
    n = len(per)
    ex = per[f"ex_{h}"]
    t = ex.mean() / (ex.std(ddof=1) / np.sqrt(n)) / np.sqrt(h / 5)
    return {"trades": len(picks), "net_bps": per[f"net_{h}"].mean() * 1e4, "vs_univ_bps": ex.mean() * 1e4,
            "vs_spy_bps": (per[f"net_{h}"] - per[f"spy_{h}"]).mean() * 1e4, "t_vs_univ": t,
            "win%": (picks[f"net_{h}"] > 0).mean() * 100, "p10_bps": picks[f"net_{h}"].quantile(0.10) * 1e4,
            "mae_med_bps": picks[f"mae_{h}"].median() * 1e4}


rules = {
    "universe (all liquid)": lambda df: df,
    "P0 current: top5 discovery score, discovered": lambda df: top_k(df, "score", pool=df["discovered"]),
    "P0b top5 momentum pts, discovered": lambda df: top_k(df, "pts_momentum", pool=df["discovered"]),
    "P1 top5 composite C, discovered": lambda df: top_k(df, "C", pool=df["discovered"]),
    "P2 top5 composite C, whole universe": lambda df: top_k(df, "C"),
    "P3 top5 mom_12_1 only, universe": lambda df: top_k(df, "mom_12_1"),
}
print("\n=== (2) per-trade outcome by selection rule (bps; t of excess vs universe, date-clustered) ===")
out = []
for h in (5, 10):
    for half in ("A", "B"):
        d = o[o["half"] == half]
        for name, f in rules.items():
            out.append({"h": h, "half": half, "rule": name, **summ(f(d), h)})
r = pd.DataFrame(out)
for c in ("net_bps", "vs_univ_bps", "vs_spy_bps", "p10_bps", "mae_med_bps", "win%"):
    r[c] = r[c].round(1)
r["t_vs_univ"] = r["t_vs_univ"].round(2)
pd.set_option("display.width", 250)
print(r.to_string(index=False))
r.to_csv(f"{SPW}/policy_eval.csv", index=False)
