"""Improvement program Batch 1, ETF-core tests M1-M6 (docs/IMPROVEMENT-PROGRAM.md Part B). PAPER research.

Ken French data, 1963-07..2024-12 (nothing later is used). Splits: TRAIN 1963-07..1999, VAL 2000-12, OOS 2013-24.
M1 profitability Hi 30, M2 value Hi 30, M3 momentum top 3 deciles, M4 equal blend of M1-M3, M5 top 10 of 49
industries by prior 12-2 return, M6 2x market only while above its 10-month average (daily). Each tilt (M1-M5)
pays a 0.25%/yr fund drag. Writes research/alpha/results/improvement/M_etf_core.json and ledger rows IP_M*.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantlab.alpha import registry  # noqa: E402
from quantlab.alpha.kf import read_zip_block  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402

from new_areas_ac import ff_daily  # noqa: E402

EXT = store_dir() / "external"
OUT = ROOT / "research" / "alpha" / "results" / "improvement"
START, END = pd.Timestamp("1963-07-01"), pd.Timestamp("2024-12-31")
SPLITS = {"TRAIN": ("1963-07-01", "1999-12-31"), "VAL": ("2000-01-01", "2012-12-31"), "OOS": ("2013-01-01", "2024-12-31")}
FUND_DRAG = 0.0025 / 12          # per month
COST = 5e-4                      # per unit of weight traded
LEV_FEE, LEV_SPREAD = 0.0095, 0.01
PUB = {"M1": "2013-01-01", "M2": "1992-01-01", "M3": "1993-01-01", "M4": "2013-01-01", "M5": "1999-01-01", "M6": "2007-01-01"}
NAMES = {"M1": "Profitability tilt (Hi 30 operating profitability)", "M2": "Value tilt (Hi 30 book-to-market)",
         "M3": "Momentum tilt (top 3 deciles of prior 12-2 return)", "M4": "Equal blend of M1-M3",
         "M5": "Industry momentum (top 10 of 49 by prior 12-2 return)", "M6": "2x market only above its 10-month average"}
PREREG = "docs/IMPROVEMENT-PROGRAM.md Part B"


def mstats(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"n_months": int(len(r))}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    ex = r - rf.reindex(r.index).fillna(0)
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {"n_months": int(len(r)), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(r.std() * math.sqrt(12)),
            "sharpe": float(ex.mean() / ex.std() * math.sqrt(12)) if ex.std() > 0 else None,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "worst_year": float(by_year.min())}


def sharpe_diff_ci(a: pd.Series, b: pd.Series, rf: pd.Series, n_boot: int = 2000, seed: int = 7) -> dict:
    """Monthly Sharpe(a) - Sharpe(b), 90% bootstrap CI resampling calendar years (12-month blocks), same years for both."""
    df = pd.DataFrame({"a": a - rf.reindex(a.index).fillna(0), "b": b - rf.reindex(b.index).fillna(0)}).dropna()
    sh = lambda x: x.mean() / x.std() * math.sqrt(12) if x.std() > 0 else np.nan  # noqa: E731
    point = float(sh(df["a"]) - sh(df["b"]))
    blocks = [g for _, g in df.groupby(df.index.year)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        diffs.append(sh(s["a"]) - sh(s["b"]))
    lo, hi = np.nanquantile(diffs, [0.05, 0.95])
    return {"diff": point, "ci90": [float(lo), float(hi)]}


def window_dd(r: pd.Series, a: str, b: str) -> float | None:
    w = r.loc[a:b].dropna()
    if w.empty:
        return None
    eq = (1 + w).cumprod()
    eq = pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)])
    return float((eq / eq.cummax() - 1).min())


def industry_momentum(ind: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Monthly return of the top-10 industries by prior 12-2 return (months t-12..t-2), equal weight, 5 bps per unit
    of weight traded (target-weight changes; drift ignored). Industries without a full window are not ranked; a held
    industry with a missing return that month is dropped and the rest re-normalised (UNKNOWN is never 0)."""
    gross = (1 + ind).rolling(11, min_periods=11).apply(np.prod, raw=True) - 1     # t-10..t, compounded
    signal = gross.shift(2)                                                           # months t-12..t-2 for holding t
    rets, turns, prev = [], [], pd.Series(0.0, index=ind.columns)
    for t in ind.index:
        s = signal.loc[t].dropna()
        if len(s) < 10:
            rets.append(np.nan)
            turns.append(0.0)
            continue
        top = s.nlargest(10).index
        w = pd.Series(0.0, index=ind.columns)
        w[top] = 0.1
        r_t = ind.loc[t, top]
        held = r_t.dropna()
        rets.append(float(held.mean()) if len(held) else np.nan)
        turns.append(float((w - prev).abs().sum()))
        prev = w
    return pd.Series(rets, index=ind.index), pd.Series(turns, index=ind.index)


def verdict(res: dict, bh: dict, key: str) -> str:
    s = res["splits"]
    ok = all((s[sp]["sharpe_minus_BH"]["diff"] or -9) > 0 for sp in SPLITS) and s["OOS"]["sharpe_minus_BH"]["ci90"][0] > 0
    if ok and s["OOS"]["cagr"] >= bh["OOS"]["cagr"]:
        return "ABSOLUTE IMPROVEMENT"
    if ok:
        return "RISK-ADJUSTED IMPROVEMENT"
    if key == "M6" and s["OOS"]["cagr"] > bh["OOS"]["cagr"]:
        return "HIGHER RETURN, HIGHER RISK"
    if s["OOS"]["max_drawdown"] > bh["OOS"]["max_drawdown"]:
        return "RISK REDUCTION ONLY"
    return "NO IMPROVEMENT"


def main() -> None:
    fac = read_zip_block(EXT / "F-F_Research_Data_Factors_CSV.zip")
    fac = fac.loc[START:END]
    rf_m = fac["RF"]
    bh_m = fac["Mkt-RF"] + fac["RF"]
    op = read_zip_block(EXT / "Portfolios_Formed_on_OP_CSV.zip", "Value Weight", "Monthly").loc[START:END]
    bm = read_zip_block(EXT / "Portfolios_Formed_on_BE-ME_CSV.zip", "Value Weight", "Monthly").loc[START:END]
    mom = read_zip_block(EXT / "10_Portfolios_Prior_12_2_CSV.zip", "Value Weight", "Monthly")
    ind_all = read_zip_block(EXT / "49_Industry_Portfolios_CSV.zip", "Value Weight", "Monthly")

    series: dict[str, pd.Series] = {"BH": bh_m}
    series["M1"] = op["Hi 30"] - FUND_DRAG
    series["M2"] = bm["Hi 30"] - FUND_DRAG
    series["M3"] = mom[["PRIOR 8", "PRIOR 9", "Hi PRIOR"]].mean(axis=1).loc[START:END] - FUND_DRAG
    series["M4"] = pd.concat([series["M1"], series["M2"], series["M3"]], axis=1).mean(axis=1, skipna=False)
    im, iturn = industry_momentum(ind_all)                     # uses pre-1963 months only as signal history
    series["M5"] = (im - iturn * COST - FUND_DRAG).loc[START:END]

    # M6 on daily data: trend from month-end total-return index vs its 10-month average, held the next month
    ff = ff_daily()
    mkt, rfd = ff["mkt"], ff["rf"]
    me = pd.DatetimeIndex(mkt.groupby([mkt.index.year, mkt.index.month]).apply(lambda s: s.index[-1]).to_numpy())
    tri_m = (1 + mkt).cumprod().reindex(me)
    on_m = (tri_m > tri_m.rolling(10, min_periods=10).mean()).where(tri_m.rolling(10, min_periods=10).mean().notna())
    on = on_m.astype(float).reindex(mkt.index).shift(1).ffill()     # decision at a month end applies from the next day
    lev_r = 2 * mkt - (rfd + LEV_SPREAD / 252) - LEV_FEE / 252
    switch = on.diff().abs().fillna(0)
    m6_d = (on * lev_r + (1 - on) * rfd - switch * COST).where(on.notna())
    month = lambda x: (1 + x).groupby([x.index.year, x.index.month]).prod(min_count=1) - 1  # noqa: E731
    m6_m, bh_dm = month(m6_d), month(mkt)
    idx = pd.DatetimeIndex([pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0) for y, m in m6_m.index])
    m6_m.index, bh_dm.index = idx, idx
    series["M6"] = m6_m.loc[START:END]
    corr = float(pd.concat([bh_dm, bh_m], axis=1).dropna().corr().iloc[0, 1])

    out: dict = {"prereg": PREREG, "git": registry.git_commit(), "data": "Ken French monthly VW portfolios + factors; daily factors for M6",
                 "period": f"{START.date()}..{END.date()}", "check_monthly_vs_daily_market_corr": corr,
                 "costs": {"fund_drag_per_year": 0.0025, "trade_cost_per_unit": COST, "M6": {"financing": "RF + 1%/yr on the borrowed 1x",
                           "fee_per_year": LEV_FEE, "formula": "on: 2*mkt_d - (rf_d + 0.01/252) - 0.0095/252; off: rf_d; 5 bps per switch"}},
                 "deviations": ["M5 turnover counts target-weight changes only (drift ignored); 0.25%/yr drag also applied to M5.",
                                "M6 is compared with buy-and-hold built from the same daily data (BH_daily); M1-M5 with the monthly file."],
                 "rules": {}, "curves_monthly": {}}
    bh_stats = {sp: mstats(bh_m.loc[a:b], rf_m) for sp, (a, b) in SPLITS.items()}
    bhd_stats = {sp: mstats(bh_dm.loc[a:b], rf_m) for sp, (a, b) in SPLITS.items()}
    out["rules"]["BH"] = {"name": "Hold the US market", "splits": bh_stats,
                          "dd_2008": window_dd(bh_m, "2007-10-31", "2009-03-31"), "dd_2022": window_dd(bh_m, "2022-01-31", "2022-12-31")}
    out["rules"]["BH_daily"] = {"name": "Hold the market (daily data, M6 baseline)", "splits": bhd_stats}
    for k in ("M1", "M2", "M3", "M4", "M5", "M6"):
        r = series[k]
        base = bh_dm if k == "M6" else bh_m
        bstats = bhd_stats if k == "M6" else bh_stats
        res: dict = {"name": NAMES[k], "splits": {}}
        for sp, (a, b) in SPLITS.items():
            rr = r.loc[a:b]
            res["splits"][sp] = mstats(rr, rf_m)
            res["splits"][sp]["sharpe_minus_BH"] = sharpe_diff_ci(rr, base.loc[a:b], rf_m)
        o = r.loc[SPLITS["OOS"][0]:SPLITS["OOS"][1]]
        ob = base.loc[o.index]
        res["oos_share_months_beating_BH"] = float((o > ob).mean())
        res["dd_2008"], res["dd_2022"] = window_dd(r, "2007-10-31", "2009-03-31"), window_dd(r, "2022-01-31", "2022-12-31")
        pub = PUB[k]
        res["post_publication"] = {"from": pub, **mstats(r.loc[pub:], rf_m), "BH": mstats(base.loc[pub:], rf_m)}
        if k == "M6":
            res["time_in_market"] = float(on.loc[START:END].mean())
        res["verdict"] = verdict(res, bstats, k)
        out["rules"][k] = res
        print(f"{k}: OOS cagr {res['splits']['OOS']['cagr']:.3f} vs {bstats['OOS']['cagr']:.3f}; sharpe "
              f"{res['splits']['OOS']['sharpe']:.2f} vs {bstats['OOS']['sharpe']:.2f}; diffs "
              f"{ {sp: round(v['sharpe_minus_BH']['diff'], 2) for sp, v in res['splits'].items()} }; "
              f"maxDD {res['splits']['OOS']['max_drawdown']:.2f} vs {bstats['OOS']['max_drawdown']:.2f}; {res['verdict']}", flush=True)
    for k, r in {"BH": bh_m, **{k: series[k] for k in ("M1", "M2", "M3", "M4", "M5", "M6")}}.items():
        eq = (1 + r.loc[START:END].fillna(0)).cumprod()
        out["curves_monthly"][k] = {str(d.date()): round(float(v), 4) for d, v in eq.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "M_etf_core.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    slug = {"M1": "profitability", "M2": "value", "M3": "momentum", "M4": "blend", "M5": "industry_momentum", "M6": "trend_2x"}
    for k, s in slug.items():
        v = out["rules"][k]
        registry.append_run(hypothesis_id=f"IP_{k}_{s}", family="improvement_etf", split="OOS",
                            spec={"prereg": PREREG, "rule": NAMES[k]},
                            metrics={"verdict": v["verdict"], "oos_sharpe_diff": v["splits"]["OOS"]["sharpe_minus_BH"]["diff"],
                                     "oos_ci90": v["splits"]["OOS"]["sharpe_minus_BH"]["ci90"], "oos_cagr": v["splits"]["OOS"]["cagr"]},
                            conclusion=v["verdict"], seed=7)
    print(f"monthly vs daily market corr {corr:.4f}; wrote {OUT / 'M_etf_core.json'}", flush=True)


if __name__ == "__main__":
    main()
