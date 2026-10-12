"""New areas A (market overlays, Ken French 1963-2024) and C (SPY overnight vs intraday, 2016-2024).
Pre-registered in docs/NEW-AREAS-PREREG.md. PAPER research; nothing from 2025 on.
Writes research/alpha/results/new_areas/A_market_overlays.json and C_spy_overnight.json."""
from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import registry  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402
from quantlab.validation.stats import newey_west_tstat  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "new_areas"
END = "2024-12-31"
A_SPLITS = {"TRAIN": ("1963-07-01", "1999-12-31"), "VAL": ("2000-01-01", "2012-12-31"), "OOS": ("2013-01-01", END)}
C_SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", END)}
COST = 0.0005


def ff_daily() -> pd.DataFrame:
    z = zipfile.ZipFile(store_dir() / "external" / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(txt) if l.strip().startswith(",Mkt-RF") or l.strip().lower().startswith(",mkt-rf"))
    rows = []
    for l in txt[start + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) < 7 or not p[0].isdigit():
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = df[c].astype(float) / 100.0
    df = df.set_index("date")
    df = df[df.index <= END]
    df["mkt"] = df["mkt_rf"] + df["rf"]
    return df


def stats(r: pd.Series, rf: pd.Series, w: pd.Series | None = None) -> dict:
    """Daily returns -> CAGR, vol, Sharpe (excess of rf), max drawdown, worst calendar year, avg exposure."""
    r = r.dropna()
    if len(r) < 50:
        return {"n_days": int(len(r))}
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {"n_days": int(len(r)), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(r.std() * math.sqrt(252)),
            "sharpe": float(ex.mean() / ex.std() * math.sqrt(252)) if ex.std() > 0 else None,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "worst_year": float(by_year.min()),
            "avg_exposure": float(w.reindex(r.index).mean()) if w is not None else 1.0}


def sharpe_diff_ci(a: pd.Series, b: pd.Series, rf: pd.Series, block_months: int = 12, n_boot: int = 2000, seed: int = 7) -> dict:
    ea, eb = a - rf.reindex(a.index).fillna(0), b - rf.reindex(b.index).fillna(0)
    df = pd.DataFrame({"a": ea, "b": eb}).dropna()
    sh = lambda x: x.mean() / x.std() * math.sqrt(252) if x.std() > 0 else np.nan  # noqa: E731
    point = float(sh(df["a"]) - sh(df["b"]))
    blocks = [g for _, g in df.groupby([df.index.year, (df.index.month - 1) // block_months])]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        diffs.append(sh(s["a"]) - sh(s["b"]))
    lo, hi = np.nanquantile(diffs, [0.05, 0.95])
    return {"diff": point, "ci90": [float(lo), float(hi)]}


def alpha_t(a: pd.Series, b: pd.Series, rf: pd.Series) -> dict:
    """Monthly excess-return regression of strategy on BH: intercept (annualised) and Newey-West t (6 lags)."""
    m = lambda x: (1 + x).groupby([x.index.year, x.index.month]).prod() - 1  # noqa: E731
    ea, eb, mr = m(a), m(b), m(rf.reindex(a.index).fillna(0))
    y, x = (ea - mr).to_numpy(), (eb - mr).to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    t = newey_west_tstat(res + beta[0], lags=6)
    return {"alpha_annual": float(beta[0] * 12), "beta": float(beta[1]), "t": float(t.t) if np.isfinite(t.t) else None}


def area_a() -> dict:
    ff = ff_daily()
    mkt, rf = ff["mkt"], ff["rf"]
    me = mkt.groupby([mkt.index.year, mkt.index.month]).apply(lambda s: s.index[-1])
    month_ends = pd.DatetimeIndex(me.to_numpy())
    tri = (1 + mkt).cumprod()
    tr_m = tri.reindex(month_ends)
    # last month's realised vol (annualised) known at each month end
    rv = mkt.groupby([mkt.index.year, mkt.index.month]).std() * math.sqrt(252)
    rv.index = month_ends
    tr_end = pd.Timestamp(A_SPLITS["TRAIN"][1])
    target = float(mkt[mkt.index <= tr_end].std() * math.sqrt(252))         # TRAIN-period market vol
    sma10 = tr_m.rolling(10, min_periods=10).mean()
    trend_up = (tr_m > sma10)
    weights = {"BH": pd.Series(1.0, index=month_ends),
               "VM1": (target / rv).clip(upper=1.0),
               "VM15": (target / rv).clip(upper=1.5),
               "TR10": trend_up.astype(float).where(sma10.notna(), 1.0)}
    weights["VM1+TR10"] = weights["VM1"] * weights["TR10"]
    out: dict = {"prereg": "docs/NEW-AREAS-PREREG.md A", "git": registry.git_commit(), "target_vol_TRAIN": target,
                 "data": f"Ken French daily, {mkt.index[0].date()}..{mkt.index[-1].date()}", "rules": {}}
    daily_ret, daily_w = {}, {}
    for k, wm in weights.items():
        # weight decided at month end applies to the NEXT month's days
        # the decision at a month end applies from the NEXT trading day (fixed 2026-10-08: the first run used
        # wm.shift(1).reindex(..., method="ffill"), which applied each decision one month late)
        w = wm.reindex(mkt.index).shift(1).ffill().fillna(1.0 if k == "BH" else np.nan)
        lev = (w - 1).clip(lower=0)
        turn = w.diff().abs().fillna(0)
        r = w * mkt + (1 - w).clip(lower=0) * rf - lev * (rf + 0.01 / 252) - turn * COST
        daily_ret[k], daily_w[k] = r, w
    for k in weights:
        r, w = daily_ret[k], daily_w[k]
        res = {"splits": {}, "post_publication": None}
        for sp, (a, b) in A_SPLITS.items():
            rr = r.loc[a:b]
            res["splits"][sp] = stats(rr, rf, w)
            if k != "BH":
                res["splits"][sp]["sharpe_minus_BH"] = sharpe_diff_ci(rr, daily_ret["BH"].loc[a:b], rf)
        if k != "BH":
            res["alpha_vs_BH_full"] = alpha_t(r.dropna(), daily_ret["BH"].reindex(r.dropna().index), rf)
            pub = "2007-01-01" if k == "TR10" else "2017-01-01"
            res["post_publication"] = {"from": pub, **stats(r.loc[pub:], rf, w),
                                       "BH": stats(daily_ret["BH"].loc[pub:], rf)}
            s = res["splits"]
            ok = all((s[sp]["sharpe_minus_BH"]["diff"] or -9) > 0 for sp in A_SPLITS) and s["OOS"]["sharpe_minus_BH"]["ci90"][0] > 0
            absolute = ok and s["OOS"]["cagr"] >= out["rules"].get("BH", {}).get("splits", {}).get("OOS", {}).get("cagr", 9)
            less_dd = s["OOS"]["max_drawdown"] > out["rules"]["BH"]["splits"]["OOS"]["max_drawdown"]
            res["verdict"] = ("ABSOLUTE IMPROVEMENT" if absolute else "RISK-ADJUSTED IMPROVEMENT" if ok
                              else "RISK REDUCTION ONLY" if less_dd else "NO IMPROVEMENT")
        out["rules"][k] = res
    # equity curves (monthly, for the dashboard)
    curves = {}
    for k, r in daily_ret.items():
        eq = (1 + r.fillna(0)).cumprod()
        curves[k] = {str(d.date()): round(float(v), 4) for d, v in eq.reindex(month_ends).dropna().items()}
    out["equity_curves_monthly"] = curves
    return out


def area_c() -> dict:
    b = pd.read_parquet(store_dir() / "bars_daily.parquet", columns=["symbol", "date", "adj_open", "adj_close"],
                        filters=[("symbol", "==", "SPY")]).sort_values("date")
    b["date"] = pd.to_datetime(b["date"])
    b = b.set_index("date").loc[:END]
    on = b["adj_open"] / b["adj_close"].shift(1) - 1
    idr = b["adj_close"] / b["adj_open"] - 1
    bh = b["adj_close"] / b["adj_close"].shift(1) - 1
    ff = ff_daily()
    rf = ff["rf"].reindex(bh.index).fillna(0)
    rules = {"BH": bh, "ON": on - 2e-4, "ID": idr - 2e-4}
    out: dict = {"prereg": "docs/NEW-AREAS-PREREG.md C", "git": registry.git_commit(), "rules": {}}
    for k, r in rules.items():
        res = {"splits": {}}
        for sp, (a, z) in C_SPLITS.items():
            res["splits"][sp] = stats(r.loc[a:z], rf)
            if k != "BH":
                res["splits"][sp]["sharpe_minus_BH"] = sharpe_diff_ci(r.loc[a:z], bh.loc[a:z], rf, block_months=1)
        if k != "BH":
            s = res["splits"]
            ok = all(s[sp]["sharpe_minus_BH"]["diff"] > 0 for sp in C_SPLITS) and s["OOS"]["sharpe_minus_BH"]["ci90"][0] > 0
            res["verdict"] = "BEATS BUY-AND-HOLD" if ok else "DOES NOT BEAT BUY-AND-HOLD"
        res["gross_total_return_2016_2024"] = float((1 + (r + (2e-4 if k != "BH" else 0)).dropna()).prod() - 1)
        out["rules"][k] = res
    return out


def fix_a() -> None:
    """Corrected Area A re-run (2026-10-08). The first run applied each month-end decision one month late. Its
    results file is kept as A_market_overlays__lagged_bug.json and its ledger row stays."""
    old = OUT / "A_market_overlays.json"
    keep = OUT / "A_market_overlays__lagged_bug.json"
    if old.exists() and not keep.exists():
        keep.write_text(old.read_text())
    a = area_a()
    a["correction"] = ("2026-10-08: timing fixed. The first run applied each month-end decision from the month end "
                       "before the previous one on 95% of days. That run is kept in A_market_overlays__lagged_bug.json.")
    old.write_text(json.dumps(registry._clean(a), indent=1, default=str))
    for k, v in a["rules"].items():
        s = v["splits"]
        print(f"A {k}: " + " | ".join(f"{sp} CAGR {s[sp]['cagr']*100:.1f}% Sh {s[sp]['sharpe']:.2f} DD {s[sp]['max_drawdown']*100:.0f}% exp {s[sp]['avg_exposure']:.2f}" for sp in s)
              + f" | {v.get('verdict', '')}", flush=True)
    registry.append_run(hypothesis_id="NA_A_market_overlays", family="new_areas", split="OOS",
                        spec={"prereg": "NEW-AREAS-PREREG", "timing": "month-end decision applies from the next trading day"},
                        metrics={k: v.get("verdict") for k, v in a["rules"].items()}, conclusion="corrected re-run; see results/new_areas",
                        seed=7, oos_override="A-TIMING-FIX: the first run applied each decision one month late (code bug, "
                                             "not a rule change); same pre-registered rules, corrected timing; first run kept")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "fix-a":
        fix_a()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    a = area_a()
    (OUT / "A_market_overlays.json").write_text(json.dumps(registry._clean(a), indent=1, default=str))
    for k, v in a["rules"].items():
        s = v["splits"]
        print(f"A {k}: " + " | ".join(f"{sp} CAGR {s[sp]['cagr']*100:.1f}% Sh {s[sp]['sharpe']:.2f} DD {s[sp]['max_drawdown']*100:.0f}% exp {s[sp]['avg_exposure']:.2f}" for sp in s)
              + f" | {v.get('verdict', '')}", flush=True)
    c = area_c()
    (OUT / "C_spy_overnight.json").write_text(json.dumps(registry._clean(c), indent=1, default=str))
    for k, v in c["rules"].items():
        s = v["splits"]
        print(f"C {k}: " + " | ".join(f"{sp} CAGR {s[sp]['cagr']*100:.1f}% Sh {s[sp]['sharpe']:.2f}" for sp in s) + f" | {v.get('verdict', '')}", flush=True)
    for hid, res in (("NA_A_market_overlays", a), ("NA_C_spy_overnight", c)):
        registry.append_run(hypothesis_id=hid, family="new_areas", split="OOS", spec={"prereg": "NEW-AREAS-PREREG"},
                            metrics={k: v.get("verdict") for k, v in res["rules"].items()}, conclusion="see results/new_areas", seed=7)


if __name__ == "__main__":
    main()
