"""Dip buying, Part I: index dips on Ken French daily US market data (docs/DIP-BUYING-PREREG.md). PAPER research.

I1 RSI(2)<10 above the 200-day average, market-on-close entry, exit above the 5-day average or after 10 days, else
T-bills; I1-lag one day later (robustness); I2 always 100% plus 100% extra (borrowed) during I1 signals; I3 always
100% plus 50% extra for 252 days after the first close 10% below the prior peak (re-armed by a new peak).
Writes research/alpha/results/dips/I_index.json and ledger rows DIP_I1/I2/I3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantlab.alpha import registry  # noqa: E402

from new_areas_ac import A_SPLITS as SPLITS, ff_daily, sharpe_diff_ci, stats  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "dips"
COST, SPREAD = 5e-4, 0.01
PREREG = "docs/DIP-BUYING-PREREG.md Part I"


def rsi(x: pd.Series, n: int = 2) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def i1_state(tri: pd.Series) -> tuple[pd.Series, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """1 after a close where the position is held (decided with data up to that close), else 0."""
    sma200, sma5, r2 = tri.rolling(200, min_periods=200).mean(), tri.rolling(5, min_periods=5).mean(), rsi(tri)
    state = np.zeros(len(tri))
    trades, inpos, held, start = [], False, 0, None
    for i in range(len(tri)):
        if inpos:
            held += 1
            if tri.iloc[i] > sma5.iloc[i] or held >= 10:
                inpos = False
                trades.append((start, tri.index[i]))
        elif np.isfinite(sma200.iloc[i]) and tri.iloc[i] > sma200.iloc[i] and r2.iloc[i] < 10:
            inpos, held, start = True, 0, tri.index[i]
        state[i] = 1.0 if inpos else 0.0
    return pd.Series(state, index=tri.index), trades


def i3_extra(tri: pd.Series, days: int = 252, extra: float = 0.5) -> pd.Series:
    peak = tri.cummax()
    dd = tri / peak - 1
    out = np.zeros(len(tri))
    armed, until = True, -1
    for i in range(len(tri)):
        if dd.iloc[i] == 0:
            armed = True
        if armed and dd.iloc[i] <= -0.10:
            armed, until = False, i + days
        out[i] = extra if i < until else 0.0
    return pd.Series(out, index=tri.index)


def run_weights(w_after_close: pd.Series, mkt: pd.Series, rf: pd.Series, lag: int = 1) -> pd.Series:
    """Weight decided at close t earns day t+lag's return; weight above 1 is borrowed at RF + 1%."""
    w = w_after_close.shift(lag).fillna(0.0)
    turn = w.diff().abs().fillna(0.0)
    lev = (w - 1).clip(lower=0)
    return w * mkt + (1 - w).clip(lower=0) * rf - lev * (rf + SPREAD / 252) - turn * COST


def main() -> None:
    ff = ff_daily()
    mkt, rf = ff["mkt"], ff["rf"]
    tri = (1 + mkt).cumprod()
    s1, trades = i1_state(tri)
    rules = {"BH": pd.Series(1.0, index=mkt.index), "I1": s1, "I2": 1.0 + s1, "I3": 1.0 + i3_extra(tri)}
    daily = {k: run_weights(w, mkt, rf) for k, w in rules.items()}
    daily["I1-lag"] = run_weights(s1, mkt, rf, lag=2)
    out: dict = {"prereg": PREREG, "git": registry.git_commit(), "data": f"Ken French daily {mkt.index[0].date()}..{mkt.index[-1].date()}",
                 "rules": {}, "curves_monthly": {}}
    for k, r in daily.items():
        w = (rules.get(k) if k in rules else s1).shift(1 if k != "I1-lag" else 2).fillna(0.0)
        res = {"splits": {}}
        for sp, (a, b) in SPLITS.items():
            res["splits"][sp] = stats(r.loc[a:b], rf, w.loc[a:b])
            if k != "BH":
                res["splits"][sp]["sharpe_minus_BH"] = sharpe_diff_ci(r.loc[a:b], daily["BH"].loc[a:b], rf)
        out["rules"][k] = res
        eq = (1 + r).cumprod()
        out["curves_monthly"][k] = {str(d.date()): round(float(v), 4) for d, v in eq.resample("ME").last().items()}
    # I1 per-trade statistics (entry at the signal close, exit at the exit close)
    tr = []
    for a, b in trades:
        ret = float(tri.loc[b] / tri.loc[a] - 1) - 2 * COST
        days = int(len(tri.loc[a:b]) - 1)
        tr.append({"entry": a, "exit": b, "ret": ret, "days": max(days, 1)})
    td = pd.DataFrame(tr)
    td["split"] = pd.cut(td["entry"], [pd.Timestamp("1900-01-01")] + [pd.Timestamp(SPLITS[s][1]) for s in SPLITS],
                         labels=list(SPLITS))
    out["I1_trades"] = {sp: {"n": int(len(g)), "mean_ret": float(g["ret"].mean()), "hit_rate": float((g["ret"] > 0).mean()),
                             "avg_days": float(g["days"].mean()), "ret_per_day": float(g["ret"].sum() / g["days"].sum()),
                             "worst": float(g["ret"].min())} for sp, g in td.groupby("split", observed=True)}
    bh = out["rules"]["BH"]["splits"]
    for k in ("I1", "I2", "I3"):
        s = out["rules"][k]["splits"]
        sh_ok = all(s[sp]["sharpe_minus_BH"]["diff"] > 0 for sp in SPLITS) and s["OOS"]["sharpe_minus_BH"]["ci90"][0] > 0
        if k == "I1":
            v = "PASSES (risk-adjusted)" if sh_ok else "FAILS"
        else:
            cagr_ok = all(s[sp]["cagr"] > bh[sp]["cagr"] for sp in SPLITS) and s["OOS"]["sharpe"] >= bh["OOS"]["sharpe"]
            v = ("ABSOLUTE IMPROVEMENT" if cagr_ok else "HIGHER RETURN, HIGHER RISK" if s["OOS"]["cagr"] > bh["OOS"]["cagr"]
                 else "NO IMPROVEMENT")
        out["rules"][k]["verdict"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "I_index.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    for k in ("I1", "I2", "I3"):
        registry.append_run(hypothesis_id=f"DIP_{k}", family="dip_buying", split="OOS", spec={"prereg": PREREG, "rule": k},
                            metrics={"verdict": out["rules"][k]["verdict"],
                                     "oos": {m: out["rules"][k]["splits"]["OOS"].get(m) for m in ("cagr", "sharpe", "max_drawdown")}},
                            conclusion=out["rules"][k]["verdict"], seed=7)
    for k, v in out["rules"].items():
        s = v["splits"]
        print(f"{k:6} " + " | ".join(f"{sp} {s[sp]['cagr']*100:5.1f}% Sh {s[sp]['sharpe']:.2f} DD {s[sp]['max_drawdown']*100:4.0f}% "
                                     f"exp {s[sp]['avg_exposure']:.2f}" for sp in SPLITS) + f" | {v.get('verdict', '')}")
    print("I1 trades:", {sp: {k2: round(x, 4) if isinstance(x, float) else x for k2, x in d.items()} for sp, d in out["I1_trades"].items()})


if __name__ == "__main__":
    main()
