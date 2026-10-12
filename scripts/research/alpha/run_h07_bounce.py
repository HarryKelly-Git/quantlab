"""H07 bounce test (audit M5; design pre-registered in docs/ALPHA-DISCOVERY-PLAN.md section 10.6).

  python scripts/research/alpha/run_h07_bounce.py sample     # draw the 40-name sample -> var/alpha/h07_bounce_sample.json
  python scripts/research/alpha/run_h07_bounce.py download   # 1-minute SIP bars 2020-2021 for the sample (Alpaca, paper keys)
  python scripts/research/alpha/run_h07_bounce.py run        # compare entries: opening print / 09:35 / 09:30-09:34 VWAP

The selected H07 book (GAP_large_equal) is built on the FULL large-cap cross-section; only the sampled
stocks' positions are re-priced. Writes research/alpha/results/H07_bounce_test.json.
"""
import json
import sys

import numpy as np
import pandas as pd

from quantlab.alpha import registry, research_data
from quantlab.alpha.store import download_minutes, store_dir

SAMPLE = store_dir() / "h07_bounce_sample.json"
SEED, N, START, END = 20261006, 40, "2020-01-01", "2021-12-31"


def sample() -> list[str]:
    d = research_data.get()
    u = d.u_large.loc[START:END]
    share = u.mean()
    elig = sorted(s for s in share[share >= 0.5].index if "@" not in s)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(elig, size=N, replace=False).tolist())
    SAMPLE.write_text(json.dumps({"seed": SEED, "n_eligible": len(elig), "sample": pick}, indent=1))
    print(len(elig), "eligible;", pick)
    return pick


def download() -> None:
    pick = json.loads(SAMPLE.read_text())["sample"]
    print(download_minutes(pick, start=START, end=END, per_minute=100))


def _entry_prices(sym: str) -> pd.DataFrame:
    """Per session: first regular-hours trade price (09:30 bar open), the last trade before 09:35 (09:34 bar
    close) and the 09:30-09:34 VWAP, from raw 1-minute bars."""
    parts = [pd.read_parquet(store_dir() / "minute" / f"{sym}_{y}.parquet") for y in (2020, 2021)
             if (store_dir() / "minute" / f"{sym}_{y}.parquet").exists()]
    if not parts:
        return pd.DataFrame()
    b = pd.concat(parts, ignore_index=True)
    b["t"] = pd.to_datetime(b["t"])
    hm = b["t"].dt.hour * 100 + b["t"].dt.minute
    b = b[(hm >= 930) & (hm <= 934)].copy()
    b["session"] = b["t"].dt.tz_localize(None).dt.normalize()
    b["pv"] = b["close"] * b["volume"]
    g = b.sort_values("t").groupby("session")
    out = pd.DataFrame({"first_0930": g["open"].first(), "last_0934": g["close"].last(),
                        "vwap_0930_0934": g["pv"].sum() / g["volume"].sum().replace(0, np.nan),
                        "n_bars": g.size()})
    return out


def run() -> dict:
    from quantlab.alpha.experiments import anomalies
    from quantlab.validation.stats import newey_west_tstat
    pick = json.loads(SAMPLE.read_text())["sample"]
    d = research_data.get()
    W = anomalies.gap_weights(d, d.u_large, "equal")           # decision row t-1 -> traded on day t
    W = W.shift(1)                                             # re-index by the trading day t
    close, opn = d.p["close"], d.p["open"]
    rows = []
    for s in pick:
        ep = _entry_prices(s)
        if ep.empty or s not in W.columns:
            continue
        w = W[s].loc[START:END]
        w = w[w != 0].dropna()
        for day, wt in w.items():
            if day not in ep.index or not np.isfinite(close.at[day, s]) or not np.isfinite(opn.at[day, s]):
                continue
            c, o = close.at[day, s], opn.at[day, s]
            e = ep.loc[day]
            rows.append({"day": day, "symbol": s, "side": float(np.sign(wt)),
                         "r_open_print": c / o - 1, "r_0935": c / e["last_0934"] - 1, "r_vwap5": c / e["vwap_0930_0934"] - 1,
                         "first_vs_daily_open": e["first_0930"] / o - 1})
    x = pd.DataFrame(rows)
    out = {"design": "docs/ALPHA-DISCOVERY-PLAN.md section 10.6", "sample": pick, "n_positions": int(len(x)),
           "n_days": int(x["day"].nunique()) if len(x) else 0}
    if x.empty:
        out["status"] = "NO DATA"
        return out
    for col in ("r_open_print", "r_0935", "r_vwap5"):
        daily = (x["side"] * x[col]).groupby(x["day"]).mean()
        t = newey_west_tstat(daily.to_numpy())
        out[col] = {"mean_bps": float(daily.mean() * 1e4), "t_nw": t.t, "n_days": int(len(daily))}
    diff = (x["side"] * (x["r_open_print"] - x["r_0935"])).groupby(x["day"]).mean()
    out["open_minus_0935"] = {"mean_bps": float(diff.mean() * 1e4), "t_nw": newey_west_tstat(diff.to_numpy()).t}
    out["median_abs_first_trade_vs_daily_open_bps"] = float(x["first_vs_daily_open"].abs().median() * 1e4)
    a, b = out["r_open_print"]["mean_bps"], out["r_0935"]["mean_bps"]
    if a <= 0:
        verdict = "INCONCLUSIVE: the opening-print gross return is not positive on this sample; H07 stays D with the caveat"
    elif b <= 0.5 * a:
        verdict = "E: entering at 09:35 keeps <= 50% of the opening-print gross return - mostly bid-ask bounce"
    else:
        verdict = "D stands: most of the gross return survives a 09:35 entry"
    out["verdict"] = verdict
    out["bounce_share"] = float(1 - b / a) if a > 0 else None
    (registry.DIR / "results" / "H07_bounce_test.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    registry.append_run(hypothesis_id="H07", family="H07_bounce_test", spec={"design": "plan 10.6", "seed": SEED, "n": N,
                        "period": f"{START}..{END}"}, split="DEV", metrics=out, data=d.manifest)
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    {"sample": sample, "download": download, "run": run}[sys.argv[1]]()
