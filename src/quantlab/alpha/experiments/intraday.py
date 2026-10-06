"""H08 market intraday momentum (Gao, Han, Li & Zhou 2018). PRE-REGISTERED (4 variants):
  r_first = price(10:00) / previous close - 1 (includes the overnight move, as in the paper)
  r_12    = price(15:30) / price(15:00) - 1;  r_last = close(16:00) / price(15:30) - 1
  signal  sign(r_first) | sign(r_first + r_12); instrument SPY | QQQ; trade at 15:30, exit at the close.
  Costs per side: 1 bp (auction-quality SPY/QQQ execution) and 7 bp (QuantLab tier: 2 + 5).
Prices from 1-minute SIP bars: the bar stamped hh:mm covers hh:mm..hh:mm+1, so price(10:00) = close of
the 09:59 bar. Half days (no 15:59 bar) are skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha import splits
from quantlab.alpha.store import store_dir
from quantlab.validation.stats import newey_west_tstat


def daily_legs(sym: str) -> pd.DataFrame:
    files = sorted((store_dir() / "minute").glob(f"{sym}_*.parquet"))
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    m["day"] = m["t"].dt.tz_localize(None).dt.normalize()
    m["hm"] = m["t"].dt.strftime("%H:%M")
    piv = m[m["hm"].isin(["09:59", "14:59", "15:29", "15:59"])].pivot_table(index="day", columns="hm", values="close", aggfunc="last")
    piv = piv.dropna(subset=["15:59"])
    out = pd.DataFrame(index=piv.index)
    out["prev_close"] = piv["15:59"].shift(1)
    out["r_first"] = piv["09:59"] / out["prev_close"] - 1
    out["r_12"] = piv["15:29"] / piv["14:59"] - 1
    out["r_last"] = piv["15:59"] / piv["15:29"] - 1
    return out.dropna()


def run() -> dict:
    res = {}
    for sym in ("SPY", "QQQ"):
        legs = daily_legs(sym)
        for sig in ("first", "first_plus_12"):
            s = np.sign(legs["r_first"] if sig == "first" else legs["r_first"] + legs["r_12"])
            gross = s * legs["r_last"]
            row = {}
            for sp in ("TRAIN", "VALIDATION", "OOS"):
                g = splits.slice_split(gross, "equity", sp)
                lg = splits.slice_split(legs, "equity", sp)
                if len(g) < 100:
                    continue
                X = np.column_stack([np.ones(len(lg)), lg["r_first"]])
                b = np.linalg.lstsq(X, lg["r_last"].to_numpy(), rcond=None)[0]
                pred = X @ b
                r2 = 1 - ((lg["r_last"] - pred) ** 2).sum() / ((lg["r_last"] - lg["r_last"].mean()) ** 2).sum()
                row[sp] = {"n_days": int(len(g)), "gross_mean_bps": float(g.mean() * 1e4), "t_gross": newey_west_tstat(g.to_numpy()).t,
                           "net_mean_bps_1bp": float(g.mean() * 1e4 - 2.0), "net_mean_bps_7bp": float(g.mean() * 1e4 - 14.0),
                           "hit_rate": float((g > 0).mean()), "slope_r_last_on_r_first": float(b[1]), "r2_in_split": float(r2)}
            res[f"{sym}_{sig}"] = row
    return res
