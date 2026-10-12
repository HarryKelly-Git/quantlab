"""H09 seasonality on SPY (Part 25). PRE-REGISTERED effect list (fixed before any result):

  dow_mon .. dow_fri       day-of-week close-to-close returns
  turn_of_month            last session of a month + first three sessions of the next
  month_end                last session of a month only
  pre_holiday              the session before a weekday market holiday (a weekday with no session)
  opex_week                the week (Mon-Fri) containing the third Friday of the month
  opex_friday              the third Friday itself
  pre_fomc_day             the session before an FOMC decision day
  fomc_day                 the FOMC decision day (statement day; scheduled meetings only)
  nfp_friday               the first Friday of each month (US jobs report rule of thumb; exceptions
                           ignored - ASSUMED schedule)
For each effect: mean SPY return on effect days vs all other days (difference, Newey-West t), by split,
plus the tradeable version: long SPY from the close before to the close of each effect day, paying the
SPY cost tier (2 bps half-spread + 5 bps slippage per side, also shown at 1 bp per side for auction
orders). Benjamini-Hochberg across the 13 effects (development period) decides which go to OOS.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantlab.alpha import splits
from quantlab.alpha.store import store_dir
from quantlab.validation.stats import benjamini_hochberg, newey_west_tstat


def effect_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.Series(dates, index=dates)
    f = pd.DataFrame(index=dates)
    for i, nm in enumerate(("mon", "tue", "wed", "thu", "fri")):
        f[f"dow_{nm}"] = dates.dayofweek == i
    month = dates.to_period("M")
    last = s.groupby(month).transform("max") == s
    rank_in_month = s.groupby(month).cumcount()
    f["month_end"] = last.to_numpy()
    f["turn_of_month"] = (last | (rank_in_month < 3)).to_numpy()
    nxt = pd.Series(dates[1:].tolist() + [pd.NaT], index=dates)
    gap_weekdays = [len(pd.bdate_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1))) if pd.notna(b) else 0
                    for a, b in zip(dates, nxt)]
    f["pre_holiday"] = np.array(gap_weekdays) > 0
    third_fri = {}
    for p in month.unique():
        d0 = p.start_time
        fridays = pd.date_range(d0, p.end_time, freq="W-FRI")
        third_fri[p] = fridays[2] if len(fridays) >= 3 else None
    tf = pd.Series([third_fri.get(p) for p in month], index=dates)
    f["opex_friday"] = (tf == s).to_numpy()
    wk = dates.to_period("W-FRI")
    f["opex_week"] = pd.Series(wk, index=dates).isin(set(pd.DatetimeIndex([t for t in third_fri.values() if t is not None]).to_period("W-FRI"))).to_numpy()
    fomc = pd.DatetimeIndex(json.loads((store_dir() / "external" / "fomc_decision_days_2016_2024.json").read_text()))
    f["fomc_day"] = dates.isin(fomc)
    f["pre_fomc_day"] = pd.Series(f["fomc_day"].to_numpy(), index=dates).shift(-1, fill_value=False).to_numpy()
    first_fri = s.groupby(month).transform(lambda x: x[x.dt.dayofweek == 4].min())
    f["nfp_friday"] = (first_fri == s).to_numpy()
    return f


def run(d) -> dict:
    r = d.p["ret_cc"]["SPY"].dropna()
    flags = effect_flags(r.index)
    out: dict = {"effects": {}}
    dev_p = {}
    for eff in flags.columns:
        on = flags[eff]
        res = {}
        for sp in ("TRAIN", "VALIDATION", "OOS"):
            rs = splits.slice_split(r, "equity", sp)
            fl = on.reindex(rs.index).fillna(False)
            diff_series = rs[fl]
            if fl.sum() < 8:
                continue
            res[sp] = {"n_days": int(fl.sum()), "mean_on_bps": float(diff_series.mean() * 1e4),
                       "mean_off_bps": float(rs[~fl].mean() * 1e4),
                       "t_on_minus_off": float((diff_series.mean() - rs[~fl].mean()) /
                                               np.sqrt(diff_series.var() / len(diff_series) + rs[~fl].var() / max(len(rs[~fl]), 1))),
                       "net_per_trade_bps_cost7": float(diff_series.mean() * 1e4 - 14.0),
                       "net_per_trade_bps_cost1": float(diff_series.mean() * 1e4 - 2.0)}
        dev = pd.concat([splits.slice_split(r, "equity", s) for s in splits.DEVELOPMENT])
        fl = on.reindex(dev.index).fillna(False)
        a, b = dev[fl], dev[~fl]
        tval = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        from scipy import stats as ss
        dev_p[eff] = float(2 * ss.norm.sf(abs(tval)))
        res["DEV"] = {"n_days": int(fl.sum()), "mean_on_bps": float(a.mean() * 1e4), "t_on_minus_off": float(tval), "p_two_sided": dev_p[eff]}
        out["effects"][eff] = res
    rej, adj = benjamini_hochberg(list(dev_p.values()), 0.05)
    out["bh_dev"] = {k: {"p": p, "q_adj": float(q), "reject": bool(rj)} for (k, p), q, rj in zip(dev_p.items(), adj, rej)}
    out["n_effects"] = len(dev_p)
    return out
