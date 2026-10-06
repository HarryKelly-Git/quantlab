"""H15 pairs trading, distance method (Gatev, Goetzmann & Rouwenhorst 2006). PRE-REGISTERED (2 variants):

  formation   at each half-year start, trailing 252 sessions; names in the large-cap universe on the
              formation date with complete data; normalised cumulative total-return price (start = 1)
  selection   the 20 pairs with the smallest sum of squared differences, within the same statistical
              sector | any sector (the 2 variants)
  trading     the next 126 sessions; spread = normalised price A - B, rebased at the formation end;
              open when |spread| > 2 x formation std (decided at the close, executed at the next open:
              the published "wait one day" variant), long the cheaper leg, short the dearer leg, equal $;
              close when the spread crosses zero (next open) or at the end of the trading period
  P&L         committed capital: each of the 20 pairs owns 1/20 of the book whether open or not; tiered
              costs on each leg each trade; 50 bps/yr borrow on the short leg; delisting closes at the last
              close with the panel's delisting return
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha import splits
from quantlab.alpha.metrics import core_metrics
from quantlab.validation.stats import newey_west_tstat

N_PAIRS, FORM, TRADE, Z = 20, 252, 126, 2.0


def _select_pairs(px: pd.DataFrame, sectors_row: pd.Series | None) -> list[tuple[str, str, float]]:
    P = px / px.iloc[0]
    X = P.to_numpy()
    names = list(P.columns)
    cand = []
    groups = (pd.Series(names, index=names).groupby(sectors_row.reindex(names).fillna("NA")).apply(list).tolist()
              if sectors_row is not None else [names])
    for g in groups:
        idx = [names.index(n) for n in g]
        if len(idx) < 2:
            continue
        A = X[:, idx]
        sq = (A ** 2).sum(axis=0)
        ssd = sq[:, None] + sq[None, :] - 2 * A.T @ A
        iu = np.triu_indices(len(idx), 1)
        for a, b, v in zip(np.array(idx)[iu[0]], np.array(idx)[iu[1]], ssd[iu]):
            cand.append((v, names[a], names[b]))
    cand.sort()
    out, used = [], set()
    for v, a, b in cand:
        if len(out) == N_PAIRS:
            break
        out.append((a, b, float(np.std(P[a].to_numpy() - P[b].to_numpy()))))
    return out


def run(d, within_sector: bool) -> dict:
    p = d.p
    ac, ao = p["adj_close"], p["adj_open"]
    roo = p["ret_oo"]
    dates = p.dates
    starts = [i for i in range(FORM, len(dates) - 1)
              if dates[i].month in (1, 7) and (i == 0 or dates[i - 1].month != dates[i].month)]
    daily = pd.Series(0.0, index=dates)
    n_trades = 0
    for s in starts:
        fd = dates[s - 1]
        uni = d.u_large.loc[fd]
        names = uni.index[uni.to_numpy()]
        win = ac.iloc[s - FORM: s][names]
        win = win.loc[:, win.notna().all()]
        if win.shape[1] < 20:
            continue
        pairs = _select_pairs(win, d.sectors.loc[fd] if within_sector else None)
        end = min(s + TRADE, len(dates) - 1)
        for a, b, sd in pairs:
            base_a, base_b = ac.at[fd, a], ac.at[fd, b]
            pos = 0                                   # +1: long A short B ; -1: short A long B (decided at close t)
            held = 0                                  # position actually held from the open of t+1
            for t in range(s, end):
                day = dates[t]
                # P&L of the position held from the open of t to the open of t+1
                if held != 0:
                    ra, rb = roo.at[day, a], roo.at[day, b]
                    ra = 0.0 if not np.isfinite(ra) else ra
                    rb = 0.0 if not np.isfinite(rb) else rb
                    daily.at[day] += held * (ra - rb) * 0.5 / N_PAIRS - 0.5 * 50e-4 / 252 / N_PAIRS
                na, nb = ac.at[day, a], ac.at[day, b]
                if not (np.isfinite(na) and np.isfinite(nb)):
                    pos = 0                           # a leg stopped trading: flat from the next open
                else:
                    spread = na / base_a - nb / base_b
                    if pos == 0 and abs(spread) > Z * sd:
                        pos = -1 if spread > 0 else 1
                    elif pos != 0 and np.sign(spread) != np.sign(-pos) and np.sign(spread) != 0 or (pos != 0 and t == end - 1):
                        pos = 0
                if pos != held and t + 1 < len(dates):     # trade at the next open: pay costs on both legs
                    ca = d.cost_bps.at[day, a] / 1e4 if np.isfinite(d.cost_bps.at[day, a]) else 30e-4
                    cb = d.cost_bps.at[day, b] / 1e4 if np.isfinite(d.cost_bps.at[day, b]) else 30e-4
                    legs = abs(pos - held)               # 1 for open/close, 2 for a flip
                    daily.at[dates[t + 1]] -= legs * 0.5 * (ca + cb) / N_PAIRS
                    n_trades += 1
                    held = pos
    r = daily.loc[dates[starts[0]]:] if starts else daily
    out = {"n_round_trip_legs": n_trades}
    for sp in ("TRAIN", "VALIDATION", "OOS"):
        x = splits.slice_split(r, "equity", sp)
        m = core_metrics(x)
        out[sp] = {"sharpe": m.get("sharpe"), "t": m.get("t_mean_nw"), "ann_return": (m.get("mean_daily") or 0) * 252,
                   "max_drawdown": m.get("max_drawdown")}
    out["daily"] = r
    return out
