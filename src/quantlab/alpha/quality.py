"""Data-quality checks (Part 42). A study does not run on a dataset that fails these; anomalies are
counted and reported, never silently "fixed" in a way that could use future information.

Equity bars: duplicates, impossible OHLC, non-positive prices, zero-volume days, stale (repeated)
closes, extreme one-day adjusted returns, extreme moves that look like UNADJUSTED SPLITS (a jump close to
a simple ratio that the adjusted series did not absorb), adjusted/raw factor sanity.

Options chain: crossed or negative spreads, zero bids, IV outliers, missing greeks, delta sign errors,
put-call parity versus the underlying close (also used to infer the SNAPSHOT TIME: if parity matches the
same-day close better than the previous close, snapshots are end-of-day for that date).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SPLIT_RATIOS = np.array([2, 3, 4, 5, 8, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 3 / 2, 2 / 3])


def equity_report(bars: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(bars)), "symbols": int(bars["symbol"].nunique())}
    out["duplicates"] = int(bars.duplicated(["symbol", "date"]).sum())
    o, h, l, c = (bars[k] for k in ("open", "high", "low", "close"))
    out["nonpositive_price"] = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
    out["high_below_low"] = int((h < l).sum())
    out["close_outside_range"] = int(((c > h * 1.0001) | (c < l * 0.9999)).sum())
    out["open_outside_range"] = int(((o > h * 1.0001) | (o < l * 0.9999)).sum())
    out["zero_volume"] = int((bars["volume"] <= 0).sum())
    b = bars.sort_values(["symbol", "date"])
    r = b.groupby("symbol")["adj_close"].pct_change()
    raw_r = b.groupby("symbol")["close"].pct_change()
    out["abs_adj_return_gt_50pct"] = int((r.abs() > 0.5).sum())
    out["abs_adj_return_gt_100pct"] = int((r > 1.0).sum())
    # unadjusted split suspects: big adjusted move whose (1 + r) is within 3% of a split ratio
    big = r.abs() > 0.4
    ratio = (1 + r[big]).to_numpy()
    near = np.abs(ratio[:, None] / SPLIT_RATIOS[None, :] - 1).min(axis=1) < 0.03
    out["unadjusted_split_suspects"] = int(near.sum())
    sus = b.loc[big[big].index[near], ["symbol", "date", "close", "adj_close"]] if near.any() else b.iloc[0:0]
    out["unadjusted_split_examples"] = sus.head(15).astype(str).to_dict("records")
    # raw vs adjusted: on split days raw moves by the ratio while adjusted does not
    split_days = ((raw_r - r).abs() > 0.2).sum()
    out["raw_vs_adjusted_split_days"] = int(split_days)
    stale = b.groupby("symbol")["close"].transform(lambda s: s.eq(s.shift()).rolling(5).sum()) >= 5
    out["stale_5day_runs"] = int(stale.sum())
    out["status"] = "PASS" if (out["duplicates"] == 0 and out["high_below_low"] < 0.0001 * len(bars)
                               and out["unadjusted_split_suspects"] < 0.0005 * out["symbols"] * 10) else "REVIEW"
    return out


def chain_report(chain: pd.DataFrame, closes: pd.DataFrame | None = None, sample: int = 200_000,
                 seed: int = 1) -> dict[str, Any]:
    """``closes``: wide raw closes (dates x symbols) from the equity store, for parity/timing checks."""
    out: dict[str, Any] = {"rows": int(len(chain)), "underlyings": int(chain["act_symbol"].nunique()),
                           "snapshots": int(chain["date"].nunique())}
    bid, ask = chain["bid"], chain["ask"]
    out["crossed_or_negative_spread"] = int((ask < bid).sum())
    out["zero_bid"] = int((bid <= 0).sum())
    out["missing_iv"] = int(chain["vol"].isna().sum())
    out["iv_gt_5"] = int((chain["vol"] > 5).sum())
    out["iv_le_0"] = int((chain["vol"] <= 0).sum())
    dc = chain["delta"]
    out["call_delta_negative"] = int(((chain["cp"] == "C") & (dc < -0.01)).sum())
    out["put_delta_positive"] = int(((chain["cp"] == "P") & (dc > 0.01)).sum())
    mid = (bid + ask) / 2
    out["median_rel_spread"] = float(((ask - bid) / mid.where(mid > 0)).median())
    if closes is not None:
        rng = np.random.default_rng(seed)
        s = chain.sample(min(sample, len(chain)), random_state=int(rng.integers(1 << 31)))
        piv = s.pivot_table(index=["date", "act_symbol", "expiration", "strike"], columns="cp",
                            values=["bid", "ask"], observed=True).dropna()
        if len(piv):
            cm = (piv[("bid", "C")] + piv[("ask", "C")]) / 2
            pm = (piv[("bid", "P")] + piv[("ask", "P")]) / 2
            idx = piv.index.to_frame(index=False)
            implied_spot = (cm - pm).to_numpy() + idx["strike"].to_numpy()   # ignores carry: short-dated, near ATM
            snap = pd.to_datetime(idx["date"])
            # same-session close vs previous-session close for each snapshot date
            cl = closes.stack(future_stack=True).rename("close").reset_index()
            cl.columns = ["date", "act_symbol", "close"]
            cl = cl.dropna()
            cl["prev_close"] = cl.groupby("act_symbol")["close"].shift(1)
            # weekend snapshots (2019 Saturdays) map to the preceding session
            sessions = pd.DatetimeIndex(sorted(cl["date"].unique()))
            pos = sessions.searchsorted(snap, side="right") - 1
            sess = sessions[np.clip(pos, 0, len(sessions) - 1)]
            j = pd.DataFrame({"date": sess, "act_symbol": idx["act_symbol"].astype(str).to_numpy(),
                              "implied": implied_spot, "strike": idx["strike"].to_numpy()})
            j = j.merge(cl, on=["date", "act_symbol"], how="inner")
            j = j[(j["strike"] / j["close"] - 1).abs() < 0.05]          # near the money only
            if len(j) > 100:
                e_same = (j["implied"] / j["close"] - 1).abs().median()
                e_prev = (j["implied"] / j["prev_close"] - 1).abs().median()
                out["parity_median_err_vs_same_close"] = float(e_same)
                out["parity_median_err_vs_prev_close"] = float(e_prev)
                out["snapshot_timing"] = "SAME_SESSION_CLOSE" if e_same < e_prev else "PREVIOUS_SESSION_CLOSE"
                out["parity_n"] = int(len(j))
    bad = out["crossed_or_negative_spread"] + out["call_delta_negative"] + out["put_delta_positive"]
    out["status"] = "PASS" if bad < 0.001 * len(chain) else "REVIEW"
    return out
