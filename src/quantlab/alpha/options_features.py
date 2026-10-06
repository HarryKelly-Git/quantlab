"""Per (snapshot, underlying, expiry) option features + outcomes, from the DoltHub chains and the
survivorship-free equity store. PAPER research.

Features (known at the snapshot; the snapshot is end-of-day, see quality.chain_report timing check):
  spot        RAW close of the snapshot's session (2019 Saturday snapshots -> the Friday session)
  k_atm       strike nearest spot with BOTH a call and a put quoted (bid > 0, ask > bid)
  straddle_bid/ask/mid at k_atm;  iv_atm = mean of the call and put IV at k_atm
  iv_put25 / iv_call25: IV of the put / call whose delta is nearest -0.25 / +0.25 (same expiry)
  dte         calendar days to expiry;  em_pct = straddle_mid / spot (implied expected absolute move)
Outcomes (OUTCOMES ONLY - never features):
  exp_close   RAW close on the expiry session (or the last session before it)
  payoff      |exp_close - k_atm|;  ret_hold_ask = payoff / straddle_ask - 1 (pay the ask, settle at intrinsic)
  rv_to_exp   annualised realised vol of daily adjusted log returns after the snapshot up to expiry
  move_to_exp |adj_close(expiry) / adj_close(snapshot) - 1|
Rows whose raw/adjusted price ratio changes between snapshot and expiry (a split or large special
dividend) are dropped from the outcome columns: OCC adjusts the contract, the raw payoff would be wrong.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.alpha.options_store import options_dir
from quantlab.alpha.store import HOLDOUT_START, load_bars


def _session_map(sessions: pd.DatetimeIndex, dates: pd.Series) -> pd.Series:
    pos = sessions.searchsorted(pd.to_datetime(dates).to_numpy(), side="right") - 1
    return pd.Series(sessions[np.clip(pos, 0, len(sessions) - 1)], index=dates.index)


def month_features(chain: pd.DataFrame, closes: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    c = chain[(chain["bid"] > 0) & (chain["ask"] > chain["bid"]) & (chain["vol"] > 0.01) & (chain["vol"] < 5)].copy()
    if c.empty:
        return pd.DataFrame()
    c["act_symbol"] = c["act_symbol"].astype(str)
    c["session"] = _session_map(sessions, c["date"]).to_numpy()
    sp = closes.stack(future_stack=True).rename("spot").reset_index()
    sp.columns = ["session", "act_symbol", "spot"]
    c = c.merge(sp.dropna(), on=["session", "act_symbol"], how="inner")
    c["mid"] = (c["bid"] + c["ask"]) / 2
    key = ["date", "session", "act_symbol", "expiration"]
    calls = c[c["cp"] == "C"].set_index(key + ["strike"])[["bid", "ask", "mid", "vol", "delta", "spot"]]
    puts = c[c["cp"] == "P"].set_index(key + ["strike"])[["bid", "ask", "mid", "vol", "delta"]]
    both = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner").reset_index()
    both["dist"] = (both["strike"] - both["spot"]).abs()
    atm = both.loc[both.groupby(key)["dist"].idxmin()]
    out = pd.DataFrame({
        "date": atm["date"], "session": atm["session"], "act_symbol": atm["act_symbol"], "expiration": atm["expiration"],
        "spot": atm["spot"], "k_atm": atm["strike"],
        "straddle_bid": atm["bid_c"] + atm["bid_p"], "straddle_ask": atm["ask_c"] + atm["ask_p"],
        "straddle_mid": atm["mid_c"] + atm["mid_p"], "iv_atm": (atm["vol_c"] + atm["vol_p"]) / 2,
        "call_delta_atm": atm["delta_c"]})
    n_strikes = both.groupby(key).size().rename("n_strikes")
    out = out.merge(n_strikes.reset_index(), on=key, how="left")
    # 25-delta wings (nearest available delta)
    cc = c[c["cp"] == "C"].assign(dd=lambda x: (x["delta"] - 0.25).abs())
    pp = c[c["cp"] == "P"].assign(dd=lambda x: (x["delta"] + 0.25).abs())
    c25 = cc.loc[cc.groupby(key)["dd"].idxmin(), key + ["vol", "dd"]].rename(columns={"vol": "iv_call25", "dd": "c25_err"})
    p25 = pp.loc[pp.groupby(key)["dd"].idxmin(), key + ["vol", "dd"]].rename(columns={"vol": "iv_put25", "dd": "p25_err"})
    out = out.merge(c25, on=key, how="left").merge(p25, on=key, how="left")
    out["dte"] = (pd.to_datetime(out["expiration"]) - pd.to_datetime(out["session"])).dt.days
    out["em_pct"] = out["straddle_mid"] / out["spot"]
    return out


def ticker_level(bars: pd.DataFrame) -> pd.DataFrame:
    """Map store entities ('TICKER' or 'TICKER@asof' for delisted entities found via the directory) to the
    plain ticker the option chain uses. Where several entities share a ticker (recycled tickers), each
    date takes the entity trading under that ticker then; ``entity`` records which one."""
    b = bars.copy()
    b["ticker"] = b["symbol"].str.split("@").str[0]
    b["prio"] = (b["symbol"] != b["ticker"]).astype(int)          # plain ticker first on overlapping dates
    b = b.sort_values(["ticker", "date", "prio"]).drop_duplicates(["ticker", "date"], keep="first")
    return b.rename(columns={"symbol": "entity"}).rename(columns={"ticker": "symbol"})


def build_features() -> pd.DataFrame:
    t0 = time.time()
    bars = ticker_level(load_bars(columns=["symbol", "date", "close", "adj_close"]))
    closes = bars.pivot(index="date", columns="symbol", values="close")
    sessions = closes.index
    files = sorted(options_dir().glob("chain_*-*.parquet"))
    feats = []
    for f in files:
        ch = pd.read_parquet(f)
        if ch["date"].max() >= pd.Timestamp(HOLDOUT_START):
            raise RuntimeError("holdout options data in the research store")
        syms = [s for s in ch["act_symbol"].astype(str).unique() if s in closes.columns]
        feats.append(month_features(ch, closes[syms], sessions))
    df = pd.concat(feats, ignore_index=True)
    print(f"features: {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
    df = add_outcomes(df, bars)
    df.to_parquet(options_dir() / "features.parquet", index=False)
    print(f"outcomes added in {time.time() - t0:.0f}s", flush=True)
    return df


def add_outcomes(df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    closes = bars.pivot(index="date", columns="symbol", values="close")
    adj = bars.pivot(index="date", columns="symbol", values="adj_close")
    ent = bars.pivot(index="date", columns="symbol", values="entity") if "entity" in bars else None
    sessions = closes.index
    logr = np.log(adj / adj.shift(1))
    exp_sess = _session_map(sessions, df["expiration"])
    df = df.assign(exp_session=exp_sess.to_numpy())
    # expiry beyond the store end -> outcome unknown
    df.loc[pd.to_datetime(df["expiration"]) > sessions[-1], "exp_session"] = pd.NaT
    ci = closes.columns.get_indexer(df["act_symbol"])
    si = sessions.get_indexer(pd.to_datetime(df["session"]))
    ei = sessions.get_indexer(pd.to_datetime(df["exp_session"]))
    C, A, L = closes.to_numpy(), adj.to_numpy(), logr.to_numpy()
    ok = (ci >= 0) & (si >= 0) & (ei >= 0)
    exp_close = np.full(len(df), np.nan)
    move = np.full(len(df), np.nan)
    rv = np.full(len(df), np.nan)
    ratio_change = np.full(len(df), np.nan)
    csum = np.nancumsum(L, axis=0)
    csum2 = np.nancumsum(L ** 2, axis=0)
    cnt = np.cumsum(np.isfinite(L), axis=0)
    idx = np.nonzero(ok)[0]
    exp_close[idx] = C[ei[idx], ci[idx]]
    move[idx] = np.abs(A[ei[idx], ci[idx]] / A[si[idx], ci[idx]] - 1)
    n = cnt[ei[idx], ci[idx]] - cnt[si[idx], ci[idx]]
    s1 = csum[ei[idx], ci[idx]] - csum[si[idx], ci[idx]]
    s2 = csum2[ei[idx], ci[idx]] - csum2[si[idx], ci[idx]]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 ** 2 / n) / (n - 1)
        rv[idx] = np.where(n >= 3, np.sqrt(var * 252), np.nan)
        ratio_change[idx] = (C[ei[idx], ci[idx]] / A[ei[idx], ci[idx]]) / (C[si[idx], ci[idx]] / A[si[idx], ci[idx]]) - 1
    split = np.abs(ratio_change) > 0.02
    if ent is not None:                       # a recycled ticker changed company inside the window
        E = ent.reindex(index=closes.index, columns=closes.columns).to_numpy()
        same = np.ones(len(df), dtype=bool)
        same[idx] = E[si[idx], ci[idx]] == E[ei[idx], ci[idx]]
        split = split | ~same
    exp_close[split] = np.nan
    move[split] = np.nan
    df["exp_close"] = exp_close
    df["move_to_exp"] = move
    df["rv_to_exp"] = rv
    df["payoff"] = (df["exp_close"] - df["k_atm"]).abs()
    df["ret_hold_ask"] = df["payoff"] / df["straddle_ask"] - 1
    df["ret_hold_mid"] = df["payoff"] / df["straddle_mid"] - 1
    df["split_or_special_div"] = split
    return df


if __name__ == "__main__":   # pragma: no cover
    build_features()
