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

OPTIONS_DATA_VERSION = ("2026-10-audit-b: ticker->company by date incl. Alpaca symbol changes; (ticker, month) "
                        "wrong-company parity test; strict-ATM rows flagged, not dropped")


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
    # put-call parity at the ATM strike implies the spot (ignoring carry, fine for short-dated near-the-money
    # options); used by build_features to reject (ticker, month) chains that belong to another company
    atm_c_mid = (atm["bid_c"] + atm["ask_c"]).to_numpy() / 2
    atm_p_mid = (atm["bid_p"] + atm["ask_p"]).to_numpy() / 2
    implied = atm_c_mid - atm_p_mid + atm["strike"].to_numpy()
    out["parity_err"] = np.abs(implied / out["spot"].to_numpy() - 1)
    out["k_dist"] = np.abs(out["k_atm"] / out["spot"] - 1)
    return out


def ticker_level(bars: pd.DataFrame) -> pd.DataFrame:
    """Map store entities to the ticker the option chain uses, choosing BY DATE the entity that traded under
    that ticker then (alpha.entities.resolve with Alpaca's symbol-change history). Audit fix C2: the first
    version preferred the plain (current) entity, so e.g. old-Caesars CZR options were priced against
    Eldorado's history, and 2023 PARA (Paramount) options against Banzai's SPAC."""
    from quantlab.alpha.entities import resolve
    from quantlab.alpha.store import load_name_changes
    r = resolve(bars, load_name_changes()).drop(columns=["symbol"])
    r["symbol"] = r["ticker"].astype(str)
    r["entity"] = r["entity"].astype(str)
    return r.drop(columns=["ticker"])


def build_features() -> pd.DataFrame:
    t0 = time.time()
    raw = load_bars(columns=["symbol", "date", "close", "adj_close", "volume"])
    bars = ticker_level(raw[raw["volume"] > 0].drop(columns=["volume"]))       # filler bars claim no ticker
    del raw
    closes = bars.pivot(index="date", columns="symbol", values="close")
    sessions = closes.index
    files = sorted(options_dir().glob("chain_*-*.parquet"))
    feats = []
    for f in files:
        ch = pd.read_parquet(f)
        if ch["date"].max() >= pd.Timestamp(HOLDOUT_START):
            from quantlab.alpha.holdout import refuse
            refuse(f"options_features.build_features: {f.name}")
        syms = [s for s in ch["act_symbol"].astype(str).unique() if s in closes.columns]
        feats.append(month_features(ch, closes[syms], sessions))
    df = pd.concat(feats, ignore_index=True)
    # wrong-company test PER (ticker, month): a chain belonging to another company misses the matched close
    # on (almost) every snapshot, so the month's MEDIAN parity error is large. A per-row rule (first audit
    # fix) mostly removed legitimate cheap, high-IV names on days without a near-the-money strike
    # (verification review): those rows stay, flagged by ``atm_ok`` for a strict-ATM sensitivity run.
    month = pd.to_datetime(df["session"]).dt.to_period("M")
    med = df.groupby([df["act_symbol"], month])["parity_err"].transform("median")
    wrong = (med > 0.05).to_numpy()
    rej = df.loc[wrong]
    report = {"rule": "drop (ticker, month) whose median put-call-parity spot error > 5%",
              "rows_before_filter": int(len(df)), "rows_rejected": int(wrong.sum()),
              "rejected_ticker_months": int(rej.groupby(["act_symbol", month[wrong]]).ngroups) if wrong.any() else 0,
              "rejected_tickers": int(rej["act_symbol"].nunique()),
              "rejected_by_year": {str(k): int(v) for k, v in pd.to_datetime(rej["session"]).dt.year.value_counts().sort_index().items()},
              "top_rejected_tickers": {str(k): int(v) for k, v in rej["act_symbol"].value_counts().head(25).items()}}
    df = df.loc[~wrong].reset_index(drop=True)
    df["atm_ok"] = (df["k_dist"] <= 0.10) & (df["parity_err"] <= 0.05)
    report["rows_not_atm_ok_kept"] = int((~df["atm_ok"]).sum())
    # the store entity behind each (ticker, session): what every equity lookup must use (audit fixes C2/M1)
    ent = bars[["symbol", "date", "entity"]].rename(columns={"symbol": "act_symbol", "date": "session"})
    df = df.merge(ent, on=["act_symbol", "session"], how="left")
    report["rows_kept"] = int(len(df))
    report["rows_on_delisted_entities"] = int(df["entity"].astype(str).str.contains("@").sum())
    import json
    (options_dir() / "features_report.json").write_text(json.dumps(report, indent=1))
    print(f"features: {len(df)} rows in {time.time() - t0:.0f}s; wrong-company months rejected {report['rows_rejected']} rows", flush=True)
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
    rv[split] = np.nan
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
