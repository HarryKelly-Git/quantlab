"""Earnings families (Parts 10-11, 15). PRE-REGISTERED before any outcome was computed.

Calendar cleaning (DoltHub earnings_calendar, 2020-2024, PIT_ASSUMED: realised announcement dates):
  * events of the same symbol within 20 calendar days of each other are one quarter's duplicates
    (a stale projected date + the real one): keep the one with the larger absolute next-session
    abnormal volume (dollar volume / its 20-day median) - the announcement leaves a volume footprint.
    This uses post-event data to CLEAN THE CALENDAR, which is acceptable only because realised dates
    are what the (backfilled) calendar represents anyway; it is never used as a trading signal.
  * 'when': AMC -> first reaction session = next session; BMO -> the announcement session itself;
    unknown -> treated as BMO for exits (the conservative choice: exit before the earlier possible release).

H20 earnings-announcement premium (Frazzini & Lamont 2007): long stocks with an announcement in the
    next 1 | 3 | 5 sessions (decided at close t, held from the next open until the open of the first
    reaction session), short SPY by beta. Universe liquid. 3 variants.
H21 EPS-surprise drift: surprise = (reported - estimate) / price at the announcement (eps_history,
    estimate as recorded by the vendor), sorted into deciles at the close of the first reaction session;
    long top / short bottom decile, hold 21 | 63 sessions starting at the next open. 2 variants.
H23 revision momentum: change in the 'Current Year' consensus EPS over the last 21 | 63 snapshot days
    divided by |consensus| (dated snapshots, PIT), deciles, long top / short bottom, hold 21. 2 variants.
H26 pre-earnings straddle (Gao, Xing & Zhang 2018 replication; options store):
    expiry = first listed expiry after the first reaction session; strike = ATM at entry; entry at the
    ASK at the snapshot k = 1 | 2 | 3 snapshots (Mon/Wed/Fri cadence, i.e. ~2-7 sessions) before the exit
    snapshot; exit at the BID at the last snapshot whose session is <= the last pre-announcement close.
    Reported: mean return, median, hit rate, NW t across events by split; vs a placebo with the same
    timing in non-earnings weeks (same stocks, 63 sessions earlier).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.alpha.options_store import options_dir


def clean_calendar(d) -> pd.DataFrame:
    cal = pd.read_parquet(options_dir() / "earnings_calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal[cal["act_symbol"].isin(d.p.symbols)].sort_values(["act_symbol", "date"]).reset_index(drop=True)
    sessions = d.p.dates
    pos = sessions.searchsorted(cal["date"].to_numpy(), side="left")
    pos = np.clip(pos, 0, len(sessions) - 1)
    cal["ann_session"] = sessions[pos]
    amc = cal["when"].fillna("").str.startswith("After")
    react_pos = np.where(amc, pos + 1, pos)
    react_pos = np.clip(react_pos, 0, len(sessions) - 1)
    cal["react_session"] = sessions[react_pos]
    cal["timing"] = np.where(amc, "AMC", np.where(cal["when"].fillna("").str.startswith("Before"), "BMO", "UNKNOWN"))
    # abnormal dollar volume on the reaction session, for de-duplication only
    dv = d.p["dollar_volume"]
    med = dv.rolling(20, min_periods=10).median().shift(1)
    ab = (dv / med)
    ci = dv.columns.get_indexer(cal["act_symbol"])
    ri = sessions.get_indexer(cal["react_session"])
    vals = ab.to_numpy()
    cal["abn_vol"] = np.where((ci >= 0) & (ri >= 0), vals[np.clip(ri, 0, None), np.clip(ci, 0, None)], np.nan)
    cal["grp"] = (cal.groupby("act_symbol")["date"].diff().dt.days.fillna(999) >= 20).cumsum()
    keep = cal.loc[cal.groupby("grp")["abn_vol"].apply(lambda s: s.fillna(-1).idxmax())]
    return keep.drop(columns=["grp"]).reset_index(drop=True)


# --- event-level returns ---------------------------------------------------------------------------
def event_returns(d, events: pd.DataFrame, entry_col: str, exit_col: str, exit_at: str = "close",
                  hedge: bool = True, cost_mult: float = 1.0) -> pd.Series:
    """Per-event net return: buy at the OPEN of ``entry_col`` session, sell at the CLOSE (or OPEN) of
    ``exit_col`` session, minus beta x SPY over the same window, minus one-way costs on each side."""
    ao, ac = d.p["adj_open"], d.p["adj_close"]
    sessions = d.p.dates
    ci = ao.columns.get_indexer(events["act_symbol"])
    ei = sessions.get_indexer(pd.to_datetime(events[entry_col]))
    xi = sessions.get_indexer(pd.to_datetime(events[exit_col]))
    ok = (ci >= 0) & (ei >= 0) & (xi >= ei)
    AO, AC = ao.to_numpy(), ac.to_numpy()
    spy = ao.columns.get_loc("SPY")
    px_exit = AC if exit_at == "close" else AO
    r = np.full(len(events), np.nan)
    rm = np.full(len(events), np.nan)
    idx = np.nonzero(ok)[0]
    r[idx] = px_exit[xi[idx], ci[idx]] / AO[ei[idx], ci[idx]] - 1
    rm[idx] = px_exit[xi[idx], spy] / AO[ei[idx], spy] - 1
    beta = d.betas["beta_mkt"].to_numpy()[np.clip(ei - 1, 0, None), np.clip(ci, 0, None)]
    beta = np.where(np.isfinite(beta), beta, 1.0)
    cost = d.cost_bps.to_numpy()[np.clip(ei - 1, 0, None), np.clip(ci, 0, None)] / 1e4 * cost_mult
    net = r - (beta * rm if hedge else 0.0) - 2 * cost
    return pd.Series(net, index=events.index)


def _shift_session(d, s: pd.Series, k: int) -> pd.Series:
    sessions = d.p.dates
    pos = sessions.get_indexer(pd.to_datetime(s))
    pos = np.where(pos >= 0, pos + k, -1)
    pos = np.where((pos >= 0) & (pos < len(sessions)), pos, -1)
    return pd.Series(np.where(pos >= 0, sessions[np.clip(pos, 0, None)], pd.NaT), index=s.index)


def h20_events(d, cal: pd.DataFrame, n_before: int, window: str) -> pd.DataFrame:
    """Decision at the close n_before sessions before the reaction session; enter next open.
    window 'pre': exit at the last pre-announcement close (AMC: announcement session; else the session
    before); 'through': exit at the close of the reaction session."""
    e = cal.copy()
    e["decision"] = _shift_session(d, e["react_session"], -n_before)
    e["entry"] = _shift_session(d, e["decision"], 1)
    last_pre = np.where(e["timing"] == "AMC", e["ann_session"], _shift_session(d, e["react_session"], -1))
    e["exit"] = last_pre if window == "pre" else e["react_session"]
    e = e.dropna(subset=["decision", "entry", "exit"])
    e = e[pd.to_datetime(e["exit"]) >= pd.to_datetime(e["entry"])]
    u = d.u_liquid
    ci = u.columns.get_indexer(e["act_symbol"])
    di = u.index.get_indexer(pd.to_datetime(e["decision"]))
    inu = (ci >= 0) & (di >= 0)
    inu[inu] = u.to_numpy()[di[inu], ci[inu]]
    return e[inu]


def summarize_events(d, e: pd.DataFrame, ret: pd.Series, date_col: str = "entry", dataset: str = "calendar") -> dict:
    from quantlab.alpha import splits
    from quantlab.validation.stats import newey_west_tstat
    out = {}
    x = pd.DataFrame({"r": ret, "dt": pd.to_datetime(e[date_col])}).dropna()
    for sp in ("TRAIN", "VALIDATION", "OOS"):
        a, b = splits.window(dataset, sp)
        xs = x[(x["dt"] >= a) & (x["dt"] <= b)]
        if len(xs) < 30:
            out[sp] = {"n_events": int(len(xs))}
            continue
        wk = xs.groupby(xs["dt"].dt.to_period("W-FRI"))["r"].mean()
        t = newey_west_tstat(wk.to_numpy(), min_obs=20)
        srt = np.sort(xs["r"].to_numpy())[::-1]
        out[sp] = {"n_events": int(len(xs)), "mean_bps": float(xs["r"].mean() * 1e4), "median_bps": float(xs["r"].median() * 1e4),
                   "hit_rate": float((xs["r"] > 0).mean()), "weekly_t_nw": t.t, "n_weeks": int(len(wk)),
                   "mean_bps_without_top5pct": float(srt[int(0.05 * len(srt)):].mean() * 1e4)}
    return out


# --- H26 pre-earnings straddle (options store) -------------------------------------------------------
def straddle_events(d, cal: pd.DataFrame, k_back: int, placebo_shift: int = 0) -> pd.DataFrame:
    """For each clean event: exit snapshot = last chain snapshot whose session <= the last
    pre-announcement close; entry snapshot = k_back snapshots earlier. Expiry = first expiry after the
    reaction session listed at entry. Strike = ATM at entry. Prices: entry ASK, exit BID of the SAME two
    contracts. ``placebo_shift`` > 0 moves every event that many sessions EARLIER (non-event control)."""
    from quantlab.alpha.options_features import _session_map
    from quantlab.alpha.options_store import options_dir
    sessions = d.p.dates
    ev = cal.copy()
    last_pre = pd.Series(np.where(ev["timing"] == "AMC", ev["ann_session"], _shift_session(d, ev["react_session"], -1)),
                         index=ev.index)
    react = pd.to_datetime(ev["react_session"])
    if placebo_shift:
        last_pre = _shift_session(d, last_pre, -placebo_shift)
        react = _shift_session(d, react, -placebo_shift)
    ev = ev.assign(last_pre=pd.to_datetime(last_pre), react=pd.to_datetime(react)).dropna(subset=["last_pre", "react"])
    rows = []
    files = sorted(options_dir().glob("chain_*-*.parquet"))
    chain = pd.concat([pd.read_parquet(f, columns=["date", "act_symbol", "expiration", "strike", "cp", "bid", "ask"])
                       for f in files], ignore_index=True)
    chain["act_symbol"] = chain["act_symbol"].astype(str)
    snaps = pd.DatetimeIndex(sorted(chain["date"].unique()))
    snap_sess = pd.Series(_session_map(sessions, pd.Series(snaps)).to_numpy(), index=snaps)
    chain = chain[chain["act_symbol"].isin(set(ev["act_symbol"]))]
    g = dict(tuple(chain.groupby("act_symbol")))
    spot = d.p["close"]
    for r in ev.itertuples():
        c = g.get(r.act_symbol)
        if c is None:
            continue
        ok_snaps = snap_sess[snap_sess <= r.last_pre]
        if ok_snaps.empty:
            continue
        exit_snap = ok_snaps.index[-1]
        if (r.last_pre - ok_snaps.iloc[-1]).days > 4:          # no snapshot close to the event
            continue
        pos = snaps.get_loc(exit_snap) - k_back
        if pos < 0:
            continue
        entry_snap = snaps[pos]
        ce = c[c["date"] == entry_snap]
        cx = c[c["date"] == exit_snap]
        if ce.empty or cx.empty:
            continue
        exps = sorted(e for e in ce["expiration"].unique() if pd.Timestamp(e) > r.react)
        if not exps:
            continue
        expiry = exps[0]
        sess_entry = snap_sess[entry_snap]
        if r.act_symbol not in spot.columns or not np.isfinite(spot.at[sess_entry, r.act_symbol]):
            continue
        s0 = spot.at[sess_entry, r.act_symbol]
        ce1 = ce[(ce["expiration"] == expiry) & (ce["bid"] > 0) & (ce["ask"] > ce["bid"])]
        strikes = ce1.groupby("strike")["cp"].nunique()
        strikes = strikes[strikes == 2].index
        if len(strikes) == 0:
            continue
        k = strikes[np.argmin(np.abs(strikes - s0))]
        leg_e = ce1[ce1["strike"] == k].set_index("cp")
        leg_x = cx[(cx["expiration"] == expiry) & (cx["strike"] == k)].set_index("cp")
        if not {"C", "P"} <= set(leg_x.index) or not {"C", "P"} <= set(leg_e.index):
            continue
        cost_in = float(leg_e.loc["C", "ask"] + leg_e.loc["P", "ask"])
        val_out = float(leg_x.loc["C", "bid"] + leg_x.loc["P", "bid"])
        mid_in = float((leg_e.loc["C", "ask"] + leg_e.loc["C", "bid"] + leg_e.loc["P", "ask"] + leg_e.loc["P", "bid"]) / 2)
        mid_out = float((leg_x.loc["C", "ask"] + leg_x.loc["C", "bid"] + leg_x.loc["P", "ask"] + leg_x.loc["P", "bid"]) / 2)
        rows.append({"act_symbol": r.act_symbol, "event": r.date, "entry_snap": entry_snap, "exit_snap": exit_snap,
                     "entry": sess_entry, "expiry": expiry, "strike": k, "spot": s0,
                     "ret_ask_bid": val_out / cost_in - 1, "ret_mid": mid_out / mid_in - 1,
                     "spread_frac_in": cost_in / mid_in - 1, "timing": r.timing})
    return pd.DataFrame(rows)


# --- runners ---------------------------------------------------------------------------------------
def _event_family(d, family: str, hid: str, variants: dict, date_col: str = "entry", oos_rerun_reason: str | None = None) -> dict:
    """variants: name -> (spec, events DataFrame, returns Series, returns at 2x costs). Selection on TRAIN
    mean; OOS reported once for the selected variant; every variant is a trial (ledger)."""
    import math
    from quantlab.alpha import registry
    from quantlab.validation.stats import benjamini_hochberg
    res, pv = {}, []
    for name, (spec, ev, r, r2) in variants.items():
        s = summarize_events(d, ev, r, date_col)
        s2 = summarize_events(d, ev, r2, date_col)
        dev_rows = []
        for sp in ("TRAIN", "VALIDATION"):
            if "weekly_t_nw" in s.get(sp, {}):
                dev_rows.append(s[sp])
        res[name] = {"summary": {k: v for k, v in s.items() if k != "OOS"},
                     "train_2x_costs_mean_bps": s2.get("TRAIN", {}).get("mean_bps")}
        registry.append_run(hypothesis_id=hid, family=family, spec={"variant": name, **spec}, split="DEV",
                            metrics=res[name], data=d.manifest)
        t = s.get("TRAIN", {}).get("weekly_t_nw")
        from scipy import stats as ss
        pv.append(float(ss.norm.sf(t)) if t is not None and np.isfinite(t) else 1.0)
    best = max(res, key=lambda k: res[k]["summary"].get("TRAIN", {}).get("mean_bps") or -1e9)
    spec, ev, r, r2 = variants[best]
    full = summarize_events(d, ev, r, date_col)
    full2 = summarize_events(d, ev, r2, date_col)
    oos = {"selected": best, **full.get("OOS", {}), "oos_2x_costs_mean_bps": full2.get("OOS", {}).get("mean_bps")}
    if oos_rerun_reason:
        oos["oos_rerun_reason"] = oos_rerun_reason
    registry.append_run(hypothesis_id=hid, family=family, spec={"variant": best, **spec}, split="OOS", metrics=oos, data=d.manifest,
                        new_variant_ok=bool(oos_rerun_reason))
    rej, adj = benjamini_hochberg(pv, 0.05)
    tr = full.get("TRAIN", {}); va = full.get("VALIDATION", {}); oo = full.get("OOS", {})
    ev_ = registry.Evidence(
        data_ok=bool(tr.get("n_events", 0) >= 100),
        dev_t_net=min(tr.get("weekly_t_nw") or -9, va.get("weekly_t_nw") or -9) if va else tr.get("weekly_t_nw"),
        dev_t_gross=None, oos_t_net=oo.get("weekly_t_nw"), oos_mean_net=(oo.get("mean_bps") or 0) / 1e4,
        oos_years_positive_frac=None, deflated_sharpe_prob=None, spa_p=None, pbo=None,
        survives_2x_costs=(full2.get("OOS", {}).get("mean_bps") or -1) > 0,
        survives_top5_removal=(oo.get("mean_bps_without_top5pct") or -1) > 0)
    cls, why = registry.classify(ev_)
    out = {"family": family, "variants": res, "selected": best, "oos": oos, "bh_train_rejections": int(np.sum(rej)),
           "evidence": ev_.__dict__,
           "classification": cls, "classification_reason": why + " (event study: DSR/SPA not applicable; BH across variants on TRAIN)"}
    from quantlab.alpha.experiment import RESULTS
    import json
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{family}.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    return out


def run_h20(d, cal: pd.DataFrame | None = None) -> dict:
    """EAP n=1 'pre' is empty by construction (entry would be at/after the last pre-announcement close)."""
    cal = clean_calendar(d) if cal is None else cal
    variants = {}
    for n in (1, 3, 5):
        for window in ("pre", "through"):
            ev = h20_events(d, cal, n, window)
            spec = {"signal": f"announcement within {n} sessions", "entry": "next open", "exit": f"{window}-announcement close",
                    "hedge": "beta x SPY", "costs": "tier, both sides", "calendar": "DoltHub, PIT_ASSUMED 2020-24"}
            variants[f"EAP_n{n}_{window}"] = (spec, ev, event_returns(d, ev, "entry", "exit"),
                                              event_returns(d, ev, "entry", "exit", cost_mult=2.0))
    return _event_family(d, "H20_earnings_announcement_premium", "H20", variants,
                         oos_rerun_reason="first run used equity splits (TRAIN 2016-19, no calendar): the recorded OOS spec "
                                          "had 0 events, so no OOS returns were ever observed; rerun on calendar splits")


def surprise_events(d, cal: pd.DataFrame) -> pd.DataFrame:
    eh = pd.read_parquet(options_dir() / "eps_history.parquet")
    eh["period_end_date"] = pd.to_datetime(eh["period_end_date"])
    eh = eh.dropna(subset=["reported", "estimate"])
    c = cal[["act_symbol", "date", "react_session", "timing"]].sort_values("date")
    m = pd.merge_asof(eh.sort_values("period_end_date"), c.rename(columns={"date": "ann_date"}),
                      left_on="period_end_date", right_on="ann_date", by="act_symbol", direction="forward",
                      tolerance=pd.Timedelta(days=100))
    m = m.dropna(subset=["react_session"])
    close = d.p["close"]
    ci = close.columns.get_indexer(m["act_symbol"])
    pre = _shift_session(d, m["react_session"], -1)
    si = d.p.dates.get_indexer(pd.to_datetime(pre))
    ok = (ci >= 0) & (si >= 0)
    px = np.where(ok, close.to_numpy()[np.clip(si, 0, None), np.clip(ci, 0, None)], np.nan)
    m["surprise"] = (pd.to_numeric(m["reported"]) - pd.to_numeric(m["estimate"])) / px
    m["entry"] = _shift_session(d, m["react_session"], 1)
    return m.dropna(subset=["surprise", "entry"])


def run_h21(d, cal: pd.DataFrame | None = None) -> dict:
    cal = clean_calendar(d) if cal is None else cal
    ev = surprise_events(d, cal).sort_values("entry").reset_index(drop=True)
    # PIT decile breakpoints: percentile of each surprise within the previous 63 sessions' surprises
    s = ev.set_index(pd.to_datetime(ev["entry"]))["surprise"]
    pct = []
    vals, times = s.to_numpy(), s.index.to_numpy()
    for i in range(len(vals)):
        lo = np.searchsorted(times, times[i] - np.timedelta64(91, "D"), side="left")
        hist = vals[lo:i]
        pct.append((hist < vals[i]).mean() if len(hist) >= 200 else np.nan)
    ev["pct"] = pct
    variants = {}
    for hold in (21, 63):
        ev_h = ev.copy()
        ev_h["exit"] = _shift_session(d, ev_h["entry"], hold - 1)
        ev_h = ev_h.dropna(subset=["exit", "pct"])
        gross = event_returns(d, ev_h, "entry", "exit", cost_mult=0.0)          # beta-hedged, no costs
        c1 = gross - event_returns(d, ev_h, "entry", "exit")                     # round-trip cost, 1x
        c2 = gross - event_returns(d, ev_h, "entry", "exit", cost_mult=2.0)      # round-trip cost, 2x
        sign = np.where(ev_h["pct"] >= 0.9, 1.0, np.where(ev_h["pct"] <= 0.1, -1.0, np.nan))
        sel = ~np.isnan(sign)
        rr = pd.Series(sign * gross - c1, index=ev_h.index)[sel]                 # both sides pay costs
        rr2 = pd.Series(sign * gross - c2, index=ev_h.index)[sel]
        spec = {"signal": "EPS surprise / price, decile vs trailing 91 days", "entry": "open after the reaction session",
                "exit": f"close after {hold} sessions", "position": "long top decile, short bottom decile (beta-hedged)",
                "costs": "tier both sides", "data": "eps_history reported vs estimate (PIT_ASSUMED)"}
        variants[f"SUE_hold{hold}"] = (spec, ev_h[sel], rr, rr2)
    return _event_family(d, "H21_eps_surprise_drift", "H21", variants,
                         oos_rerun_reason="first run used equity splits; it DID show OOS 2022-24 for SUE_hold21 once "
                                          "(mean +89.6 bps, weekly t -0.01, -228 bps without the top 5%): CONTAMINATED, reported as such")


# --- H23 analyst revision momentum (cross-sectional, daily) --------------------------------------------
def revision_signal(d, window: int) -> pd.DataFrame:
    """Change in the 'Current Year' consensus EPS over ``window`` sessions, divided by price, only where
    the fiscal period (period_end_date) is the same at both ends (a fiscal-year roll is not a revision).
    Snapshots are mapped to the first session on/after their date (forward-filled up to 10 sessions)."""
    est = pd.read_parquet(options_dir() / "eps_estimate.parquet", columns=["date", "act_symbol", "period", "period_end_date", "consensus"])
    est = est[(est["period"] == "Current Year") & est["act_symbol"].isin(d.p.symbols)].copy()
    est["date"] = pd.to_datetime(est["date"])
    est["consensus"] = pd.to_numeric(est["consensus"], errors="coerce")
    est["pend"] = pd.to_datetime(est["period_end_date"]).astype("int64")
    sessions = d.p.dates
    pos = sessions.searchsorted(est["date"].to_numpy(), side="left")
    est = est[pos < len(sessions)]
    est["session"] = sessions[pos[pos < len(sessions)]]
    est = est.sort_values("date").drop_duplicates(["session", "act_symbol"], keep="last")
    cons = est.pivot(index="session", columns="act_symbol", values="consensus").reindex(sessions).ffill(limit=10)
    pend = est.pivot(index="session", columns="act_symbol", values="pend").reindex(sessions).ffill(limit=10)
    same = pend == pend.shift(window)
    rev = (cons - cons.shift(window)).where(same) / d.p["close"].reindex(columns=cons.columns)
    return rev.reindex(columns=d.p.symbols)


def run_h23(d) -> dict:
    from quantlab.alpha.engine import quantile_weights
    from quantlab.alpha.experiment import Variant, run_family
    from quantlab.alpha.experiments.reversal import context
    vs = []
    for w in (21, 63):
        spec = {"universe": "liquid survivorship-free", "signal": f"change in Current-Year consensus EPS over {w} sessions / price",
                "entry": "next open", "position": "decile long-short", "sizing": "equal", "exit": "overlapping 21-session hold",
                "costs": "CostModel tiers; borrow 50bps/yr", "benchmark": "cash",
                "data": "DoltHub eps_estimate (archive import before 2021-04, commit-verified after)"}
        vs.append(Variant(f"REV_{w}", spec, (lambda w=w: quantile_weights(revision_signal(d, w), d.u_liquid, q=0.1)), holding=21))
    return run_family(family="H23_revision_momentum", hypothesis_id="H23", variants=vs, ctx=context(d))


def run_h26(d, cal: pd.DataFrame | None = None) -> dict:
    """Pre-earnings straddle replication: k_back = 1 | 2 | 3 snapshots; placebo 63 sessions earlier."""
    from quantlab.alpha import registry, splits
    from quantlab.validation.stats import newey_west_tstat
    import json
    cal = clean_calendar(d) if cal is None else cal
    out = {"family": "H26_pre_earnings_straddle", "variants": {}}
    for k in (1, 2, 3):
        ev = straddle_events(d, cal, k)
        pl = straddle_events(d, cal, k, placebo_shift=63)
        res = {}
        for name, x in (("event", ev), ("placebo", pl)):
            if x.empty:
                continue
            x = x.assign(dt=pd.to_datetime(x["entry"]))
            for sp in ("TRAIN", "VALIDATION", "OOS"):
                a, b = splits.window("options", sp)
                xs = x[(x["dt"] >= a) & (x["dt"] <= b)]
                if len(xs) < 30:
                    continue
                wk = xs.groupby(xs["dt"].dt.to_period("W-FRI"))["ret_ask_bid"].mean()
                res[f"{name}_{sp}"] = {"n": int(len(xs)), "mean_ask_bid": float(xs["ret_ask_bid"].mean()),
                                       "median_ask_bid": float(xs["ret_ask_bid"].median()), "mean_mid": float(xs["ret_mid"].mean()),
                                       "hit_rate": float((xs["ret_ask_bid"] > 0).mean()), "t_weekly_nw": newey_west_tstat(wk.to_numpy(), min_obs=15).t,
                                       "median_spread_frac_in": float(xs["spread_frac_in"].median())}
            x.to_parquet(registry.DIR.parent.parent / "var" / "alpha" / f"h26_{name}_k{k}.parquet") if False else None
        out["variants"][f"k{k}"] = res
        registry.append_run(hypothesis_id="H26", family="H26_pre_earnings_straddle",
                            spec={"variant": f"k{k}", "entry": f"{k} snapshots before the last pre-announcement close, at the ask",
                                  "exit": "last pre-announcement snapshot, at the bid", "structure": "ATM straddle, first expiry after the event"},
                            split="ALL", metrics=res, data=d.manifest)
    (registry.DIR / "results").mkdir(parents=True, exist_ok=True)
    (registry.DIR / "results" / "H26_pre_earnings_straddle.json").write_text(json.dumps(registry._clean(out), indent=1, default=str))
    return out
