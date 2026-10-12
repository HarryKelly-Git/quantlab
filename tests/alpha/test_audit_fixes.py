"""Tests for the 2026-10 adversarial-audit fixes (C1, C2, M1-M6 and the point-in-time gaps it found):
entity resolution by date, twin de-duplication, the one-time OOS override, PBO -> G, Driscoll-Kraay
standard errors, and truncation-invariance of every signal the audit listed as untested
(gap_weights, conditioned_weights, revision_signal, the H21 surprise percentile, iv_pctile, opt_mom,
earn_in_window, straddle_events)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha import registry, research_data
from quantlab.alpha.entities import apply_aliases, dedupe_twins, map_events, resolve
from quantlab.alpha.experiments import anomalies, earnings, momentum
from quantlab.alpha.experiments import options_vol as ov
from quantlab.alpha.synthetic import make_panel, truncate


def _rd(p):
    rd = research_data.from_panel(p, manifest={"synthetic": True})
    rd.u_liquid = (p["close"].notna() & (p["n_hist"] >= 60)) & pd.DataFrame(
        np.broadcast_to(p.master.set_index("symbol")["sec_type"].reindex(p.symbols).eq("COMMON").to_numpy(), p["close"].shape),
        index=p.dates, columns=p.symbols)
    rd.u_large = rd.u_liquid
    return rd


@pytest.fixture(scope="module")
def full():
    p = make_panel(n_stocks=40, n_days=600, seed=5)
    return p, _rd(p)


# --- entity resolution (C2 / M1) -----------------------------------------------------------------------
def test_resolve_uses_the_entity_that_held_the_ticker_on_each_date():
    d = pd.bdate_range("2019-01-01", periods=10)
    bars = pd.concat([
        pd.DataFrame({"symbol": "CZR@2019-01-08", "date": d[:6]}),       # old company used CZR until it delisted
        pd.DataFrame({"symbol": "CZR", "date": d}),                      # current company (history under ERI)
        pd.DataFrame({"symbol": "AAA", "date": d}),
    ])
    r = resolve(bars).set_index(["ticker", "date"])["entity"]
    assert r[("CZR", d[0])] == "CZR@2019-01-08"
    assert r[("CZR", d[5])] == "CZR@2019-01-08"
    assert r[("CZR", d[6])] == "CZR"
    assert r[("AAA", d[3])] == "AAA"
    ev = pd.DataFrame({"act_symbol": ["CZR", "CZR", "ZZZ"], "session": [d[2], d[8], d[2]]})
    m = map_events(ev, resolve(bars), "act_symbol", "session")
    assert list(m.iloc[:2]) == ["CZR@2019-01-08", "CZR"] and pd.isna(m.iloc[2])


def test_dedupe_twins_rename_contained_and_coincidence():
    d = pd.bdate_range("2019-01-01", periods=30)
    rng = np.random.default_rng(0)
    px = 20 + rng.normal(0, 1, 30).cumsum()
    vol = rng.integers(10_000, 50_000, 30).astype(float)
    rows = [
        pd.DataFrame({"symbol": "NEW", "date": d, "close": px, "volume": vol}),                  # renamed company, full history
        pd.DataFrame({"symbol": "OLD", "date": d[:12], "close": px[:12], "volume": vol[:12]}),    # old ticker: fully contained
        # merger survivor: its own history, then identical to SURV for the last 10 days
        pd.DataFrame({"symbol": "PART", "date": d[5:], "close": np.r_[np.linspace(5, 6, 15), px[20:]],
                      "volume": np.r_[np.full(15, 999.0), vol[20:]]}),
        # coincidence: same close/volume as NEW on 2 days only -> not twins
        pd.DataFrame({"symbol": "COIN", "date": d, "close": np.where(np.arange(30) < 2, px, 99.0), "volume": np.where(np.arange(30) < 2, vol, 5.0)}),
    ]
    bars = pd.concat(rows, ignore_index=True)
    out, aliases, renamed, rep = dedupe_twins(bars, min_days=5)
    left = out.groupby("symbol").size()
    assert "OLD" not in left.index                              # contained twin dropped entirely
    assert left["NEW"] == 30                                    # the key with most bars is kept
    assert left["COIN"] == 30                                   # 2 identical days = coincidence
    assert left["PART"] == 15                                   # loses its 10 identical days only
    assert "PART" in renamed                                    # its last bar was a twin day: not a delisting
    assert "OLD" not in renamed                                 # dropped entirely instead
    a = aliases.set_index(["symbol", "date"])["keeper"]
    assert a[("OLD", d[3])] == "NEW" and a[("PART", d[25])] == "NEW"
    ent = apply_aliases(pd.Series(["OLD", "PART", "PART", "COIN"]), pd.Series([d[3], d[25], d[6], d[6]]), aliases)
    assert list(ent) == ["NEW", "NEW", "PART", "COIN"]


# --- registry: one-time override (M6) and PBO (minor) ----------------------------------------------------
def test_registry_override_is_one_time_and_data_changes_need_it(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "DIR", tmp_path)
    monkeypatch.setattr(registry, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(registry, "QUEUE", tmp_path / "queue.json")
    kw = dict(hypothesis_id="H9", family="f", split="OOS", metrics={})
    registry.append_run(spec={"v": 1}, data={"panel": 1}, **kw)
    registry.append_run(spec={"v": 1}, data={"panel": 1}, **kw)                 # identical rerun: allowed
    with pytest.raises(registry.OOSReuseError):
        registry.append_run(spec={"v": 1}, data={"panel": 2}, **kw)             # same spec, new data: needs a reason
    with pytest.raises(registry.OOSReuseError):
        registry.append_run(spec={"v": 1}, data={"panel": 2}, oos_override="bug fix", **kw)   # too short
    reason = "audit-2026-10 M2: twin entities removed from the panel; same pre-registered spec"
    row = registry.append_run(spec={"v": 1}, data={"panel": 2}, oos_override=reason, **kw)
    assert row["oos_override"] == reason
    with pytest.raises(registry.OOSReuseError):
        registry.append_run(spec={"v": 2}, data={"panel": 3}, oos_override=reason, **kw)       # reason reused
    E = registry.Evidence
    assert registry.classify(E(True, 2.5, 3.0, None, None, None, 0.97, 0.01, 0.6, True, True))[0] == "G"


# --- Driscoll-Kraay (M4) -------------------------------------------------------------------------------
def test_driscoll_kraay_lag0_is_week_cluster_and_grows_with_overlap():
    rng = np.random.default_rng(1)
    T, n = 300, 20
    wk = np.repeat(np.arange(T), n)
    shock = np.convolve(rng.normal(0, 1, T + 5), np.ones(6), "valid")[:T]          # 6-week overlapping outcome
    x = rng.normal(0, 1, T * n)
    y = 0.5 * x + np.repeat(shock, n) + rng.normal(0, 0.5, T * n)
    X = np.column_stack([np.ones(T * n), x + np.repeat(shock, n) * 0.3])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ coef
    se0 = ov.driscoll_kraay_se(X, res, wk, 0)
    meat = np.zeros((2, 2))
    for w in np.unique(wk):
        sc = X[wk == w].T @ res[wk == w]
        meat += np.outer(sc, sc)
    XtX_inv = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(se0, np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv)), rtol=1e-10)
    assert ov.driscoll_kraay_se(X, res, wk, 6)[0] > 1.5 * se0[0]        # serial correlation widens the SE


# --- options-lab features: PIT (opt_mom, iv_pctile, earn_in_window) ------------------------------------
def _obs(seed=2, n_ent=4, n_weeks=160):
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2019-01-04", periods=n_weeks * 5)[::5]
    rows = []
    for e in range(n_ent):
        for s in sessions:
            exp = s + pd.Timedelta(days=30)
            rows.append({"entity": f"E{e}", "act_symbol": f"T{e}", "session": s, "expiration": exp, "exp_session": exp,
                         "iv_atm": 0.3 + 0.05 * rng.normal(), "ret_hold_mid": rng.normal(-0.05, 0.4)})
    return pd.DataFrame(rows)


def test_option_features_truncation_invariant():
    m = _obs()
    m = m.sort_values(["entity", "session"])
    full = pd.concat([ov.option_momentum(m), ov.iv_percentile(m).rename("iv_pctile")], axis=1)
    checked = 0
    for D in (m["session"].sort_values().unique()[[70, 120, 159]]):
        D = pd.Timestamp(D)
        t = m[m["session"] <= D].copy()
        t.loc[pd.to_datetime(t["exp_session"]) > D, "ret_hold_mid"] = np.nan    # outcomes unknown at D
        tr = pd.concat([ov.option_momentum(t), ov.iv_percentile(t).rename("iv_pctile")], axis=1)
        rows = t.index[t["session"] == D]
        a, b = full.loc[rows], tr.loc[rows]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
        checked += int(a.notna().sum().sum())
    assert checked > 0


def test_earnings_in_window_matches_the_company_not_the_ticker():
    s = pd.Timestamp("2021-03-01")
    m = pd.DataFrame({"entity": ["A@2021-06-01", "A"], "act_symbol": ["A", "A"], "session": [s, s],
                      "expiration": [s + pd.Timedelta(days=30)] * 2})
    cal = pd.DataFrame({"entity": ["A@2021-06-01"], "react_session": [s + pd.Timedelta(days=10)]})
    assert list(ov.earnings_in_window(m, cal)) == [True, False]


# --- equity signals the audit listed as untested ----------------------------------------------------------
def test_gap_weights_use_nothing_after_the_open(full):
    """Decision row D-1 trades at the open of D using the gap into D's open. Scrambling everything about D
    except its open (close, high, low, volume, intraday return) and everything after D must not change it."""
    p, d = full
    for D in p.dates[[300, 450]]:
        i = p.dates.get_loc(D)
        w_full = anomalies.gap_weights(d, d.u_large, "equal").iloc[i - 1]
        pt = truncate(p, D)
        f = dict(pt.f)
        for k in ("close", "high", "low", "volume", "adj_close", "adj_high", "adj_low", "ret_id", "ret_cc", "dollar_volume"):
            x = f[k].copy()
            x.iloc[-1] = x.iloc[-1] * 1.7 + 3.0
            f[k] = x
        pt.f = f
        dt = _rd(pt)
        w_t = anomalies.gap_weights(dt, dt.u_large, "equal").iloc[i - 1]
        np.testing.assert_allclose(w_full.to_numpy(), w_t.reindex(w_full.index).to_numpy(), atol=1e-12)
        assert (w_full.abs() > 0).sum() > 0


@pytest.mark.parametrize("cond", ["sector_leadership", "volume_rising", "calm_market", "strong_breadth"])
def test_conditioned_momentum_weights_truncation_invariant(full, cond):
    p, d = full
    w_full = momentum.conditioned_weights(d, cond)
    for D in p.dates[[420, 599]]:
        w_t = momentum.conditioned_weights(_rd(truncate(p, D)), cond)
        np.testing.assert_allclose(w_full.loc[D].to_numpy(), w_t.loc[D].reindex(w_full.columns).to_numpy(), atol=1e-12)


def _estimates(p, seed=4):
    rng = np.random.default_rng(seed)
    snaps = p.dates[::3]
    rows = []
    for s in [x for x in p.symbols if x.startswith("S")]:
        lvl = 2.0
        for t in snaps:
            lvl += rng.normal(0, 0.05)
            fy = pd.Timestamp(f"{t.year}-12-31")
            rows.append({"date": t + pd.Timedelta(days=int(rng.integers(0, 2))), "act_symbol": s, "period": "Current Year",
                         "period_end_date": fy, "consensus": lvl})
    return pd.DataFrame(rows)


def test_revision_signal_truncation_invariant(full):
    p, d = full
    est = _estimates(p)
    r_full = earnings.revision_signal(d, 21, est=est)
    checked = 0
    for D in p.dates[[300, 450, 599]]:
        dt = _rd(truncate(p, D))
        r_t = earnings.revision_signal(dt, 21, est=est[est["date"] <= D])
        a, b = r_full.loc[D], r_t.loc[D].reindex(r_full.columns)
        assert (a.notna() == b.notna()).all()
        np.testing.assert_allclose(a[a.notna()].to_numpy(), b[a.notna()].to_numpy(), rtol=1e-12)
        checked += int(a.notna().sum())
    assert checked > 0


def test_surprise_percentile_truncation_invariant():
    rng = np.random.default_rng(7)
    ent = pd.Timestamp("2020-01-02") + pd.to_timedelta(np.sort(rng.integers(0, 900, 3000)), unit="D")
    ev = pd.DataFrame({"entry": ent, "surprise": rng.normal(0, 0.01, 3000)}).sample(frac=1.0, random_state=1)
    full_pct = earnings.surprise_percentile(ev)
    for D in (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-03-15")):
        t = ev[ev["entry"] <= D]
        tp = earnings.surprise_percentile(t)
        np.testing.assert_allclose(full_pct.loc[t.index].to_numpy(), tp.to_numpy(), equal_nan=True)
        assert tp.notna().sum() > 0


# --- H26 straddle events: decisions use only entry-time data; placebo avoids real events (C1) ----------------
def _chain_and_calendar(p, stock):
    from quantlab.options.pricing import bs_price
    close = p["close"][stock]
    snaps = [t for t in p.dates if t.weekday() in (0, 2, 4)]
    rows = []
    for t in snaps:
        s0 = float(close.loc[t])
        if not np.isfinite(s0):
            continue
        fridays = [t + pd.Timedelta(days=k) for k in range(1, 40) if (t + pd.Timedelta(days=k)).weekday() == 4][:4]
        for ex in fridays:
            T_ = (ex - t).days / 365
            for k in np.arange(np.floor(0.8 * s0), np.ceil(1.2 * s0) + 1):
                for cp, kind in (("C", "call"), ("P", "put")):
                    mid = float(bs_price(s0, k, T_, 0.35, 0.0, kind))
                    rows.append({"date": t, "act_symbol": stock, "expiration": ex, "strike": float(k), "cp": cp,
                                 "bid": max(mid * 0.97, 0.01), "ask": mid * 1.03 + 0.01})
    chain = pd.DataFrame(rows)
    react = p.dates[100::63]
    cal = pd.DataFrame({"act_symbol": stock, "entity": stock, "date": react, "ann_session": react, "react_session": react,
                        "timing": "BMO"})
    return chain, cal


def test_straddle_events_decision_pit_and_placebo_exclusion(full):
    p, d = full
    stock = "S001"
    chain, cal = _chain_and_calendar(p, stock)
    ev = earnings.straddle_events(d, cal, 2, chain=chain)
    assert len(ev) >= 4
    # perturb every quote that is neither on an entry nor an exit snapshot, and every close after entry
    keep = set(ev["entry_snap"]) | set(ev["exit_snap"])
    ch2 = chain.copy()
    other = ~ch2["date"].isin(keep)
    ch2.loc[other, ["bid", "ask"]] *= 3.0
    ev2 = earnings.straddle_events(d, cal, 2, chain=ch2)
    pd.testing.assert_frame_equal(ev.reset_index(drop=True), ev2.reset_index(drop=True))
    # placebo: 31 sessions before each real event, mid-quarter -> kept; add a real event 10 sessions from
    # one placebo date -> that placebo is excluded
    pl = earnings.straddle_events(d, cal, 2, placebo=True, chain=chain)
    assert len(pl) >= 3
    first_pl = pd.Timestamp(pl["entry"].min())
    i = p.dates.get_loc(first_pl)
    extra = p.dates[i + 10]
    cal2 = pd.concat([cal, pd.DataFrame({"act_symbol": stock, "entity": stock, "date": [extra], "ann_session": [extra],
                                         "react_session": [extra], "timing": "BMO"})], ignore_index=True)
    real = {stock: np.sort(pd.to_datetime(cal2["react_session"]).to_numpy())}
    pl3 = earnings.straddle_events(d, cal, 2, placebo=True, chain=chain, real_events=real)
    assert first_pl not in set(pd.to_datetime(pl3["entry"]))
    assert len(pl3) == len(pl) - 1


# --- verification-review follow-ups: symbol changes, false twins, override tags ------------------------------
def _renames(rows):
    return pd.DataFrame(rows, columns=["old_symbol", "new_symbol", "old_cusip", "new_cusip", "process_date"]).assign(
        process_date=lambda x: pd.to_datetime(x["process_date"]))


def test_resolve_follows_symbol_changes_both_ways():
    d = pd.bdate_range("2019-06-03", "2023-12-29")
    bars = pd.concat([
        pd.DataFrame({"symbol": "PARA", "date": d[d >= "2021-01-04"]}),      # today's PARA: traded as BNZI until 2026
        pd.DataFrame({"symbol": "CBS@2019-12-01", "date": d}),               # CBS -> VIAC -> PARA (old Paramount)
        pd.DataFrame({"symbol": "OLD@2020-01-10", "date": d[d <= "2021-06-30"]}),   # last seen 2020-01-10, no rename
    ])
    rn = _renames([("BNZI", "PARA", "X1", "X1", "2026-08-07"), ("CBS", "VIAC", "Y1", "Y1", "2020-02-13"),
                   ("VIAC", "PARA", "Y1", "Y1", "2022-02-17")])
    r = resolve(bars, rn).set_index(["ticker", "date"])["entity"]
    assert r[("PARA", pd.Timestamp("2023-06-01"))] == "CBS@2019-12-01"     # not Banzai
    assert r[("BNZI", pd.Timestamp("2023-06-01"))] == "PARA"
    assert r[("VIAC", pd.Timestamp("2021-06-01"))] == "CBS@2019-12-01"
    assert r[("CBS", pd.Timestamp("2019-07-01"))] == "CBS@2019-12-01"
    assert ("CBS", pd.Timestamp("2021-06-01")) not in r.index
    assert r[("OLD", pd.Timestamp("2020-01-16"))] == "OLD@2020-01-10"     # last_seen + 7 days tolerance
    assert ("OLD", pd.Timestamp("2020-03-02")) not in r.index              # later bars claim nothing
    # CUSIP-only records change no ticker
    rn2 = pd.concat([rn, _renames([("PARA", "PARA", "Z9", "X1", "2026-09-01")])], ignore_index=True)
    r2 = resolve(bars, rn2).set_index(["ticker", "date"])["entity"]
    assert r2[("PARA", pd.Timestamp("2023-06-01"))] == "CBS@2019-12-01"


def test_dedupe_requires_most_common_days_identical():
    d = pd.bdate_range("2019-01-01", periods=100)
    rng = np.random.default_rng(3)
    a = pd.DataFrame({"symbol": "FUNDA", "date": d, "close": 10 + rng.normal(0, 0.1, 100).round(2), "volume": rng.integers(100, 900, 100).astype(float)})
    b = a.copy()
    b["symbol"] = "FUNDB"
    scattered = np.arange(100) % 15 == 0                                   # identical on 7 scattered days only
    b.loc[~scattered, "close"] = b.loc[~scattered, "close"] + 0.37
    out, aliases, renamed, rep = dedupe_twins(pd.concat([a, b], ignore_index=True), min_days=5)
    assert len(out) == 200 and len(aliases) == 0


def test_registry_override_tags_and_identical_reruns(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "DIR", tmp_path)
    monkeypatch.setattr(registry, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(registry, "QUEUE", tmp_path / "queue.json")
    kw = dict(hypothesis_id="H8", family="f", split="OOS", metrics={})
    registry.append_run(spec={"v": 1}, data={"panel": 1}, **kw)
    registry.append_run(spec={"v": 1}, data={"panel": 2}, oos_override="audit-2026-10: twins removed, same spec", **kw)
    registry.append_run(spec={"v": 1}, data={"panel": 2}, **kw)              # identical to an earlier run: allowed
    registry.append_run(spec={"v": 1}, data={"panel": 1}, **kw)              # identical to the first run: allowed
    with pytest.raises(registry.OOSReuseError):                               # same tag, reworded: refused
        registry.append_run(spec={"v": 1}, data={"panel": 3}, oos_override="audit-2026-10: reworded reason, new data", **kw)
    with pytest.raises(registry.OOSReuseError):                               # no tag
        registry.append_run(spec={"v": 1}, data={"panel": 3}, oos_override="a long reason without any tag at all here", **kw)
    registry.append_run(spec={"v": 1}, data={"panel": 3}, oos_override="audit-2026-10b: zero-volume filler dropped", **kw)
    registry.append_run(spec={"v": 1}, data={"panel": 3}, split="ALL", hypothesis_id="H8", family="f", metrics={})
    assert registry.oos_looks("H8") == 6
