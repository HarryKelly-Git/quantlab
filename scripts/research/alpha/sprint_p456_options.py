"""Sprint P4-P6: implied vs forecast volatility across maturities, the options execution engine at four fill
levels, and the four pre-registered option hypotheses (docs/ALPHA-SPRINT-PREREG.md sections 4-6). PAPER research.

Needs P2 (var/alpha/cache/sprint_p2_models.pkl) and var/alpha/options/strangle25.parquet.
Writes research/alpha/results/sprint/P4_P6_options.json and var/alpha/cache/sprint_p6_trades.parquet.
"""
from __future__ import annotations

import gc
import json
import math
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import opt_exec as ox  # noqa: E402
from quantlab.alpha import ca_fixes, registry, research_data  # noqa: E402
from quantlab.alpha import volforecast as vf  # noqa: E402
from quantlab.alpha.entities import apply_aliases  # noqa: E402
from quantlab.alpha.experiments import options_vol as ov  # noqa: E402
from quantlab.alpha.options_store import options_dir  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "sprint"
CACHE = store_dir() / "cache"
OSPLIT = {"TRAIN": ("2019-02-01", "2021-12-31"), "VAL": ("2022-01-01", "2022-12-31"), "OOS": ("2023-01-01", "2024-12-31")}
BUCKETS = ((5, 12), (13, 25), (26, 45), (46, 75))
K_GRID = (0.10, 0.20, 0.30)
MODELS = ("HV21", "HV63", "EWMA", "GARCH", "HAR", "QL")


def mem() -> str:
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak"


def osplit(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), "NONE", dtype=object)
    for k, (a, b) in OSPLIT.items():
        out[((s >= a) & (s <= b)).to_numpy()] = k
    return out


def interp_sigma(sig_h: dict[int, np.ndarray], n: np.ndarray) -> np.ndarray:
    """ln sigma^2 interpolated linearly in h between the fitted horizons; flat beyond 20, below 1."""
    hs = np.array(vf.HORIZONS, dtype=float)
    lv = np.column_stack([np.log(np.maximum(sig_h[h], 1e-6) ** 2) for h in vf.HORIZONS])
    nn = np.clip(n.astype(float), hs[0], hs[-1])
    j = np.clip(np.searchsorted(hs, nn, side="right") - 1, 0, len(hs) - 2)
    w = (nn - hs[j]) / (hs[j + 1] - hs[j])
    rows = np.arange(len(n))
    out = lv[rows, j] * (1 - w) + lv[rows, j + 1] * w
    return np.sqrt(np.exp(out))


def nearest_h(n: np.ndarray) -> np.ndarray:
    hs = np.array(vf.HORIZONS)
    return hs[np.abs(hs[None, :] - np.clip(n, 1, 20)[:, None]).argmin(axis=1)]


def first_per_week(x: pd.DataFrame, target_dte: int, lo: int, hi: int) -> pd.DataFrame:
    f = x[(x["dte"] >= lo) & (x["dte"] <= hi)].copy()
    f["dd"] = (f["dte"] - target_dte).abs()
    f = f.loc[f.groupby(["date", "act_symbol"])["dd"].idxmin()]
    f["week"] = pd.to_datetime(f["session"]).dt.to_period("W-FRI")
    return f.sort_values("date").groupby(["week", "act_symbol"], observed=True).head(1).reset_index(drop=True)


def qlike_tab(x: pd.DataFrame, cols: list[str]) -> dict:
    ok = np.all([np.isfinite(x[c].to_numpy()) & (x[c].to_numpy() > 0) for c in cols + ["rv_to_exp"]], axis=0)
    y = x.loc[ok]
    return {"n": int(ok.sum()), **{c: float(np.mean(vf.qlike(y["rv_to_exp"].to_numpy(), y[c].to_numpy()))) for c in cols}}


def encompassing(x: pd.DataFrame, f: str, train: pd.DataFrame) -> dict:
    def clean(z):
        z = z.dropna(subset=["rv_to_exp", "iv_atm", f])
        return z[(z["rv_to_exp"] > 0) & (z["iv_atm"] > 0) & (z[f] > 0)]
    tr, ev = clean(train), clean(x)
    if len(tr) < 200 or len(ev) < 200:
        return {"n": int(len(ev))}
    def X(z, with_f):
        cols = [np.ones(len(z)), np.log(z["iv_atm"].to_numpy())] + ([np.log(z[f].to_numpy())] if with_f else [])
        return np.column_stack(cols)
    y_tr, y_ev = np.log(tr["rv_to_exp"].to_numpy()), np.log(ev["rv_to_exp"].to_numpy())
    b_iv = np.linalg.lstsq(X(tr, False), y_tr, rcond=None)[0]
    b_f = np.linalg.lstsq(X(tr, True), y_tr, rcond=None)[0]
    sst = np.sum((y_ev - y_ev.mean()) ** 2)
    r2_iv = 1 - np.sum((y_ev - X(ev, False) @ b_iv) ** 2) / sst
    r2_f = 1 - np.sum((y_ev - X(ev, True) @ b_f) ** 2) / sst
    Xe = X(ev, True)
    coef = np.linalg.lstsq(Xe, y_ev, rcond=None)[0]
    res = y_ev - Xe @ coef
    wk = pd.to_datetime(ev["session"]).dt.to_period("W-FRI").astype(str).to_numpy()
    se = ov.driscoll_kraay_se(Xe, res, wk, 6)
    return {"n": int(len(ev)), "n_weeks": int(len(np.unique(wk))), "oos_r2_iv_only": float(r2_iv),
            "oos_r2_iv_plus_forecast": float(r2_f), "oos_r2_gain": float(r2_f - r2_iv),
            "train_coef_iv_plus_forecast": b_f.round(4).tolist(), "in_split_b_iv": float(coef[1]),
            "in_split_b_forecast": float(coef[2]), "t_b_forecast_dk6": float(coef[2] / se[2]),
            "t_b_iv_dk6": float(coef[1] / se[1])}


def straddle_levels(x: pd.DataFrame, side: str = "buy") -> dict:
    k = x["k_atm"].to_numpy()
    st = x["exp_close"].to_numpy()
    itm = (st != k).astype(int)                         # exactly one leg finishes ITM unless S_T == K
    args = (x["straddle_bid"].to_numpy(), x["straddle_ask"].to_numpy(), 2, st, x["payoff"].to_numpy(), itm,
            x["mdv20"].to_numpy())
    return ox.long_structure_returns(*args) if side == "buy" else ox.short_structure_returns(*args)


def main() -> None:
    t0 = time.time()
    models = pickle.load((CACHE / "sprint_p2_models.pkl").open("rb"))
    sel = models["selected"]
    ql_name = {h: next(iter([k for k in ("QL_OLS", "QL_HGB")])) for h in vf.HORIZONS}
    p2 = json.loads((OUT / "P2_vol_forecast.json").read_text())
    for h in vf.HORIZONS:
        ql_name[h] = p2["horizons"][str(h)]["QL_selected_on_VAL"]
    d = research_data.get()
    ca_info = ca_fixes.patch_research_data(d)
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps", "buckets"):
        setattr(d, k, None)
    gc.collect()
    p, u = d.p, d.u_liquid
    cols = u.columns[u.any(axis=0).to_numpy()]
    W = vf.features(p, u, d.mdv20, cols)
    lr2, v252 = W.f["lr2"].to_numpy(), W.f["v252"].to_numpy()
    g = models["garch"]
    W.f["g_s2next"] = pd.DataFrame(vf.garch_var(lr2, v252, g["a"], g["b"]), index=p.dates, columns=cols)
    del W.f["lr2"], lr2
    gc.collect()
    print(f"features {time.time() - t0:.0f}s {mem()}", flush=True)

    feat = ov.load_features()
    feat = feat[(feat["dte"] >= 5) & (feat["dte"] <= 80) & ~feat["split_or_special_div"].astype(bool)].copy()
    feat["entity"] = apply_aliases(feat["entity"], feat["session"], p.meta.get("twin_aliases"))
    feat = feat[feat["entity"].isin(cols)].reset_index(drop=True)
    si = p.dates.get_indexer(pd.to_datetime(feat["session"]))
    ei = p.dates.get_indexer(pd.to_datetime(feat["exp_session"]))
    feat["n_sess"] = ei - si
    feat = feat[(si >= 0) & (ei > si)].reset_index(drop=True)
    # sprint-c: an option whose life spans a flagged corporate-action day has an UNKNOWN outcome; truncated
    # entities (merger ticker reuse) have no valid prices after their last good date
    fl = p.meta.get("ca_flags")
    n0 = len(feat)
    if fl is not None and len(fl):
        z_ = feat[["entity", "session", "exp_session"]].reset_index().merge(fl.rename(columns={"date": "fdate"}), on="entity")
        bad = z_.loc[(pd.to_datetime(z_["fdate"]) > pd.to_datetime(z_["session"])) &
                     (pd.to_datetime(z_["fdate"]) <= pd.to_datetime(z_["exp_session"])), "index"].unique()
        feat = feat.drop(index=bad).reset_index(drop=True)
    print(f"dropped {n0 - len(feat)} option rows spanning flagged corporate-action days", flush=True)
    ci = cols.get_indexer(feat["entity"])
    feat["mdv20"] = d.mdv20[cols].to_numpy()[p.dates.get_indexer(pd.to_datetime(feat["session"])), ci]
    feat["split"] = osplit(pd.to_datetime(feat["session"]))
    feat["spread_pct"] = (feat["straddle_ask"] - feat["straddle_bid"]) / feat["straddle_mid"]
    feat["liquid"] = (feat["mdv20"] >= 20e6) & (feat["spread_pct"] <= 0.10)
    pairs = pd.DataFrame({"date": pd.to_datetime(feat["session"]).to_numpy(), "symbol": feat["entity"].to_numpy()})
    X = vf.sample(W, pairs)
    assert len(X) == len(feat)
    del W
    gc.collect()
    n = feat["n_sess"].to_numpy()
    sig = {m: {} for m in MODELS}
    for h in vf.HORIZONS:
        fc = vf.predict(models["fitted"][h], X)
        for m in MODELS:
            sig[m][h] = fc[ql_name[h] if m == "QL" else m].to_numpy()
    for m in MODELS:
        feat[f"f_{m}"] = interp_sigma(sig[m], n)
    nh = nearest_h(n)
    selk = "QL" if sel.startswith("QL") else sel
    feat["nu_sel"] = [models["fitted"][h].nu[ql_name[h] if selk == "QL" else selk] for h in nh]
    feat["conf_sel"] = [models["fitted"][h].log_err_sd[ql_name[h] if selk == "QL" else selk] for h in nh]
    print(f"option rows {len(feat):,} with forecasts; {time.time() - t0:.0f}s {mem()}", flush=True)
    res: dict = {"prereg": "docs/ALPHA-SPRINT-PREREG.md sections 4-6", "git": registry.git_commit(), "selected_forecast": sel,
                 "ca_patch": ca_info,
                 "ql_spec_by_horizon": ql_name, "rows": int(len(feat)), "liquid_rows": int(feat["liquid"].sum()),
                 "unknown_fields": list(ox.UNKNOWN_FIELDS)}

    # ---- P4 (a) accuracy by maturity bucket, (b) encompassing, (c) residual deciles -----------------------
    fcols = ["iv_atm"] + [f"f_{m}" for m in MODELS]
    p4: dict = {"accuracy_qlike": {}, "encompassing": {}, "residual_deciles": {}}
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}"
        x = first_per_week(feat, (lo + hi) // 2, lo, hi)
        p4["accuracy_qlike"][b] = {}
        p4["encompassing"][b] = {}
        trn = x[(x["split"] == "TRAIN") & (pd.to_datetime(x["exp_session"]) <= pd.Timestamp(OSPLIT["TRAIN"][1]))]
        for sp in ("VAL", "OOS"):
            for sub in ("liquid", "all"):
                y = x[(x["split"] == sp) & (x["liquid"] if sub == "liquid" else True)]
                p4["accuracy_qlike"][b][f"{sp}_{sub}"] = qlike_tab(y, fcols)
            y = x[(x["split"] == sp) & x["liquid"]]
            tl = trn[trn["liquid"]]
            p4["encompassing"][b][sp] = {m: encompassing(y, f"f_{m}", tl) for m in MODELS}
    o1s = first_per_week(feat, 30, 21, 45)
    f_sel = f"f_{selk}"
    for sp in ("VAL", "OOS"):
        y = o1s[(o1s["split"] == sp) & o1s["liquid"]].dropna(subset=[f_sel, "iv_atm", "payoff", "exp_close"]).copy()
        y["G"] = np.log(y[f_sel] / y["iv_atm"])
        y["dec"] = pd.qcut(y["G"].rank(method="first"), 10, labels=False)
        tab = {}
        for dcl, z in y.groupby("dec"):
            buy, sell = straddle_levels(z, "buy"), straddle_levels(z, "sell")
            dd = pd.to_datetime(z["session"]).to_numpy()
            tab[int(dcl)] = {"n": int(len(z)), "mean_G": float(z["G"].mean()),
                             "mean_log_rv_over_iv": float(np.log(z["rv_to_exp"] / z["iv_atm"]).mean()),
                             "buy": {lv: float(np.nanmean(r["ret"])) for lv, r in buy.items()},
                             "sell": {lv: float(np.nanmean(r["ret"])) for lv, r in sell.items()}}
            if dcl in (0, 9):
                tab[int(dcl)]["buy_class"] = ox.classify(ox.summarize_levels(buy, dd))
                tab[int(dcl)]["sell_class"] = ox.classify(ox.summarize_levels(sell, dd))
        p4["residual_deciles"][sp] = tab
    p4["rule"] = "QuantLab adds information beyond IV only if OOS R^2 gain > 0 AND OOS t(b_forecast) >= 3 (DK6)"
    adds = {b: {m: (v["OOS"][m].get("oos_r2_gain", -1) > 0 and v["OOS"][m].get("t_b_forecast_dk6", 0) >= 3)
                for m in MODELS} for b, v in p4["encompassing"].items()}
    p4["adds_information_beyond_IV"] = adds
    res["P4"] = p4
    print(f"P4 done {time.time() - t0:.0f}s: adds beyond IV {adds}", flush=True)

    # ---- Part-49 model comparison (labels O1 a replication if the P2 model is not better) ----------------
    try:
        m49 = ov.build_obs(d)
        model49 = ov.fit_rv_model(m49)
        m49["fc49"] = ov.predict_rv(m49, model49, kind="median")
        key = ["date", "act_symbol", "expiration"]
        cmp_ = o1s.merge(m49[key + ["fc49"]], on=key, how="inner")
        lab = {}
        for sp in ("VAL", "OOS"):
            z = cmp_[(cmp_["split"] == sp) & cmp_["liquid"]]
            lab[sp] = qlike_tab(z, [f_sel, "fc49"])
        res["part49_model_comparison"] = lab
        replication = not (lab["VAL"].get(f_sel, 9) < lab["VAL"].get("fc49", 0))
    except Exception as exc:                           # report, never guess
        res["part49_model_comparison"] = {"error": repr(exc)}
        replication = None
    res["O1_is_replication_of_part49"] = replication

    # ---- P6 O1: long ATM straddle when model E|move| / implied move >= 1 + k ---------------------------
    z = o1s[o1s["liquid"]].dropna(subset=[f_sel, "em_pct", "payoff", "exp_close", "straddle_bid", "straddle_ask"]).copy()
    nu = z["nu_sel"].to_numpy().astype(int)
    em_model = np.array([vf.expected_abs_move(np.array([s]), int(h), int(v))[0]
                         for s, h, v in zip(z[f_sel].to_numpy(), z["n_sess"].to_numpy(), nu)])
    z["exp_move_model"] = em_model
    z["ratio"] = z["exp_move_model"] / z["em_pct"]
    trades = []
    o1: dict = {"by_k_VAL": {}}
    for k in K_GRID:
        v = z[(z["split"] == "VAL") & (z["ratio"] >= 1 + k)]
        lv = straddle_levels(v, "buy") if len(v) else {}
        o1["by_k_VAL"][str(k)] = ox.summarize_levels(lv, pd.to_datetime(v["session"]).to_numpy()) if len(v) else {"n": 0}
    elig = {k: o1["by_k_VAL"][str(k)]["CONSERVATIVE"]["mean"] for k in K_GRID
            if o1["by_k_VAL"][str(k)].get("CONSERVATIVE", {}).get("n", 0) >= 30}
    k_star = max(elig, key=elig.get) if elig else None
    o1["k_chosen_on_VAL"] = k_star
    for sp in ("TRAIN", "OOS"):
        if k_star is None:
            break
        v = z[(z["split"] == sp) & (z["ratio"] >= 1 + k_star)].copy()
        lv = straddle_levels(v, "buy")
        o1[sp] = ox.summarize_levels(lv, pd.to_datetime(v["session"]).to_numpy())
        o1[f"{sp}_class"] = ox.classify(o1[sp])
        if sp == "OOS":
            spot = v["spot"].to_numpy()
            paid_c = lv["CONSERVATIVE"]["paid"]
            v["implied_move"] = v["em_pct"]
            v["confidence_log_sd"] = v["conf_sel"]
            for lvl, r in lv.items():
                v[f"cost_{lvl}"] = r["paid"]
                v[f"ret_{lvl}"] = r["ret"]
            v["spread_cost_CONSERVATIVE"] = paid_c - v["straddle_mid"].to_numpy()
            v["breakeven_move"] = paid_c / spot
            v["model_p_profit"] = ox.prob_profit_long(spot, v["k_atm"].to_numpy(), v["k_atm"].to_numpy(),
                                                      v[f_sel].to_numpy(), v["n_sess"].to_numpy(), int(np.median(nu)), paid_c)
            v["model_ev"] = ox.expected_payoff(spot, v["k_atm"].to_numpy(), v["k_atm"].to_numpy(), v[f_sel].to_numpy(),
                                               v["n_sess"].to_numpy(), int(np.median(nu))) - paid_c
            v["max_loss"] = paid_c
            v["max_profit"] = np.inf
            v["hypothesis"] = "O1"
            trades.append(v)
    o1["rule"] = "k chosen on VAL from {0.10, 0.20, 0.30} by mean net return at CONSERVATIVE (>= 30 trades); one OOS look"
    res["O1_long_straddle"] = o1
    # P7 input: the same O1 rule driven by EVERY model's forecast (k chosen on VAL per model)
    by_model = {}
    for m in MODELS:
        zm = o1s[o1s["liquid"]].dropna(subset=[f"f_{m}", "em_pct", "payoff", "exp_close"]).copy()
        nh_m = nearest_h(zm["n_sess"].to_numpy())
        nu_m = np.array([models["fitted"][h].nu[ql_name[h] if m == "QL" else m] for h in nh_m])
        e_m = np.array([vf.expected_abs_move(np.array([s_]), int(h_), int(v_))[0]
                        for s_, h_, v_ in zip(zm[f"f_{m}"].to_numpy(), zm["n_sess"].to_numpy(), nu_m)])
        zm["ratio"] = e_m / zm["em_pct"].to_numpy()
        row = {"VAL_by_k": {}}
        for k in K_GRID:
            v = zm[(zm["split"] == "VAL") & (zm["ratio"] >= 1 + k)]
            row["VAL_by_k"][str(k)] = ox.summarize_levels(straddle_levels(v, "buy"), pd.to_datetime(v["session"]).to_numpy()) \
                if len(v) else {"CONSERVATIVE": {"n": 0}}
        el = {k: row["VAL_by_k"][str(k)]["CONSERVATIVE"]["mean"] for k in K_GRID
              if row["VAL_by_k"][str(k)]["CONSERVATIVE"].get("n", 0) >= 30}
        kk = max(el, key=el.get) if el else None
        row["k"] = kk
        if kk is not None:
            v = zm[(zm["split"] == "OOS") & (zm["ratio"] >= 1 + kk)]
            row["OOS"] = ox.summarize_levels(straddle_levels(v, "buy"), pd.to_datetime(v["session"]).to_numpy())
            row["OOS_class"] = ox.classify(row["OOS"])
        by_model[m] = row
    res["O1_by_model"] = by_model

    # ---- P7 input: IV + QuantLab ensemble (log-linear, fit on TRAIN with embargo; kept only if VAL improves) -------
    e = o1s[o1s["liquid"]].dropna(subset=["f_QL", "iv_atm", "rv_to_exp", "em_pct", "payoff", "exp_close"]).copy()
    e = e[(e["f_QL"] > 0) & (e["iv_atm"] > 0) & (e["rv_to_exp"] > 0)]
    tr_e = e[(e["split"] == "TRAIN") & (pd.to_datetime(e["exp_session"]) <= pd.Timestamp(OSPLIT["TRAIN"][1]))]
    ly = np.log(tr_e["rv_to_exp"].to_numpy())
    Xi = np.column_stack([np.ones(len(tr_e)), np.log(tr_e["iv_atm"])])
    Xe = np.column_stack([Xi, np.log(tr_e["f_QL"])])
    bi, be = np.linalg.lstsq(Xi, ly, rcond=None)[0], np.linalg.lstsq(Xe, ly, rcond=None)[0]
    si2, se2 = float(np.var(ly - Xi @ bi)), float(np.var(ly - Xe @ be))
    li, lq = np.log(e["iv_atm"].to_numpy()), np.log(e["f_QL"].to_numpy())
    e["f_IVcal"] = np.exp(bi[0] + bi[1] * li + si2 / 2)
    e["f_ENS"] = np.exp(be[0] + be[1] * li + be[2] * lq + se2 / 2)
    ens = {"train_coef_iv_only": bi.round(4).tolist(), "train_coef_ens": be.round(4).tolist(), "n_train": int(len(tr_e))}
    for sp in ("VAL", "OOS"):
        ens[f"{sp}_qlike"] = qlike_tab(e[e["split"] == sp], ["f_IVcal", "f_ENS", "f_QL", "iv_atm"])
    keep = ens["VAL_qlike"]["f_ENS"] < ens["VAL_qlike"]["f_IVcal"]
    ens["status"] = ("KEPT: lower VAL QLIKE than TRAIN-calibrated IV" if keep else
                     "DISCARDED: the ensemble did not improve VAL QLIKE over TRAIN-calibrated IV")
    if keep:
        nh_e = nearest_h(e["n_sess"].to_numpy())
        nu_e = np.array([models["fitted"][h].nu[ql_name[h]] for h in nh_e])
        em_e = np.array([vf.expected_abs_move(np.array([s_]), int(h_), int(v_))[0]
                         for s_, h_, v_ in zip(e["f_ENS"].to_numpy(), e["n_sess"].to_numpy(), nu_e)])
        e["ratio"] = em_e / e["em_pct"].to_numpy()
        vb = {}
        for k in K_GRID:
            v = e[(e["split"] == "VAL") & (e["ratio"] >= 1 + k)]
            vb[str(k)] = ox.summarize_levels(straddle_levels(v, "buy"), pd.to_datetime(v["session"]).to_numpy()) \
                if len(v) else {"CONSERVATIVE": {"n": 0}}
        el = {k: vb[str(k)]["CONSERVATIVE"]["mean"] for k in K_GRID if vb[str(k)]["CONSERVATIVE"].get("n", 0) >= 30}
        kk = max(el, key=el.get) if el else None
        ens["O1_VAL_by_k"], ens["O1_k"] = vb, kk
        if kk is not None:
            v = e[(e["split"] == "OOS") & (e["ratio"] >= 1 + kk)]
            ens["O1_OOS"] = ox.summarize_levels(straddle_levels(v, "buy"), pd.to_datetime(v["session"]).to_numpy())
            ens["O1_OOS_class"] = ox.classify(ens["O1_OOS"])
            res["O1_by_model"]["ENSEMBLE"] = {"k": kk, "OOS": ens["O1_OOS"], "OOS_class": ens["O1_OOS_class"]}
    res["ensemble"] = ens
    print(f"O1 done: k*={k_star}, OOS class {o1.get('OOS_class')}", flush=True)

    # ---- P6 O2: long 25-delta strangle when model E[payoff] - cost(CONSERVATIVE) > 0 ---------------------
    q = pd.read_parquet(options_dir() / "strangle25.parquet")
    q["date"] = pd.to_datetime(q["date"])
    zz = o1s[o1s["liquid"]].merge(q, on=["date", "act_symbol", "expiration"], how="inner").dropna(
        subset=[f_sel, "exp_close", "bid_c25", "ask_c25", "bid_p25", "ask_p25"])
    zz = zz[(zz["strike_c25"] > zz["spot"]) & (zz["strike_p25"] < zz["spot"])].copy()
    bid = (zz["bid_c25"] + zz["bid_p25"]).to_numpy()
    ask = (zz["ask_c25"] + zz["ask_p25"]).to_numpy()
    st_ = zz["exp_close"].to_numpy()
    payoff = np.maximum(st_ - zz["strike_c25"].to_numpy(), 0) + np.maximum(zz["strike_p25"].to_numpy() - st_, 0)
    itm = (st_ > zz["strike_c25"].to_numpy()).astype(int) + (st_ < zz["strike_p25"].to_numpy()).astype(int)
    nu2 = int(np.median(zz["nu_sel"]))
    e_pay = ox.expected_payoff(zz["spot"].to_numpy(), zz["strike_c25"].to_numpy(), zz["strike_p25"].to_numpy(),
                               zz[f_sel].to_numpy(), zz["n_sess"].to_numpy(), nu2)
    paid_c = ox.fill(bid, ask, "CONSERVATIVE") + 2 * ox.FEE_PER_SHARE
    zz["model_ev"] = e_pay - paid_c
    sigm = zz["model_ev"].to_numpy() > 0
    o2: dict = {"rule": "enter when model E[payoff at expiry] - cost at CONSERVATIVE > 0", "nu": nu2}
    lv_all = ox.long_structure_returns(bid, ask, 2, st_, payoff, itm, zz["mdv20"].to_numpy())
    for sp in ("TRAIN", "VAL", "OOS"):
        msk = (zz["split"] == sp).to_numpy()
        for name, mm in (("signal", msk & sigm), ("all_liquid_strangles", msk)):
            sub = {lvl: {kk: vv[mm] for kk, vv in r.items()} for lvl, r in lv_all.items()}
            o2[f"{sp}_{name}"] = ox.summarize_levels(sub, pd.to_datetime(zz.loc[mm, "session"]).to_numpy())
        o2[f"{sp}_signal_class"] = ox.classify(o2[f"{sp}_signal"])
        o2[f"{sp}_spread_pct_median"] = float(np.median(((ask - bid) / ((ask + bid) / 2))[msk])) if msk.any() else None
    v = zz[(zz["split"] == "OOS").to_numpy() & sigm].copy()
    if len(v):
        mm = ((zz["split"] == "OOS").to_numpy() & sigm)
        for lvl, r in lv_all.items():
            v[f"ret_{lvl}"] = r["ret"][mm]
            v[f"cost_{lvl}"] = r["paid"][mm]
        v["hypothesis"] = "O2"
        v["max_loss"] = v["cost_CONSERVATIVE"]
        v["max_profit"] = np.inf
        trades.append(v)
    res["O2_long_strangle"] = o2
    print(f"O2 done: OOS class {o2.get('OOS_signal_class')}", flush=True)

    # ---- O3: not run by rule; O4 precondition (term structure) on TRAIN+VAL -----------------------------
    res["O3_debit_spread"] = {"status": "NOT RUN BY RULE",
                              "reason": "needs a directional model with OOS AUC >= 0.55; best is 0.51-0.52 (H12) and every "
                                        "directional family is class E/D"}
    fr = first_per_week(feat, 30, 13, 45)
    bk = feat[(feat["dte"] >= 40) & (feat["dte"] <= 80)].copy()
    bk["dd"] = (bk["dte"] - 60).abs()
    bk = bk.loc[bk.groupby(["date", "act_symbol"])["dd"].idxmin(), ["date", "act_symbol", "expiration", "iv_atm"]]
    tv = fr.merge(bk.rename(columns={"iv_atm": "iv_back", "expiration": "exp_back"}), on=["date", "act_symbol"], how="inner")
    tv = tv[(tv["exp_back"] > tv["expiration"]) & tv["split"].isin(["TRAIN", "VAL"]) & tv["liquid"]]
    tv = tv.dropna(subset=["rv_to_exp", "iv_atm", "iv_back"])
    tv = tv[(tv["rv_to_exp"] > 0) & (tv["iv_atm"] > 0) & (tv["iv_back"] > 0)]
    y = np.log(tv["rv_to_exp"] / tv["iv_atm"]).to_numpy()
    Xs = np.column_stack([np.ones(len(tv)), np.log(tv["iv_atm"]), np.log(tv["iv_back"] / tv["iv_atm"])])
    coef = np.linalg.lstsq(Xs, y, rcond=None)[0]
    se = ov.driscoll_kraay_se(Xs, y - Xs @ coef, pd.to_datetime(tv["session"]).dt.to_period("W-FRI").astype(str).to_numpy(), 6)
    t_slope = float(coef[2] / se[2])
    res["O4_calendar"] = {"precondition": "TRAIN+VAL: log(front RV / front IV) on log front IV and the log IV term slope; "
                                          "slope DK6 |t| >= 3 needed",
                          "n": int(len(tv)), "coef_slope": float(coef[2]), "t_slope_dk6": t_slope,
                          "status": "PRECONDITION MET: calendar test required" if abs(t_slope) >= 3 else "NOT RUN: precondition failed"}
    print(f"O4 precondition t={t_slope:.2f}", flush=True)

    # ---- baselines for P5: every liquid 30-DTE straddle bought / sold --------------------------------------
    base = {}
    for sp in ("VAL", "OOS"):
        b_ = z[z["split"] == sp]
        dd = pd.to_datetime(b_["session"]).to_numpy()
        base[sp] = {"buy_all": ox.summarize_levels(straddle_levels(b_, "buy"), dd),
                    "sell_all": ox.summarize_levels(straddle_levels(b_, "sell"), dd),
                    "median_spread_pct": float(b_["spread_pct"].median())}
        base[sp]["buy_all_class"] = ox.classify(base[sp]["buy_all"])
        base[sp]["sell_all_class"] = ox.classify(base[sp]["sell_all"])
    res["P5_baselines"] = base
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P4_P6_options.json").write_text(json.dumps(registry._clean(res), indent=1, default=str))
    if trades:
        tdf = pd.concat(trades, ignore_index=True)
        keep = [c for c in tdf.columns if not c.startswith("iv_pctile")]
        tdf[keep].to_parquet(CACHE / "sprint_p6_trades.parquet", index=False)
    registry.append_run(hypothesis_id="S6_options_set", family="sprint_options", split="OOS",
                        spec={"prereg": "ALPHA-SPRINT-PREREG sections 4-6", "forecast": sel, "k_grid": list(K_GRID)},
                        metrics={"O1": {"k": k_star, "class": o1.get("OOS_class"),
                                        "mean_by_level": {k: v.get("mean") for k, v in (o1.get("OOS") or {}).items()}},
                                 "O2": {"class": o2.get("OOS_signal_class"),
                                        "mean_by_level": {k: v.get("mean") for k, v in o2.get("OOS_signal", {}).items()}},
                                 "O4": res["O4_calendar"]["status"]},
                        conclusion="see P4_P6_options.json", seed=7)
    print(f"done {time.time() - t0:.0f}s {mem()}", flush=True)


if __name__ == "__main__":
    main()
