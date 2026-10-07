"""Sprint P2: volatility / move-magnitude forecast engine on the survivorship-free panel (PAPER research).

Pre-registered in docs/ALPHA-SPRINT-PREREG.md section 2. Writes:
  research/alpha/results/sprint/P2_vol_forecast.json   every metric, VAL and OOS, every horizon
  var/alpha/cache/sprint_p2_models.pkl                 fitted models (TRAIN only) for P3/P4/P6/P7
  var/alpha/cache/sprint_p2_daily.parquet              daily forecasts (h = 5, 20) on the liquid universe

Usage: .venv/bin/python scripts/research/alpha/sprint_p2_volforecast.py
"""
from __future__ import annotations

import gc
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import registry, research_data  # noqa: E402
from quantlab.alpha import volforecast as vf  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "sprint"
CACHE = store_dir() / "cache"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
NON_IV = ("HV21", "HV63", "EWMA", "GARCH", "HAR")


def mem() -> str:
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB peak"


def split_of(dates: pd.Series) -> pd.Series:
    out = pd.Series("NONE", index=dates.index)
    for k, (a, b) in SPLITS.items():
        out[(dates >= a) & (dates <= b)] = k
    return out


def main() -> None:
    t0 = time.time()
    d = research_data.get()
    for k in ("resid", "betas", "sec_ret", "alt_ret_oo", "u_large", "u_survivor", "vol20", "vol60", "cost_bps"):
        setattr(d, k, None)
    gc.collect()
    p, u = d.p, d.u_liquid
    cols = u.columns[u.any(axis=0).to_numpy()]
    print(f"loaded {time.time() - t0:.0f}s, {len(cols)} ever-liquid symbols, {mem()}", flush=True)

    W = vf.features(p, u, d.mdv20, cols)
    print(f"features {time.time() - t0:.0f}s {mem()}", flush=True)
    lr2, v252 = W.f["lr2"].to_numpy(), W.f["v252"].to_numpy()
    tr = p.dates <= pd.Timestamp(SPLITS["TRAIN"][1])
    garch = vf.fit_garch(lr2[tr], v252[tr])
    print(f"GARCH fit (TRAIN) {garch} {time.time() - t0:.0f}s", flush=True)
    W.f["g_s2next"] = pd.DataFrame(vf.garch_var(lr2, v252, garch["a"], garch["b"]), index=p.dates, columns=cols)
    del W.f["lr2"], lr2
    gc.collect()

    # weekly origins on the liquid universe
    origins = vf.weekly_origins(p.dates)
    st = u.loc[origins, cols].stack(future_stack=True)
    pairs = st[st].reset_index().iloc[:, :2]
    pairs.columns = ["date", "symbol"]
    df = vf.sample(W, pairs)
    T = vf.targets(d.ret_cc_pnl, cols)
    di, si = p.dates.get_indexer(df["date"]), cols.get_indexer(df["symbol"])
    for h in vf.HORIZONS:
        df[f"rv{h}"] = T[h]["rv"].to_numpy()[di, si]
        df[f"move{h}"] = T[h]["move"].to_numpy()[di, si]
    del T
    gc.collect()
    df["split"] = split_of(df["date"]).to_numpy()
    df["year"] = df["date"].dt.year
    df["spy_vol"] = d.regimes["spy_vol"].reindex(df["date"]).to_numpy()
    liq = d.buckets["liquidity"]
    df["liq"] = liq.to_numpy()[liq.index.get_indexer(df["date"]), liq.columns.get_indexer(df["symbol"])]
    print(f"weekly sample {len(df):,} rows; by split {df['split'].value_counts().to_dict()} {time.time() - t0:.0f}s {mem()}",
          flush=True)

    res: dict = {"prereg": "docs/ALPHA-SPRINT-PREREG.md section 2", "git": registry.git_commit(),
                 "data": {"panel_version": d.manifest.get("panel_version"), "rows": int(len(df)),
                          "origins": int(df["date"].nunique()), "symbols": int(df["symbol"].nunique()),
                          "by_split": df["split"].value_counts().to_dict(),
                          "note": "the store starts 2016-01 and the universe needs 252 sessions of history, so TRAIN "
                                  "origins effectively start in 2017"},
                 "garch": garch, "horizons": {}}
    models: dict = {"garch": garch, "fitted": {}}
    sel_votes = []
    for h in vf.HORIZONS:
        th = time.time()
        trn = df[(df["split"] == "TRAIN") & df[f"rv{h}"].notna()]
        m = vf.fit(trn, h, garch)
        fc_tr = vf.predict(m, trn)
        m.nu = {k: vf.fit_nu(trn[f"move{h}"].to_numpy(), fc_tr[k].to_numpy(), h) for k in fc_tr.columns}
        # sensitivity: level-free comparison, each model rescaled by its TRAIN QLIKE-optimal scalar
        m.qlike_scalar = {k: float(np.sqrt(np.nanmean(np.maximum(trn[f"rv{h}"], vf.FLOOR) ** 2 /
                                                      np.maximum(fc_tr[k], vf.FLOOR) ** 2))) for k in fc_tr.columns}
        log_err_sd = {k: float(np.nanstd(np.log(np.maximum(trn[f"rv{h}"], vf.FLOOR) / np.maximum(fc_tr[k], vf.FLOOR))))
                      for k in fc_tr.columns}
        fc_raw = vf.predict(m, df, calibrated=False)
        fc = vf.predict(m, df)
        hr: dict = {"nu": m.nu, "calibration_scalars_scale_baselines": m.scal, "smearing_var": m.smear,
                    "qlike_optimal_scalar_TRAIN": m.qlike_scalar, "log_error_sd_TRAIN": log_err_sd,
                    "ols_coef_har": m.har.round(4).tolist(), "splits": {}}
        for sp in ("TRAIN", "VAL", "OOS"):
            mk = (df["split"] == sp).to_numpy()
            sub, f = df[mk], fc[mk]
            ev = vf.evaluate(sub, f, h, m.nu, ref="HV21")
            qd = ev.pop("ql_by_date")
            ev["raw_scale_baselines_QLIKE"] = {k: float(np.nanmean(vf.qlike(sub[f"rv{h}"].to_numpy(), fc_raw.loc[mk, k].to_numpy())))
                                               for k in ("HV21", "HV63", "EWMA", "GARCH")}
            ev["level_free_QLIKE"] = {k: float(np.nanmean(vf.qlike(sub[f"rv{h}"].to_numpy(), f[k].to_numpy() * m.qlike_scalar[k])))
                                      for k in f.columns}
            if sp != "TRAIN":
                ev["stability_vs_HV21"] = vf.stability(sub, f, h, "HV21", {"year": sub["year"].to_numpy(),
                                                                            "spy_vol": sub["spy_vol"].to_numpy(),
                                                                            "liquidity": sub["liq"].to_numpy()})
            ev["_ql_by_date"] = qd
            hr["splits"][sp] = ev
        # QuantLab specification chosen on VAL (pre-registered), then DM vs every baseline on OOS
        qv = {k: hr["splits"]["VAL"]["models"][k]["QLIKE"] for k in ("QL_OLS", "QL_HGB")}
        ql = min(qv, key=qv.get)
        hr["QL_selected_on_VAL"] = ql
        for sp in ("VAL", "OOS"):
            qd = hr["splits"][sp]["_ql_by_date"]
            hr["splits"][sp]["DM_QL_vs"] = {b: vf.dm(qd[ql], qd[b]) for b in NON_IV}
        oos = hr["splits"]["OOS"]
        beats = all(oos["models"][ql]["QLIKE"] < oos["models"][b]["QLIKE"] and (oos["DM_QL_vs"][b]["t"] or 0) <= -2
                    for b in NON_IV)
        hr["QL_beats_all_baselines_OOS"] = bool(beats)
        sel_votes.append((h, ql, beats))
        for sp in hr["splits"]:
            hr["splits"][sp].pop("_ql_by_date", None)
        res["horizons"][str(h)] = hr
        m.log_err_sd = log_err_sd
        models["fitted"][h] = m
        print(f"h={h}: VAL QLIKE " + ", ".join(f"{k} {v['QLIKE']:.4f}" for k, v in hr["splits"]["VAL"]["models"].items())
              + f" | OOS QL={ql} beats_all={beats} ({time.time() - th:.0f}s) {mem()}", flush=True)

    # pre-registered decision: QuantLab wins only if it beats every non-IV baseline at >= 3 of 5 horizons
    n_beat = sum(b for _, _, b in sel_votes)
    if n_beat >= 3:
        ql_choice = max({q for _, q, _ in sel_votes}, key=[q for _, q, _ in sel_votes].count)
        selected, why = ql_choice, f"QuantLab beat every non-IV baseline (QLIKE lower, DM t <= -2) at {n_beat}/5 horizons"
    else:
        mean_q = {k: np.mean([res["horizons"][str(h)]["splits"]["OOS"]["models"][k]["QLIKE"] for h in vf.HORIZONS])
                  for k in NON_IV}
        best = min(mean_q.values())
        simple_order = ("HV21", "HV63", "EWMA", "GARCH", "HAR")     # simplest first
        selected = next(k for k in simple_order if mean_q[k] <= best * 1.01)
        why = (f"QuantLab beat every baseline at only {n_beat}/5 horizons; the simplest baseline within 1% of the best "
               f"mean OOS QLIKE is selected (mean OOS QLIKE: { {k: round(v, 4) for k, v in mean_q.items()} })")
    res["selection"] = {"selected_forecast": selected, "rule": why, "votes": [list(map(str, v)) for v in sel_votes]}
    models["selected"] = selected
    print("SELECTED:", selected, "|", why, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P2_vol_forecast.json").write_text(json.dumps(registry._clean(res), indent=1, default=str))
    with (CACHE / "sprint_p2_models.pkl").open("wb") as fh:
        pickle.dump(models, fh)
    for sp in ("VAL", "OOS"):
        registry.append_run(hypothesis_id="S2_vol_forecast", family="sprint_vol", split=sp,
                            spec={"prereg": "ALPHA-SPRINT-PREREG section 2", "horizons": list(vf.HORIZONS),
                                  "features": list(vf.QL_FEATURES), "hgb": vf.HGB_PARAMS},
                            metrics={str(h): {k: v["QLIKE"] for k, v in res["horizons"][str(h)]["splits"][sp]["models"].items()}
                                     for h in vf.HORIZONS},
                            conclusion=f"selected={selected}; {why}" if sp == "OOS" else "", seed=7,
                            data={"panel_version": d.manifest.get("panel_version")})

    # daily forecasts (h = 5, 20) on the liquid universe for P3/P7 sizing
    del df
    gc.collect()
    st = u[cols].stack(future_stack=True)
    dp = st[st].reset_index().iloc[:, :2]
    dp.columns = ["date", "symbol"]
    parts = []
    for chunk in np.array_split(np.arange(len(dp)), 8):
        sub = vf.sample(W, dp.iloc[chunk])
        keep = sub[["date", "symbol"]].copy()
        for h in (5, 20):
            fc = vf.predict(models["fitted"][h], sub)
            for k in fc.columns:
                keep[f"{k}_h{h}"] = fc[k].astype("float32").to_numpy()
        parts.append(keep)
    daily = pd.concat(parts, ignore_index=True)
    daily.to_parquet(CACHE / "sprint_p2_daily.parquet", index=False)
    print(f"daily forecasts {len(daily):,} rows saved; total {time.time() - t0:.0f}s {mem()}", flush=True)


if __name__ == "__main__":
    main()
