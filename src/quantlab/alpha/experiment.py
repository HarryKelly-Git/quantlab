"""Run a pre-registered hypothesis FAMILY end to end (Part 3, 4, 5, 39 of the alpha brief).

A family = one hypothesis with a fixed grid of variants written down before TRAIN is run. For every
variant the same pipeline runs; the family-level statistics then account for how many were tried.

Stages (each fills a different part of ``registry.Evidence``, so no single stage can declare success):
  QUANT        backtest every variant (next-open execution, tiered costs, borrow)
  STATISTICIAN TRAIN/VALIDATION metrics, Newey-West t, Deflated Sharpe with the family trial count,
               Hansen SPA + White RC across variants (development period), PBO (CSCV), effective trials
  SELECTION    best variant by TRAIN net Sharpe (VALIDATION only confirms; nothing is re-tuned)
  SKEPTIC      robustness of the selected variant: 2x costs, next-close execution, delisting-return
               sensitivity, top 1/5/10% trade removal, by year / regime / liquidity / volatility bucket
  OOS          ONE evaluation of the locked spec on 2022-2024 (registry refuses a different spec)
  DIRECTOR     A-G classification; ledger + queue updated
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.alpha import mht, registry, splits
from quantlab.alpha.engine import BacktestResult, run_weights
from quantlab.alpha.metrics import by_label, by_period, core_metrics, remove_top_trades, spells, trade_stats
from quantlab.validation.stats import benjamini_hochberg, deflated_sharpe, newey_west_tstat

RESULTS = registry.DIR / "results"


@dataclass
class Variant:
    name: str
    spec: dict[str, Any]                      # universe, signal, entry, position, sizing, exit, costs, benchmark...
    build: Callable[[], pd.DataFrame]          # returns target weights (decision-date indexed)
    holding: int = 1
    execution: str = "open"


@dataclass
class Context:
    ret_oo: pd.DataFrame
    ret_cc: pd.DataFrame
    cost_bps: pd.DataFrame
    bench: pd.Series                           # SPY P&L aligned to decision dates (open t+1 -> open t+2)
    regimes: dict[str, pd.Series] = field(default_factory=dict)
    buckets: dict[str, pd.DataFrame] = field(default_factory=dict)
    alt_ret_oo: dict[str, pd.DataFrame] = field(default_factory=dict)   # delisting-return sensitivities
    dataset: str = "equity"
    data_manifest: dict[str, Any] = field(default_factory=dict)
    intraday_only: bool = False                # books opened at the open and closed at the close each day


def _split_stats(net: pd.Series, gross: pd.Series, ds: str) -> dict[str, Any]:
    out = {}
    for sp in ("TRAIN", "VALIDATION"):
        n = splits.slice_split(net, ds, sp).dropna()
        g = splits.slice_split(gross, ds, sp).dropna()
        if len(n) < 60:
            out[sp] = {"n_days": len(n)}
            continue
        t = newey_west_tstat(n.to_numpy())
        tg = newey_west_tstat(g.to_numpy())
        out[sp] = {"n_days": len(n), "mean_bps": float(n.mean() * 1e4), "ann_return": float(n.mean() * 252),
                   "sharpe": float(n.mean() / n.std() * math.sqrt(252)) if n.std() > 0 else None,
                   "t_net": t.t, "t_gross": tg.t, "mean_gross_bps": float(g.mean() * 1e4)}
    dev = pd.concat([splits.slice_split(net, ds, "TRAIN"), splits.slice_split(net, ds, "VALIDATION")]).dropna()
    devg = pd.concat([splits.slice_split(gross, ds, "TRAIN"), splits.slice_split(gross, ds, "VALIDATION")]).dropna()
    if len(dev) >= 60:
        out["DEV"] = {"n_days": len(dev), "mean_bps": float(dev.mean() * 1e4),
                      "sharpe": float(dev.mean() / dev.std() * math.sqrt(252)) if dev.std() > 0 else None,
                      "t_net": newey_west_tstat(dev.to_numpy()).t, "t_gross": newey_west_tstat(devg.to_numpy()).t}
    return out


def _first_active(r: BacktestResult):
    """First decision date with a non-zero book. Audit minor fix: TRAIN starts in 2016 but universes need
    252 sessions of history, so the first ~252 days were zeros that diluted TRAIN mean/Sharpe (~25%)."""
    nz = r.gross_exposure[r.gross_exposure > 0]
    return nz.index[0] if len(nz) else None


def _from(x: pd.Series, start) -> pd.Series:
    return x if start is None else x.loc[start:]


def _bt(v: Variant, ctx: Context, w: pd.DataFrame, **kw) -> BacktestResult:
    return run_weights(w, kw.pop("ret_oo", ctx.ret_oo), ctx.cost_bps, holding=v.holding,
                       execution=kw.pop("execution", v.execution), ret_cc=ctx.ret_cc,
                       roundtrip_each_period=ctx.intraday_only, **kw)


def run_family(*, family: str, hypothesis_id: str, variants: list[Variant], ctx: Context,
               prior_trials: int = 0, run_oos: bool = True, select_on: str = "TRAIN",
               notes: str = "", oos_override: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    ds = ctx.dataset
    results: dict[str, dict[str, Any]] = {}
    nets: dict[str, pd.Series] = {}
    for v in variants:
        w = v.build()                          # weights are rebuilt for the selected variant later: keeping
        r = _bt(v, ctx, w)                     # every variant's (dates x names) book would need ~100 MB each
        act = _first_active(r)
        nets[v.name] = _from(r.net, act)
        st = _split_stats(_from(r.net, act), _from(r.gross, act), ds)
        st["first_active"] = str(act.date()) if act is not None else None
        st["turnover_daily"] = float(r.turnover.mean())
        st["gross_exposure"] = float(r.gross_exposure.mean())
        st["avg_names"] = float((r.n_long + r.n_short).mean())
        results[v.name] = st
        registry.append_run(hypothesis_id=hypothesis_id, family=family, spec={"variant": v.name, **v.spec,
                            "holding": v.holding, "execution": v.execution}, split="DEV", metrics=st,
                            data=ctx.data_manifest)
        del w, r
    # --- statistician: family-level, development period only -----------------------------------------
    dev_mat = pd.DataFrame({k: pd.concat([splits.slice_split(s, ds, "TRAIN"), splits.slice_split(s, ds, "VALIDATION")])
                            for k, s in nets.items()}).dropna(how="all").fillna(0.0)
    fam: dict[str, Any] = {"n_variants": len(variants), "prior_trials": prior_trials}
    n_trials = len(variants) + prior_trials
    if len(variants) >= 2:
        fam["spa"] = mht.hansen_spa(dev_mat.to_numpy(), n_resamples=1000).to_dict()
        fam["white_rc"] = mht.white_reality_check(dev_mat.to_numpy(), n_resamples=1000).to_dict()
        fam["pbo"] = mht.pbo_cscv(dev_mat.to_numpy(), n_blocks=12)
        fam["effective_trials"] = mht.effective_trials(dev_mat.to_numpy())
        pvals = [newey_west_tstat(dev_mat[c].to_numpy()).p_one_sided for c in dev_mat.columns]
        rej, adj = benjamini_hochberg(pvals, 0.05)
        fam["bh_rejections"] = int(np.sum(rej))
    # --- selection on TRAIN ------------------------------------------------------------------------------
    key = lambda k: (results[k].get(select_on, {}).get("sharpe") or -1e9)
    best = max(results, key=key)
    bv = next(v for v in variants if v.name == best)
    trial_sr = [results[k].get("DEV", {}).get("sharpe") for k in results]
    trial_sr = [x / math.sqrt(252) for x in trial_sr if x is not None]
    dev_best = dev_mat[best]
    dsr = deflated_sharpe(dev_best.to_numpy(), n_trials, trial_sharpes=trial_sr if len(trial_sr) >= 2 else None)
    fam["selected"] = best
    fam["dsr_selected_dev"] = dsr.to_dict()
    # --- skeptic: robustness of the selected variant (development + OOS views computed, OOS reported once)
    wbest = bv.build()
    base = _bt(bv, ctx, wbest, keep_contrib=True)
    act = _first_active(base)

    def dev_of(x: pd.Series) -> pd.Series:
        return _from(pd.concat([splits.slice_split(x, ds, s) for s in splits.DEVELOPMENT]), act).dropna()

    rob: dict[str, Any] = {}
    x2 = _bt(bv, ctx, wbest, cost_mult=2.0)
    rob["dev_t_net_2x_costs"] = newey_west_tstat(dev_of(x2.net).to_numpy()).t
    if bv.execution == "open" and not ctx.intraday_only:
        mc = _bt(bv, ctx, wbest, execution="close")
        rob["dev_t_net_next_close_exec"] = newey_west_tstat(dev_of(mc.net).to_numpy()).t
    for nm, alt in ctx.alt_ret_oo.items():
        a = _bt(bv, ctx, wbest, ret_oo=alt)
        rob[f"dev_t_net_delist_{nm}"] = newey_west_tstat(dev_of(a.net).to_numpy()).t
    dev_idx = _from(pd.concat([splits.slice_split(base.net, ds, s) for s in splits.DEVELOPMENT]), act).index
    rob["dev_top_removal"] = remove_top_trades(base.net.loc[dev_idx], base.weights.loc[dev_idx], base.contrib.loc[dev_idx])
    rob["dev_by_year"] = by_period(base.net.loc[dev_idx], "Y").reset_index().astype(str).to_dict("records")
    for nm, lab in ctx.regimes.items():
        rob[f"dev_by_{nm}"] = by_label(base.net.loc[dev_idx], lab).reset_index().astype(str).to_dict("records")
    for nm, b in ctx.buckets.items():
        from quantlab.alpha.metrics import bucket_contrib
        bc = bucket_contrib(base.contrib.loc[dev_idx], b.loc[dev_idx])
        rob[f"dev_gross_by_{nm}_bps_per_day"] = {str(k): float(v.mean() * 1e4) for k, v in bc.items()}
    sp = spells(base.weights.loc[dev_idx], base.contrib.loc[dev_idx])
    rob["dev_trade_stats"] = trade_stats(sp)
    # --- OOS: once --------------------------------------------------------------------------------------
    oos: dict[str, Any] = {}
    if run_oos:
        on = splits.slice_split(base.net, ds, "OOS").dropna()
        og = splits.slice_split(base.gross, ds, "OOS").dropna()
        ob = splits.slice_split(ctx.bench, ds, "OOS").reindex(on.index)
        oos = core_metrics(on, ob)
        oos["t_gross"] = newey_west_tstat(og.to_numpy()).t
        oos["by_year"] = by_period(on, "Y").reset_index().astype(str).to_dict("records")
        yrs = by_period(on, "Y")["return"]
        oos["years_positive_frac"] = float((yrs > 0).mean()) if len(yrs) else None
        x2o = splits.slice_split(x2.net, ds, "OOS").dropna()
        oos["t_net_2x_costs"] = newey_west_tstat(x2o.to_numpy()).t
        oidx = on.index
        oos["top_removal"] = remove_top_trades(base.net.loc[oidx], base.weights.loc[oidx], base.contrib.loc[oidx])
        for nm, lab in ctx.regimes.items():
            oos[f"by_{nm}"] = by_label(on, lab).reset_index().astype(str).to_dict("records")
        registry.append_run(hypothesis_id=hypothesis_id, family=family, spec={"variant": best, **bv.spec,
                            "holding": bv.holding, "execution": bv.execution}, split="OOS", metrics=oos,
                            data=ctx.data_manifest, oos_override=oos_override)
    # --- director ----------------------------------------------------------------------------------------
    dsel = results[best].get("DEV", {})
    top5 = rob["dev_top_removal"].get("without_top_5pct", {})
    ev = registry.Evidence(
        data_ok=True, dev_t_net=dsel.get("t_net"), dev_t_gross=dsel.get("t_gross"),
        oos_t_net=oos.get("t_mean_nw") if oos else None, oos_mean_net=oos.get("mean_daily") if oos else None,
        oos_years_positive_frac=oos.get("years_positive_frac") if oos else None,
        deflated_sharpe_prob=dsr.dsr if dsr.status == "OK" else None,
        spa_p=fam.get("spa", {}).get("p_value"), pbo=fam.get("pbo", {}).get("pbo"),
        survives_2x_costs=(oos.get("t_net_2x_costs") or -9) >= 2 if oos else (rob["dev_t_net_2x_costs"] or -9) >= 2,
        survives_top5_removal=(oos.get("top_removal", {}).get("without_top_5pct", {}).get("mean_daily") or -1) > 0
        if oos else (top5.get("mean_daily") or -1) > 0)
    cls, why = registry.classify(ev)
    summary = {"family": family, "hypothesis_id": hypothesis_id, "notes": notes, "variants": results, "family_stats": fam,
               "robustness_dev": rob, "oos": oos, "classification": cls, "classification_reason": why,
               "evidence": ev.__dict__, "seconds": round(time.time() - t0, 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{family}.json").write_text(json.dumps(registry._clean(summary), indent=1, default=str))
    return summary


def run_oos_only(*, family: str, hypothesis_id: str, variants: list[Variant], ctx: Context, dev_family: str,
                 oos_override: str | None = None) -> dict[str, Any]:
    """Spend the family's single OOS evaluation on the variant ALREADY selected (on TRAIN) by a
    development run (``dev_family`` results file). Nothing is re-selected; the dev statistics are reused."""
    dev = json.loads((RESULTS / f"{dev_family}.json").read_text())
    best = dev["family_stats"]["selected"]
    bv = next(v for v in variants if v.name == best)
    ds = ctx.dataset
    w = bv.build()
    base = _bt(bv, ctx, w, keep_contrib=True)
    x2 = _bt(bv, ctx, w, cost_mult=2.0)
    on = splits.slice_split(base.net, ds, "OOS").dropna()
    og = splits.slice_split(base.gross, ds, "OOS").dropna()
    ob = splits.slice_split(ctx.bench, ds, "OOS").reindex(on.index)
    oos = core_metrics(on, ob)
    oos["t_gross"] = newey_west_tstat(og.to_numpy()).t
    oos["mean_gross_bps"] = float(og.mean() * 1e4)
    oos["by_year"] = by_period(on, "Y").reset_index().astype(str).to_dict("records")
    yrs = by_period(on, "Y")["return"]
    oos["years_positive_frac"] = float((yrs > 0).mean()) if len(yrs) else None
    oos["t_net_2x_costs"] = newey_west_tstat(splits.slice_split(x2.net, ds, "OOS").dropna().to_numpy()).t
    oidx = on.index
    oos["top_removal"] = remove_top_trades(base.net.loc[oidx], base.weights.loc[oidx], base.contrib.loc[oidx])
    for nm, lab in ctx.regimes.items():
        oos[f"by_{nm}"] = by_label(on, lab).reset_index().astype(str).to_dict("records")
    registry.append_run(hypothesis_id=hypothesis_id, family=family, spec={"variant": best, **bv.spec, "holding": bv.holding,
                        "execution": bv.execution}, split="OOS", metrics=oos, data=ctx.data_manifest,
                        oos_override=oos_override)
    ev = dict(dev["evidence"])
    ev.update(oos_t_net=oos.get("t_mean_nw"), oos_mean_net=oos.get("mean_daily"), oos_years_positive_frac=oos.get("years_positive_frac"),
              survives_2x_costs=(oos.get("t_net_2x_costs") or -9) >= 2,
              survives_top5_removal=(oos.get("top_removal", {}).get("without_top_5pct", {}).get("mean_daily") or -1) > 0)
    cls, why = registry.classify(registry.Evidence(**ev))
    dev.update({"family": family, "oos": oos, "classification": cls, "classification_reason": why, "evidence": ev,
                "oos_note": f"OOS spent once on the TRAIN-selected variant of {dev_family}"})
    (RESULTS / f"{family}.json").write_text(json.dumps(registry._clean(dev), indent=1, default=str))
    return dev
