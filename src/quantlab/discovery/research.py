"""Discovery research: what happens AFTER discovery? (research only; never changes a rule)

1. REPLAY discovery point-in-time on past sessions: at each sampled session D the exact live
   scoring core (``DiscoveryEngine.core``) runs on features that only look back from D.
2. OUTCOMES from the NEXT-SESSION OPEN (the first executable price after a close-of-D signal) to the
   close of D+h, h in 1/3/5/10/20: gross, modeled round-trip cost (CostModel, same formula as the
   backtester), net, excess vs SPY over the same window, MFE/MAE from the entry. Outcomes are joined
   AFTER the scan and never used as discovery inputs.
3. STATISTICS per family and per a SMALL fixed list of combinations, against the same-date baseline
   (all research-universe symbols). Minimum samples apply. Consistency is checked in two halves of
   the period. The verdict uses a date-clustered t-statistic with a Bonferroni threshold across all
   groups tested; horizons >= the sampling interval overlap, so 10d/20d statistics are optimistic.
4. REDUNDANCY of the five scored families: per-date Spearman correlation of their points, and the
   overlap of their firing sets.

Holdout: sessions on/after ``validation.holdout.start`` are never used, neither as signal dates
nor as outcome bars. Universe = currently listed stocks (survivorship-biased: absolute returns are
flattered, comparisons against the same-date baseline much less so).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.costs import CostModel
from quantlab.core.types import new_id
from quantlab.data.panel import DataBundle, Panel
from quantlab.db.database import to_json, utcnow_iso
from quantlab.discovery.engine import DiscoveryEngine
from quantlab.discovery.families import SCORED, stabilising
from quantlab.features.base import FeatureSet
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

HORIZONS = (1, 3, 5, 10, 20)
PRIMARY_H = 5
COMBOS: dict[str, tuple[str, ...]] = {
    "Momentum + relative strength": ("momentum", "relative_strength"),
    "Momentum + volume": ("momentum", "volume_activity"),
    "Breakout + volume": ("breakout_compression", "volume_activity"),
    "Breakout + relative strength": ("breakout_compression", "relative_strength"),
    "Relative strength + volume": ("relative_strength", "volume_activity"),
    "Mean reversion + stabilisation": ("mean_reversion", "stabilising"),
    "High momentum + abnormal volume": ("high_momentum", "abnormal_volume"),
    "High momentum + breakout": ("high_momentum", "breakout_compression"),
}
FAMILY_LABEL = {"momentum": "Momentum", "relative_strength": "Relative strength", "volume_activity": "Volume/activity",
                "breakout_compression": "Breakout/compression", "mean_reversion": "Mean reversion"}


def _window(bundle: DataBundle, i0: int, i1: int) -> DataBundle:
    p = Panel({k: v.iloc[i0:i1 + 1] for k, v in bundle.panel.fields.items()}, dict(bundle.panel.meta))
    return replace(bundle, panel=p)


def replay(config, bundle: DataBundle, dates, universe_mask: pd.DataFrame | None = None,
           block_sessions: int = 60) -> pd.DataFrame:
    """PIT discovery observations for every basic-filter symbol at each date in ``dates``."""
    eng = DiscoveryEngine(config)
    all_dates = bundle.panel.dates
    pos = {d: i for i, d in enumerate(all_dates)}
    dates = sorted(pd.Timestamp(d) for d in dates if pd.Timestamp(d) in pos)
    benches = {bundle.market_symbol, *bundle.sector_etfs}
    lb = eng.s.lookback_sessions
    rows = []
    for k in range(0, len(dates), max(1, block_sessions // 5)):
        block = dates[k:k + max(1, block_sessions // 5)]
        i0, i1 = max(0, pos[block[0]] - lb), pos[block[-1]]
        tb = _window(bundle, i0, i1)
        fs = FeatureSet(tb, dtype="float32")
        for d in block:
            c = eng.core(tb.panel, fs, d, benches)
            syms = c["syms"]
            if len(syms) == 0:
                continue
            m, comp, xs = c["masks"], c["comp"], c["xs"]
            df = pd.DataFrame({"date": d, "symbol": syms, "score": c["score"].to_numpy(),
                               "coverage": c["coverage"].to_numpy()})
            for fam in SCORED:
                df[fam] = m[fam].to_numpy()
                df[f"pts_{fam}"] = comp[fam].to_numpy()
            df["stabilising"] = stabilising(xs).to_numpy()
            df["rel_volume_1d"] = pd.to_numeric(xs["rel_volume_1d"], errors="coerce").to_numpy()
            df["abnormal_volume"] = df["rel_volume_1d"] >= 3.0
            df["high_momentum"] = df["pts_momentum"] >= 18.0
            df["adv20"] = c["adv20"].reindex(syms).to_numpy()
            df["discovered"] = m.any(axis=1).to_numpy()
            if universe_mask is not None and d in universe_mask.index:
                df["in_universe"] = universe_mask.loc[d].reindex(syms).fillna(False).to_numpy()
            else:
                df["in_universe"] = True
            rows.append(df)
        log_event(log, "discovery replay block", block_start=str(block[0].date()), block_end=str(block[-1].date()),
                  rows=sum(len(r) for r in rows))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _one_way(adv: np.ndarray, costs: CostModel) -> np.ndarray:
    hs = np.full(adv.shape, costs.half_spread_tiers[-1][1], dtype=float)      # unknown -> worst tier
    done = np.zeros(adv.shape, dtype=bool)
    for threshold, bps in costs.half_spread_tiers:                             # descending thresholds
        hit = ~done & np.isfinite(adv) & (adv >= threshold)
        hs[hit] = bps
        done |= hit
    return (hs + costs.slippage_bps) / 1e4


def attach_outcomes(obs: pd.DataFrame, bundle: DataBundle, costs: CostModel, horizons=HORIZONS,
                    stop_before: pd.Timestamp | None = None) -> pd.DataFrame:
    """Next-open entry outcomes for each observation. Bars on/after ``stop_before`` are never used."""
    p = bundle.panel
    dates = p.dates
    last = len(dates) - 1
    if stop_before is not None:
        last = min(last, int(dates.searchsorted(stop_before, side="left")) - 1)
    pos = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(p.symbols)}
    i = obs["date"].map(pos).to_numpy()
    c = obs["symbol"].map(col).to_numpy()
    ao, ac, ah, al = (p.aopen.to_numpy(float), p.aclose.to_numpy(float), p.ahigh.to_numpy(float), p.alow.to_numpy(float))
    mkt = col.get(bundle.market_symbol)
    one_way = _one_way(obs["adv20"].to_numpy(float), costs)
    out = obs.copy()
    entry_i = i + 1
    ok_entry = entry_i <= last
    entry = np.where(ok_entry, ao[np.minimum(entry_i, len(dates) - 1), c], np.nan)
    spy_entry = np.where(ok_entry, ao[np.minimum(entry_i, len(dates) - 1), mkt], np.nan) if mkt is not None else None
    for h in horizons:
        j = i + h
        ok = ok_entry & (j <= last)
        jj = np.minimum(j, len(dates) - 1)
        exit_ = np.where(ok, ac[jj, c], np.nan)
        r = exit_ / entry - 1
        cost = one_way * (1 + exit_ / entry)
        out[f"gross_{h}"] = r
        out[f"cost_{h}"] = cost
        out[f"net_{h}"] = r - cost
        if spy_entry is not None:
            spy = np.where(ok, ac[jj, mkt], np.nan) / spy_entry - 1
            out[f"spy_{h}"] = spy
            out[f"xs_net_{h}"] = out[f"net_{h}"] - spy
        # MFE/MAE from the entry over sessions i+1..j
        hi = pd.DataFrame(ah).iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1].to_numpy()
        lo = pd.DataFrame(al).iloc[::-1].rolling(h, min_periods=1).min().iloc[::-1].to_numpy()
        e = np.minimum(entry_i, len(dates) - 1)
        out[f"mfe_{h}"] = np.where(ok, hi[e, c] / entry - 1, np.nan)
        out[f"mae_{h}"] = np.where(ok, lo[e, c] / entry - 1, np.nan)
    return out


def group_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    g: dict[str, pd.Series] = {"Baseline: all research-universe symbols": pd.Series(True, index=df.index),
                               "Any discovery family": df["discovered"],
                               "High-ranked (score >= 70)": df["discovered"] & (df["score"] >= 70) & (df["coverage"] >= 0.8)}
    for fam, lbl in FAMILY_LABEL.items():
        g[lbl] = df[fam]
    for name, parts in COMBOS.items():
        m = pd.Series(True, index=df.index)
        for part in parts:
            m &= df[part].fillna(False).astype(bool)
        g[name] = m
    return g


def analyse(df: pd.DataFrame, horizons=HORIZONS, min_obs: int = 200, min_dates: int = 30) -> dict[str, Any]:
    """Statistics per group x horizon vs the same-date baseline, period split, clustered t-stat and a
    Bonferroni-adjusted verdict. Research labels only (never a promotion)."""
    df = df[df["in_universe"]].copy()
    if df.empty:
        return {"groups": [], "notes": ["no research-universe observations"]}
    dates = np.array(sorted(df["date"].unique()))
    split = dates[len(dates) // 2] if len(dates) else None
    masks = group_masks(df)
    n_tests = len(masks) - 1
    z_crit = _bonferroni_z(n_tests)
    groups = []
    for name, m in masks.items():
        g = df[m.to_numpy()]
        rec: dict[str, Any] = {"group": name, "n_obs": int(len(g)), "n_dates": int(g["date"].nunique()),
                               "baseline": name.startswith("Baseline"), "horizons": {}}
        for h in horizons:
            net, gross = g[f"net_{h}"], g[f"gross_{h}"]
            valid = net.notna()
            if valid.sum() == 0:
                rec["horizons"][h] = {"n": 0}
                continue
            base_by_date = df.groupby("date")[f"net_{h}"].mean()
            grp_by_date = g[valid].groupby("date")[f"net_{h}"].mean()
            diff = (grp_by_date - base_by_date.reindex(grp_by_date.index)).dropna()
            t = (diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))) if len(diff) > 2 and diff.std(ddof=1) > 0 else float("nan")
            first = diff[diff.index < split] if split is not None else diff
            second = diff[diff.index >= split] if split is not None else diff
            rec["horizons"][h] = {
                "n": int(valid.sum()), "n_dates": int(g[valid]["date"].nunique()),
                "mean_gross": float(gross[valid].mean()), "median_gross": float(gross[valid].median()),
                "mean_net": float(net[valid].mean()), "median_net": float(net[valid].median()),
                "mean_cost": float(g[valid][f"cost_{h}"].mean()),
                "win_rate_net": float((net[valid] > 0).mean()),
                "mean_excess_spy_net": float(g[valid][f"xs_net_{h}"].mean()) if f"xs_net_{h}" in g else None,
                "median_mfe": float(g[valid][f"mfe_{h}"].median()), "median_mae": float(g[valid][f"mae_{h}"].median()),
                "vs_baseline_mean": float(diff.mean()) if len(diff) else None,
                "vs_baseline_t_clustered": None if not math.isfinite(t) else float(t),
                "vs_baseline_first_half": float(first.mean()) if len(first) else None,
                "vs_baseline_second_half": float(second.mean()) if len(second) else None,
            }
        rec["verdict"], rec["verdict_reason"] = _verdict(rec, min_obs, min_dates, z_crit)
        groups.append(rec)
    return {"groups": groups, "split_date": str(pd.Timestamp(split).date()) if split is not None else None,
            "n_tests": n_tests, "bonferroni_z": z_crit, "min_obs": min_obs, "min_dates": min_dates,
            "primary_horizon": PRIMARY_H}


def _bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / (2 * max(n_tests, 1))))


def _verdict(rec: dict[str, Any], min_obs: int, min_dates: int, z_crit: float) -> tuple[str, str]:
    if rec["baseline"]:
        return "BASELINE", "reference group"
    h = rec["horizons"].get(PRIMARY_H) or {}
    if rec["n_obs"] < min_obs or rec["n_dates"] < min_dates or not h.get("n"):
        return "INSUFFICIENT_SAMPLE", f"needs >= {min_obs} observations on >= {min_dates} dates (has {rec['n_obs']} on {rec['n_dates']})"
    t, a, b = h.get("vs_baseline_t_clustered"), h.get("vs_baseline_first_half"), h.get("vs_baseline_second_half")
    if t is None or a is None or b is None:
        return "INSUFFICIENT_SAMPLE", "period split or t-statistic not computable"
    if t >= z_crit and a > 0 and b > 0:
        return "PROMISING", f"5d net return beats the same-date baseline in both halves; clustered t {t:.2f} >= {z_crit:.2f} (Bonferroni)"
    if t <= -z_crit and a < 0 and b < 0:
        return "POOR", f"5d net return trails the same-date baseline in both halves; clustered t {t:.2f} <= -{z_crit:.2f}"
    return "FLAT", f"no consistent, multiple-testing-robust difference from the baseline (t {t:.2f}, halves {a:+.2%} / {b:+.2%})"


def redundancy(df: pd.DataFrame) -> dict[str, Any]:
    """How much the five scored families measure the same thing."""
    cols = [f"pts_{f}" for f in SCORED]
    per_date = []
    for _, g in df.groupby("date"):
        x = g[cols].dropna()
        if len(x) >= 30:
            per_date.append(x.rank().corr().to_numpy())
    corr = np.nanmean(np.stack(per_date), axis=0) if per_date else np.full((5, 5), np.nan)
    names = [FAMILY_LABEL[f] for f in SCORED]
    pairs = []
    for a in range(5):
        for b in range(a + 1, 5):
            fa, fb = df[SCORED[a]].astype(bool), df[SCORED[b]].astype(bool)
            inter, union = int((fa & fb).sum()), int((fa | fb).sum())
            pairs.append({"a": names[a], "b": names[b], "spearman_points": float(corr[a, b]),
                          "jaccard_fired": inter / union if union else None,
                          "share_of_a_also_b": inter / int(fa.sum()) if fa.sum() else None,
                          "share_of_b_also_a": inter / int(fb.sum()) if fb.sum() else None})
    pairs.sort(key=lambda x: -abs(x["spearman_points"]) if x["spearman_points"] == x["spearman_points"] else 0)
    return {"families": names, "spearman_matrix": [[None if not np.isfinite(v) else float(v) for v in row] for row in corr],
            "pairs": pairs, "n_dates": len(per_date),
            "note": "Within one date, rs_spy_n = ret_n minus a constant (SPY's return), so their cross-sectional "
                    "ranks are identical: relative strength vs SPY and momentum share two of their inputs."}


def render_markdown(summary: dict[str, Any], red: dict[str, Any], meta: dict[str, Any]) -> str:
    L = [f"# Discovery research: forward outcomes ({meta['period_start']} to {meta['period_end']})", "",
         "Research only. Nothing here changes a discovery rule or promotes a strategy.", "",
         f"- Point-in-time replay of the live discovery scoring on {meta['n_dates']} sessions (every "
         f"{meta['every']} sessions), {meta['n_obs']:,} research-universe observations.",
         "- Entry at the NEXT session's open; exit at the close of D+h. Costs: the backtester's CostModel "
         "(half-spread tier + slippage, both legs).",
         f"- Verdicts compare each group with the same-date baseline (all research-universe symbols) on 5-day net "
         f"returns, need >= {summary.get('min_obs')} observations on >= {summary.get('min_dates')} dates, consistency "
         f"in both halves (split {summary.get('split_date')}) and a date-clustered t >= {summary.get('bonferroni_z', 0):.2f} "
         f"(Bonferroni across {summary.get('n_tests')} groups).",
         "- Caveats: currently listed stocks only (survivorship bias flatters absolute returns); 10d/20d windows "
         "overlap across sampled dates; no locked-holdout data used.", "",
         "| group | verdict | obs | dates | 5d mean gross | 5d median net | 5d mean net | win rate | vs baseline 5d | halves | t |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda x: "n/a" if x is None else f"{x:+.2%}"   # noqa: E731
    for g in summary["groups"]:
        h = g["horizons"].get(PRIMARY_H) or {}
        L.append(f"| {g['group']} | {g['verdict']} | {g['n_obs']:,} | {g['n_dates']} | {pct(h.get('mean_gross'))} | "
                 f"{pct(h.get('median_net'))} | {pct(h.get('mean_net'))} | "
                 f"{'n/a' if h.get('win_rate_net') is None else f'{h['win_rate_net']:.0%}'} | {pct(h.get('vs_baseline_mean'))} | "
                 f"{pct(h.get('vs_baseline_first_half'))} / {pct(h.get('vs_baseline_second_half'))} | "
                 f"{'n/a' if h.get('vs_baseline_t_clustered') is None else f'{h['vs_baseline_t_clustered']:.2f}'} |")
    L += ["", "## All horizons (mean net return, median net return, win rate)", "",
          "| group | 1d | 3d | 5d | 10d | 20d |", "|---|---|---|---|---|---|"]
    for g in summary["groups"]:
        cells = []
        for h in HORIZONS:
            x = g["horizons"].get(h) or {}
            cells.append("n/a" if not x.get("n") else f"{x['mean_net']:+.2%} / {x['median_net']:+.2%} / {x['win_rate_net']:.0%}")
        L.append(f"| {g['group']} | " + " | ".join(cells) + " |")
    L += ["", "## Score redundancy (per-date Spearman of family points, averaged)", "",
          "| pair | Spearman | Jaccard of firing sets | share of A also B |", "|---|---|---|---|"]
    for pr in red["pairs"]:
        L.append(f"| {pr['a']} / {pr['b']} | {pr['spearman_points']:.2f} | "
                 f"{'n/a' if pr['jaccard_fired'] is None else f'{pr['jaccard_fired']:.2f}'} | "
                 f"{'n/a' if pr['share_of_a_also_b'] is None else f'{pr['share_of_a_also_b']:.0%}'} |")
    L += ["", red["note"], ""]
    return "\n".join(L)


def run_research(ctx, bundle: DataBundle, start, end, every: int = 5, min_obs: int = 200,
                 min_dates: int = 30) -> dict[str, Any]:
    """Replay -> outcomes -> statistics -> redundancy -> persisted summary + parquet + report."""
    from quantlab.universe import UniverseEngine
    from quantlab.data.validation import quarantine_map
    holdout = pd.Timestamp(ctx.config.get("validation.holdout.start", "2025-01-01"))
    dates = bundle.panel.dates
    lb = int(ctx.config.get("discovery.lookback_sessions", 300))
    sel = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    last_ok = dates[dates < holdout]
    if len(last_ok) > max(HORIZONS) + 1:
        sel = sel[sel <= last_ok[-(max(HORIZONS) + 2)]]        # every outcome bar stays before the holdout
    sel = sel[sel >= dates[min(lb, len(dates) - 1)]]
    sample = sel[::every]
    if len(sample) == 0:
        raise ValueError("no research dates in range (check the holdout, lookback and data range)")
    umask = UniverseEngine(ctx.config).membership(bundle.truncate(sample[-1]), exclude=quarantine_map(ctx.db))
    obs = replay(ctx.config, bundle, sample, universe_mask=umask)
    n_basic = int(len(obs))
    obs = obs[obs["in_universe"]].reset_index(drop=True)       # analysis = research universe (tradeable)
    obs = attach_outcomes(obs, bundle, CostModel.from_config(ctx.config), stop_before=holdout)
    summary = analyse(obs, min_obs=min_obs, min_dates=min_dates)
    red = redundancy(obs[obs["in_universe"]])
    rid = new_id("research")
    out_dir = ctx.config.path("project.report_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{rid}_observations.parquet"
    obs.to_parquet(data_path, index=False)
    meta = {"period_start": str(sample[0].date()), "period_end": str(sample[-1].date()), "n_dates": int(len(sample)),
            "n_obs": int(obs["in_universe"].sum()), "n_basic_scanned_obs": n_basic, "every": every}
    md = render_markdown(summary, red, meta)
    report_path = out_dir / f"{rid}.md"
    report_path.write_text(md, encoding="utf-8")
    ctx.db.insert("discovery_research_runs", {
        "research_id": rid, "created_at": utcnow_iso(), "is_synthetic": int(bundle.is_synthetic),
        "period_start": meta["period_start"], "period_end": meta["period_end"], "n_dates": meta["n_dates"],
        "n_observations": meta["n_obs"], "params_json": to_json({"every": every, "min_obs": min_obs,
                                                                  "min_dates": min_dates, "horizons": HORIZONS,
                                                                  "holdout_start": str(holdout.date())}),
        "summary_json": to_json(summary), "redundancy_json": to_json(red), "report_path": str(report_path),
        "data_path": str(data_path)})
    return {"research_id": rid, "meta": meta, "summary": summary, "redundancy": red, "report": str(report_path)}


__all__ = ["COMBOS", "HORIZONS", "analyse", "attach_outcomes", "group_masks", "redundancy", "replay", "run_research"]
