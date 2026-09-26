"""Catalyst research (research only; never changes a rule or promotes anything).

Point-in-time replay of a SMALL, pre-registered set of catalyst setups, with forward outcomes from
the next-session open. Everything below was fixed before any result was seen:

  A  Earnings + positive reaction            reaction session D, reaction_z_1d >= 1.5
  B  Earnings + positive reaction + volume   A and reaction-day dollar volume >= 2.0x normal
  C  Material catalyst + reaction + volume   (earnings reaction today OR company-specific material
                                             news OR material 8-K usable at D) and today's move
                                             z >= 1.5 and dollar volume >= 2.0x
  D  Catalyst + strong industry RS           the same catalyst set, industry_rank_63 >= 0.7 and the
                                             stock ahead of its industry over 20 sessions
  controls (reported with verdicts, same test family): all earnings reaction days; earnings with a
  negative reaction (z <= -1.5) -- a check that the machinery sees post-earnings drift at all.

Statistics per group x horizon (1/3/5/10/20): n, dates, mean/median gross and net (CostModel,
same formula as the backtester), win rate (net > 0), MFE/MAE, excess vs SPY, and excess vs the
SAME-DATE baseline (mean net of all research-universe members). Inference: date-clustered t of the
per-date mean excess; PRIMARY horizons 5 and 20 sessions; Bonferroni over (groups x 2 primary
horizons). Verdict (the project's criteria, extended to two primary horizons):
  PROMISING    t >= z_crit at a primary horizon AND both period halves > 0 at that horizon
  NEGATIVE     t <= -z_crit AND both halves < 0 (same horizon)
  FLAT         otherwise
  INCONCLUSIVE fewer than min_obs observations or min_dates dates
Period consistency is also reported per calendar year. Overlapping horizons (10d/20d on daily
signals) make per-observation statistics optimistic; the date-clustered t and the halves are the
evidence. Holdout: no signal date and no outcome bar on/after ``validation.holdout.start``.
Universe: currently listed stocks (survivorship-biased: absolute returns are flattered; the
same-date baseline comparison much less so).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.costs import CostModel
from quantlab.core.types import new_id
from quantlab.db.database import to_json, utcnow_iso
from quantlab.discovery.research import HORIZONS, _bonferroni_z, _one_way, attach_outcomes
from quantlab.features.base import FeatureSet

PRIMARY = (5, 20)
FEATURES = ("days_since_earnings", "reaction_z_1d", "reaction_ret_1d", "event_rel_volume", "news_material_1d",
            "sec_material_1d", "ret_z_1d", "ret_1d", "rel_volume_1d", "industry_rank_63", "rs_industry_20",
            "rev_growth_yoy", "eps_growth_yoy", "adv20")
GROUPS = ("A: Earnings + positive reaction", "B: Earnings + positive reaction + abnormal volume",
          "C: Material catalyst + positive reaction + abnormal volume", "D: Catalyst + strong industry RS",
          "Control: all earnings reaction days", "Control: earnings + negative reaction")


def group_frames(fs: FeatureSet, z: float = 1.5, vol: float = 2.0, strong: float = 0.7) -> dict[str, pd.DataFrame]:
    """Boolean dates x symbols frames per pre-registered group (NaN never satisfies a condition)."""
    g = {n: fs.get(n) for n in FEATURES if n in fs.registry}
    day = g["days_since_earnings"] == 0
    rz, erv = g["reaction_z_1d"], g["event_rel_volume"]
    catalyst = day | (g["news_material_1d"] >= 1) | (g["sec_material_1d"] >= 1)
    a = day & (rz >= z)
    return {
        GROUPS[0]: a,
        GROUPS[1]: a & (erv >= vol),
        GROUPS[2]: catalyst & (g["ret_z_1d"] >= z) & (g["rel_volume_1d"] >= vol),
        GROUPS[3]: catalyst & (g["industry_rank_63"] >= strong) & (g["rs_industry_20"] > 0),
        GROUPS[4]: day,
        GROUPS[5]: day & (rz <= -z),
    }


def baseline(bundle, costs: CostModel, umask: pd.DataFrame, dates, horizons=HORIZONS,
             stop_before: pd.Timestamp | None = None) -> dict[int, pd.Series]:
    """Per-date mean NET outcome of all research-universe members (next-open entry), per horizon."""
    p = bundle.panel
    D = p.dates
    last = len(D) - 1 if stop_before is None else int(D.searchsorted(stop_before, side="left")) - 1
    ao, ac = p.aopen.to_numpy(float), p.aclose.to_numpy(float)
    adv = p.dollar_volume.rolling(20, min_periods=20).median().to_numpy(float)
    ow = _one_way(adv, costs)
    um = umask.reindex(index=D, columns=p.symbols).fillna(False).to_numpy(bool)
    rows = D.get_indexer(pd.DatetimeIndex(dates))
    out = {}
    for h in horizons:
        vals = []
        for i in rows:
            if i < 0 or i + h > last or i + 1 > last:
                vals.append(np.nan)
                continue
            e, x = ao[i + 1], ac[i + h]
            net = x / e - 1 - ow[i] * (1 + x / e)
            v = net[um[i] & np.isfinite(net)]
            vals.append(float(v.mean()) if len(v) else np.nan)
        out[h] = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return out


def analyse(obs: pd.DataFrame, base: dict[int, pd.Series], min_obs: int = 200, min_dates: int = 50,
            horizons=HORIZONS) -> dict[str, Any]:
    n_tests = sum(1 for _ in GROUPS) * len(PRIMARY)
    z_crit = _bonferroni_z(n_tests)
    out = []
    all_dates = np.array(sorted(obs["date"].unique())) if len(obs) else np.array([])
    for name in GROUPS:
        df = obs[obs[name]] if name in obs.columns else obs.iloc[0:0]
        rec: dict[str, Any] = {"group": name, "n_obs": int(len(df)), "n_dates": int(df["date"].nunique()) if len(df) else 0,
                               "n_symbols": int(df["symbol"].nunique()) if len(df) else 0, "horizons": {}}
        for h in horizons:
            net = df[f"net_{h}"]
            ok = net.notna()
            d = df[ok]
            if d.empty:
                rec["horizons"][h] = {"n": 0}
                continue
            ex = d[f"net_{h}"] - d["date"].map(base[h]).astype(float)
            per_date = ex.groupby(d["date"]).mean().dropna()
            nd = len(per_date)
            t = float(per_date.mean() / (per_date.std(ddof=1) / np.sqrt(nd))) if nd > 2 and per_date.std(ddof=1) > 0 else None
            split = all_dates[len(all_dates) // 2] if len(all_dates) else None
            a = per_date[per_date.index < split].mean() if split is not None else np.nan
            b = per_date[per_date.index >= split].mean() if split is not None else np.nan
            years = per_date.groupby(per_date.index.year).agg(["mean", "size"])
            rec["horizons"][h] = {
                "n": int(len(d)), "n_dates": nd,
                "mean_gross": float(d[f"gross_{h}"].mean()), "median_gross": float(d[f"gross_{h}"].median()),
                "mean_net": float(d[f"net_{h}"].mean()), "median_net": float(d[f"net_{h}"].median()),
                "win_rate_net": float((d[f"net_{h}"] > 0).mean()),
                "mean_mfe": float(d[f"mfe_{h}"].mean()), "mean_mae": float(d[f"mae_{h}"].mean()),
                "mean_xs_spy": float(d[f"xs_net_{h}"].mean()) if f"xs_net_{h}" in d else None,
                "vs_baseline": float(ex.mean()), "vs_baseline_date_weighted": float(per_date.mean()),
                "vs_baseline_t_clustered": t,
                "vs_baseline_first_half": None if not np.isfinite(a) else float(a),
                "vs_baseline_second_half": None if not np.isfinite(b) else float(b),
                "by_year": {int(y): {"vs_baseline": float(r["mean"]), "dates": int(r["size"])} for y, r in years.iterrows()},
            }
        rec["verdict"], rec["why"] = verdict(rec, min_obs, min_dates, z_crit)
        out.append(rec)
    return {"groups": out, "z_crit": z_crit, "n_tests": n_tests, "primary_horizons": list(PRIMARY),
            "min_obs": min_obs, "min_dates": min_dates}


def verdict(rec: dict[str, Any], min_obs: int, min_dates: int, z_crit: float) -> tuple[str, str]:
    if rec["n_obs"] < min_obs or rec["n_dates"] < min_dates:
        return "INCONCLUSIVE", f"needs >= {min_obs} observations on >= {min_dates} dates (has {rec['n_obs']} on {rec['n_dates']})"
    notes = []
    for h in PRIMARY:
        s = rec["horizons"].get(h) or {}
        t, a, b = s.get("vs_baseline_t_clustered"), s.get("vs_baseline_first_half"), s.get("vs_baseline_second_half")
        if t is None or a is None or b is None:
            notes.append(f"{h}d: not computable")
            continue
        if t >= z_crit and a > 0 and b > 0:
            return "PROMISING", (f"{h}d net beats the same-date baseline in both halves ({a:+.2%} / {b:+.2%}); "
                                 f"clustered t {t:.2f} >= {z_crit:.2f} (Bonferroni over {len(GROUPS) * len(PRIMARY)} tests)")
        if t <= -z_crit and a < 0 and b < 0:
            return "NEGATIVE", (f"{h}d net trails the same-date baseline in both halves ({a:+.2%} / {b:+.2%}); "
                                f"clustered t {t:.2f} <= -{z_crit:.2f}")
        notes.append(f"{h}d t {t:+.2f}, halves {a:+.2%} / {b:+.2%}")
    return "FLAT", "no consistent, multiple-testing-robust difference from the baseline: " + "; ".join(notes)


def research_bundle(ctx, start: str, end: str, symbols: list[str] | None = None, news_since: str | None = None,
                    facts_since: str | None = None):
    """A lean real-data bundle: bars/actions/events as stored, news with only the columns the
    classifier needs (tags counted over ALL rows first), fundamentals for the as-of replay. Nothing
    after ``end`` is loaded; ``news_since`` / ``facts_since`` drop older history a caller does not
    need (loading less never changes what is known at D, only how far back the history reaches)."""
    import pyarrow.parquet as pq
    store = ctx.store
    snap = store.snapshot(synthetic=False)
    lean = {k: v for k, v in snap.items() if k in ("bars", "corporate_actions", "reference", "events", "fundamentals")}
    b = store.load_bundle(ctx.config.section("benchmarks"), symbols=symbols, start=start, end=end, snapshot=lean,
                          synthetic=False)
    frames = []
    cols = ["news_id", "symbol", "headline", "source", "url", "created_at", "updated_at", "available_at", "pit_status",
            "retrieved_at"]
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    lo = pd.Timestamp(news_since or start, tz="UTC")
    for ds in snap.get("news", []):
        row = store.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (ds,))
        t = pq.read_table(store.data_dir / row["path"], columns=cols).to_pandas()
        a = pd.to_datetime(t["available_at"], utc=True)
        frames.append(t[(a >= lo) & (a < hi)])
    news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    if len(news):
        news = news.sort_values("retrieved_at", kind="mergesort").drop_duplicates(["news_id", "symbol"], keep="last")
        news["n_tags"] = news.groupby("news_id")["symbol"].transform("size").astype("float64")
        news["summary"] = ""
        news["provider"] = "alpaca"
    b = replace(b, news=news.reset_index(drop=True))
    ev = b.events
    if len(ev):
        b = replace(b, events=ev[pd.to_datetime(ev["available_at"], utc=True) < hi].reset_index(drop=True))
    fu = b.fundamentals
    if len(fu):
        keep = ["symbol", "concept", "period_start", "period_end", "fiscal_period", "value", "available_at", "pit_status"]
        fav = pd.to_datetime(fu["available_at"], utc=True)
        keep_rows = (fav < hi) & ((fav >= pd.Timestamp(facts_since, tz="UTC")) if facts_since else True)
        fu = fu[keep_rows][keep]
        b = replace(b, fundamentals=fu.reset_index(drop=True))
    return b


def run_catalyst_research(ctx, bundle, start, end, min_obs: int = 200, min_dates: int = 50,
                          report: bool = True) -> dict[str, Any]:
    from quantlab.data.validation import quarantine_map
    from quantlab.universe import UniverseEngine
    holdout = pd.Timestamp(ctx.config.get("validation.holdout.start", "2025-01-01"))
    dates = bundle.panel.dates
    sel = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    ok = dates[dates < holdout]
    if len(ok) > max(HORIZONS) + 1:
        sel = sel[sel <= ok[-(max(HORIZONS) + 2)]]            # every outcome bar stays before the holdout
    umask = UniverseEngine(ctx.config).membership(bundle, exclude=quarantine_map(ctx.db))
    fs = FeatureSet(bundle, universe=umask, dtype="float32")
    groups = group_frames(fs)
    um = umask.reindex(index=dates, columns=bundle.panel.symbols).fillna(False)
    any_g = None
    for m in groups.values():
        m = m.reindex(index=dates, columns=bundle.panel.symbols).fillna(False).astype(bool) & um
        any_g = m if any_g is None else (any_g | m)
    any_g = any_g.loc[sel]
    st = any_g.stack()
    st = st[st]
    obs = pd.DataFrame({"date": st.index.get_level_values(0), "symbol": st.index.get_level_values(1)})
    for name, m in groups.items():
        m = m.reindex(index=dates, columns=bundle.panel.symbols).fillna(False).astype(bool)
        obs[name] = m.to_numpy()[dates.get_indexer(obs["date"]), bundle.panel.symbols.get_indexer(obs["symbol"])]
    for n in ("reaction_z_1d", "reaction_ret_1d", "event_rel_volume", "ret_z_1d", "rel_volume_1d", "industry_rank_63",
              "rev_growth_yoy", "eps_growth_yoy", "adv20"):
        f = fs.get(n)
        obs[n] = f.to_numpy()[f.index.get_indexer(obs["date"]), f.columns.get_indexer(obs["symbol"])]
    costs = CostModel.from_config(ctx.config)
    obs = attach_outcomes(obs, bundle, costs, stop_before=holdout)
    base = baseline(bundle, costs, umask, sel, stop_before=holdout)
    res = analyse(obs, base, min_obs=min_obs, min_dates=min_dates)
    rid = new_id("catres")
    rep_dir = Path(ctx.config.path("project.report_dir"))
    rep_dir.mkdir(parents=True, exist_ok=True)
    data_path = rep_dir / f"{rid}.parquet"
    obs.to_parquet(data_path, index=False)
    md = render(res, sel, len(obs)) if report else ""
    rep_path = rep_dir / f"{rid}.md"
    if report:
        rep_path.write_text(md, encoding="utf-8")
    summary = {"kind": "catalyst", **res}
    ctx.db.insert("discovery_research_runs", {
        "research_id": rid, "created_at": utcnow_iso(), "is_synthetic": int(bool(bundle.is_synthetic)),
        "period_start": str(sel[0].date()) if len(sel) else str(start), "period_end": str(sel[-1].date()) if len(sel) else str(end),
        "n_dates": int(len(sel)), "n_observations": int(len(obs)),
        "params_json": to_json({"kind": "catalyst", "groups": list(GROUPS), "primary": list(PRIMARY), "min_obs": min_obs,
                                "min_dates": min_dates, "holdout": str(holdout.date())}),
        "summary_json": to_json(summary), "redundancy_json": to_json({}),
        "report_path": str(rep_path) if report else None, "data_path": str(data_path)})
    return {"research_id": rid, "n_dates": int(len(sel)), "n_obs": int(len(obs)), **res,
            "report_path": str(rep_path) if report else None}


def render(res: dict[str, Any], sel, n_obs: int) -> str:
    L = ["# Catalyst research (point-in-time replay, research only)", "",
         f"Sessions {sel[0].date() if len(sel) else '?'} .. {sel[-1].date() if len(sel) else '?'} ({len(sel)}), "
         f"{n_obs} observations. Entry at the next open; net of modelled costs. Baseline = mean net of all "
         f"research-universe members on the same date. Bonferroni z = {res['z_crit']:.2f} over {res['n_tests']} tests; "
         f"primary horizons {res['primary_horizons']}. Minimum {res['min_obs']} observations on {res['min_dates']} dates.",
         "", "| group | verdict | n | dates | h | mean net | median net | win | vs baseline | t | halves | MFE | MAE |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in res["groups"]:
        for h in HORIZONS:
            s = g["horizons"].get(h) or {}
            if not s.get("n"):
                continue
            t = s.get("vs_baseline_t_clustered")
            a, b = s.get("vs_baseline_first_half"), s.get("vs_baseline_second_half")
            L.append(f"| {g['group']} | {g['verdict'] if h == 5 else ''} | {s['n']} | {s['n_dates']} | {h}d | "
                     f"{s['mean_net']:+.2%} | {s['median_net']:+.2%} | {s['win_rate_net']:.0%} | {s['vs_baseline']:+.2%} | "
                     f"{'' if t is None else f'{t:+.2f}'} | {'' if a is None else f'{a:+.2%}'} / {'' if b is None else f'{b:+.2%}'} | "
                     f"{s['mean_mfe']:+.1%} | {s['mean_mae']:+.1%} |")
    L += ["", "## Verdicts", ""] + [f"* **{g['group']}**: {g['verdict']}. {g['why']}" for g in res["groups"]]
    L += ["", "## Period consistency (vs baseline, 5d, per year)", ""]
    for g in res["groups"]:
        y = (g["horizons"].get(5) or {}).get("by_year") or {}
        L.append(f"* {g['group']}: " + ", ".join(f"{k} {v['vs_baseline']:+.2%} ({v['dates']} dates)" for k, v in y.items()))
    L += ["", "Research only. No finding is promoted automatically; a PROMISING group becomes a hypothesis for a "
          "formal strategy experiment (PIT backtest, walk-forward, locked holdout, prospective paper)."]
    return "\n".join(L) + "\n"


def scan_bundle(ctx, end=None):
    """Real-data bundle for an end-of-day scan at ``end`` (default: today): ~520 calendar days of
    bars (>= the 300-session feature window), ~150 days of news, ~3 years of facts."""
    e = pd.Timestamp(end or pd.Timestamp.now().normalize())
    return research_bundle(ctx, str((e - pd.Timedelta(days=520)).date()), str(e.date()),
                           news_since=str((e - pd.Timedelta(days=150)).date()),
                           facts_since=str((e - pd.Timedelta(days=3 * 366)).date()))


__all__ = ["GROUPS", "PRIMARY", "analyse", "baseline", "group_frames", "research_bundle", "run_catalyst_research",
           "scan_bundle",
           "verdict"]
