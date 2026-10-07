"""Bot-database importer and missed-opportunity analysis (Phase 3, Part 4). PAPER research, read-only.

Input: a COPY of the paper bot's SQLite database (``var/quantlab.db`` on the PC; see docs/BOT-DB-IMPORT.md).
Only the shadow tables are read. They hold every candidate the bot scored, what it decided and why it
rejected the rest (``shadow_opportunities``), and the bot's own forward measurement of what happened next
(``shadow_outcomes``: plan and buy-and-hold returns, MFE/MAE, stop/target hits, excess over the benchmark).

No market data is loaded here. The outcomes are the bot's own forward records, so the sealed 2025+
historical holdout is never touched.

Rules (Part 4): report, never tune. Opportunities scored on the same day share one market, so the
effective sample is the number of distinct as-of DATES, not opportunities. No verdict is given below
``MIN_DATES`` dates, and nothing here may be used to change a filter on a single sample.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED = {
    "shadow_opportunities": ["opportunity_id", "as_of_date", "symbol", "strategy_id", "score", "bot_decision",
                             "reject_stage", "reject_reason", "is_synthetic"],
    "shadow_outcomes": ["opportunity_id", "horizon_sessions", "ret", "ret_hold", "mfe", "mae", "hit_stop",
                        "hit_target", "benchmark_ret", "excess_ret", "status"],
}
OPTIONAL = ("shadow_outcome_details", "risk_checks", "decisions")
MIN_DATES = 20            # distinct as-of dates before any group comparison is called more than descriptive

# Filter ablation (sprint P1). Fixed BEFORE any bot data was seen (docs/ALPHA-SPRINT-PREREG.md section 1).
# "Remove filter X" re-admits rejected candidates whose ONLY blocking failures are in X.
ABLATIONS = {"B_remove_liquidity": {"no_trade.liquidity"}, "C_remove_volatility": {"no_trade.volatility"}}
# F: relaxed thresholds. check -> (measured field, threshold field, comparison, relaxation factor).
RELAX = {"no_trade.liquidity": ("adv20", "min", ">=", 0.5), "no_trade.volatility": ("vol_20d", "max", "<=", 1.25),
         "risk.ev": ("ev_bps", "min_ev_bps", ">", 0.0)}


class SchemaError(ValueError):
    pass


def load(path: str | Path) -> dict[str, pd.DataFrame]:
    """Open the exported database READ-ONLY, validate the schema, return the shadow tables."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        missing = [t for t in REQUIRED if t not in tables]
        if missing:
            raise SchemaError(f"missing tables {missing}; found {sorted(tables)[:20]}")
        out = {}
        for t, cols in REQUIRED.items():
            have = [r[1] for r in con.execute(f"pragma table_info({t})")]
            lack = [c for c in cols if c not in have]
            if lack:
                raise SchemaError(f"{t}: missing columns {lack}")
            out[t] = pd.read_sql_query(f"select * from {t}", con)
        for t in OPTIONAL:
            if t in tables:
                out[t] = pd.read_sql_query(f"select * from {t}", con)
        return out
    finally:
        con.close()


def quality(f: dict[str, pd.DataFrame]) -> dict[str, Any]:
    o, r = f["shadow_opportunities"], f["shadow_outcomes"]
    real = o[o["is_synthetic"].fillna(0).astype(int) == 0]
    q: dict[str, Any] = {
        "opportunities": int(len(o)), "synthetic_excluded": int(len(o) - len(real)),
        "distinct_dates": int(real["as_of_date"].nunique()), "date_range": [str(real["as_of_date"].min()), str(real["as_of_date"].max())],
        "duplicate_opportunity_ids": int(o["opportunity_id"].duplicated().sum()),
        "decisions": real["bot_decision"].value_counts().to_dict(),
        "outcome_rows": int(len(r)), "outcome_status": r["status"].value_counts().to_dict(),
        "horizons": sorted(int(h) for h in r["horizon_sessions"].dropna().unique()),
        "opportunities_without_any_outcome": int((~real["opportunity_id"].isin(r["opportunity_id"])).sum()),
        "outcomes_without_opportunity": int((~r["opportunity_id"].isin(o["opportunity_id"])).sum()),
    }
    c = r[r["status"] == "complete"]
    q["impossible_values"] = {"abs_ret_gt_300pct": int((c["ret_hold"].abs() > 3).sum()), "mfe_negative": int((c["mfe"] < -1e-9).sum()),
                              "mae_positive": int((c["mae"] > 1e-9).sum())}
    q["status"] = "PASS" if (q["duplicate_opportunity_ids"] == 0 and q["outcomes_without_opportunity"] == 0
                              and sum(q["impossible_values"].values()) == 0) else "REVIEW"
    return q


def _cluster_ci(x: pd.Series, dates: pd.Series, n_boot: int = 2000, seed: int = 7) -> tuple[float, float] | None:
    """95% bootstrap CI of the mean, resampling whole as-of DATES (same-day candidates are correlated)."""
    d = pd.DataFrame({"x": x.to_numpy(), "d": dates.to_numpy()}).dropna()
    groups = [g["x"].to_numpy() for _, g in d.groupby("d")]
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    sums = np.array([g.sum() for g in groups]); counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summ(g: pd.DataFrame) -> dict[str, Any]:
    c = g[g["status"] == "complete"]
    if c.empty:
        return {"n": 0}
    ci = _cluster_ci(c["excess_ret"], c["as_of_date"])
    return {"n": int(len(c)), "n_dates": int(c["as_of_date"].nunique()),
            "mean_ret_hold": float(c["ret_hold"].mean()), "median_ret_hold": float(c["ret_hold"].median()),
            "mean_plan_ret": float(c["ret"].mean()), "mean_excess": float(c["excess_ret"].mean()),
            "excess_ci95_by_date": ci, "hit_rate_hold": float((c["ret_hold"] > 0).mean()),
            "mean_mfe": float(c["mfe"].mean()), "mean_mae": float(c["mae"].mean()),
            "stop_rate": float(c["hit_stop"].fillna(0).mean()), "target_rate": float(c["hit_target"].fillna(0).mean())}


def analyse(f: dict[str, pd.DataFrame]) -> dict[str, Any]:
    o = f["shadow_opportunities"]
    o = o[o["is_synthetic"].fillna(0).astype(int) == 0]
    r = f["shadow_outcomes"]
    m = r.merge(o, on="opportunity_id", how="inner")
    m["rejected"] = m["bot_decision"] != "TRADE"
    out: dict[str, Any] = {"rule": f"descriptive only; verdicts need >= {MIN_DATES} distinct as-of dates; never tune a filter on one sample",
                           "by_horizon": {}}
    for h, g in m.groupby("horizon_sessions"):
        res: dict[str, Any] = {"all": _summ(g), "traded": _summ(g[~g["rejected"]]), "rejected": _summ(g[g["rejected"]]),
                               "by_decision": {k: _summ(x) for k, x in g.groupby("bot_decision")},
                               "by_reject_stage": {k: _summ(x) for k, x in g.groupby("reject_stage")},
                               "by_strategy": {k: _summ(x) for k, x in g.groupby("strategy_id")}}
        top = g["reject_reason"].fillna("NONE").value_counts().head(15).index
        res["by_reject_reason_top15"] = {k: _summ(g[g["reject_reason"].fillna("NONE") == k]) for k in top}
        c = g[(g["status"] == "complete") & g["score"].notna()]
        res["score_rank_corr_with_excess"] = float(c["score"].rank().corr(c["excess_ret"].rank())) if len(c) > 20 else None
        n_dates = int(g["as_of_date"].nunique())
        t, rj = res["traded"], res["rejected"]
        if n_dates < MIN_DATES:
            verdict = f"INSUFFICIENT EVIDENCE: {n_dates} distinct as-of dates (< {MIN_DATES}); descriptive only"
        elif rj.get("excess_ci95_by_date") and rj["excess_ci95_by_date"][0] > (t.get("mean_excess") or 0):
            verdict = "REJECTED CANDIDATES OUTPERFORMED TRADED ONES (date-clustered CI above the traded mean): review the filters, do not change them yet"
        else:
            verdict = "no evidence that the bot rejects better trades than it takes"
        res["too_conservative_verdict"] = verdict
        out["by_horizon"][str(int(h))] = res
    return out


def _blocking(f: dict[str, pd.DataFrame], o: pd.DataFrame) -> tuple[pd.Series, str]:
    """opportunity_id -> {names of blocking (failed CRITICAL) checks}. Prefers the per-check records
    (``risk_checks``); falls back to the ``reject_reason`` text ("name: reason; name: reason")."""
    rc = f.get("risk_checks")
    if rc is not None and "candidate_id" in o.columns and not rc.empty:
        b = rc[(rc["passed"].astype(int) == 0) & (rc["severity"] == "CRITICAL")]
        names = b.groupby("candidate_id")["check_name"].agg(lambda x: frozenset(x))
        out = o["candidate_id"].map(names)
        return out.where(out.notna(), frozenset()).set_axis(o["opportunity_id"]), "risk_checks"

    def parse(txt: Any) -> frozenset:
        parts = [p.split(":")[0].strip() for p in str(txt or "").split("; ")]
        return frozenset(p if p.startswith(("no_trade.", "risk.")) else ("ai" if p.startswith("AI") else "other")
                         for p in parts if p)
    return pd.Series([parse(t) for t in o["reject_reason"]], index=o["opportunity_id"]), "reject_reason_text"


def _check_details(f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """(candidate_id, check_name) -> details dict (first record), for threshold relaxation."""
    rc = f.get("risk_checks")
    if rc is None or rc.empty or "details_json" not in rc.columns:
        return pd.DataFrame(columns=["candidate_id", "check_name", "details"])
    d = rc.drop_duplicates(["candidate_id", "check_name"])[["candidate_id", "check_name", "details_json"]]
    return d.assign(details=[json.loads(x) if isinstance(x, str) and x else {} for x in d["details_json"]])


def _ev_bps(f: dict[str, pd.DataFrame]) -> pd.Series:
    """candidate_id -> EV after costs in bps, from ``decisions.ev_json`` (computed for every candidate,
    even when the risk chain stopped before its EV stage). Missing -> NaN (UNKNOWN fails closed)."""
    d = f.get("decisions")
    if d is None or d.empty or "ev_json" not in d.columns:
        return pd.Series(dtype=float)
    ev = [(json.loads(x) or {}).get("ev") if isinstance(x, str) and x else None for x in d["ev_json"]]
    return pd.Series(pd.to_numeric(pd.Series(ev), errors="coerce").to_numpy() * 1e4, index=d["candidate_id"]).groupby(level=0).first()


def _relaxed_pass(name: str, det: dict, min_ev_bps: float) -> bool:
    if name not in RELAX:
        return False
    field_, thr, op, k = RELAX[name]
    v, t = det.get(field_), det.get(thr, min_ev_bps if name == "risk.ev" else None)
    if v is None or t is None:
        return False                                      # UNKNOWN stays failed
    lim = float(t) * k if name != "risk.ev" else k
    return {"<=": v <= lim, ">=": v >= lim, ">": v > lim}[op]


def ablation(f: dict[str, pd.DataFrame], min_ev_bps: float = 10.0) -> dict[str, Any]:
    """Sprint P1: what the bot would have added under each filter ablation, and how those candidates did.

    EXPLORATORY with the forward sample available (a handful of dates): it reports, it never picks a winner.
    Re-admitted candidates are an UPPER BOUND on what would have traded: portfolio capacity, sector caps and
    sizing were not replayed, and checks after the first failed stage were never evaluated (EV is re-checked
    from ``decisions.ev_json``; the rest is unknown). D (remove the score threshold) is not identifiable here:
    sub-threshold names never become candidates. It is tested on 2016-2024 history instead."""
    o = f["shadow_opportunities"]
    o = o[o["is_synthetic"].fillna(0).astype(int) == 0].copy()
    blk, src = _blocking(f, o)
    o["blocking"] = o["opportunity_id"].map(blk)
    o["blocking"] = [b if isinstance(b, frozenset) else frozenset() for b in o["blocking"]]
    ev = _ev_bps(f)
    o["ev_bps"] = o["candidate_id"].map(ev) if "candidate_id" in o.columns and not ev.empty else np.nan
    det = _check_details(f)
    dmap = {(r.candidate_id, r.check_name): r.details for r in det.itertuples()} if not det.empty else {}
    rejected = o["bot_decision"] != "TRADE"
    ev_known = o["ev_bps"].notna().any()

    def readmit(removed: set[str] | None = None, relax: bool = False) -> pd.Series:
        keep = []
        for r in o.itertuples():
            if r.bot_decision == "TRADE" or not r.blocking:
                keep.append(False)
                continue
            left = set(r.blocking) - (removed or set())
            if relax:
                left = {n for n in left if not _relaxed_pass(n, dmap.get((getattr(r, "candidate_id", None), n), {}) |
                                                              ({"ev_bps": r.ev_bps} if n == "risk.ev" else {}), min_ev_bps)}
            ok = not left
            if ok and ev_known and "risk.ev" not in set(r.blocking):  # EV stage may not have run: re-check it
                lim = 0.0 if relax else min_ev_bps
                ok = bool(np.isfinite(r.ev_bps) and r.ev_bps > lim)
            keep.append(ok)
        return pd.Series(keep, index=o.index)

    scen = {"A_current": pd.Series(False, index=o.index)}
    scen.update({k: readmit(v) for k, v in ABLATIONS.items()})
    seen = sorted({n for b in o.loc[rejected, "blocking"] for n in b})
    scen.update({f"E_remove_{n}": readmit({n}) for n in seen if n not in set().union(*ABLATIONS.values())})
    scen["F_relaxed_thresholds"] = readmit(relax=True)
    r = f["shadow_outcomes"]
    m = r.merge(o[["opportunity_id", "as_of_date", "strategy_id", "bot_decision"]], on="opportunity_id", how="inner")
    n_dates = int(o["as_of_date"].nunique())
    out: dict[str, Any] = {
        "status": (f"EXPLORATORY: {n_dates} distinct as-of dates. No winner is declared below {MIN_DATES} dates, and no "
                   "filter is changed on this sample." if n_dates < MIN_DATES else
                   "descriptive; a filter change still needs a pre-registered forward test"),
        "blocking_source": src, "ev_recheck": "decisions.ev_json" if ev_known else "UNKNOWN (no decisions table)",
        "D_remove_score_threshold": "NOT IDENTIFIABLE from the bot database (sub-threshold names are never recorded); "
                                    "see the historical threshold ablation",
        "relaxation_factors": {k: {"field": v[0], "factor_or_limit": v[3]} for k, v in RELAX.items()},
        "scenarios": {}}
    traded_ids = set(o.loc[~rejected, "opportunity_id"])
    for name, mask in scen.items():
        added = set(o.loc[mask, "opportunity_id"])
        res = {"n_added": len(added), "added_by_strategy": o.loc[mask, "strategy_id"].value_counts().to_dict(),
               "by_horizon": {}}
        for h, g in m.groupby("horizon_sessions"):
            res["by_horizon"][str(int(h))] = {"traded_now": _summ(g[g["opportunity_id"].isin(traded_ids)]),
                                              "added": _summ(g[g["opportunity_id"].isin(added)]),
                                              "book_after": _summ(g[g["opportunity_id"].isin(traded_ids | added)])}
        out["scenarios"][name] = res
    return out


def run(path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    f = load(path)
    res = {"source": str(path), "quality": quality(f), "analysis": analyse(f), "ablation": ablation(f)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "botdb_missed_opportunities.json").write_text(json.dumps(res, indent=1, default=str))
    return res
