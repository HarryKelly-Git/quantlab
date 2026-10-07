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
OPTIONAL = ("shadow_outcome_details",)
MIN_DATES = 20            # distinct as-of dates before any group comparison is called more than descriptive


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


def run(path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    f = load(path)
    res = {"source": str(path), "quality": quality(f), "analysis": analyse(f)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "botdb_missed_opportunities.json").write_text(json.dumps(res, indent=1, default=str))
    return res
