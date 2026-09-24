"""LLM monitoring: operational health of the optional AI layer and, above all, the forward test of
"does AI filtering improve results?".

Everything here READS the audit trail (``ai_calls``, ``ai_assessments``, ``shadow_opportunities``,
``shadow_outcomes``, ``candidates``). Nothing is written.

POINT-IN-TIME / CONTAMINATION RULES for the forward-performance test (ARCHITECTURE.md 2.9):
  * An AI decision about a candidate as of session D is CONTAMINATED when D <= the model's
    knowledge cutoff (``ai.knowledge_cutoffs``): the model may already know what happened next.
    A model with an unknown (null / missing) cutoff is treated as contaminated. Model names are
    matched exactly; a served snapshot name that is not in config is therefore excluded (and
    reported) rather than guessed.
  * Only near-real-time decisions count: an assessment recorded more than
    ``monitoring.llm_max_decision_lag_days`` after D is excluded as LATE. Provider model ids can be
    moving aliases, so a retroactive call may be answered by a newer model that knows the outcome.
  * Synthetic candidates/opportunities are never evidence.
  * The first assessment per (candidate, role, model) is the decision; later retries are ignored.
  Every exclusion is counted and reported so the operator can see how much evidence was dropped.

Statistics: group means with a stationary block bootstrap over SESSION clusters (candidates on the
same date share market moves; adjacent dates share holding periods), percentile CIs, and an
explicit INSUFFICIENT_SAMPLE label below ``monitoring.llm_min_forward_sample`` (default:
``validation.min_trades_for_conclusion``). Results are MODEL_OUTPUT, never FACT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.types import AIDecision, InfoKind
from quantlab.db.database import Database, from_json
from quantlab.logging_setup import get_logger

log = get_logger(__name__)

ATTEMPTED_STATUSES = ("ok", "error", "invalid_schema")      # a real provider call was made
FAILURE_STATUSES = ("error", "invalid_schema")
USABLE_OUTCOME_STATUSES = ("complete", "delisted")           # delisted carries costs.delisting_return
GUARD_KEYS = ("invalid_references", "unverified_numbers")
DECISIONS = [d.value for d in AIDecision]

INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


# =============================================================================================
# Shared statistics helpers (also used by research.promotion)
# =============================================================================================
def stationary_bootstrap_counts(n: int, n_resamples: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(n_resamples, n) matrix: how often each of n ordered clusters appears in each resample.

    Politis-Romano stationary bootstrap: blocks start at uniform positions and have geometric
    lengths with mean ``mean_block`` (circular wrap). ``mean_block=1`` is the iid bootstrap.
    """
    if n <= 0 or n_resamples <= 0:
        return np.zeros((max(n_resamples, 0), max(n, 0)), dtype=np.int64)
    p = 1.0 / max(1.0, min(float(mean_block), float(n)))
    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_resamples)
    new_block = rng.random((n_resamples, n)) < p
    jumps = rng.integers(0, n, (n_resamples, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % n)
    flat = idx + (np.arange(n_resamples)[:, None] * n)
    return np.bincount(flat.ravel(), minlength=n_resamples * n).reshape(n_resamples, n)


def bootstrap_group_means(
    values: np.ndarray | pd.Series,
    clusters: np.ndarray | pd.Series,
    groups: dict[str, np.ndarray | pd.Series],
    diffs: list[tuple[str, str]] | None = None,
    n_resamples: int = 2000,
    mean_block: float = 20.0,
    seed: int = 7,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Mean of ``values`` within each boolean group mask, with cluster-bootstrap percentile CIs.

    Clusters (e.g. session dates) are resampled jointly for all groups, so differences between
    overlapping groups (ACCEPT vs ALL) keep their dependence. Cluster keys must sort
    chronologically (ISO date strings do). CIs are None with fewer than 2 clusters.
    """
    vals = np.asarray(values, dtype=float)
    keys = np.asarray(clusters)
    if vals.shape[0] != keys.shape[0]:
        raise ValueError("values and clusters must have the same length")
    if np.isnan(vals).any():
        raise ValueError("values contain NaN: filter them out before bootstrapping")
    uniq, inv = np.unique(keys, return_inverse=True) if len(keys) else (np.array([]), np.array([], dtype=int))
    n_c = len(uniq)
    alpha = 1.0 - float(ci_level)
    boot_w = None
    if n_c >= 2 and n_resamples > 0:
        boot_w = stationary_bootstrap_counts(n_c, int(n_resamples), mean_block, np.random.default_rng(seed)).astype(float)

    out: dict[str, Any] = {"groups": {}, "diffs": {}, "n_clusters": int(n_c), "ci_level": float(ci_level),
                           "n_resamples": int(n_resamples) if boot_w is not None else 0}
    boots: dict[str, np.ndarray | None] = {}
    for name, mask in groups.items():
        m = np.asarray(mask, dtype=bool)
        n = int(m.sum())
        sums = np.bincount(inv, weights=np.where(m, vals, 0.0), minlength=n_c) if n_c else np.array([])
        cnts = np.bincount(inv, weights=m.astype(float), minlength=n_c) if n_c else np.array([])
        mean = float(sums.sum() / cnts.sum()) if n else None
        ci = None
        boot = None
        if boot_w is not None and n:
            c = boot_w @ cnts
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = np.where(c > 0, (boot_w @ sums) / np.where(c > 0, c, 1.0), np.nan)
            ci = _percentile_ci(boot, alpha)
        boots[name] = boot
        out["groups"][name] = {
            "n": n,
            "n_clusters": int((cnts > 0).sum()) if n_c else 0,
            "mean": mean,
            "win_rate": float((vals[m] > 0).mean()) if n else None,
            "ci": ci,
        }
    for a, b in diffs or []:
        ga, gb = out["groups"].get(a), out["groups"].get(b)
        est = None if ga is None or gb is None or ga["mean"] is None or gb["mean"] is None else ga["mean"] - gb["mean"]
        ci = None
        if est is not None and boots.get(a) is not None and boots.get(b) is not None:
            ci = _percentile_ci(boots[a] - boots[b], alpha)
        out["diffs"][f"{a}-{b}"] = {"estimate": est, "ci": ci}
    return out


def _percentile_ci(boot: np.ndarray, alpha: float) -> list[float] | None:
    finite = boot[np.isfinite(boot)]
    # Too many resamples without the group => the CI would describe a different population.
    if len(finite) < max(10, 0.9 * len(boot)):
        return None
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)]


def select_primary_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """One usable outcome row per opportunity.

    Input columns: opportunity_id, horizon_sessions, status, holding_sessions (plan horizon) and the
    metric columns. The PLAN horizon (horizon_sessions == holding_sessions) is preferred because it
    is what the trade plan would actually have held; otherwise the longest usable horizon.
    Rows whose status is not complete/delisted (e.g. no_data) are never used.
    """
    if outcomes.empty:
        return outcomes.copy()
    df = outcomes[outcomes["status"].isin(USABLE_OUTCOME_STATUSES)].copy()
    if df.empty:
        return df
    df["_plan"] = (df["horizon_sessions"] == df["holding_sessions"]).astype(int)
    df = df.sort_values(["opportunity_id", "_plan", "horizon_sessions"], ascending=[True, False, False])
    return df.drop_duplicates("opportunity_id", keep="first").drop(columns="_plan").reset_index(drop=True)


# =============================================================================================
# Time windows
# =============================================================================================
def _to_utc(x: Any, end: bool = False) -> pd.Timestamp | None:
    """Window bound -> UTC timestamp. A date-only END bound includes that whole UTC day."""
    if x is None:
        return None
    date_only = (isinstance(x, date) and not isinstance(x, datetime)) or (isinstance(x, str) and len(x.strip()) == 10)
    ts = pd.Timestamp(x)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if end:
        ts = ts + (pd.Timedelta(days=1) if date_only else pd.Timedelta(microseconds=1))
    return ts


def _window_mask(series: pd.Series, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    mask = ts.notna()
    if lo is not None:
        mask &= ts >= lo
    if hi is not None:
        mask &= ts < hi
    return mask


def _rate(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


# =============================================================================================
# Report
# =============================================================================================
@dataclass
class LLMReport:
    window: dict[str, str | None]
    calls: list[dict[str, Any]]
    totals: dict[str, Any]
    decisions: list[dict[str, Any]]
    disagreement: list[dict[str, Any]]
    hallucination_guard: list[dict[str, Any]]
    forward_performance: list[dict[str, Any]]
    ai_filter_verdict: str
    answer: str
    info_kind: str = InfoKind.MODEL_OUTPUT.value
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        t = self.totals
        cost = "UNKNOWN" if t.get("cost_usd") is None else f"{t['cost_usd']:.4f}"
        if t.get("cost_estimated"):
            cost += " (partly estimated from token counts)"
        lines = [
            "## AI layer monitor",
            f"_{self.info_kind}. Window: {self.window.get('start') or 'beginning'} .. {self.window.get('end') or 'now'}_",
            "",
            f"**Does AI filtering improve results?** {self.ai_filter_verdict}. {self.answer}",
            "",
            f"Calls attempted {t.get('n_attempted', 0)}, failure rate {_fmt_pct(t.get('failure_rate'))}, "
            f"invalid-schema rate {_fmt_pct(t.get('invalid_schema_rate'))}, tokens in/out "
            f"{t.get('input_tokens', 0)}/{t.get('output_tokens', 0)}, cost USD {cost}.",
            "",
            "| role | model | clean n | ACCEPT mean | REJECT mean | ACCEPT-ALL (CI) | verdict | excluded |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for fp in self.forward_performance:
            g = fp["primary"]["groups"]
            d = fp["primary"]["diffs"].get("ACCEPT-ALL", {})
            lines.append(
                f"| {fp['role']} | {fp['model']} | {fp['n_clean']} | {_fmt_ret(g.get('ACCEPT', {}).get('mean'))} "
                f"(n={g.get('ACCEPT', {}).get('n', 0)}) | {_fmt_ret(g.get('REJECT', {}).get('mean'))} "
                f"(n={g.get('REJECT', {}).get('n', 0)}) | {_fmt_ret(d.get('estimate'))} {_fmt_ci(d.get('ci'))} | "
                f"{fp['verdict']} | {', '.join(f'{k}={v}' for k, v in fp['excluded'].items() if v) or '-'} |"
            )
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines) + "\n"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _fmt_ret(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.2f}%"


def _fmt_ci(ci: list[float] | None) -> str:
    return "" if not ci else f"[{100 * ci[0]:+.2f}%, {100 * ci[1]:+.2f}%]"


class LLMMonitor:
    """Read-only monitor over the AI audit tables. See module docstring for the rules."""

    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config

    # -- configuration --------------------------------------------------------------------------
    def knowledge_cutoff(self, model: str) -> date | None:
        """Configured knowledge cutoff for an exact model id; None = unknown (=> contaminated)."""
        raw = (self.config.get("ai.knowledge_cutoffs", {}) or {}).get(model)
        if raw in (None, "", "null"):
            return None
        try:
            return pd.Timestamp(raw).date()
        except (ValueError, TypeError):
            log.warning("unparseable knowledge cutoff for %s: %r (treated as UNKNOWN)", model, raw)
            return None

    def _pricing(self, model: str) -> tuple[float, float] | None:
        p = (self.config.get("ai.pricing_per_mtok", {}) or {}).get(model)
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p):
            return float(p[0]), float(p[1])
        return None

    def _min_sample(self) -> int:
        return int(self.config.get("monitoring.llm_min_forward_sample",
                                   self.config.get("validation.min_trades_for_conclusion", 100)))

    def active_judge_model(self) -> str | None:
        prov = self.config.get("ai.active_provider", None)
        return self.config.get(f"ai.providers.{prov}.models.judge", None) if prov else None

    # -- operational call statistics ------------------------------------------------------------
    def call_stats(self, start: Any = None, end: Any = None) -> dict[str, Any]:
        """Per (role, provider, model) call statistics plus totals, from ``ai_calls``."""
        df = self.db.query_df(
            "SELECT role, provider, model, status, started_at, latency_ms, input_tokens, output_tokens, "
            "cached_tokens, cost_usd FROM ai_calls"
        )
        lo, hi = _to_utc(start), _to_utc(end, end=True)
        if not df.empty:
            df = df[_window_mask(df["started_at"], lo, hi)]
        rows = [self._call_row(g, dict(zip(("role", "provider", "model"), k)))
                for k, g in df.groupby(["role", "provider", "model"], sort=True)] if not df.empty else []
        return {"by_model": rows, "totals": self._call_row(df, {})}

    def _call_row(self, g: pd.DataFrame, key: dict[str, Any]) -> dict[str, Any]:
        status = g["status"] if not g.empty else pd.Series(dtype=object)
        n_attempted = int(status.isin(ATTEMPTED_STATUSES).sum())
        n_error = int((status == "error").sum())
        n_invalid = int((status == "invalid_schema").sum())
        lat = pd.to_numeric(g.loc[status.isin(ATTEMPTED_STATUSES), "latency_ms"], errors="coerce").dropna() if not g.empty else pd.Series(dtype=float)
        cost, estimated, n_unknown_cost = self._cost(g)
        row = dict(key)
        row.update({
            "n_calls": int(len(g)),
            "n_attempted": n_attempted,
            "n_ok": int((status == "ok").sum()),
            "n_error": n_error,
            "n_invalid_schema": n_invalid,
            "n_cache_hit": int((status == "cache_hit").sum()),
            "n_budget_exceeded": int((status == "budget_exceeded").sum()),
            "n_disabled": int((status == "disabled").sum()),
            "error_rate": _rate(n_error, n_attempted),
            "invalid_schema_rate": _rate(n_invalid, n_attempted),
            "failure_rate": _rate(n_error + n_invalid, n_attempted),
            "input_tokens": int(pd.to_numeric(g.get("input_tokens"), errors="coerce").fillna(0).sum()) if not g.empty else 0,
            "output_tokens": int(pd.to_numeric(g.get("output_tokens"), errors="coerce").fillna(0).sum()) if not g.empty else 0,
            "cached_tokens": int(pd.to_numeric(g.get("cached_tokens"), errors="coerce").fillna(0).sum()) if not g.empty else 0,
            "cost_usd": cost,
            "cost_estimated": estimated,
            "n_unknown_cost": n_unknown_cost,
            "latency_ms": {
                "n": int(len(lat)),
                "p50": float(np.percentile(lat, 50)) if len(lat) else None,
                "p90": float(np.percentile(lat, 90)) if len(lat) else None,
                "p99": float(np.percentile(lat, 99)) if len(lat) else None,
            },
        })
        return row

    def _cost(self, g: pd.DataFrame) -> tuple[float | None, bool, int]:
        """Recorded cost, estimated from tokens x configured pricing when missing. Calls whose cost
        cannot be established make the total UNKNOWN (None) rather than silently understated."""
        if g.empty:
            return 0.0, False, 0
        total, estimated, unknown = 0.0, False, 0
        for r in g.itertuples(index=False):
            if r.cost_usd is not None and not pd.isna(r.cost_usd):
                total += float(r.cost_usd)
                continue
            if r.status in ("cache_hit", "disabled", "budget_exceeded") and not (r.input_tokens or r.output_tokens):
                continue                                     # no billable call was made
            price = self._pricing(r.model)
            if price is None or r.input_tokens is None or pd.isna(r.input_tokens):
                unknown += 1
                continue
            out_tok = 0 if r.output_tokens is None or pd.isna(r.output_tokens) else float(r.output_tokens)
            total += (float(r.input_tokens) * price[0] + out_tok * price[1]) / 1e6
            estimated = True
        return (None if unknown else total), estimated, unknown

    # -- assessments ----------------------------------------------------------------------------
    def _assessments(self, start: Any, end: Any) -> pd.DataFrame:
        df = self.db.query_df(
            "SELECT assessment_id, candidate_id, role, provider, model, decision, assessment_json, created_at "
            "FROM ai_assessments"
        )
        if df.empty:
            return df
        df = df[_window_mask(df["created_at"], _to_utc(start), _to_utc(end, end=True))].copy()
        df["assessed_ts"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
        # The FIRST assessment per (candidate, role, model) is the decision that was acted on.
        df = df.sort_values(["assessed_ts", "assessment_id"]).drop_duplicates(["candidate_id", "role", "model"], keep="first")
        return df.reset_index(drop=True)

    @staticmethod
    def _decision_rates(a: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        if a.empty:
            return out
        for (role, model), g in a.groupby(["role", "model"], sort=True):
            dec = g["decision"].where(g["decision"].isin(DECISIONS), AIDecision.UNKNOWN.value)
            counts = {d: int((dec == d).sum()) for d in DECISIONS}
            n = int(len(g))
            out.append({"role": role, "model": model, "n": n, "counts": counts,
                        **{f"{d.lower()}_rate": _rate(counts[d], n) for d in DECISIONS}})
        return out

    @staticmethod
    def _disagreement(a: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        if a.empty:
            return out
        known = a[a["decision"].isin([AIDecision.ACCEPT.value, AIDecision.REJECT.value, AIDecision.WATCH.value])]
        for role, g in known.groupby("role", sort=True):
            wide = g.pivot_table(index="candidate_id", columns="model", values="decision", aggfunc="first")
            multi = wide[wide.notna().sum(axis=1) >= 2]
            n_disagree = int((multi.nunique(axis=1, dropna=True) > 1).sum()) if len(multi) else 0
            pairs = []
            for m1, m2 in combinations(sorted(wide.columns), 2):
                both = wide[[m1, m2]].dropna()
                if len(both):
                    pairs.append({"model_a": m1, "model_b": m2, "n": int(len(both)),
                                  "agreement_rate": float((both[m1] == both[m2]).mean())})
            out.append({"role": role, "n_candidates_compared": int(len(multi)), "n_disagree": n_disagree,
                        "disagreement_rate": _rate(n_disagree, len(multi)), "pairs": pairs})
        return out

    @staticmethod
    def _hallucination_guard(a: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        if a.empty:
            return out
        for (role, model), g in a.groupby(["role", "model"], sort=True):
            n_unparseable = 0
            present = {k: 0 for k in GUARD_KEYS}
            flagged = {k: 0 for k in GUARD_KEYS}
            n_with_guard, n_any_flag = 0, 0
            for text in g["assessment_json"]:
                try:
                    obj = from_json(text, {})
                except (ValueError, TypeError):
                    n_unparseable += 1
                    continue
                if not isinstance(obj, dict):
                    n_unparseable += 1
                    continue
                has_any, any_flag = False, False
                for k in GUARD_KEYS:
                    if k in obj:
                        has_any = True
                        present[k] += 1
                        if _truthy_flag(obj[k]):
                            flagged[k] += 1
                            any_flag = True
                n_with_guard += int(has_any)
                n_any_flag += int(any_flag)
            out.append({
                "role": role, "model": model, "n_assessments": int(len(g)), "n_with_guard_fields": n_with_guard,
                "n_unparseable": n_unparseable,
                **{f"{k}_rate": _rate(flagged[k], present[k]) for k in GUARD_KEYS},
                "any_flag_rate": _rate(n_any_flag, n_with_guard),
            })
        return out

    # -- forward performance ---------------------------------------------------------------------
    def _forward_frame(self, a: pd.DataFrame) -> pd.DataFrame:
        """Assessments joined to their shadow opportunity + primary outcome, with an exclusion reason."""
        if a.empty:
            return a.assign(exclusion=pd.Series(dtype=object))
        opps = self.db.query_df(
            "SELECT o.opportunity_id, o.candidate_id, o.as_of_date, o.holding_sessions, o.is_synthetic AS opp_synthetic, "
            "o.created_at AS opp_created_at FROM shadow_opportunities o "
            "WHERE o.candidate_id IN (SELECT candidate_id FROM ai_assessments)"
        )
        if not opps.empty:   # one opportunity per candidate: the first one recorded
            opps = opps.sort_values(["opp_created_at", "opportunity_id"]).drop_duplicates("candidate_id", keep="first")
        cands = self.db.query_df(
            "SELECT candidate_id, as_of_date AS cand_as_of, is_synthetic AS cand_synthetic FROM candidates "
            "WHERE candidate_id IN (SELECT candidate_id FROM ai_assessments)"
        )
        outs = self.db.query_df(
            "SELECT so.opportunity_id, so.horizon_sessions, so.status, so.ret, so.excess_ret, o.holding_sessions "
            "FROM shadow_outcomes so JOIN shadow_opportunities o ON o.opportunity_id = so.opportunity_id "
            "WHERE o.candidate_id IN (SELECT candidate_id FROM ai_assessments)"
        )
        df = a.merge(opps, on="candidate_id", how="left") if not opps.empty else a.assign(
            opportunity_id=None, as_of_date=None, holding_sessions=None, opp_synthetic=None, opp_created_at=None)
        df = df.merge(cands, on="candidate_id", how="left") if not cands.empty else df.assign(cand_as_of=None, cand_synthetic=None)
        df["as_of_date"] = df["as_of_date"].where(df["as_of_date"].notna(), df["cand_as_of"])
        prim = select_primary_outcomes(outs) if not outs.empty else pd.DataFrame(
            columns=["opportunity_id", "horizon_sessions", "status", "ret", "excess_ret"])
        df = df.merge(prim[["opportunity_id", "horizon_sessions", "ret", "excess_ret"]], on="opportunity_id", how="left")

        lag_days = int(self.config.get("monitoring.llm_max_decision_lag_days", 5))
        reasons = []
        for r in df.itertuples(index=False):
            reasons.append(self._exclusion(r, lag_days))
        df["exclusion"] = reasons
        return df

    def _exclusion(self, r: Any, lag_days: int) -> str | None:
        if _flag(getattr(r, "opp_synthetic", None)) or _flag(getattr(r, "cand_synthetic", None)):
            return "synthetic"
        if r.as_of_date is None or pd.isna(r.as_of_date):
            return "no_opportunity"
        as_of = pd.Timestamp(r.as_of_date).date()
        cutoff = self.knowledge_cutoff(r.model)
        if cutoff is None:
            return "unknown_cutoff"
        if as_of <= cutoff:
            return "contaminated"
        if (r.assessed_ts.date() - as_of).days > lag_days:
            return "late_decision"
        if r.opportunity_id is None or pd.isna(r.opportunity_id):
            return "no_opportunity"
        if r.ret is None or pd.isna(r.ret):
            return "no_outcome"
        return None

    def _performance_block(self, df: pd.DataFrame, metric: str) -> dict[str, Any]:
        d = df[df[metric].notna()]
        dec = d["decision"].where(d["decision"].isin(DECISIONS), AIDecision.UNKNOWN.value).to_numpy()
        groups = {name: dec == name for name in DECISIONS}
        groups["ALL"] = np.ones(len(d), dtype=bool)
        res = bootstrap_group_means(
            d[metric].to_numpy(dtype=float), d["as_of_date"].astype(str).to_numpy(), groups,
            diffs=[("ACCEPT", "REJECT"), ("ACCEPT", "ALL")],
            n_resamples=int(self.config.get("validation.bootstrap.n_resamples", 2000)),
            mean_block=float(self.config.get("validation.bootstrap.block_length", 20)),
            seed=int(self.config.get("validation.bootstrap.seed", 7)),
            ci_level=float(self.config.get("monitoring.llm_ci_level", 0.95)),
        )
        min_n = self._min_sample()
        for g in res["groups"].values():
            g["label"] = INSUFFICIENT_SAMPLE if g["n"] < min_n else "OK"
        n = {k: v["n"] for k, v in res["groups"].items()}
        res["diffs"]["ACCEPT-REJECT"]["label"] = INSUFFICIENT_SAMPLE if min(n["ACCEPT"], n["REJECT"]) < min_n else "OK"
        res["diffs"]["ACCEPT-ALL"]["label"] = INSUFFICIENT_SAMPLE if min(n["ACCEPT"], n["REJECT"]) < min_n else "OK"
        res["metric"] = metric
        res["min_sample"] = min_n
        return res

    @staticmethod
    def _verdict(block: dict[str, Any]) -> str:
        d = block["diffs"]["ACCEPT-ALL"]
        if d["label"] == INSUFFICIENT_SAMPLE:
            return INSUFFICIENT_SAMPLE
        if d["ci"] is None:
            return "UNKNOWN"
        lo, hi = d["ci"]
        if lo > 0:
            return "IMPROVES"
        if hi < 0:
            return "HURTS"
        return "NO_SIGNIFICANT_EFFECT"

    def forward_performance(self, start: Any = None, end: Any = None) -> list[dict[str, Any]]:
        """AI-accepted vs AI-rejected (and vs all assessed) forward shadow outcomes per (role, model)."""
        a = self._assessments(start, end)
        df = self._forward_frame(a)
        if df.empty:
            return []
        metric = str(self.config.get("monitoring.llm_outcome_metric", "ret"))
        out = []
        for (role, model), g in df.groupby(["role", "model"], sort=True):
            clean = g[g["exclusion"].isna()]
            excluded = {k: int((g["exclusion"] == k).sum()) for k in
                        ("synthetic", "unknown_cutoff", "contaminated", "late_decision", "no_opportunity", "no_outcome")}
            primary = self._performance_block(clean, metric)
            cutoff = self.knowledge_cutoff(model)
            out.append({
                "role": role, "model": model,
                "knowledge_cutoff": cutoff.isoformat() if cutoff else None,
                "n_assessed": int(len(g)), "n_clean": int(len(clean)), "excluded": excluded,
                "primary": primary,
                "secondary_excess": self._performance_block(clean, "excess_ret") if metric != "excess_ret" else None,
                "verdict": self._verdict(primary),
            })
        return out

    # -- full report ----------------------------------------------------------------------------
    def report(self, start: Any = None, end: Any = None) -> LLMReport:
        cs = self.call_stats(start, end)
        a = self._assessments(start, end)
        fwd = self.forward_performance(start, end)
        judge = self.active_judge_model()
        notes: list[str] = []
        head = next((f for f in fwd if f["role"] == "judge" and f["model"] == judge), None)
        if head is None:
            verdict = INSUFFICIENT_SAMPLE
            answer = (f"No clean forward evidence for the active judge model ({judge or 'none configured'}). "
                      "AI filtering is UNPROVEN.")
        else:
            verdict = head["verdict"]
            g = head["primary"]["groups"]
            d = head["primary"]["diffs"]["ACCEPT-ALL"]
            answer = (f"judge {judge}: ACCEPT n={g['ACCEPT']['n']}, REJECT n={g['REJECT']['n']} "
                      f"(need {head['primary']['min_sample']} each); ACCEPT minus ALL = {_fmt_ret(d['estimate'])} "
                      f"{_fmt_ci(d['ci'])}.")
            if verdict != "IMPROVES":
                answer += " AI filtering is UNPROVEN." if verdict in (INSUFFICIENT_SAMPLE, "NO_SIGNIFICANT_EFFECT", "UNKNOWN") else ""
        excluded_total = {}
        for f in fwd:
            for k, v in f["excluded"].items():
                excluded_total[k] = excluded_total.get(k, 0) + v
        if excluded_total.get("unknown_cutoff"):
            notes.append(f"{excluded_total['unknown_cutoff']} decisions excluded: model knowledge cutoff unknown "
                         "(add it to ai.knowledge_cutoffs only from provider documentation).")
        if excluded_total.get("contaminated"):
            notes.append(f"{excluded_total['contaminated']} decisions excluded as CONTAMINATED (as_of_date <= model cutoff).")
        lo, hi = _to_utc(start), _to_utc(end, end=True)
        return LLMReport(
            window={"start": lo.isoformat() if lo is not None else None, "end": hi.isoformat() if hi is not None else None},
            calls=cs["by_model"], totals=cs["totals"],
            decisions=self._decision_rates(a), disagreement=self._disagreement(a),
            hallucination_guard=self._hallucination_guard(a), forward_performance=fwd,
            ai_filter_verdict=verdict, answer=answer, notes=notes,
        )


def _flag(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and bool(int(x))


def _truthy_flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, (list, tuple, dict, str)):
        return len(v) > 0
    return v is not None


def utc_days_ago(days: float) -> str:
    """Convenience for callers building windows, e.g. ``report(start=utc_days_ago(30))``."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
