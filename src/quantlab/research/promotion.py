"""Strategy promotion: IDEA -> RESEARCH -> BACKTEST -> WALK_FORWARD -> LOCKED_HOLDOUT -> SHADOW ->
PAPER -> PROMOTED (``core.types.STRATEGY_STAGE_ORDER``).

A strategy's ``stage`` is where it currently is; to LEAVE a stage it must meet that stage's gate,
and gates are cumulative (moving SHADOW -> PAPER still requires the backtest, walk-forward and
holdout gates, re-checked against the current evidence).

    leave IDEA / RESEARCH  : registered (strategies row with params)
    leave BACKTEST         : >= promotion.min_backtest_trades trades in ONE qualifying backtest
    leave WALK_FORWARD     : latest qualifying walk-forward has >= min_walk_forward_windows OOS
                             windows, OOS Sharpe >= min_oos_sharpe, and a deflated-Sharpe p-value
                             <= max_deflated_sharpe_pvalue (missing p-value => unmet)
    leave LOCKED_HOLDOUT   : FIRST holdout evaluation succeeded, was logged in holdout_access_log,
                             has >= min_holdout_trades trades and Sharpe >= min_holdout_sharpe;
                             no unlogged holdout access exists
    leave SHADOW           : >= min_shadow_sessions forward-recorded sessions, >= min_shadow_outcomes
                             measured outcomes, mean outcome > min_shadow_expectancy
    leave PAPER            : >= min_paper_trades closed BOT paper trades, mean return >
                             min_paper_expectancy, and no degradation breach
    all forward moves      : strategy status is ACTIVE or SHADOW (not PAUSED/RETIRED)

Evidence rules (why promotion can never rest on a backtest return alone):
  * Experiments with synthetic data (flag, or a synthetic dataset in their snapshot) are excluded.
  * Historical experiments that used AI models are CONTAMINATED_HISTORICAL and excluded.
  * Failed experiments are listed but never count. Nothing is summed across reruns: the backtest
    gate uses the single best-populated run; walk-forward uses the LATEST run (not the best one);
    the holdout uses the FIRST evaluation (re-running the holdout cannot improve the verdict).
  * An experiment whose period reaches the holdout start is holdout evidence, never backtest or
    walk-forward evidence, and must have a holdout_access_log entry.
  * Shadow sessions count only when recorded within ``promotion.max_shadow_record_lag_days`` of
    their as-of date (backfilled "forward" records are not forward evidence).
  * The last two gates need forward (shadow/paper) evidence.

Writes: ``strategies`` (stage/status/updated_at) and the append-only ``strategy_status_log`` in
one transaction, log first (migration 096 enforces the order). Stages can never be skipped.
Unmet requirements block a forward move unless the actor is HUMAN and passes an explicit
``override_reason``; the override is logged as such. AI actors can never promote or activate.
Moving backwards (demotion) is always allowed with a reason: it is the conservative direction.
History is never deleted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.types import STRATEGY_STAGE_ORDER, InfoKind, StrategyStage, StrategyStatus
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event
from quantlab.monitoring.killswitch import is_ai_actor, is_human_actor, require_actor
from quantlab.monitoring.llm_monitor import bootstrap_group_means, select_primary_outcomes

log = get_logger(__name__)

# Metric names looked up in experiment_results.metrics_json (strategy-scoped sub-dicts
# metrics["strategies"][strategy_id] or metrics[strategy_id] are searched first).
SHARPE_KEYS = ("oos_sharpe", "sharpe", "sharpe_ratio")
HOLDOUT_SHARPE_KEYS = ("holdout_sharpe", "sharpe", "oos_sharpe", "sharpe_ratio")
DSR_PVALUE_KEYS = ("deflated_sharpe_pvalue", "dsr_pvalue")
DSR_PROB_KEYS = ("deflated_sharpe", "dsr", "deflated_sharpe_ratio", "dsr_probability")
WINDOW_KEYS = ("n_windows", "walk_forward_windows", "n_oos_windows")
TRADE_KEYS = ("n_trades", "trades", "num_trades")
CONTAMINATED = "CONTAMINATED_HISTORICAL"

FORWARD_OK_STATUSES = (StrategyStatus.ACTIVE, StrategyStatus.SHADOW)
TRADABLE_STAGES = (StrategyStage.PAPER, StrategyStage.PROMOTED)


class PromotionError(ValueError):
    """Invalid promotion request."""


class StageSkipError(PromotionError):
    """A forward transition may only move to the next stage."""


class PromotionBlockedError(PromotionError):
    """Requirements for the target stage are unmet (and no valid human override was given)."""

    def __init__(self, message: str, unmet: list["Requirement"]):
        super().__init__(message)
        self.unmet = unmet


class PromotionPermissionError(PermissionError):
    """The actor may not make this change (AI promotion, non-human resume/override, ...)."""


class StrategyNotRegisteredError(KeyError):
    pass


# =============================================================================================
# Records
# =============================================================================================
@dataclass
class Requirement:
    name: str
    gate: str                   # the stage this requirement lets a strategy LEAVE
    met: bool
    value: Any
    threshold: Any
    detail: str = ""


@dataclass
class StrategyEvidence:
    strategy_id: str
    version: str
    backtest: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    holdout: dict[str, Any] = field(default_factory=dict)
    shadow: dict[str, Any] = field(default_factory=dict)
    paper: dict[str, Any] = field(default_factory=dict)
    excluded_experiments: list[dict[str, Any]] = field(default_factory=list)
    integrity_issues: list[str] = field(default_factory=list)
    gathered_at: str = ""
    info_kind: str = InfoKind.FACT.value      # counts/metrics read from the audit trail

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DegradationResult:
    strategy_id: str
    version: str
    recommend_pause: bool
    reasons: list[str]
    metrics: dict[str, Any]
    label: str                  # OK | BREACH | INSUFFICIENT_SAMPLE
    info_kind: str = InfoKind.MODEL_OUTPUT.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionEvaluation:
    strategy_id: str
    version: str
    current_stage: str
    current_status: str
    next_stage: str | None
    recommended_stage: str
    requirements: list[Requirement]
    evidence: StrategyEvidence
    recommendation: str

    @property
    def unmet(self) -> list[Requirement]:
        return [r for r in self.requirements if not r.met]

    @property
    def can_promote(self) -> bool:
        return self.next_stage is not None and not self.unmet

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unmet"] = [r.name for r in self.unmet]
        d["can_promote"] = self.can_promote
        return d


# =============================================================================================
# Helpers
# =============================================================================================
def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _metric(metrics: dict[str, Any], keys: tuple[str, ...], strategy_id: str) -> float | None:
    scopes: list[dict[str, Any]] = []
    strat = metrics.get("strategies")
    if isinstance(strat, dict) and isinstance(strat.get(strategy_id), dict):
        scopes.append(strat[strategy_id])
    if isinstance(metrics.get(strategy_id), dict):
        scopes.append(metrics[strategy_id])
    scopes.append(metrics)
    for scope in scopes:
        for k in keys:
            v = _num(scope.get(k))
            if v is not None:
                return v
    return None


def _mentions(strategy_versions_json: str | None, strategy_id: str, version: str) -> bool:
    try:
        obj = from_json(strategy_versions_json, None)
    except (ValueError, TypeError):
        return False
    if isinstance(obj, dict):
        v = obj.get(strategy_id)
        if isinstance(v, str):
            return v == version
        if isinstance(v, dict):
            return v.get("version") == version
        if isinstance(v, list):
            return version in v
        return False
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and item in (f"{strategy_id}@{version}", f"{strategy_id}:{version}"):
                return True
            if isinstance(item, dict) and (item.get("strategy_id") or item.get("id")) == strategy_id \
                    and item.get("version") == version:
                return True
    return False


def _nonempty_json(text: str | None) -> bool:
    try:
        obj = from_json(text, None)
    except (ValueError, TypeError):
        return bool(text and text.strip())
    return bool(obj)


def _stage_index(stage: StrategyStage) -> int:
    return STRATEGY_STAGE_ORDER.index(stage)


# =============================================================================================
# Manager
# =============================================================================================
class PromotionManager:
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(f"promotion.{key}", default)

    # -- registry -------------------------------------------------------------------------------
    def register(self, strategy_id: str, version: str, family: str, params: dict[str, Any],
                 description: str | None = None, feature_deps: list[str] | None = None,
                 stage: StrategyStage | str = StrategyStage.IDEA, actor: str = "system") -> bool:
        """Create the strategies row if missing (status SHADOW). Returns True if created.
        New strategies can only enter at IDEA or RESEARCH; everything later must be earned."""
        st = StrategyStage(stage)
        if st not in (StrategyStage.IDEA, StrategyStage.RESEARCH):
            raise PromotionError("a strategy can only be registered at IDEA or RESEARCH")
        actor = require_actor(actor)
        if self._row(strategy_id, version, required=False) is not None:
            return False
        now = utcnow_iso()
        with self.db.transaction():
            self.db.insert("strategies", {
                "strategy_id": strategy_id, "version": version, "family": family, "description": description,
                "params_json": to_json(params or {}), "feature_deps_json": to_json(feature_deps or []),
                "status": StrategyStatus.SHADOW.value, "stage": st.value, "created_at": now, "updated_at": now,
            })
            self._log(strategy_id, version, None, StrategyStatus.SHADOW.value, None, st.value,
                      "registered", {"family": family}, now, actor)
        return True

    def get(self, strategy_id: str, version: str) -> dict[str, Any]:
        return self._row(strategy_id, version, required=True)

    def history(self, strategy_id: str, version: str) -> list[dict[str, Any]]:
        return self.db.fetchall("SELECT * FROM strategy_status_log WHERE strategy_id = ? AND version = ? ORDER BY id",
                                (strategy_id, version))

    def tradable_strategies(self) -> list[tuple[str, str]]:
        """(strategy_id, version) allowed to place BOT paper orders: status ACTIVE at stage PAPER/PROMOTED."""
        rows = self.db.fetchall(
            "SELECT strategy_id, version FROM strategies WHERE status = ? AND stage IN (?, ?) ORDER BY strategy_id, version",
            (StrategyStatus.ACTIVE.value, *[s.value for s in TRADABLE_STAGES]))
        return [(r["strategy_id"], r["version"]) for r in rows]

    # -- evidence -------------------------------------------------------------------------------
    def gather_evidence(self, strategy_id: str, version: str) -> StrategyEvidence:
        ev = StrategyEvidence(strategy_id, version, gathered_at=utcnow_iso())
        holdout_start = pd.Timestamp(str(self.config.get("validation.holdout.start"))).date()
        synthetic_datasets = {r["dataset_id"] for r in self.db.fetchall(
            "SELECT dataset_id FROM datasets WHERE is_synthetic = 1")}
        linked = {r["experiment_id"] for r in self.db.fetchall(
            "SELECT DISTINCT experiment_id FROM backtest_trades WHERE strategy_id = ? AND strategy_version = ?",
            (strategy_id, version))}
        experiments = [e for e in self.db.fetchall("SELECT * FROM experiments ORDER BY created_at, experiment_id")
                       if e["experiment_id"] in linked or _mentions(e["strategy_versions_json"], strategy_id, version)]
        backtests, walkforwards, holdouts = [], [], []
        for e in experiments:
            eid = e["experiment_id"]
            if self._is_synthetic(e, synthetic_datasets):
                ev.excluded_experiments.append({"experiment_id": eid, "kind": e["kind"], "reason": "synthetic"})
                continue
            result = self.db.fetchone(
                "SELECT status, metrics_json FROM experiment_results WHERE experiment_id = ? ORDER BY id DESC LIMIT 1", (eid,))
            metrics = from_json(result["metrics_json"], {}) if result else {}
            metrics = metrics if isinstance(metrics, dict) else {}
            if _nonempty_json(e["ai_models_json"]) or CONTAMINATED in (e["notes"] or "") or CONTAMINATED in to_json(metrics):
                ev.excluded_experiments.append({"experiment_id": eid, "kind": e["kind"], "reason": CONTAMINATED})
                continue
            touches = bool(e["touches_holdout"]) or (
                e["period_end"] is not None and pd.Timestamp(e["period_end"]).date() >= holdout_start)
            if touches:
                logged = self.db.fetchone("SELECT COUNT(*) AS n FROM holdout_access_log WHERE experiment_id = ?", (eid,))["n"]
                if not logged:
                    ev.integrity_issues.append(f"experiment {eid} touches the locked holdout without a holdout_access_log entry")
            if result is None or result["status"] != "succeeded":
                ev.excluded_experiments.append({"experiment_id": eid, "kind": e["kind"],
                                                "reason": "no_result" if result is None else f"result_{result['status']}"})
                if touches:
                    holdouts.append((e, None, False))
                continue
            if touches:
                holdouts.append((e, metrics, bool(logged)))
            elif e["kind"] == "backtest":
                backtests.append((e, metrics))
            elif e["kind"] == "walk_forward":
                walkforwards.append((e, metrics))
            else:
                ev.excluded_experiments.append({"experiment_id": eid, "kind": e["kind"], "reason": "kind_not_used_for_promotion"})
        ev.backtest = self._backtest_evidence(strategy_id, version, backtests)
        ev.walk_forward = self._walk_forward_evidence(strategy_id, version, walkforwards)
        ev.holdout = self._holdout_evidence(strategy_id, version, holdouts)
        ev.shadow = self._shadow_evidence(strategy_id, version)
        ev.paper = self._paper_evidence(strategy_id, version)
        return ev

    @staticmethod
    def _is_synthetic(e: dict[str, Any], synthetic_datasets: set[str]) -> bool:
        if e["uses_synthetic_data"]:
            return True
        try:
            snap = from_json(e["dataset_ids_json"], None)
        except (ValueError, TypeError):
            return False
        ids: list[str] = []
        if isinstance(snap, dict):
            for v in snap.values():
                ids.extend(v if isinstance(v, list) else [v])
        elif isinstance(snap, list):
            ids = list(snap)
        return any(isinstance(i, str) and (i in synthetic_datasets or i == "synthetic" or i.startswith("synthetic"))
                   for i in ids)

    def _trade_count(self, eid: str, strategy_id: str, version: str, metrics: dict[str, Any],
                     segment_sql: str) -> tuple[int, str]:
        n = self.db.fetchone(
            f"SELECT COUNT(*) AS n FROM backtest_trades WHERE experiment_id = ? AND strategy_id = ? "
            f"AND strategy_version = ? AND {segment_sql}", (eid, strategy_id, version))["n"]
        if n:
            return int(n), "backtest_trades"
        m = _metric(metrics, TRADE_KEYS, strategy_id)
        return (int(m), "metrics") if m is not None else (0, "none")

    def _backtest_evidence(self, sid: str, ver: str, rows: list[tuple[dict, dict]]) -> dict[str, Any]:
        runs = []
        for e, metrics in rows:
            n, src = self._trade_count(e["experiment_id"], sid, ver, metrics, "segment != 'holdout'")
            runs.append({"experiment_id": e["experiment_id"], "created_at": e["created_at"], "n_trades": n, "source": src})
        best = max(runs, key=lambda r: (r["n_trades"], r["created_at"]), default=None)
        return {"n_experiments": len(runs), "experiments": runs,
                "best_experiment_id": best["experiment_id"] if best else None,
                "max_trades": best["n_trades"] if best else 0}

    def _walk_forward_evidence(self, sid: str, ver: str, rows: list[tuple[dict, dict]]) -> dict[str, Any]:
        if not rows:
            return {"n_experiments": 0, "experiment_id": None, "n_windows": 0, "oos_sharpe": None,
                    "deflated_sharpe_pvalue": None}
        e, metrics = rows[-1]                       # LATEST, never the best
        eid = e["experiment_id"]
        n_win = self.db.fetchone(
            "SELECT COUNT(DISTINCT segment) AS n FROM backtest_trades WHERE experiment_id = ? AND strategy_id = ? "
            "AND strategy_version = ? AND segment LIKE 'oos:%'", (eid, sid, ver))["n"]
        win_src = "backtest_trades"
        if not n_win:
            m = _metric(metrics, WINDOW_KEYS, sid)
            n_win, win_src = (int(m), "metrics") if m is not None else (0, "none")
        p = _metric(metrics, DSR_PVALUE_KEYS, sid)
        p_src = "metrics:pvalue" if p is not None else None
        if p is None:
            prob = _metric(metrics, DSR_PROB_KEYS, sid)
            if prob is not None and 0.0 <= prob <= 1.0:
                p, p_src = 1.0 - prob, "metrics:1-DSR"
        n_trades, _ = self._trade_count(eid, sid, ver, metrics, "segment LIKE 'oos:%'")
        return {"n_experiments": len(rows), "experiment_id": eid, "created_at": e["created_at"],
                "n_windows": int(n_win), "windows_source": win_src,
                "oos_sharpe": _metric(metrics, SHARPE_KEYS, sid), "deflated_sharpe_pvalue": p,
                "pvalue_source": p_src, "n_variants_tested": e["n_variants_tested"], "n_oos_trades": n_trades}

    def _holdout_evidence(self, sid: str, ver: str, rows: list[tuple[dict, dict | None, bool]]) -> dict[str, Any]:
        n_access = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM holdout_access_log WHERE experiment_id IN "
            f"({','.join('?' * len(rows)) or 'NULL'})", [e["experiment_id"] for e, _, _ in rows])["n"]
        first = next(((e, m, lg) for e, m, lg in rows if m is not None), None)   # FIRST successful evaluation
        out: dict[str, Any] = {"n_evaluations": len(rows), "n_access_log_entries": int(n_access),
                               "experiment_id": None, "logged": False, "sharpe": None, "n_trades": 0}
        if first is None:
            return out
        e, metrics, logged = first
        n, src = self._trade_count(e["experiment_id"], sid, ver, metrics, "1 = 1")
        out.update({"experiment_id": e["experiment_id"], "created_at": e["created_at"], "logged": logged,
                    "sharpe": _metric(metrics, HOLDOUT_SHARPE_KEYS, sid), "n_trades": n, "trades_source": src,
                    "later_evaluations_ignored": sum(1 for x in rows if x[1] is not None) - 1})
        return out

    def _shadow_evidence(self, sid: str, ver: str) -> dict[str, Any]:
        lag = int(self._cfg("max_shadow_record_lag_days", 4))
        opps = self.db.query_df(
            "SELECT opportunity_id, as_of_date, holding_sessions, created_at, is_synthetic FROM shadow_opportunities "
            "WHERE strategy_id = ? AND strategy_version = ?", (sid, ver))
        out: dict[str, Any] = {"n_opportunities": int(len(opps)), "n_synthetic": 0, "n_backfilled": 0,
                               "n_sessions": 0, "n_outcomes": 0, "mean_ret": None, "ci": None, "first_session": None,
                               "last_session": None}
        if opps.empty:
            return out
        synth = opps["is_synthetic"].fillna(0).astype(int) == 1
        created = pd.to_datetime(opps["created_at"], utc=True, errors="coerce", format="ISO8601")
        as_of = pd.to_datetime(opps["as_of_date"], errors="coerce")
        lag_days = (created.dt.tz_convert(None).dt.normalize() - as_of).dt.days
        forward = lag_days.between(0, lag) & ~synth
        out["n_synthetic"] = int(synth.sum())
        out["n_backfilled"] = int((~synth & ~lag_days.between(0, lag)).sum())
        fwd = opps[forward]
        out["n_sessions"] = int(fwd["as_of_date"].nunique())
        if not fwd.empty:
            out["first_session"], out["last_session"] = str(fwd["as_of_date"].min()), str(fwd["as_of_date"].max())
        outs = self._outcomes(fwd)
        if not outs.empty:
            stats = self._mean_ci(outs["ret"].to_numpy(float), outs["as_of_date"].astype(str).to_numpy())
            out.update({"n_outcomes": int(len(outs)), "mean_ret": stats["mean"], "ci": stats["ci"],
                        "win_rate": stats["win_rate"]})
        return out

    def _outcomes(self, opps: pd.DataFrame) -> pd.DataFrame:
        if opps.empty:
            return pd.DataFrame(columns=["opportunity_id", "as_of_date", "ret"])
        ids = opps["opportunity_id"].tolist()
        frames = []
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            frames.append(self.db.query_df(
                "SELECT opportunity_id, horizon_sessions, status, ret, excess_ret FROM shadow_outcomes "
                f"WHERE opportunity_id IN ({','.join('?' * len(chunk))})", chunk))
        outs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if outs.empty:
            return pd.DataFrame(columns=["opportunity_id", "as_of_date", "ret"])
        outs = outs.merge(opps[["opportunity_id", "holding_sessions", "as_of_date"]], on="opportunity_id", how="left")
        prim = select_primary_outcomes(outs)
        return prim[prim["ret"].notna()].reset_index(drop=True)

    def _paper_evidence(self, sid: str, ver: str) -> dict[str, Any]:
        trades = self.db.query_df(
            "SELECT t.trade_id, t.status, t.entry_date, t.exit_date, t.ret, t.net_pnl, c.is_synthetic "
            "FROM trades t LEFT JOIN candidates c ON c.candidate_id = t.candidate_id "
            "WHERE t.book = 'BOT' AND t.strategy_id = ? AND t.strategy_version = ?", (sid, ver))
        out: dict[str, Any] = {"n_trades": int(len(trades)), "n_closed": 0, "n_open": 0, "n_synthetic": 0,
                               "n_ret_unknown": 0, "n_pnl_unknown": 0, "mean_ret": None, "ci": None,
                               "total_net_pnl": None, "max_drawdown": None}
        if trades.empty:
            return out
        synth = trades["is_synthetic"].fillna(0).astype(int) == 1
        out["n_synthetic"] = int(synth.sum())
        t = trades[~synth]
        closed = t[t["status"] == "CLOSED"].sort_values(["exit_date", "trade_id"])
        out["n_open"] = int((t["status"] == "OPEN").sum())
        out["n_closed"] = int(len(closed))
        rets = closed[closed["ret"].notna()]
        out["n_ret_unknown"] = int(closed["ret"].isna().sum())
        out["n_pnl_unknown"] = int(closed["net_pnl"].isna().sum())
        if not rets.empty:
            stats = self._mean_ci(rets["ret"].to_numpy(float), rets["entry_date"].fillna("").astype(str).to_numpy())
            out.update({"mean_ret": stats["mean"], "ci": stats["ci"], "win_rate": stats["win_rate"]})
        if len(closed) and out["n_pnl_unknown"] == 0:
            cum = closed["net_pnl"].astype(float).cumsum().to_numpy()
            peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
            capital = float(self.config.get("paper.bot.starting_cash", 100000))
            out["total_net_pnl"] = float(cum[-1])
            out["max_drawdown"] = float(np.max(peak - cum) / capital) if capital > 0 else None
        return out

    def _mean_ci(self, values: np.ndarray, clusters: np.ndarray, ci_level: float | None = None) -> dict[str, Any]:
        res = bootstrap_group_means(
            values, clusters, {"all": np.ones(len(values), dtype=bool)},
            n_resamples=int(self.config.get("validation.bootstrap.n_resamples", 2000)),
            mean_block=float(self.config.get("validation.bootstrap.block_length", 20)),
            seed=int(self.config.get("validation.bootstrap.seed", 7)),
            ci_level=float(ci_level if ci_level is not None else self._cfg("ci_level", 0.90)),
        )
        return res["groups"]["all"]

    # -- requirements ---------------------------------------------------------------------------
    def _gate(self, stage: StrategyStage, ev: StrategyEvidence, row: dict[str, Any]) -> list[Requirement]:
        g = stage.value
        if stage in (StrategyStage.IDEA, StrategyStage.RESEARCH):
            return [Requirement("registered", g, True, True, True, "strategies row with params exists")]
        if stage is StrategyStage.BACKTEST:
            need = int(self._cfg("min_backtest_trades", 100))
            n = ev.backtest.get("max_trades", 0)
            return [Requirement("backtest_trades", g, n >= need, n, need,
                                f"best qualifying backtest {ev.backtest.get('best_experiment_id')}")]
        if stage is StrategyStage.WALK_FORWARD:
            wf = ev.walk_forward
            need_w = int(self._cfg("min_walk_forward_windows", 4))
            need_s = float(self._cfg("min_oos_sharpe", 0.40))
            max_p = float(self._cfg("max_deflated_sharpe_pvalue", 0.10))
            s, p = wf.get("oos_sharpe"), wf.get("deflated_sharpe_pvalue")
            return [
                Requirement("walk_forward_windows", g, wf.get("n_windows", 0) >= need_w, wf.get("n_windows", 0), need_w,
                            f"latest walk-forward {wf.get('experiment_id')}"),
                Requirement("oos_sharpe", g, s is not None and s >= need_s, s, need_s,
                            "UNKNOWN (not reported)" if s is None else ""),
                Requirement("deflated_sharpe_pvalue", g, p is not None and p <= max_p, p, max_p,
                            "UNKNOWN (not reported in experiment metrics)" if p is None else f"source {wf.get('pvalue_source')}"),
            ]
        if stage is StrategyStage.LOCKED_HOLDOUT:
            h = ev.holdout
            need_t = int(self._cfg("min_holdout_trades", 30))
            need_s = float(self._cfg("min_holdout_sharpe", 0.0))
            unlogged = [i for i in ev.integrity_issues if "holdout" in i]
            return [
                Requirement("holdout_evaluated", g, h.get("experiment_id") is not None and h.get("logged", False),
                            h.get("experiment_id"), "first successful holdout evaluation, logged",
                            "" if h.get("logged") else "no logged holdout evaluation"),
                Requirement("holdout_access_logged", g, not unlogged, len(unlogged), 0, "; ".join(unlogged)),
                Requirement("holdout_trades", g, h.get("n_trades", 0) >= need_t, h.get("n_trades", 0), need_t),
                Requirement("holdout_sharpe", g, h.get("sharpe") is not None and h["sharpe"] >= need_s, h.get("sharpe"),
                            need_s, "UNKNOWN (not reported)" if h.get("sharpe") is None else ""),
            ]
        if stage is StrategyStage.SHADOW:
            sh = ev.shadow
            need_d = int(self._cfg("min_shadow_sessions", 60))
            need_o = int(self._cfg("min_shadow_outcomes", 30))
            min_e = float(self._cfg("min_shadow_expectancy", 0.0))
            n_o, m = sh.get("n_outcomes", 0), sh.get("mean_ret")
            return [
                Requirement("shadow_sessions", g, sh.get("n_sessions", 0) >= need_d, sh.get("n_sessions", 0), need_d,
                            f"{sh.get('n_backfilled', 0)} backfilled record(s) ignored"),
                Requirement("shadow_outcomes", g, n_o >= need_o, n_o, need_o),
                Requirement("shadow_expectancy", g, n_o >= need_o and m is not None and m > min_e, m, min_e,
                            "INSUFFICIENT_SAMPLE" if n_o < need_o else f"CI {sh.get('ci')}"),
            ]
        if stage is StrategyStage.PAPER:
            pp = ev.paper
            need_t = int(self._cfg("min_paper_trades", 30))
            min_e = float(self._cfg("min_paper_expectancy", 0.0))
            n, m = pp.get("n_closed", 0), pp.get("mean_ret")
            deg = self.degradation_check(row["strategy_id"], row["version"], evidence=ev)
            return [
                Requirement("paper_trades", g, n >= need_t, n, need_t),
                Requirement("paper_expectancy", g, n >= need_t and m is not None and m > min_e, m, min_e,
                            "INSUFFICIENT_SAMPLE" if n < need_t else f"CI {pp.get('ci')}"),
                Requirement("no_degradation", g, not deg.recommend_pause, deg.label, "OK", "; ".join(deg.reasons)),
            ]
        return []

    def requirements_for(self, strategy_id: str, version: str, to_stage: StrategyStage | str,
                         evidence: StrategyEvidence | None = None) -> list[Requirement]:
        """Cumulative requirements to ENTER ``to_stage``: every gate of every earlier stage."""
        target = StrategyStage(to_stage)
        row = self.get(strategy_id, version)
        ev = evidence or self.gather_evidence(strategy_id, version)
        reqs: list[Requirement] = []
        for st in STRATEGY_STAGE_ORDER[:_stage_index(target)]:
            if st is StrategyStage.RESEARCH:
                continue                      # same 'registered' gate as IDEA
            reqs.extend(self._gate(st, ev, row))
        status = StrategyStatus(row["status"])
        reqs.append(Requirement("status_ok", "ALL", status in FORWARD_OK_STATUSES, status.value,
                                [s.value for s in FORWARD_OK_STATUSES],
                                "" if status in FORWARD_OK_STATUSES else "PAUSED/RETIRED strategies need human review"))
        return reqs

    def evaluate(self, strategy_id: str, version: str) -> PromotionEvaluation:
        row = self.get(strategy_id, version)
        cur = StrategyStage(row["stage"])
        idx = _stage_index(cur)
        nxt = STRATEGY_STAGE_ORDER[idx + 1] if idx + 1 < len(STRATEGY_STAGE_ORDER) else None
        ev = self.gather_evidence(strategy_id, version)
        if nxt is None:
            reqs = self.requirements_for(strategy_id, version, cur, ev) + self._gate(StrategyStage.PAPER, ev, row)
        else:
            reqs = self.requirements_for(strategy_id, version, nxt, ev)
        unmet = [r for r in reqs if not r.met]
        if nxt is None:
            rec = cur
            text = "PROMOTED; maintenance checks " + ("pass" if not unmet else "FAIL: " + ", ".join(r.name for r in unmet))
        elif unmet:
            rec = cur
            text = f"stay at {cur.value}: unmet " + ", ".join(r.name for r in unmet)
        else:
            rec = nxt
            text = f"eligible for {nxt.value}"
        return PromotionEvaluation(strategy_id, version, cur.value, row["status"], nxt.value if nxt else None,
                                   rec.value, reqs, ev, text)

    # -- changes --------------------------------------------------------------------------------
    def transition(self, strategy_id: str, version: str, to_stage: StrategyStage | str, reason: str, actor: str,
                   evidence: dict[str, Any] | None = None, override_reason: str | None = None) -> dict[str, Any]:
        """Move a strategy one stage forward (requirements enforced) or any number of stages back.
        Returns the strategy_status_log row written."""
        actor = require_actor(actor)
        if not isinstance(reason, str) or not reason.strip():
            raise PromotionError("a transition requires a non-empty reason")
        target = StrategyStage(to_stage)
        row = self.get(strategy_id, version)
        cur = StrategyStage(row["stage"])
        ci, ti = _stage_index(cur), _stage_index(target)
        if ti == ci:
            raise PromotionError(f"{strategy_id}@{version} is already at {cur.value}")
        payload: dict[str, Any] = {"caller_evidence": evidence or {}, "direction": "forward" if ti > ci else "demotion"}
        text = reason.strip()
        if ti > ci:
            if ti != ci + 1:
                raise StageSkipError(f"cannot skip stages: {cur.value} -> {target.value} (next is "
                                     f"{STRATEGY_STAGE_ORDER[ci + 1].value})")
            if is_ai_actor(actor):
                raise PromotionPermissionError("AI actors can never promote strategies")
            if target is StrategyStage.PROMOTED and bool(self._cfg("require_human_for_promoted", True)) \
                    and not is_human_actor(actor):
                raise PromotionPermissionError("promotion to PROMOTED requires a human actor")
            ev = self.gather_evidence(strategy_id, version)
            reqs = self.requirements_for(strategy_id, version, target, ev)
            unmet = [r for r in reqs if not r.met]
            payload.update({"requirements": [asdict(r) for r in reqs], "evidence": ev.to_dict(),
                            "unmet": [r.name for r in unmet], "override": False})
            if unmet:
                if not (is_human_actor(actor) and isinstance(override_reason, str) and override_reason.strip()):
                    log_event(log, "promotion refused", strategy_id=strategy_id, version=version,
                              to_stage=target.value, unmet=[r.name for r in unmet], actor=actor)
                    raise PromotionBlockedError(
                        f"cannot move {strategy_id}@{version} to {target.value}: unmet "
                        + ", ".join(f"{r.name} (value={r.value!r}, need {r.threshold!r})" for r in unmet), unmet)
                payload.update({"override": True, "override_reason": override_reason.strip()})
                text = (f"HUMAN OVERRIDE of unmet requirements [{', '.join(r.name for r in unmet)}]: "
                        f"{override_reason.strip()} | {text}")
        now = utcnow_iso()
        with self.db.transaction():
            self._log(strategy_id, version, row["status"], row["status"], cur.value, target.value, text, payload, now, actor)
            self.db.execute("UPDATE strategies SET stage = ?, updated_at = ? WHERE strategy_id = ? AND version = ?",
                            (target.value, now, strategy_id, version))
        log_event(log, "strategy stage changed", strategy_id=strategy_id, version=version, from_stage=cur.value,
                  to_stage=target.value, actor=actor, override=payload.get("override", False))
        return self.history(strategy_id, version)[-1]

    def set_status(self, strategy_id: str, version: str, status: StrategyStatus | str, reason: str,
                   actor: str = "system", evidence: dict[str, Any] | None = None) -> bool:
        """Change ACTIVE/SHADOW/PAUSED/RETIRED. Returns False when already in that status.

        ACTIVE (places paper orders) needs stage PAPER/PROMOTED and a non-AI actor. Leaving PAUSED
        for ACTIVE, or leaving RETIRED at all, requires a human: the cause needs human review.
        """
        actor = require_actor(actor)
        new = StrategyStatus(status)
        if not isinstance(reason, str) or not reason.strip():
            raise PromotionError("a status change requires a non-empty reason")
        row = self.get(strategy_id, version)
        old = StrategyStatus(row["status"])
        if new is old:
            return False
        stage = StrategyStage(row["stage"])
        if new is StrategyStatus.ACTIVE:
            if is_ai_actor(actor):
                raise PromotionPermissionError("AI actors can never activate a strategy")
            if stage not in TRADABLE_STAGES:
                raise PromotionError(f"ACTIVE requires stage PAPER or PROMOTED (is {stage.value}); promote first")
            if old is StrategyStatus.PAUSED and not is_human_actor(actor):
                raise PromotionPermissionError("resuming a PAUSED strategy requires a human actor")
        if old is StrategyStatus.RETIRED and not is_human_actor(actor):
            raise PromotionPermissionError("un-retiring a strategy requires a human actor")
        now = utcnow_iso()
        with self.db.transaction():
            self._log(strategy_id, version, old.value, new.value, stage.value, stage.value, reason.strip(),
                      {"caller_evidence": evidence or {}}, now, actor)
            self.db.execute("UPDATE strategies SET status = ?, updated_at = ? WHERE strategy_id = ? AND version = ?",
                            (new.value, now, strategy_id, version))
        log_event(log, "strategy status changed", strategy_id=strategy_id, version=version, old=old.value,
                  new=new.value, actor=actor)
        return True

    # -- degradation ----------------------------------------------------------------------------
    def degradation_check(self, strategy_id: str, version: str, evidence: StrategyEvidence | None = None) -> DegradationResult:
        """Pause recommendation from FORWARD evidence (paper trades, shadow outcomes).

        * Expectancy breach: with >= degradation.min_trades samples, the upper bound of the
          bootstrap CI (level promotion.degradation.ci_level) of the mean return is below
          degradation.min_expectancy, i.e. the forward edge is credibly worse than the threshold.
        * Drawdown breach: peak-to-trough of the strategy's cumulative closed paper P&L exceeds
          degradation.max_drawdown x paper.bot.starting_cash. A realized loss is a fact, not an
          estimate, so no minimum sample applies.
        * Closed paper trades with unknown P&L are a breach (fail safe: unknown is not OK).
        """
        ev = evidence or self.gather_evidence(strategy_id, version)
        min_n = int(self.config.get("promotion.degradation.min_trades", 30))
        min_e = float(self.config.get("promotion.degradation.min_expectancy", 0.0))
        max_dd = float(self.config.get("promotion.degradation.max_drawdown", 0.10))
        level = float(self.config.get("promotion.degradation.ci_level", 0.90))
        reasons: list[str] = []
        metrics: dict[str, Any] = {"min_samples": min_n, "min_expectancy": min_e, "max_drawdown": max_dd, "ci_level": level}
        enough = False
        pp = ev.paper
        paper_ci = None
        if pp.get("n_closed", 0) - pp.get("n_ret_unknown", 0) >= min_n:
            enough = True
            paper_ci = self._forward_ci(strategy_id, version, "paper", level)
            if paper_ci is not None and paper_ci[1] < min_e:
                reasons.append(f"paper expectancy CI upper {paper_ci[1]:+.4f} < {min_e:+.4f} (n={pp['n_closed']})")
        if pp.get("n_pnl_unknown", 0):
            reasons.append(f"{pp['n_pnl_unknown']} closed paper trade(s) with UNKNOWN P&L")
        if pp.get("max_drawdown") is not None and pp["max_drawdown"] > max_dd:
            reasons.append(f"paper drawdown {pp['max_drawdown']:.2%} > {max_dd:.2%} of starting capital")
        sh = ev.shadow
        shadow_ci = None
        if sh.get("n_outcomes", 0) >= min_n:
            enough = True
            shadow_ci = self._forward_ci(strategy_id, version, "shadow", level)
            if shadow_ci is not None and shadow_ci[1] < min_e:
                reasons.append(f"shadow expectancy CI upper {shadow_ci[1]:+.4f} < {min_e:+.4f} (n={sh['n_outcomes']})")
        metrics.update({"paper_n_closed": pp.get("n_closed", 0), "paper_mean_ret": pp.get("mean_ret"),
                        "paper_ci": paper_ci, "paper_max_drawdown": pp.get("max_drawdown"),
                        "shadow_n_outcomes": sh.get("n_outcomes", 0), "shadow_mean_ret": sh.get("mean_ret"),
                        "shadow_ci": shadow_ci})
        label = "BREACH" if reasons else ("OK" if enough else "INSUFFICIENT_SAMPLE")
        return DegradationResult(strategy_id, version, bool(reasons), reasons, metrics, label)

    def _forward_ci(self, sid: str, ver: str, source: str, level: float) -> list[float] | None:
        if source == "paper":
            t = self.db.query_df(
                "SELECT t.ret, t.entry_date, c.is_synthetic FROM trades t LEFT JOIN candidates c "
                "ON c.candidate_id = t.candidate_id WHERE t.book = 'BOT' AND t.strategy_id = ? AND "
                "t.strategy_version = ? AND t.status = 'CLOSED' AND t.ret IS NOT NULL", (sid, ver))
            t = t[t["is_synthetic"].fillna(0).astype(int) != 1]
            vals, clusters = t["ret"].to_numpy(float), t["entry_date"].fillna("").astype(str).to_numpy()
        else:
            lag = int(self._cfg("max_shadow_record_lag_days", 4))
            opps = self.db.query_df(
                "SELECT opportunity_id, as_of_date, holding_sessions, created_at, is_synthetic FROM shadow_opportunities "
                "WHERE strategy_id = ? AND strategy_version = ? AND is_synthetic = 0", (sid, ver))
            if not opps.empty:
                created = pd.to_datetime(opps["created_at"], utc=True, errors="coerce", format="ISO8601")
                lag_days = (created.dt.tz_convert(None).dt.normalize() - pd.to_datetime(opps["as_of_date"])).dt.days
                opps = opps[lag_days.between(0, lag)]
            outs = self._outcomes(opps)
            vals, clusters = outs["ret"].to_numpy(float), outs["as_of_date"].astype(str).to_numpy()
        if len(vals) == 0:
            return None
        return self._mean_ci(vals, clusters, ci_level=level)["ci"]

    def enforce_degradation(self, strategy_id: str, version: str) -> DegradationResult:
        """Run :meth:`degradation_check` and PAUSE the strategy (actor system) on a breach."""
        res = self.degradation_check(strategy_id, version)
        row = self.get(strategy_id, version)
        if res.recommend_pause and row["status"] in (StrategyStatus.ACTIVE.value, StrategyStatus.SHADOW.value):
            self.set_status(strategy_id, version, StrategyStatus.PAUSED,
                            "degradation: " + "; ".join(res.reasons), actor="system:degradation",
                            evidence=res.to_dict())
        return res

    # -- internals ------------------------------------------------------------------------------
    def _row(self, strategy_id: str, version: str, required: bool) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM strategies WHERE strategy_id = ? AND version = ?", (strategy_id, version))
        if row is None and required:
            raise StrategyNotRegisteredError(f"strategy {strategy_id}@{version} is not registered")
        return row

    def _log(self, strategy_id: str, version: str, from_status: str | None, to_status: str, from_stage: str | None,
             to_stage: str, reason: str, evidence: dict[str, Any], at: str, actor: str) -> None:
        self.db.insert("strategy_status_log", {
            "strategy_id": strategy_id, "version": version, "from_status": from_status, "to_status": to_status,
            "from_stage": from_stage, "to_stage": to_stage, "reason": reason, "evidence_json": to_json(evidence),
            "changed_at": at, "changed_by": actor,
        })


def today_iso() -> str:
    return date.today().isoformat()
