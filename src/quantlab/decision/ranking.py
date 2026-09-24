"""Opportunity ranking: orders one day's strategy candidates into symbol-level opportunities.

``opportunity_score`` lies in [0, 1] and is a RANKING index. It is NOT a probability of profit
and must never be read or displayed as one. (Profitability is the EV engine's job, from empirical
outcomes only.) It is a weighted mean of four components, each in [0, 1]:

  * ``signal``: the percentile of the candidate's raw score among the SAME strategy's candidates
    that day (midrank: (rank - 0.5) / n). Raw scores of different strategies live on different
    scales. Percentiles make strategies comparable and are invariant to any monotone rescaling of
    a strategy's score. A lone candidate gets 0.5, which honestly means "no relative information".
  * ``quality``: the strategy's validated track record, shrunk toward neutral 0.5:
    ``0.5 + (Phi(t) - 0.5) * n_eff / (n_eff + k)`` with t the t-statistic of net per-trade returns,
    n_eff the evidence count (in-sample discounted) and k = ``ranking.quality_prior_trades``.
    Only used when a stats provider is given.
  * ``ml``: the mean CALIBRATED probability of ACTIVE ML models (neutral 0.5 when a record has none).
    Only used when at least one usable prediction exists in the batch. Uncalibrated or non-ACTIVE
    models are shown as evidence but never scored.
  * ``confirmation``: evidence from OTHER, distinct strategies on the same symbol/direction,
    ``1 - prod_j(1 - e_j * (1 - max(0, rho_bj)))``. It is bounded, saturates, and is discounted by
    the (optional) correlation rho between the best strategy and strategy j. This is an INPUT
    with a small weight, NOT a vote count. Three mediocre signals do not beat one strong signal.

Grouping: one record per (as_of_date, symbol, direction). Opposite directions stay separate records
(the no-trade engine flags the conflict). Percentiles and ranks are computed per as_of_date, so a
day's ranking never depends on candidates from other days (point-in-time; see tests).

Every evidence item is labeled with :class:`~quantlab.core.types.InfoKind`.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Protocol

import pandas as pd

from quantlab.config import Config
from quantlab.core.types import (
    AIReview,
    Candidate,
    Direction,
    ExpectedValue,
    InfoKind,
    MLPrediction,
    PitStatus,
)
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.decision.stats_provider import StrategyStats
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

SCORE_KIND = "ranking"   # opportunity_score semantics. It is never a probability.

_EVENT = {"days_since_earnings", "ear_3d", "ear_z", "event_rel_volume", "est_sessions_to_earnings", "sue"}
_FUNDAMENTAL = {"rev_growth_yoy", "ni_margin", "roe", "leverage", "eps_growth_yoy", "ep_ttm", "fundamental_age"}
_MARKET = {"market_trend_200", "market_mom_60", "market_vol_20", "market_drawdown", "breadth_50", "breadth_200",
           "sector_dispersion_63"}
_MODEL_FEATURES = {"est_sessions_to_earnings": "MODEL estimate (63 - days_since_earnings; no PIT schedule)",
                   "sue": "statistical surprise model"}


class StatsSource(Protocol):
    def get(self, strategy_id: str, version: str, regime: str | None = None, as_of: Any = None) -> StrategyStats: ...


@dataclass
class EvidenceItem:
    kind: InfoKind
    label: str
    text: str
    value: Any = None
    source: str = ""


@dataclass
class StrategyEvidence:
    candidate_id: str
    strategy_id: str
    strategy_version: str
    score: float
    percentile: float          # within this strategy's candidates that day
    group_size: int
    quality: float | None      # shrunk validated quality in [0, 1] (None when no stats provider)
    quality_n_eff: float
    stats_source: str
    evidence: float            # combined per-strategy evidence e_i in [0, 1]


@dataclass
class OpportunityRecord:
    symbol: str
    as_of_date: date
    direction: Direction
    candidates: list[Candidate]
    strategies: list[StrategyEvidence]
    primary_candidate_id: str
    opportunity_score: float
    rank: int = 0
    components: dict[str, float | None] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    quantitative_evidence: list[EvidenceItem] = field(default_factory=list)
    event_evidence: list[EvidenceItem] = field(default_factory=list)
    fundamental_evidence: list[EvidenceItem] = field(default_factory=list)
    news_evidence: list[EvidenceItem] = field(default_factory=list)
    ml: list[EvidenceItem] = field(default_factory=list)
    market_context: list[EvidenceItem] = field(default_factory=list)
    risks: list[EvidenceItem] = field(default_factory=list)
    uncertainty: list[EvidenceItem] = field(default_factory=list)
    expected_value: ExpectedValue | None = None     # filled by the pipeline (EVEngine)
    ai: AIReview | None = None                      # filled later by the AI layer
    pit_status: PitStatus = PitStatus.PIT           # weakest over the record's candidates
    score_kind: str = SCORE_KIND

    @property
    def candidate_ids(self) -> list[str]:
        return [c.candidate_id for c in self.candidates]

    @property
    def primary_candidate(self) -> Candidate:
        return next(c for c in self.candidates if c.candidate_id == self.primary_candidate_id)

    @property
    def strategy_ids(self) -> list[str]:
        return [s.strategy_id for s in self.strategies]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (enums -> values, dates -> ISO) for persistence and the dashboard."""
        return json.loads(to_json(asdict(self)))


def feature_group(name: str) -> str:
    """Catalog group of a feature: from the live registry when registered, else the static
    catalog of ARCHITECTURE.md section 4 (so ranking never imports other subsystems' modules)."""
    try:
        from quantlab.features.base import FEATURES
        if name in FEATURES:
            return FEATURES.spec(name).group
    except Exception:  # registry unavailable -> static catalog
        pass
    if name in _EVENT:
        return "event"
    if name in _FUNDAMENTAL:
        return "fundamental"
    if name.startswith("news_"):
        return "news"
    if name in _MARKET:
        return "market"
    return "price"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _phi(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


class OpportunityRanker:
    DEFAULT_WEIGHTS = {"signal": 0.5, "quality": 0.2, "ml": 0.2, "confirmation": 0.1}

    def __init__(self, config: Config):
        self.config = config
        w = {**self.DEFAULT_WEIGHTS, **(config.get("ranking.weights", {}) or {})}
        if any(float(v) < 0 for v in w.values()) or float(w["signal"]) <= 0:
            raise ValueError("ranking.weights must be >= 0 with a positive 'signal' weight")
        self.weights = {k: float(w[k]) for k in self.DEFAULT_WEIGHTS}
        self.quality_k = float(config.get("ranking.quality_prior_trades", config.get("expected_value.prior_trades", 50)))
        self.in_sample_weight = float(config.get("expected_value.in_sample_weight", 0.25))

    # -- components -----------------------------------------------------------------------------
    def strategy_quality(self, st: StrategyStats | None) -> tuple[float, float]:
        """(quality in [0,1], n_eff). No evidence or no dispersion -> neutral 0.5."""
        if st is None or st.n <= 0 or st.t_stat is None or not _finite(st.t_stat):
            return 0.5, 0.0
        n_eff = float(st.n) if st.validated else float(st.n) * self.in_sample_weight
        if n_eff <= 0:
            return 0.5, 0.0
        return 0.5 + (_phi(float(st.t_stat)) - 0.5) * n_eff / (n_eff + self.quality_k), n_eff

    @staticmethod
    def usable_ml(preds: Iterable[MLPrediction]) -> list[MLPrediction]:
        return [p for p in preds if p.calibrated and str(p.status).upper() == "ACTIVE" and _finite(p.probability)
                and 0.0 <= float(p.probability) <= 1.0]

    @staticmethod
    def _percentiles(cands: list[Candidate]) -> dict[str, tuple[float, int]]:
        """candidate_id -> (midrank percentile, group size) within (as_of_date, strategy, version)."""
        out: dict[str, tuple[float, int]] = {}
        groups: dict[tuple, list[Candidate]] = {}
        for c in cands:
            groups.setdefault((c.as_of_date, c.strategy_id, c.strategy_version), []).append(c)
        for members in groups.values():
            finite = [c for c in members if _finite(c.score)]
            for c in members:
                if not _finite(c.score):
                    out[c.candidate_id] = (0.0, len(finite))
            if not finite:
                continue
            s = pd.Series([float(c.score) for c in finite], index=[c.candidate_id for c in finite])
            r = s.rank(method="average")
            n = len(finite)
            for cid, rk in r.items():
                out[cid] = ((float(rk) - 0.5) / n, n)
        return out

    # -- main -----------------------------------------------------------------------------------
    def rank(self, candidates: list[Candidate], ml: Mapping[str, MLPrediction | list[MLPrediction]] | None = None,
             stats: StatsSource | None = None, market_context: Mapping[str, Any] | None = None,
             strategy_correlation: Mapping[tuple[str, str], float] | None = None) -> list[OpportunityRecord]:
        if not candidates:
            return []
        ml = ml or {}
        preds_of = {cid: (v if isinstance(v, list) else [v]) for cid, v in ml.items()}
        batch_has_ml = any(self.usable_ml(p) for p in preds_of.values())
        use_quality = stats is not None
        weights = {k: v for k, v in self.weights.items()
                   if (k != "quality" or use_quality) and (k != "ml" or batch_has_ml)}
        pct = self._percentiles(candidates)
        stats_cache: dict[tuple, StrategyStats | None] = {}

        groups: dict[tuple, list[Candidate]] = {}
        for c in candidates:
            groups.setdefault((c.as_of_date, c.symbol, Direction(c.direction)), []).append(c)

        records: list[OpportunityRecord] = []
        for (d, sym, direction), members in groups.items():
            evs: list[StrategyEvidence] = []
            stats_of: dict[str, StrategyStats | None] = {}
            for c in members:
                st = None
                if use_quality:
                    key = (c.strategy_id, c.strategy_version, d)
                    if key not in stats_cache:
                        stats_cache[key] = stats.get(c.strategy_id, c.strategy_version, as_of=d)
                    st = stats_cache[key]
                stats_of[c.candidate_id] = st
                q, n_eff = self.strategy_quality(st) if use_quality else (None, 0.0)
                p, n = pct[c.candidate_id]
                e = (weights["signal"] * p + weights["quality"] * q) / (weights["signal"] + weights["quality"]) \
                    if use_quality else p
                evs.append(StrategyEvidence(c.candidate_id, c.strategy_id, c.strategy_version, float(c.score), p, n,
                                            q, n_eff, st.source if st is not None else "none", e))
            evs.sort(key=lambda s: (-s.evidence, -s.percentile, s.strategy_id, s.candidate_id))
            best = evs[0]
            # confirmation from DISTINCT other strategies (versions of one family do not confirm each other)
            others: dict[str, float] = {}
            for s in evs[1:]:
                if s.strategy_id == best.strategy_id:
                    continue
                rho = 0.0
                if strategy_correlation:
                    rho = strategy_correlation.get((best.strategy_id, s.strategy_id),
                                                   strategy_correlation.get((s.strategy_id, best.strategy_id), 0.0))
                others[s.strategy_id] = max(others.get(s.strategy_id, 0.0), s.evidence * (1.0 - max(0.0, float(rho))))
            conf = 1.0 - math.prod(1.0 - e for e in others.values()) if others else 0.0

            usable = [p for c in members for p in self.usable_ml(preds_of.get(c.candidate_id, []))]
            ml_p = sum(float(p.probability) for p in usable) / len(usable) if usable else None
            comps: dict[str, float | None] = {"signal": best.percentile, "quality": best.quality if use_quality else None,
                                              "ml": (ml_p if ml_p is not None else 0.5) if batch_has_ml else None,
                                              "confirmation": conf}
            score = sum(weights[k] * float(comps[k]) for k in weights) / sum(weights.values())
            rec = OpportunityRecord(
                symbol=sym, as_of_date=d, direction=direction, candidates=list(members), strategies=evs,
                primary_candidate_id=best.candidate_id, opportunity_score=float(min(1.0, max(0.0, score))),
                components=comps, weights=dict(weights),
                pit_status=PitStatus.weakest([PitStatus(c.pit_status) for c in members]),
            )
            self._fill_evidence(rec, members, stats_of, preds_of, usable, ml_p, market_context)
            records.append(rec)

        records.sort(key=lambda r: (r.as_of_date, -r.opportunity_score, r.symbol, r.direction.value))
        rank, last_day = 0, None
        for r in records:
            rank = 1 if r.as_of_date != last_day else rank + 1
            r.rank, last_day = rank, r.as_of_date
        log_event(log, "opportunities ranked", n_candidates=len(candidates), n_records=len(records),
                  weights=weights)
        return records

    # -- evidence labeling ----------------------------------------------------------------------
    def _fill_evidence(self, rec: OpportunityRecord, members: list[Candidate],
                       stats_of: Mapping[str, StrategyStats | None], preds_of: Mapping[str, list[MLPrediction]],
                       usable: list[MLPrediction], ml_p: float | None, market_context: Mapping[str, Any] | None) -> None:
        M, F, U = InfoKind.MODEL_OUTPUT, InfoKind.FACT, InfoKind.UNCERTAINTY
        rec.quantitative_evidence.append(EvidenceItem(
            M, "opportunity_score", f"opportunity_score {rec.opportunity_score:.3f}: a RANKING within this day's "
            "candidates, NOT a probability of profit", rec.opportunity_score, "decision.ranking"))
        for s in rec.strategies:
            rec.quantitative_evidence.append(EvidenceItem(
                M, f"{s.strategy_id}.score", f"{s.strategy_id}@{s.strategy_version} score {s.score:.4g} "
                f"(percentile {s.percentile:.2f} of {s.group_size} same-strategy candidates)", s.score, s.strategy_id))
            if s.group_size <= 1:
                rec.uncertainty.append(EvidenceItem(U, f"{s.strategy_id}.percentile", f"{s.strategy_id}: only "
                                                    f"{s.group_size} candidate(s) today, percentile uninformative"))
            st = stats_of.get(s.candidate_id)
            if st is not None:
                if st.n > 0:
                    rec.quantitative_evidence.append(EvidenceItem(
                        M, f"{s.strategy_id}.history", f"{s.strategy_id} history: n={st.n} ({st.source}), net expectancy "
                        f"{(st.expectancy or 0) * 1e4:.1f} bps/trade, t={st.t_stat if st.t_stat is not None else float('nan'):.2f}; "
                        f"shrunk quality {s.quality:.3f}", st.expectancy, st.source))
                if not st.validated:
                    rec.uncertainty.append(EvidenceItem(U, f"{s.strategy_id}.unvalidated",
                                                        f"{s.strategy_id}: no out-of-sample/forward evidence "
                                                        f"(source={st.source}, n={st.n})"))
        if len(rec.strategies) > 1:
            rec.quantitative_evidence.append(EvidenceItem(
                M, "confirmation", f"{len({s.strategy_id for s in rec.strategies})} distinct strategies agree; "
                f"confirmation input {rec.components['confirmation']:.3f} (bounded, not a vote count)",
                rec.components["confirmation"]))

        seen: set[str] = set()
        for c in members:
            for reason in c.reasons:
                rec.quantitative_evidence.append(EvidenceItem(M, f"{c.strategy_id}.reason", reason, None, c.strategy_id))
            for name, value in c.features.items():
                if name in seen:
                    continue
                seen.add(name)
                group = feature_group(name)
                if not _finite(value):
                    item = EvidenceItem(U, name, f"{name} = UNKNOWN", None, "features")
                    rec.uncertainty.append(item)
                elif name in _MODEL_FEATURES:
                    item = EvidenceItem(M, name, f"{name} = {float(value):.4g} ({_MODEL_FEATURES[name]})", float(value),
                                        "features")
                else:
                    item = EvidenceItem(F, name, f"{name} = {float(value):.4g}", float(value), "features")
                {"event": rec.event_evidence, "fundamental": rec.fundamental_evidence, "news": rec.news_evidence,
                 "market": rec.market_context}.get(group, rec.quantitative_evidence).append(item)

        for c in members:
            for p in preds_of.get(c.candidate_id, []):
                ok = p in usable
                kind = M if ok else U
                prob = f"{float(p.probability):.3f}" if _finite(p.probability) else "UNKNOWN"
                rec.ml.append(EvidenceItem(kind, f"ml.{p.model_id}", f"{p.model_id}@{p.model_version} P({p.target})={prob} "
                                           f"[{'calibrated' if p.calibrated else 'UNCALIBRATED'}, {p.status}]"
                                           + ("" if ok else " (shown, not used in ranking)"), p.probability, p.model_id))
        if ml_p is None:
            rec.uncertainty.append(EvidenceItem(U, "ml", "no calibrated ACTIVE ML probability for this opportunity"))

        for k, v in (market_context or {}).items():
            rec.market_context.append(EvidenceItem(M, str(k), f"{k} = {v}", v, "market_context"))

        primary = rec.primary_candidate
        plan = primary.plan
        if _finite(plan.entry_ref_price) and _finite(plan.stop_price) and plan.entry_ref_price:
            dist = abs(plan.entry_ref_price - plan.stop_price) / plan.entry_ref_price
            rec.risks.append(EvidenceItem(F, "stop_distance", f"stop {plan.stop_price:.2f} vs ref {plan.entry_ref_price:.2f} "
                                          f"({dist:.1%} risk per share)", dist, "plan"))
        else:
            rec.uncertainty.append(EvidenceItem(U, "stop", "no protective stop in the plan (portfolio uses a fallback)"))
        for name, value in primary.risk.items():
            if _finite(value):
                rec.risks.append(EvidenceItem(F, name, f"{name} = {float(value):.4g}", float(value), "candidate.risk"))
            else:
                rec.uncertainty.append(EvidenceItem(U, name, f"{name} = UNKNOWN", None, "candidate.risk"))
        if rec.pit_status is not PitStatus.PIT:
            rec.uncertainty.append(EvidenceItem(U, "pit_status", f"weakest PIT status of inputs: {rec.pit_status.value}"))


def persist_ranking(db: Database, records: list[OpportunityRecord], run_id: str | None = None,
                    is_synthetic: bool = False) -> int:
    """Append the ranking to ``opportunity_rankings`` (080). Returns rows written."""
    now = utcnow_iso()
    rows = [{
        "run_id": run_id, "as_of_date": r.as_of_date.isoformat(), "symbol": r.symbol, "direction": r.direction.value,
        "rank": r.rank, "opportunity_score": r.opportunity_score, "primary_candidate_id": r.primary_candidate_id,
        "candidate_ids_json": to_json(r.candidate_ids),
        "strategies_json": to_json([asdict(s) for s in r.strategies]),
        "components_json": to_json(r.components), "record_json": to_json(r.to_dict()),
        "is_synthetic": int(is_synthetic), "created_at": now,
    } for r in records]
    return db.insert_many("opportunity_rankings", rows)
