"""Shared vocabulary for every layer. Changing these is an architectural change — see ARCHITECTURE.md.

All enums are ``str`` enums so they serialize to readable values in the database and JSON.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------------
# Point-in-time status
# --------------------------------------------------------------------------------------------
class PitStatus(str, Enum):
    """How well we can establish that a datum was available at decision time.

    Ordered strongest -> weakest. Research code must declare which statuses it accepts; reports
    always print the weakest status that influenced a result.
    """

    PIT = "PIT"                            # availability timestamp known from the source itself
    PIT_CONSERVATIVE = "PIT_CONSERVATIVE"  # availability inferred with a deliberately late assumption
    ASSUMED_STATIC = "ASSUMED_STATIC"      # current value assumed unchanged historically (e.g. security type)
    UNKNOWN = "UNKNOWN"                    # availability cannot be established -> unusable for historical research by default

    @property
    def rank(self) -> int:
        return _PIT_ORDER.index(self)

    @staticmethod
    def weakest(statuses: "list[PitStatus] | set[PitStatus]") -> "PitStatus":
        statuses = list(statuses)
        if not statuses:
            return PitStatus.UNKNOWN
        return max(statuses, key=lambda s: _PIT_ORDER.index(PitStatus(s)))


_PIT_ORDER = [PitStatus.PIT, PitStatus.PIT_CONSERVATIVE, PitStatus.ASSUMED_STATIC, PitStatus.UNKNOWN]

HISTORICAL_RESEARCH_PIT = frozenset({PitStatus.PIT, PitStatus.PIT_CONSERVATIVE, PitStatus.ASSUMED_STATIC})


class InfoKind(str, Enum):
    """Transparency label: every statement shown to a human is one of these."""

    FACT = "FACT"                  # directly from a data source, with provenance
    MODEL_OUTPUT = "MODEL_OUTPUT"  # computed by a quantitative/statistical model
    AI_OPINION = "AI_OPINION"      # produced by an LLM
    HYPOTHESIS = "HYPOTHESIS"      # untested idea
    UNCERTAINTY = "UNCERTAINTY"    # explicitly unknown / not establishable


# --------------------------------------------------------------------------------------------
# Modes, books, states
# --------------------------------------------------------------------------------------------
class Mode(str, Enum):
    RESEARCH = "RESEARCH"
    BOT_PAPER = "BOT_PAPER"
    HUMAN_PAPER = "HUMAN_PAPER"


class Book(str, Enum):
    BOT = "BOT"
    HUMAN = "HUMAN"
    SHADOW = "SHADOW"
    BACKTEST = "BACKTEST"


class SystemState(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "SYSTEM_PAUSED"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


# --------------------------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------------------------
class AIDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    WATCH = "WATCH"
    UNKNOWN = "UNKNOWN"


class ObjectionSeverity(str, Enum):
    HARD_FAIL = "HARD_FAIL"
    MATERIAL_CONCERN = "MATERIAL_CONCERN"
    MINOR_CONCERN = "MINOR_CONCERN"
    UNKNOWN = "UNKNOWN"


class FinalDecision(str, Enum):
    """Bot's final decision on a candidate after every layer (incl. risk)."""

    TRADE = "TRADE"          # an order will be generated
    NO_TRADE = "NO_TRADE"    # rejected somewhere in the chain (see reject_stage)
    WATCH = "WATCH"
    UNKNOWN = "UNKNOWN"      # could not decide (e.g. missing data / AI failure with UNKNOWN fallback)


class RejectStage(str, Enum):
    """Where a candidate was stopped. Used by the shadow book and counterfactual engine."""

    NONE = "NONE"
    DATA = "DATA"
    SIGNAL = "SIGNAL"
    STRATEGY = "STRATEGY"
    RANKING = "RANKING"        # below the cut for AI review / capacity
    ML = "ML"
    AI = "AI"
    NO_TRADE = "NO_TRADE"
    EV = "EV"
    PORTFOLIO = "PORTFOLIO"
    RISK = "RISK"
    EXECUTION = "EXECUTION"
    SYSTEM_PAUSED = "SYSTEM_PAUSED"
    HUMAN = "HUMAN"


class HumanAction(str, Enum):
    BUY = "BUY"
    PASS = "PASS"
    WATCH = "WATCH"
    SELL = "SELL"
    WAIT = "WAIT"


class ExitReason(str, Enum):
    STOP = "STOP"
    TARGET = "TARGET"
    TIME = "TIME"
    INVALIDATION = "INVALIDATION"
    STRATEGY_REVERSAL = "STRATEGY_REVERSAL"
    MARKET_RISK = "MARKET_RISK"
    THESIS_INVALIDATION = "THESIS_INVALIDATION"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    DELISTED = "DELISTED"
    MANUAL = "MANUAL"
    END_OF_TEST = "END_OF_TEST"


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "pending_submit"   # internal: created, not yet acknowledged
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    UNKNOWN = "unknown"                 # broker state could not be established -> stop + reconcile

    @property
    def is_terminal(self) -> bool:
        return self in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}


# --------------------------------------------------------------------------------------------
# Research lifecycle
# --------------------------------------------------------------------------------------------
class StrategyStage(str, Enum):
    IDEA = "IDEA"
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    WALK_FORWARD = "WALK_FORWARD"
    LOCKED_HOLDOUT = "LOCKED_HOLDOUT"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    PROMOTED = "PROMOTED"


STRATEGY_STAGE_ORDER = list(StrategyStage)


class StrategyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SHADOW = "SHADOW"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


class ModelStatus(str, Enum):
    ACTIVE = "ACTIVE"
    MONITORED = "MONITORED"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    TESTING = "TESTING"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CheckSeverity(str, Enum):
    CRITICAL = "CRITICAL"   # failure => NO TRADE (or SYSTEM_PAUSED for health checks)
    WARNING = "WARNING"
    INFO = "INFO"


# --------------------------------------------------------------------------------------------
# Core records passed between layers
# --------------------------------------------------------------------------------------------
@dataclass
class TradePlan:
    """How a candidate would be traded. Prices are RAW (unadjusted) USD as of the signal date."""

    entry: str = "next_open"                 # execution convention
    entry_ref_price: float | None = None     # last raw close known at decision time
    stop_price: float | None = None          # protective stop / invalidation level
    target_price: float | None = None        # optional profit target
    holding_sessions: int = 20               # maximum holding period
    invalidation: str = ""                   # human-readable thesis invalidation condition


@dataclass
class Candidate:
    """One strategy's opinion about one symbol on one session date. Immutable once recorded."""

    symbol: str
    as_of_date: date
    strategy_id: str
    strategy_version: str
    score: float
    direction: Direction = Direction.LONG
    features: dict[str, float] = field(default_factory=dict)
    plan: TradePlan = field(default_factory=TradePlan)
    risk: dict[str, float] = field(default_factory=dict)       # e.g. atr, vol_20d, adv
    pit_status: PitStatus = PitStatus.PIT
    reasons: list[str] = field(default_factory=list)            # quantitative reasoning (FACT/MODEL_OUTPUT)
    candidate_id: str = field(default_factory=lambda: new_id("cand"))
    created_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of_date"] = self.as_of_date.isoformat()
        d["created_at"] = self.created_at.isoformat()
        d["direction"] = self.direction.value
        d["pit_status"] = self.pit_status.value
        return d


@dataclass
class CheckResult:
    """Result of one gate in the data->signal->strategy->EV->portfolio->risk->execution chain,
    a no-trade rule, or a health check."""

    name: str
    passed: bool
    severity: CheckSeverity = CheckSeverity.CRITICAL
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return (not self.passed) and self.severity is CheckSeverity.CRITICAL


@dataclass
class Objection:
    category: str
    severity: ObjectionSeverity
    text: str
    source: str = "ai"          # ai | no_trade_engine | risk_engine | human


@dataclass
class AIAssessment:
    """Structured output of one LLM role (researcher | adversary | judge). AI_OPINION, never FACT.

    ``qualitative_confidence`` is a label (low/medium/high/unknown) and must NEVER be converted
    into a probability without separate empirical validation.
    """

    role: str
    provider: str
    model: str
    decision: AIDecision
    summary: str = ""
    key_evidence: list[str] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    qualitative_confidence: str = "unknown"
    ok: bool = True                   # False => call failed / invalid; decision forced to UNKNOWN
    error: str | None = None
    call_id: str | None = None


@dataclass
class AIReview:
    """All AI roles for one candidate plus the aggregated AI decision."""

    candidate_id: str
    researcher: AIAssessment | None = None
    adversary: AIAssessment | None = None
    judge: AIAssessment | None = None
    decision: AIDecision = AIDecision.UNKNOWN
    enabled: bool = True              # False => AI layer disabled; decision is UNKNOWN by design

    @property
    def objections(self) -> list[Objection]:
        out: list[Objection] = []
        for a in (self.researcher, self.adversary, self.judge):
            if a is not None:
                out.extend(a.objections)
        return out


@dataclass
class MLPrediction:
    model_id: str
    model_version: str
    target: str
    probability: float | None         # calibrated probability when calibrated=True
    calibrated: bool = False
    status: str = "ACTIVE"            # model status at prediction time


@dataclass
class ExpectedValue:
    """Quantitative EV. Probability comes from validated historical statistics ONLY."""

    p_win: float | None
    avg_win: float | None          # fractional return
    avg_loss: float | None         # fractional return (negative)
    cost: float                    # round-trip cost as fraction
    ev: float | None               # expected net return per trade (fraction)
    n_obs: int = 0
    method: str = ""
    pit_status: PitStatus = PitStatus.PIT
    notes: list[str] = field(default_factory=list)
