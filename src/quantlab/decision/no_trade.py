"""No-trade engine: "why should we NOT trade this?"

Every rule returns exactly one :class:`~quantlab.core.types.CheckResult`, whether it passes or not, so
the audit trail shows which rules ran. The severity is stated on each result:
  * CRITICAL failure -> the candidate cannot be traded (the final decision is NO_TRADE; AI cannot override)
  * WARNING failure  -> recorded and shown, does not block by itself
  * INFO             -> rule passed, or was not applicable (the reason says which)

Rules fail CLOSED where the unknown quantity guards against real harm (unknown liquidity,
volatility, price or signal-day bar -> CRITICAL). Rules built on MODEL ESTIMATES (the estimated
earnings date) are labeled as estimates. Each result carries ``details["info_kind"]``
(core.types.InfoKind) so the report can label it.

Point-in-time: the engine only reads the candidate and the context. :meth:`NoTradeContext.build`
derives missing liquidity/volatility/bar facts from ``panel`` rows up to ``as_of`` only, using exactly
the feature definitions in ARCHITECTURE.md section 4 (adv20 = median of 20 sessions of raw dollar
volume, vol_20d = std(ret, 20) * sqrt(252)). tests/decision proves this is truncation-invariant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.core.types import (
    AIReview,
    Candidate,
    CheckResult,
    CheckSeverity,
    InfoKind,
    MLPrediction,
    ObjectionSeverity,
    PitStatus,
)
from quantlab.db.database import Database
from quantlab.decision.stats_provider import StrategyStats

_ADV_WINDOW = 20
_VOL_WINDOW = 20
_ANNUALIZE = math.sqrt(252.0)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class NoTradeContext:
    """Everything the no-trade rules look at besides the candidate. Unknown = None."""

    as_of: date | None = None
    adv20: float | None = None
    vol_20d: float | None = None
    est_sessions_to_earnings: float | None = None     # MODEL estimate (63 - days_since_earnings)
    sessions_to_known_earnings: int | None = None     # only from a PIT-confirmed schedule (none configured today)
    same_day_candidates: list[Candidate] = field(default_factory=list)
    stats: StrategyStats | None = None
    regime: str | None = None
    quarantined: bool = False
    quarantine_reasons: list[str] = field(default_factory=list)
    has_bar_on_signal_day: bool | None = None
    last_raw_close: float | None = None
    halted: bool | None = None
    ml_predictions: list[MLPrediction] = field(default_factory=list)
    ai_review: AIReview | None = None
    sources: dict[str, str] = field(default_factory=dict)   # where each value came from (audit)

    @classmethod
    def build(cls, candidate: Candidate, *, panel: Any = None, features: Mapping[str, Any] | None = None,
              **explicit: Any) -> "NoTradeContext":
        """Resolve each value, in order: explicit kwarg, then ``features`` (feature values of this
        symbol at as_of), then ``candidate.features`` / ``candidate.risk``, then values derived from
        ``panel`` rows <= as_of. Anything still unknown stays None, and the rules treat it
        fail-safe."""
        as_of = explicit.pop("as_of", None) or candidate.as_of_date
        ctx = cls(as_of=as_of, **explicit)
        feats = dict(features or {})
        pools: list[tuple[str, Mapping[str, Any]]] = [("features", feats), ("candidate.features", candidate.features),
                                                      ("candidate.risk", candidate.risk)]
        for attr, name in (("adv20", "adv20"), ("vol_20d", "vol_20d"),
                           ("est_sessions_to_earnings", "est_sessions_to_earnings")):
            if attr in explicit and explicit[attr] is not None:
                ctx.sources[attr] = "explicit"
                continue
            for label, pool in pools:
                v = _num(pool.get(name)) if pool else None
                if v is not None:
                    setattr(ctx, attr, v)
                    ctx.sources[attr] = label
                    break
        if panel is not None:
            facts = panel_facts(panel, candidate.symbol, as_of)
            for attr in ("adv20", "vol_20d", "has_bar_on_signal_day", "last_raw_close"):
                if getattr(ctx, attr) is None and facts.get(attr) is not None:
                    setattr(ctx, attr, facts[attr])
                    ctx.sources[attr] = "panel<=as_of"
        return ctx


def panel_facts(panel: Any, symbol: str, as_of: Any) -> dict[str, Any]:
    """PIT facts for one symbol from panel rows up to and including ``as_of``.

    Mirrors the catalog definitions (ARCHITECTURE section 4) with full windows required, like
    ``rolling(n)`` with ``min_periods=n``. Partial windows give None, never a guess."""
    d = to_session(as_of)
    out: dict[str, Any] = {"adv20": None, "vol_20d": None, "has_bar_on_signal_day": None, "last_raw_close": None}
    if symbol not in panel.symbols:
        out["has_bar_on_signal_day"] = False
        return out
    close = panel.close[symbol].loc[:d]
    if len(close) == 0:
        out["has_bar_on_signal_day"] = False
        return out
    on_day = close.index[-1] == d and _num(close.iloc[-1]) is not None
    out["has_bar_on_signal_day"] = bool(on_day)
    if on_day:
        out["last_raw_close"] = float(close.iloc[-1])
    dv = panel.dollar_volume[symbol].loc[:d].iloc[-_ADV_WINDOW:]
    if len(dv) == _ADV_WINDOW and dv.notna().all():
        out["adv20"] = float(dv.median())
    r = panel.ret[symbol].loc[:d].iloc[-_VOL_WINDOW:]
    if len(r) == _VOL_WINDOW and r.notna().all():
        out["vol_20d"] = float(r.std(ddof=1) * _ANNUALIZE)
    return out


def active_quarantine(db: Database, symbol: str, as_of: Any) -> list[str]:
    """Reasons ``symbol`` is quarantined on ``as_of`` according to ``symbol_quarantine`` (002).

    Rows are append-only. For each check_name, the latest row that has started by ``as_of`` decides.
    It is active while its ``to_date`` is NULL or >= as_of (inclusive, the conservative reading).
    If the data subsystem publishes an authoritative helper, pass its answer as ``quarantined=``
    instead."""
    d = to_session(as_of).date().isoformat()
    rows = db.fetchall(
        "SELECT id, check_name, reason, from_date, to_date FROM symbol_quarantine "
        "WHERE symbol = ? AND (from_date IS NULL OR from_date <= ?) ORDER BY id",
        (symbol, d),
    )
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        latest[r["check_name"]] = r
    return [f"{r['check_name']}: {r['reason']}" for r in latest.values() if r["to_date"] is None or r["to_date"] >= d]


def _res(name: str, passed: bool, severity: CheckSeverity, reason: str, kind: InfoKind, **details: Any) -> CheckResult:
    return CheckResult(name=name, passed=passed, severity=CheckSeverity.INFO if passed else severity,
                       reason=reason, details={"info_kind": kind.value, **details})


class NoTradeEngine:
    def __init__(self, config: Config):
        self.config = config
        nt = lambda k, d: config.get(f"no_trade.{k}", d)  # noqa: E731
        self.min_adv = float(nt("min_median_dollar_volume", 5_000_000))
        self.max_vol = float(nt("max_annualized_vol", 1.20))
        self.blackout = int(nt("earnings_blackout_sessions", 2))
        self.estimate_severity = CheckSeverity(str(nt("earnings_estimate_severity", "WARNING")).upper())
        self.max_disagreement = float(nt("max_model_disagreement", 0.35))
        self.min_trust = int(nt("min_strategy_trades_for_trust", 30))
        self.min_regime_trades = int(nt("min_regime_trades", self.min_trust))
        self.regime_t = float(nt("regime_t_threshold", 2.0))
        self.block_unknown_pit = bool(nt("block_on_unknown_pit", True))
        self.min_price = float(config.get("universe.min_price", 5.0))

    def evaluate(self, candidate: Candidate, context: NoTradeContext | None = None) -> list[CheckResult]:
        c = context or NoTradeContext.build(candidate)
        return [
            self._liquidity(c),
            self._volatility(c),
            self._earnings(c),
            self._conflicts(candidate, c),
            self._regime(c),
            self._strategy_history(c),
            self._pit(candidate),
            self._quarantine(c),
            self._missing_features(candidate),
            self._signal_day_bar(candidate, c),
            self._min_price(candidate, c),
            self._halted(c),
            self._model_disagreement(c),
            *self._ai(c),
        ]

    # -- liquidity / volatility / events ----------------------------------------------------------
    def _liquidity(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.liquidity"
        if c.adv20 is None:
            return _res(n, False, CheckSeverity.CRITICAL, "liquidity UNKNOWN (adv20 missing): fail closed",
                        InfoKind.UNCERTAINTY)
        ok = c.adv20 >= self.min_adv
        return _res(n, ok, CheckSeverity.CRITICAL,
                    f"adv20 ${c.adv20:,.0f} {'>=' if ok else '<'} minimum ${self.min_adv:,.0f}", InfoKind.FACT,
                    adv20=c.adv20, min=self.min_adv)

    def _volatility(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.volatility"
        if c.vol_20d is None:
            return _res(n, False, CheckSeverity.CRITICAL, "volatility UNKNOWN (vol_20d missing): fail closed",
                        InfoKind.UNCERTAINTY)
        ok = c.vol_20d <= self.max_vol
        return _res(n, ok, CheckSeverity.CRITICAL,
                    f"vol_20d {c.vol_20d:.2f} {'<=' if ok else '>'} max {self.max_vol:.2f} (annualized)",
                    InfoKind.FACT, vol_20d=c.vol_20d, max=self.max_vol)

    def _earnings(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.earnings_proximity"
        if c.sessions_to_known_earnings is not None:
            ok = c.sessions_to_known_earnings > self.blackout
            return _res(n, ok, CheckSeverity.CRITICAL,
                        f"confirmed earnings in {c.sessions_to_known_earnings} sessions "
                        f"({'outside' if ok else 'inside'} {self.blackout}-session blackout)", InfoKind.FACT,
                        sessions=c.sessions_to_known_earnings, estimate=False)
        if c.est_sessions_to_earnings is not None:
            ok = c.est_sessions_to_earnings > self.blackout
            return _res(n, ok, self.estimate_severity,
                        f"ESTIMATED earnings in ~{c.est_sessions_to_earnings:.0f} sessions (model estimate: 63 - "
                        f"days_since_earnings, no PIT schedule) {'outside' if ok else 'inside'} "
                        f"{self.blackout}-session blackout", InfoKind.MODEL_OUTPUT,
                        sessions=c.est_sessions_to_earnings, estimate=True)
        return _res(n, True, CheckSeverity.INFO, "next earnings date UNKNOWN: rule not applicable",
                    InfoKind.UNCERTAINTY, estimate=None)

    def _conflicts(self, cand: Candidate, c: NoTradeContext) -> CheckResult:
        n = "no_trade.conflicting_signals"
        opp = [o for o in c.same_day_candidates
               if o.symbol == cand.symbol and o.as_of_date == cand.as_of_date and o.direction != cand.direction
               and o.candidate_id != cand.candidate_id]
        if opp:
            return _res(n, False, CheckSeverity.CRITICAL,
                        f"{len(opp)} opposite-direction signal(s) on the same symbol/day: "
                        + ", ".join(f"{o.strategy_id} {o.direction.value}" for o in opp), InfoKind.MODEL_OUTPUT,
                        conflicting=[o.candidate_id for o in opp])
        return _res(n, True, CheckSeverity.INFO, "no opposite-direction signals", InfoKind.MODEL_OUTPUT)

    # -- strategy evidence ----------------------------------------------------------------------
    def _regime(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.regime"
        if c.regime is None:
            return _res(n, True, CheckSeverity.INFO, "current regime UNKNOWN: rule not applied", InfoKind.UNCERTAINTY)
        rs = c.stats.by_regime.get(c.regime) if c.stats is not None else None
        if rs is None:
            return _res(n, True, CheckSeverity.INFO, f"no history in regime {c.regime!r}: rule not applied",
                        InfoKind.UNCERTAINTY, regime=c.regime)
        if rs.n < self.min_regime_trades or rs.t_stat is None or rs.expectancy is None:
            return _res(n, True, CheckSeverity.INFO,
                        f"regime {c.regime!r}: n={rs.n} < {self.min_regime_trades} (or no dispersion): too few "
                        "samples to judge", InfoKind.MODEL_OUTPUT, regime=c.regime, n=rs.n)
        bad = rs.expectancy < 0 and rs.t_stat < -self.regime_t
        return _res(n, not bad, CheckSeverity.CRITICAL,
                    f"regime {c.regime!r}: n={rs.n}, net expectancy {rs.expectancy * 1e4:.1f} bps, t={rs.t_stat:.2f}"
                    + (f" (significantly negative, t < -{self.regime_t:g})" if bad else ""), InfoKind.MODEL_OUTPUT,
                    regime=c.regime, n=rs.n, t_stat=rs.t_stat, source=c.stats.source)

    def _strategy_history(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.strategy_history"
        s = c.stats
        if s is None or s.n == 0:
            return _res(n, False, CheckSeverity.WARNING, "unproven: no matured trade history for this strategy version",
                        InfoKind.UNCERTAINTY, n=0, source=s.source if s else "none")
        det = dict(n=s.n, source=s.source, expectancy=s.expectancy, t_stat=s.t_stat)
        exp_txt = f"{s.expectancy * 1e4:.1f} bps" if s.expectancy is not None else "UNKNOWN"
        if s.n >= self.min_trust and s.expectancy is not None and s.expectancy < 0:
            label = "validated" if s.validated else f"{s.source} (optimistic)"
            return _res(n, False, CheckSeverity.CRITICAL,
                        f"negative {label} net expectancy {exp_txt} over n={s.n} trades", InfoKind.MODEL_OUTPUT, **det)
        if not s.validated:
            return _res(n, False, CheckSeverity.WARNING,
                        f"unproven: only {s.source} evidence (n={s.n}), not validated out-of-sample",
                        InfoKind.MODEL_OUTPUT, **det)
        if s.n < self.min_trust:
            return _res(n, False, CheckSeverity.WARNING,
                        f"unproven: n={s.n} < {self.min_trust} validated trades", InfoKind.MODEL_OUTPUT, **det)
        return _res(n, True, CheckSeverity.INFO, f"validated history n={s.n} ({s.source}), net expectancy {exp_txt}",
                    InfoKind.MODEL_OUTPUT, **det)

    # -- data uncertainty -------------------------------------------------------------------------
    def _pit(self, cand: Candidate) -> CheckResult:
        n = "no_trade.pit_status"
        status = PitStatus(cand.pit_status)
        if status is PitStatus.UNKNOWN:
            sev = CheckSeverity.CRITICAL if self.block_unknown_pit else CheckSeverity.WARNING
            return _res(n, False, sev, "candidate PIT status UNKNOWN: availability of its inputs cannot be established",
                        InfoKind.UNCERTAINTY, pit_status=status.value)
        return _res(n, True, CheckSeverity.INFO, f"PIT status {status.value}", InfoKind.FACT, pit_status=status.value)

    def _quarantine(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.quarantine"
        if c.quarantined:
            return _res(n, False, CheckSeverity.CRITICAL,
                        "symbol quarantined by data validation: " + ("; ".join(c.quarantine_reasons) or "no reason given"),
                        InfoKind.FACT, reasons=c.quarantine_reasons)
        return _res(n, True, CheckSeverity.INFO, "symbol not quarantined", InfoKind.FACT)

    def _missing_features(self, cand: Candidate) -> CheckResult:
        n = "no_trade.missing_features"
        missing = sorted(k for k, v in cand.features.items() if _num(v) is None)
        if missing:
            return _res(n, False, CheckSeverity.WARNING, f"{len(missing)} feature(s) UNKNOWN: {', '.join(missing)}",
                        InfoKind.UNCERTAINTY, missing=missing)
        return _res(n, True, CheckSeverity.INFO, "all candidate features present", InfoKind.FACT)

    # -- execution uncertainty --------------------------------------------------------------------
    def _signal_day_bar(self, cand: Candidate, c: NoTradeContext) -> CheckResult:
        n = "no_trade.signal_day_bar"
        has = c.has_bar_on_signal_day
        src = "context"
        if has is None:
            ref = _num(cand.plan.entry_ref_price)
            has, src = (True, "plan.entry_ref_price") if ref is not None and ref > 0 else (None, "unknown")
        if has is None:
            return _res(n, False, CheckSeverity.CRITICAL, "signal-day bar UNKNOWN: fail closed", InfoKind.UNCERTAINTY)
        return _res(n, bool(has), CheckSeverity.CRITICAL,
                    "bar present on the signal day" if has else "NO bar on the signal day (halt/delisting/data gap)",
                    InfoKind.FACT, source=src)

    def _min_price(self, cand: Candidate, c: NoTradeContext) -> CheckResult:
        n = "no_trade.min_price"
        px = c.last_raw_close if c.last_raw_close is not None else _num(cand.plan.entry_ref_price)
        if px is None:
            return _res(n, False, CheckSeverity.CRITICAL, "raw price UNKNOWN: fail closed", InfoKind.UNCERTAINTY)
        ok = px >= self.min_price
        return _res(n, ok, CheckSeverity.CRITICAL, f"raw close ${px:.2f} {'>=' if ok else '<'} ${self.min_price:.2f}",
                    InfoKind.FACT, price=px, min=self.min_price)

    def _halted(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.halted"
        if c.halted:
            return _res(n, False, CheckSeverity.CRITICAL, "trading halted", InfoKind.FACT)
        if c.halted is None:
            return _res(n, True, CheckSeverity.INFO, "halt status not available (a halt would show as a missing bar)",
                        InfoKind.UNCERTAINTY)
        return _res(n, True, CheckSeverity.INFO, "not halted", InfoKind.FACT)

    # -- models / AI ------------------------------------------------------------------------------
    def _model_disagreement(self, c: NoTradeContext) -> CheckResult:
        n = "no_trade.model_disagreement"
        preds = [p for p in c.ml_predictions if _num(p.probability) is not None]
        used = [p for p in preds if p.calibrated and str(p.status).upper() == "ACTIVE"]
        for group, sev, label in ((used, CheckSeverity.CRITICAL, "active calibrated"),
                                  (preds, CheckSeverity.WARNING, "all")):
            if len(group) >= 2:
                probs = [float(p.probability) for p in group]
                spread = max(probs) - min(probs)
                if spread > self.max_disagreement:
                    return _res(n, False, sev,
                                f"{label} ML models disagree: probability spread {spread:.2f} > "
                                f"{self.max_disagreement:.2f}", InfoKind.MODEL_OUTPUT, spread=spread,
                                models={f"{p.model_id}@{p.model_version}": p.probability for p in group})
        if len(preds) < 2:
            return _res(n, True, CheckSeverity.INFO, f"{len(preds)} ML prediction(s): disagreement not measurable",
                        InfoKind.UNCERTAINTY)
        return _res(n, True, CheckSeverity.INFO, "ML models agree within tolerance", InfoKind.MODEL_OUTPUT)

    def _ai(self, c: NoTradeContext) -> list[CheckResult]:
        r = c.ai_review
        if r is None or not r.enabled:
            return [_res("no_trade.ai_hard_fail", True, CheckSeverity.INFO, "no AI review (layer disabled or not run)",
                         InfoKind.UNCERTAINTY)]
        hard, material = [], []
        for a in (r.researcher, r.adversary, r.judge):
            if a is None or not a.ok:
                continue
            for o in a.objections:
                sev = ObjectionSeverity(o.severity)
                if sev is ObjectionSeverity.HARD_FAIL:
                    hard.append(f"[{a.role}] {o.category}: {o.text}")
                elif sev is ObjectionSeverity.MATERIAL_CONCERN:
                    material.append(f"[{a.role}] {o.category}: {o.text}")
        out = [_res("no_trade.ai_hard_fail", not hard, CheckSeverity.CRITICAL,
                    ("AI HARD_FAIL objection(s): " + " | ".join(hard)) if hard else "no AI HARD_FAIL objections",
                    InfoKind.AI_OPINION, objections=hard)]
        if material:
            out.append(_res("no_trade.ai_material_concern", False, CheckSeverity.WARNING,
                            "AI material concern(s): " + " | ".join(material), InfoKind.AI_OPINION,
                            objections=material))
        return out


def blocking(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.blocking]


def warnings(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if not r.passed and r.severity is CheckSeverity.WARNING]


def results_to_dicts(results: list[CheckResult]) -> list[dict[str, Any]]:
    return [{"name": r.name, "passed": r.passed, "severity": r.severity.value, "reason": r.reason,
             "details": r.details} for r in results]


__all__ = ["NoTradeContext", "NoTradeEngine", "active_quarantine", "blocking", "panel_facts",
           "results_to_dicts", "warnings"]
