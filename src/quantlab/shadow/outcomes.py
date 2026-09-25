"""Shadow outcome tracking: what each recorded opportunity actually went on to do.

For every ``shadow_opportunities`` row whose horizon has MATURED on the given panel, compute the
outcome exactly once and append it to ``shadow_outcomes`` (+ an audit row in
``shadow_outcome_details``). Both legs use :func:`quantlab.core.tradesim.simulate_plan` with
:meth:`CostModel.from_config`, i.e. the same fill model as the backtester and the paper books:

  * plan leg (``ret``)      — follows the recorded TradePlan (stop/target on closes, time exit),
                              next-open entry, round-trip modeled costs;
  * hold leg (``ret_hold``) — plain buy-and-hold over the same horizon (no stop/target), same
                              entry and costs, so "did the plan's exits help?" is answerable;
  * benchmark / excess      — the market benchmark over the plan leg's entry->exit window.

Point-in-time rules (why the numbers can be trusted):
  * An outcome is written only when the time-exit fill session (signal + 1 + horizon) is already
    in the panel AND both legs have closed. Anything else stays pending — nothing is extrapolated
    and no session beyond the panel's last date is ever touched (``as_of`` truncates first).
  * Once written a row is never updated (DB triggers); rerunning ``update`` is a no-op.
  * A symbol that stops printing is only declared DELISTED after ``execution.delisting_missing_sessions``
    consecutive sessions of absence (enforced by core.tradesim itself, booked on that session), so a
    one-day data gap at the panel's edge cannot be frozen forever as a -30% delisting.
  * A symbol missing from the panel entirely is SKIPPED (maybe a partial panel), never marked
    ``no_data``; ``no_data`` is reserved for "the panel covers it and there was nothing to trade".
  * Synthetic and real data never mix: the panel's provenance decides which opportunities it may
    evaluate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import pandas as pd

from quantlab.config import Config
from quantlab.core.costs import CostModel
from quantlab.core.tradesim import PlanOutcome, simulate_plan
from quantlab.core.types import Direction, ExitReason, TradePlan
from quantlab.data.panel import DataBundle, Panel
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event
from quantlab.shadow.book import parse_quant_reasoning

log = get_logger(__name__)

METHOD = "tradesim.simulate_plan/v1"
FINAL_STATUSES = ("complete", "delisted", "no_data")


@dataclass
class OutcomeEvaluation:
    """Result of evaluating one opportunity on one panel (pure; nothing written)."""

    opportunity_id: str
    status: str                        # complete | delisted | no_data | pending | skipped
    reason: str = ""
    horizon_sessions: int | None = None
    row: dict[str, Any] | None = None       # shadow_outcomes row when final
    details: dict[str, Any] | None = None   # shadow_outcome_details row when final
    plan_outcome: PlanOutcome | None = field(default=None, repr=False)
    hold_outcome: PlanOutcome | None = field(default=None, repr=False)

    @property
    def is_final(self) -> bool:
        return self.status in FINAL_STATUSES


def panel_is_synthetic(panel: Panel) -> bool:
    """True when every provider recorded in the panel's metadata is the synthetic generator."""
    providers = panel.meta.get("providers") or []
    return bool(providers) and all(str(p) == "synthetic" for p in providers)


def _iso(ts: Any) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).date().isoformat()


class ShadowOutcomeTracker:
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config
        self.costs = CostModel.from_config(config)
        self.benchmark: str = str(config.get("benchmarks.market", "SPY"))
        self.default_horizon = int(config.get("shadow.default_horizon_sessions", 20))
        self.delisting_grace = self.costs.delisting_missing_sessions   # single shared rule (tradesim)
        self.last_stats: dict[str, int] = {}

    # -- selection ---------------------------------------------------------------------------
    def horizon_for(self, opp: Mapping[str, Any]) -> int:
        h = opp.get("holding_sessions")
        try:
            h = int(h) if h is not None else 0
        except (TypeError, ValueError):
            h = 0
        return h if h > 0 else self.default_horizon

    def pending(self, synthetic: bool = False, up_to: str | None = None) -> list[dict[str, Any]]:
        """Opportunities with no outcome row yet at their horizon (optionally signal date <= up_to)."""
        sql = (
            "SELECT o.*, c.direction AS cand_direction FROM shadow_opportunities o "
            "LEFT JOIN candidates c ON c.candidate_id = o.candidate_id "
            "WHERE o.is_synthetic = ? AND NOT EXISTS ("
            "  SELECT 1 FROM shadow_outcomes so WHERE so.opportunity_id = o.opportunity_id"
            "  AND so.horizon_sessions = CASE WHEN o.holding_sessions > 0 THEN o.holding_sessions ELSE ? END)"
        )
        params: list[Any] = [int(bool(synthetic)), self.default_horizon]
        if up_to is not None:
            sql += " AND o.as_of_date <= ?"
            params.append(str(up_to)[:10])
        sql += " ORDER BY o.as_of_date, o.created_at"
        return self.db.fetchall(sql, params)

    @staticmethod
    def direction_for(opp: Mapping[str, Any]) -> Direction | None:
        """Trade direction from the candidates table, else from the ShadowBook JSON. Never guessed."""
        for raw in (opp.get("cand_direction"), parse_quant_reasoning(opp.get("quant_reasoning")).get("direction")):
            if raw:
                try:
                    return Direction(str(raw).upper())
                except ValueError:
                    return None
        return None

    # -- evaluation (pure) ---------------------------------------------------------------------
    def evaluate(self, panel: Panel, opp: Mapping[str, Any]) -> OutcomeEvaluation:
        oid = str(opp["opportunity_id"])
        horizon = self.horizon_for(opp)
        ev = OutcomeEvaluation(oid, "pending", horizon_sessions=horizon)
        symbol = str(opp["symbol"])
        signal = pd.Timestamp(opp["as_of_date"]).normalize()
        dates = panel.dates
        n = len(dates)
        if n == 0 or signal > dates[-1]:
            ev.reason = "signal date is after the panel's last session"
            return ev
        if signal < dates[0]:
            ev.status, ev.reason = "skipped", "signal date precedes the panel"
            return ev
        if symbol not in panel.symbols:
            ev.status, ev.reason = "skipped", "symbol not in panel (partial panel?)"
            return ev
        if signal not in dates:
            sessions_after = n - int(dates.searchsorted(signal, side="right"))
            if sessions_after >= horizon + 1:
                return self._no_data(ev, panel, "signal date is not a session of the panel")
            ev.reason = "horizon not matured"
            return ev
        i0 = dates.get_loc(signal)
        if n - 1 < i0 + 1 + horizon:
            ev.reason = f"horizon not matured: need session index {i0 + 1 + horizon}, panel ends at {n - 1}"
            return ev
        direction = self.direction_for(opp)
        if direction is None:
            ev.status, ev.reason = "skipped", "trade direction unknown (not guessed)"
            return ev

        ref = opp.get("entry_ref_price")
        plan = TradePlan(entry=opp.get("entry_convention") or "next_open", entry_ref_price=ref,
                         stop_price=opp.get("stop_price"), target_price=opp.get("target_price"),
                         holding_sessions=horizon)
        hold = TradePlan(entry=plan.entry, entry_ref_price=ref, holding_sessions=horizon)
        po = simulate_plan(panel, symbol, signal, plan, self.costs, direction, benchmark=self.benchmark)
        if po.status == "no_entry":
            return self._no_data(ev, panel, "no raw close at the signal session or no later session to enter")
        if po.status == "open":
            ev.reason = "plan leg still open (missing bars inside the window)"
            return ev
        ho = simulate_plan(panel, symbol, signal, hold, self.costs, direction, benchmark=self.benchmark)
        if ho.status not in ("complete", "delisted"):
            ev.reason = "buy-and-hold leg still open (missing bars inside the window)"
            return ev
        measured_at = utcnow_iso()
        ev.status, ev.reason = po.status, "matured"
        ev.plan_outcome, ev.hold_outcome = po, ho
        ev.row = {
            "opportunity_id": oid, "horizon_sessions": horizon, "measured_at": measured_at,
            "entry_date": _iso(po.entry_date), "entry_price": po.entry_price_raw,
            "exit_date": _iso(po.exit_date), "exit_price": po.exit_price_raw,
            "ret": po.net_ret, "ret_hold": ho.net_ret, "mfe": po.mfe, "mae": po.mae,
            "hit_stop": int(po.exit_reason == ExitReason.STOP.value),
            "hit_target": int(po.exit_reason == ExitReason.TARGET.value),
            "benchmark_ret": po.benchmark_ret, "excess_ret": po.excess_ret, "status": po.status,
        }
        ev.details = self._details(oid, horizon, panel, measured_at, direction, po, ho)
        return ev

    def _no_data(self, ev: OutcomeEvaluation, panel: Panel, reason: str) -> OutcomeEvaluation:
        measured_at = utcnow_iso()
        ev.status, ev.reason = "no_data", reason
        ev.row = {"opportunity_id": ev.opportunity_id, "horizon_sessions": ev.horizon_sessions,
                  "measured_at": measured_at, "status": "no_data"}
        ev.details = self._details(ev.opportunity_id, ev.horizon_sessions or 0, panel, measured_at, None, None, None)
        return ev

    def _details(self, oid: str, horizon: int, panel: Panel, measured_at: str, direction: Direction | None,
                 po: PlanOutcome | None, ho: PlanOutcome | None) -> dict[str, Any]:
        return {
            "opportunity_id": oid, "horizon_sessions": horizon, "method": METHOD,
            "direction": direction.value if direction else None,
            "exit_reason": po.exit_reason if po else None,
            "gross_ret": po.gross_ret if po else None, "cost_ret": po.cost_ret if po else None,
            "holding_sessions": po.holding_sessions if po else None,
            "hold_exit_date": _iso(ho.exit_date) if ho else None,
            "hold_exit_reason": ho.exit_reason if ho else None,
            "hold_gross_ret": ho.gross_ret if ho else None,
            "benchmark": self.benchmark if (po and po.benchmark_ret is not None) else None,
            "panel_last_date": _iso(panel.dates[-1]),
            "cost_model_json": to_json(asdict(self.costs)),
            "measured_at": measured_at,
        }

    # -- persistence ---------------------------------------------------------------------------
    def update(self, panel: Panel | DataBundle, as_of=None, synthetic: bool | None = None) -> int:
        """Evaluate every pending opportunity on ``panel`` (truncated at ``as_of`` if given) and
        append the matured outcomes. Returns the number of NEW outcome rows. Idempotent."""
        if isinstance(panel, DataBundle):
            if synthetic is None:
                synthetic = bool(panel.is_synthetic)
            panel = panel.panel
        if as_of is not None:
            panel = panel.truncate(as_of)
        if synthetic is None:
            synthetic = panel_is_synthetic(panel)
        stats: Counter[str] = Counter()
        if len(panel.dates) == 0:
            self.last_stats = {}
            return 0
        inserted = 0
        for opp in self.pending(synthetic=synthetic, up_to=_iso(panel.dates[-1])):
            ev = self.evaluate(panel, opp)
            stats[ev.status] += 1
            if ev.is_final and self._insert(ev):
                inserted += 1
        self.last_stats = dict(stats)
        log_event(log, "shadow outcomes updated", inserted=inserted, panel_last_date=_iso(panel.dates[-1]),
                  synthetic=synthetic, **{f"n_{k}": v for k, v in stats.items()})
        return inserted

    def _insert(self, ev: OutcomeEvaluation) -> bool:
        assert ev.row is not None and ev.details is not None
        with self.db.transaction():
            self.db.insert("shadow_outcomes", ev.row, or_ignore=True)
            if int(self.db.fetchone("SELECT changes() AS c")["c"]) == 0:
                return False          # someone else already measured it: never overwrite
            self.db.insert("shadow_outcome_details", ev.details)
        return True

    def outcomes(self, opportunity_id: str) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT so.*, d.exit_reason, d.gross_ret, d.cost_ret, d.hold_gross_ret, d.panel_last_date, d.method "
            "FROM shadow_outcomes so LEFT JOIN shadow_outcome_details d "
            "ON d.opportunity_id = so.opportunity_id AND d.horizon_sessions = so.horizon_sessions "
            "WHERE so.opportunity_id=? ORDER BY so.horizon_sessions", (opportunity_id,))
