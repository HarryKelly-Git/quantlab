"""Daily paper-trading pipeline (BOT book) — one run per session D, resumable per step.

Order of work for session D (information cutoff = D 16:00 ET; every step sees bundle.truncate(D)):
  1. preflight      run row, system state
  2. data           load the stored bundle, truncate to D (point-in-time view)
  3. validate       data integrity; CRITICAL failure -> SYSTEM_PAUSED (research continues, no orders)
  4. execution      corporate actions -> fills of orders placed at D-1 (at D's open) -> mark at D's close
  5. exits          exit rules evaluated on D's close -> exit orders for D+1's open
  6. research       universe -> features -> regime -> strategy candidates (persisted, immutable)
  7. decide         stats -> EV -> no-trade -> AI (optional) -> portfolio -> risk -> final decision;
                    EVERY candidate is written to decisions + risk_checks + shadow_opportunities
  8. orders         paper entry orders for TRADE decisions (refused if SYSTEM_PAUSED)
  9. outcomes       matured shadow outcomes (what happened to everything we did / did not trade)
 10. risk           book drawdown breach -> SYSTEM_PAUSED
 11. discover       market discovery (research only): what looks interesting today, ranked and explained,
                    with what the unchanged decision chain decided for each setup. Never places orders;
                    a discovery failure is recorded and never blocks risk or the report; it re-runs
                    on resume (idempotent per run)
 12. report         daily Markdown report

Paper trading only: the broker is SimBroker (default) or AlpacaPaperBroker (paper endpoint only).
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.context import AppContext
from quantlab.core.costs import CostModel
from quantlab.core.types import (
    AIDecision, AIReview, Book, CheckResult, FinalDecision, RejectStage, StrategyStage, StrategyStatus, SystemState,
)
from quantlab.data.panel import DataBundle
from quantlab.data.validation import DataValidator, quarantine_map
from quantlab.db.database import to_json, utcnow_iso
from quantlab.decision.expected_value import EVEngine
from quantlab.decision.final import FinalDecisionEngine, persist_decision
from quantlab.decision.no_trade import NoTradeContext, NoTradeEngine
from quantlab.decision.stats_provider import StrategyStatsProvider
from quantlab.execution.exits import build_exit_engine
from quantlab.execution.ledger import Ledger, bind_book
from quantlab.execution.service import PaperExecutionService
from quantlab.execution.sim_broker import SimBroker
from quantlab.features.base import FeatureSet
from quantlab.logging_setup import get_logger, log_event, set_run_id
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.pipeline.records import load_candidates, persist_candidates
from quantlab.portfolio.construction import BookState, PortfolioConstructor
from quantlab.regime import RegimeEngine
from quantlab.risk.engine import DecisionContext, RiskEngine, record_checks
from quantlab.sectors import sector_map
from quantlab.shadow.book import ShadowBook
from quantlab.shadow.outcomes import ShadowOutcomeTracker
from quantlab.strategies.arena import StrategyArena
from quantlab.strategies.registry import build_strategies, register_strategies
from quantlab.universe import UniverseEngine

log = get_logger(__name__)

STEPS = ("preflight", "data", "validate", "execution", "exits", "research", "decide", "orders", "outcomes",
         "risk", "discover", "explore", "report")


@dataclass
class PipelineResult:
    run_id: str
    as_of: str
    system_state: str
    steps: dict[str, str] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None
    errors: list[str] = field(default_factory=list)


class DailyPipeline:
    def __init__(self, ctx: AppContext, synthetic: bool | None = None, broker=None, book: str = Book.BOT.value,
                 bundle: DataBundle | None = None, order_guard=None, exploration_submit: str = "now"):
        self.ctx = ctx
        self.cfg = ctx.config
        self.db = ctx.db
        self.synthetic = synthetic
        self.book = book
        self.costs = CostModel.from_config(self.cfg)
        self.broker = broker or SimBroker(book, self.costs, config=self.cfg, db=self.db,
                                          starting_cash=float(self.cfg.get(f"paper.{book.lower()}.starting_cash", 100000)))
        bind_book(self.db, book, self.broker.name)      # a book never mixes sim and Alpaca-paper history
        self.ledger = Ledger(self.db, book, config=self.cfg, broker_name=self.broker.name)
        self.exec = PaperExecutionService(self.db, self.cfg, book, self.broker, self.ledger)
        # extra per-order gate from the paper runner (execution window, broker verified, reconciled)
        self.exec.submission_guard = order_guard
        self.killswitch = KillSwitch(self.db)
        self._full_bundle = bundle
        # "now": plan and (EXPLORATION mode) submit in this run (sim replays: simulated time just after
        # the close). "preopen": plan only; the paper runner revalidates and submits before the open.
        self.exploration_submit = exploration_submit

    # -- data ------------------------------------------------------------------------------------
    def full_bundle(self) -> DataBundle:
        if self._full_bundle is None:
            snap = self.ctx.store.snapshot(synthetic=self.synthetic)
            self._full_bundle = self.ctx.store.load_bundle(self.cfg.section("benchmarks"), snapshot=snap,
                                                           synthetic=self.synthetic)
        return self._full_bundle

    # -- step bookkeeping --------------------------------------------------------------------------
    def _step(self, run_id: str, name: str, status: str, output: dict | None = None, error: str | None = None) -> None:
        now = utcnow_iso()
        self.db.upsert("pipeline_steps", {"run_id": run_id, "step": name, "step_order": STEPS.index(name),
                                          "status": status, "started_at": now if status == "running" else None,
                                          "finished_at": None if status == "running" else now,
                                          "error": error, "output_json": to_json(output or {})}, ["run_id", "step"])

    def _done(self, run_id: str) -> set[str]:
        return {r["step"] for r in self.db.fetchall(
            "SELECT step FROM pipeline_steps WHERE run_id=? AND status='succeeded'", (run_id,))}

    # -- run -------------------------------------------------------------------------------------
    def run(self, as_of=None, resume_run_id: str | None = None) -> PipelineResult:
        full = self.full_bundle()
        d = pd.Timestamp(as_of) if as_of is not None else full.panel.dates[-1]
        if d not in full.panel.dates:
            raise ValueError(f"{d.date()} is not a session in the stored data")
        run_id = resume_run_id or self.ctx.start_run("pipeline", mode="BOT_PAPER", as_of_date=str(d.date()))
        set_run_id(run_id)
        done = self._done(run_id) if resume_run_id else set()
        res = PipelineResult(run_id, str(d.date()), self.killswitch.state()[0].value)
        state: dict[str, Any] = {"as_of": d}
        steps = [("preflight", self._preflight), ("data", self._data), ("validate", self._validate),
                 ("execution", self._execution), ("exits", self._exits), ("research", self._research),
                 ("decide", self._decide), ("orders", self._orders), ("outcomes", self._outcomes),
                 ("risk", self._risk), ("discover", self._discover), ("explore", self._explore),
                 ("report", self._report)]
        # in-memory steps are always recomputed on resume (deterministic); durable ones are skipped
        # 'discover' is deliberately NOT durable: it re-runs on resume (its persistence is idempotent
        # per run_id), so a failed discovery is retried and the report always gets its section
        durable = {"execution", "exits", "decide", "orders", "outcomes", "report"}
        for name, fn in steps:
            if name in done and name in durable:
                res.steps[name] = "skipped (already done)"
                if name == "decide":
                    state["decisions"] = self._load_decisions(run_id)
                continue
            self._step(run_id, name, "running")
            try:
                out = fn(run_id, state) or {}
                self._step(run_id, name, "succeeded", out)
                res.steps[name] = "succeeded"
                res.counts.update({f"{name}.{k}": v for k, v in out.items() if isinstance(v, (int, float, str))})
            except Exception as exc:   # a failed step stops the run visibly; resumable later
                tb = traceback.format_exc()[-3000:]
                self._step(run_id, name, "failed", error=tb)
                res.steps[name] = "failed"
                res.errors.append(f"{name}: {exc!r}")
                log_event(log, "pipeline step failed", level=40, step=name, error=repr(exc))
                self.ctx.finish_run(run_id, "failed", f"{name}: {exc!r}")
                res.system_state = self.killswitch.state()[0].value
                return res
        self.ctx.finish_run(run_id, "succeeded")
        res.system_state = self.killswitch.state()[0].value
        res.report_path = state.get("report_path")
        return res

    # -- steps -----------------------------------------------------------------------------------
    def _preflight(self, run_id, st) -> dict:
        s, reason, _ = self.killswitch.state()
        return {"system_state": s.value, "reason": reason or ""}

    def _data(self, run_id, st) -> dict:
        full = self.full_bundle()
        view = full.truncate(st["as_of"])
        st["view"] = view
        return {"sessions": len(view.panel.dates), "symbols": len(view.panel.symbols), "synthetic": int(view.is_synthetic)}

    def _validate(self, run_id, st) -> dict:
        if "view" not in st:
            self._data(run_id, st)
        v = DataValidator(self.cfg, self.db)
        rep = v.check_bundle(st["view"], expected_last_session=st["as_of"])
        v.record(rep, run_id=run_id)
        st["quarantine"] = quarantine_map(self.db, rep)
        if not rep.ok:
            self.killswitch.pause("critical data validation failure: " + "; ".join(c.name for c in rep.critical_failures),
                                  trigger="data_validation", details={"run_id": run_id})
        return {"ok": int(rep.ok), "critical": len(rep.critical_failures), "quarantined": len(rep.quarantined)}

    def _execution(self, run_id, st) -> dict:
        p = st["view"].panel
        d = st["as_of"]
        self.ledger.apply_corporate_actions(d, p)
        sync = self.exec.sync(d, p)
        mtm = self.ledger.mark_to_market(d, p)
        if sync.unknown:
            self.killswitch.pause("broker returned UNKNOWN order state; reconcile before trading",
                                  trigger="broker_unknown_state", details={"orders": sync.unknown[:20]})
        st["mtm"] = mtm
        return {"filled": len(sync.filled), "rejected": len(sync.rejected), "expired": len(sync.expired),
                "unknown": len(sync.unknown), "equity": float(mtm.get("equity", np.nan))}

    def _exits(self, run_id, st) -> dict:
        eng = build_exit_engine(self.cfg, market_symbol=st["view"].market_symbol)
        signals = eng.evaluate(self.ledger.open_trades(), st["as_of"], st["view"].panel)
        n = 0
        for sig in signals:
            r = self.exec.submit_exit(sig.trade_id, sig.reason.value, detail=sig.detail,
                                      session_date=str(st["as_of"].date()))
            n += 0 if r.get("refused") else 1
        return {"exit_signals": len(signals), "exit_orders": n}

    def _research(self, run_id, st) -> dict:
        view, d = st["view"], st["as_of"]
        if "quarantine" not in st:
            st["quarantine"] = quarantine_map(self.db)
        uni = UniverseEngine(self.cfg)
        u = uni.membership(view, exclude=st["quarantine"])
        uni.snapshot(view, d, db=self.db, run_id=run_id, exclude=st["quarantine"])
        fs = FeatureSet(view, universe=u)
        regime = RegimeEngine(self.cfg).persist(self.db, fs, d, run_id=run_id)
        strategies = build_strategies(self.cfg)
        register_strategies(self.db, strategies)
        cands = load_candidates(self.db, run_id)          # resumed run: reuse the recorded candidates
        if not cands:
            cands = StrategyArena(strategies).candidates(fs, u, d)
            persist_candidates(self.db, cands, run_id, view.is_synthetic)
        st.update(universe=u, fs=fs, regime=regime, strategies={s.strategy_id: s for s in strategies}, candidates=cands)
        return {"universe": int(u.loc[d].sum()), "candidates": len(cands), "regime": regime["label"]}

    def _strategy_status(self, sid: str, ver: str) -> StrategyStatus | None:
        row = self.db.fetchone("SELECT status FROM strategies WHERE strategy_id=? AND version=?", (sid, ver))
        return StrategyStatus(row["status"]) if row else None

    def _strategy_stage(self, sid: str, ver: str) -> StrategyStage | None:
        row = self.db.fetchone("SELECT stage FROM strategies WHERE strategy_id=? AND version=?", (sid, ver))
        return StrategyStage(row["stage"]) if row else None

    def _book_state(self, st) -> BookState:
        led = self.ledger.state()
        p, d = st["view"].panel, st["as_of"]
        smap = st.setdefault("sector_map", sector_map(st["view"], self.cfg))
        positions, open_risk = {}, 0.0
        mtm = st.get("mtm") or {}
        equity = float(mtm.get("equity") if mtm.get("equity") is not None else led.get("cash", 0.0))
        for t in self.ledger.open_trades():
            px = float(p.close.at[d, t.symbol]) if t.symbol in p.symbols and np.isfinite(p.close.at[d, t.symbol]) else float(t.entry_price or 0)
            positions[t.symbol] = {"qty": t.qty, "price": px, "sector": smap.get(t.symbol), "stop": t.stop_price}
            if t.stop_price and equity > 0:
                open_risk += max(px - t.stop_price, 0.0) * t.qty / equity
        return BookState(equity=equity, cash=float(led.get("cash", 0.0)), positions=positions, open_risk=open_risk)

    def _decide(self, run_id, st) -> dict:
        if "candidates" not in st:
            self._research(run_id, st)
        cands, fs, view, d = st["candidates"], st["fs"], st["view"], st["as_of"]
        stats_p = StrategyStatsProvider(self.db, self.cfg)
        ev_eng, nt_eng = EVEngine(self.cfg, self.costs), NoTradeEngine(self.cfg)
        final, risk_eng = FinalDecisionEngine(self.cfg), RiskEngine(self.cfg)
        system_state = self.killswitch.state()[0]
        ai_enabled = bool(self.cfg.get("ai.enabled", False))
        pre: list[dict] = []
        for c in cands:
            stats = stats_p.get(c.strategy_id, c.strategy_version, as_of=d)   # overall; by_regime inside
            ev = ev_eng.estimate(c, stats, c.risk.get("adv20"))
            ntc = NoTradeContext.build(c, panel=view.panel, features=c.features, as_of=d.date(), stats=stats,
                                       regime=st["regime"]["label"], same_day_candidates=cands,
                                       quarantined=c.symbol in st["quarantine"] and
                                       (st["quarantine"][c.symbol] is None or st["quarantine"][c.symbol] <= d))
            nt = nt_eng.evaluate(c, ntc)
            ai = AIReview(candidate_id=c.candidate_id, enabled=ai_enabled, decision=AIDecision.UNKNOWN)
            pre.append({"c": c, "stats": stats, "ev": ev, "nt": nt, "ai": ai})
        # portfolio sizing only for candidates that survived hard no-trade rules and belong to ACTIVE strategies
        eligible = [x["c"] for x in pre if not any(r.blocking for r in x["nt"])
                    and self._strategy_status(x["c"].strategy_id, x["c"].strategy_version) is StrategyStatus.ACTIVE]
        intents, rejections = PortfolioConstructor(self.cfg).build(self._book_state(st), eligible, view.panel, d,
                                                                   st.setdefault("sector_map", sector_map(view, self.cfg)))
        by_cand = {i.candidate_id: i for i in intents}
        rej_by = {r.candidate_id: r for r in rejections}
        mtm = st.get("mtm") or {}
        # orders decided for THIS session (not wall-clock today: replays run many sessions in one day)
        daily_orders = self.db.fetchone("SELECT COUNT(*) AS n FROM order_intents WHERE book=? AND session_date=?",
                                        (self.book, str(d.date())))["n"]
        shadow = ShadowBook(self.db)
        broker_ok = self.broker.is_available()     # once per session, not once per candidate
        decisions, counts = [], {"TRADE": 0, "NO_TRADE": 0, "WATCH": 0, "UNKNOWN": 0}
        for x in pre:
            c = x["c"]
            rctx = DecisionContext(candidate=c, book=Book(self.book), no_trade_checks=x["nt"], ev=x["ev"],
                                   sizing=by_cand.get(c.candidate_id), portfolio_rejection=rej_by.get(c.candidate_id),
                                   strategy_status=self._strategy_status(c.strategy_id, c.strategy_version),
                                   strategy_stage=self._strategy_stage(c.strategy_id, c.strategy_version),
                                   system_state=system_state, broker_available=broker_ok,
                                   daily_order_count=int(daily_orders), book_drawdown=mtm.get("drawdown"), as_of=d.date())
            risk = risk_eng.run(rctx)
            outcome = final.decide(c, x["ai"], x["nt"], x["ev"], risk)
            dec_id = persist_decision(self.db, c.candidate_id, outcome, run_id=run_id, ev=x["ev"], no_trade=x["nt"],
                                      sizing=by_cand.get(c.candidate_id))
            record_checks(self.db, c.candidate_id, dec_id, list(x["nt"]) + list(risk.checks))
            shadow.record(c, outcome.decision, outcome.reject_stage, "; ".join(outcome.reasons)[:1000],
                          ai_decision=outcome.ai_decision, objections=x["ai"].objections, is_synthetic=view.is_synthetic)
            counts[outcome.decision.value] = counts.get(outcome.decision.value, 0) + 1
            decisions.append({"candidate": c, "decision_id": dec_id, "outcome": outcome, "intent": by_cand.get(c.candidate_id)})
        st["decisions"] = decisions
        stages = pd.Series([x["outcome"].reject_stage.value for x in decisions]).value_counts().to_dict() if decisions else {}
        return {**{k.lower(): v for k, v in counts.items()}, "reject_stages": to_json(stages)}

    def _load_decisions(self, run_id) -> list[dict]:
        """Resume support: rebuild TRADE decisions (with their sizing) recorded by this run."""
        from quantlab.db.database import from_json
        from quantlab.decision.final import DecisionOutcome
        cands = {c.candidate_id: c for c in load_candidates(self.db, run_id)}
        out = []
        for r in self.db.fetchall("SELECT * FROM decisions WHERE run_id=? AND decision=?", (run_id, FinalDecision.TRADE.value)):
            sizing = from_json(r["sizing_json"], None)
            c = cands.get(r["candidate_id"])
            if c is None or not sizing:
                continue
            intent = type("Intent", (), {"qty": sizing.get("qty"), "sizing": sizing.get("sizing", {})})()
            out.append({"candidate": c, "decision_id": r["decision_id"], "intent": intent,
                        "outcome": DecisionOutcome(FinalDecision.TRADE, RejectStage.NONE)})
        return out

    def _orders(self, run_id, st) -> dict:
        placed = refused = 0
        for x in st.get("decisions", []):
            if x["outcome"].decision is not FinalDecision.TRADE or x["intent"] is None:
                continue
            c, it = x["candidate"], x["intent"]
            r = self.exec.submit_entry(c.symbol, it.qty, candidate_id=c.candidate_id, decision_id=x["decision_id"],
                                       plan=c.plan, strategy_id=c.strategy_id, strategy_version=c.strategy_version,
                                       journal={"sizing": it.sizing, "reasons": c.reasons, "features": c.features},
                                       session_date=str(st["as_of"].date()))
            refused += int(bool(r.get("refused")))
            placed += int(not r.get("refused"))
        return {"orders_placed": placed, "orders_refused": refused}

    def _outcomes(self, run_id, st) -> dict:
        n = ShadowOutcomeTracker(self.db, self.cfg).update(st["view"].panel, as_of=st["as_of"],
                                                            synthetic=st["view"].is_synthetic)
        from quantlab.exploration import ExplorationOutcomeTracker
        m = ExplorationOutcomeTracker(self.db, self.cfg).update(st["view"].panel, st["as_of"])
        return {"outcomes_recorded": int(n), "exploration_outcomes": int(m)}

    def _risk(self, run_id, st) -> dict:
        mtm = st.get("mtm") or {}
        dd = mtm.get("drawdown")
        limit = float(self.cfg.get("risk.max_drawdown_pause", 0.2))
        if dd is not None and np.isfinite(dd) and -abs(dd) <= -limit:
            self.killswitch.pause(f"{self.book} book drawdown {dd:.1%} breached {limit:.0%}", trigger="drawdown",
                                  details={"drawdown": dd})
        return {"drawdown": float(dd) if dd is not None and np.isfinite(dd) else 0.0}

    def _discover(self, run_id, st) -> dict:
        """Research-only market discovery for this session (quantlab/discovery). It reads the
        decisions and orders this run already made; it cannot create or change either."""
        if not bool(self.cfg.get("discovery.enabled", True)):
            return {"skipped": "discovery.enabled is false"}
        if "view" not in st:
            self._data(run_id, st)
        from quantlab.discovery import run_discovery, strategy_links
        try:
            dr = run_discovery(self.ctx, st["view"], st["as_of"], run_id=run_id, quarantine=st.get("quarantine"),
                               links=strategy_links(self.db, st["as_of"], run_id))
        except Exception as exc:          # research layer: record loudly, never block risk/report
            log_event(log, "discovery failed", level=40, error=repr(exc))
            st["discovery_error"] = repr(exc)
            return {"error": repr(exc)[:500]}
        st["discovery"] = dr
        f = dr.assessment.funnel
        return {"discovery_run_id": dr.discovery_run_id, "discovered": f["discovered"], "high_ranked": f["high_ranked"],
                "watchlist": f["watchlist"], "paper_eligible": f["paper_eligible"],
                "diagnostics": len(dr.assessment.diagnostics), "outcomes_written": dr.outcomes_written}

    def _explore(self, run_id, st) -> dict:
        """PAPER_EXPLORATION planning (and, in EXPLORATION mode with ``exploration_submit="now"``,
        submission through the same execution service). STRICT mode: selections are tracked as SHADOW
        and never traded. It never touches the strict decisions or their orders."""
        from quantlab.exploration import paper_mode, plan_exploration, preopen_submit
        dr = st.get("discovery")
        if dr is None:
            return {"skipped": "no discovery run this session"}
        mode = paper_mode(self.cfg)
        sim_now = self.cfg_cutoff(st["as_of"]) + pd.Timedelta(minutes=5)
        mtm = st.get("mtm") or {}
        plan = plan_exploration(self.ctx, book=self.book, run_id=dr.discovery_run_id, equity=mtm.get("equity"),
                                mode=mode, now=sim_now)
        out = {"mode": mode, **{f"planned_{k.lower()}": v for k, v in (plan.get("counts") or {}).items()}}
        if mode == "EXPLORATION" and self.exploration_submit == "now":
            sub_ = preopen_submit(self.ctx, self.exec, now=sim_now, session=plan.get("session"))
            out.update({"submitted": sub_.get("submitted", 0), "cancelled": sub_.get("cancelled", 0),
                        "refused": sub_.get("refused", 0)})
        return out

    def cfg_cutoff(self, d) -> pd.Timestamp:
        return self.full_bundle().calendar.cutoff(d)

    def _report(self, run_id, st) -> dict:
        from quantlab.pipeline.report import write_daily_report
        path = write_daily_report(self.ctx, run_id, st)
        st["report_path"] = str(path)
        return {"report": str(path)}


def replay(ctx: AppContext, start, end, synthetic: bool | None = None, broker=None) -> list[PipelineResult]:
    """Run the daily pipeline for every stored session in [start, end], in order (a forward
    paper-trading simulation; each day only sees data up to its own cutoff)."""
    pipe = DailyPipeline(ctx, synthetic=synthetic, broker=broker)
    dates = pipe.full_bundle().panel.dates
    out = []
    for d in dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]:
        out.append(pipe.run(d))
        if out[-1].errors:
            break
    return out
