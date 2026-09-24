"""Trade journal: persists every paper trade with a full audit snapshot and explains it end to end.

Why a snapshot AND joins: ``journal_json`` freezes what was known when the trade was opened
(candidate evidence, strategy/model/AI versions, objections, risk checks, EV, sizing rationale,
regime, sector). Later edits elsewhere cannot rewrite history. ``explain()`` also re-joins the
append-only source tables (candidates, decisions, risk_checks, ai_assessments, ai_objections,
ml_predictions, orders, order_events, fills, trades) so every trade can be traced
DATA -> FEATURES -> STRATEGY -> MODEL -> AI -> OBJECTIONS -> EV -> RISK -> DECISION -> ORDER ->
FILL -> EXIT -> RESULT. Each stage carries a transparency label (``core.types.InfoKind``). A stage
with no data is labeled UNCERTAINTY. It is never filled in with made-up content.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from quantlab.core.types import InfoKind, TradePlan, new_id
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

EXPLAIN_STAGES = ("DATA", "FEATURES", "STRATEGY", "MODEL", "AI", "OBJECTIONS", "EV", "RISK", "DECISION",
                  "ORDER", "FILL", "EXIT", "RESULT")


class JournalError(RuntimeError):
    pass


def _plan_dict(plan: TradePlan | dict | None) -> dict[str, Any]:
    if plan is None:
        return {}
    if is_dataclass(plan):
        return asdict(plan)
    return dict(plan)


def _num(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


class TradeJournal:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------------------------------
    # snapshot of everything known about the decision behind a trade
    # ------------------------------------------------------------------------------------------
    def build_snapshot(self, candidate_id: str | None = None, decision_id: str | None = None,
                       human_decision_id: str | None = None, symbol: str | None = None) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        cand = self.db.fetchone("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)) if candidate_id else None
        if cand:
            snap["candidate"] = {
                "candidate_id": cand["candidate_id"], "as_of_date": cand["as_of_date"], "symbol": cand["symbol"],
                "score": cand["score"], "rank": cand["rank"], "opportunity_score": cand["opportunity_score"],
                "features": from_json(cand["features_json"], {}), "reasons": from_json(cand["reasons_json"], []),
                "risk": from_json(cand["risk_json"], {}), "pit_status": cand["pit_status"],
                "is_synthetic": bool(cand["is_synthetic"]),
            }
            snap["strategy"] = {"id": cand["strategy_id"], "version": cand["strategy_version"]}
            symbol = symbol or cand["symbol"]
        if candidate_id:
            preds = self.db.fetchall("SELECT model_id, model_version, target, probability, prediction FROM ml_predictions "
                                     "WHERE candidate_id=? ORDER BY id", (candidate_id,))
            if preds:
                snap["model_predictions"] = preds
                snap["model_versions"] = sorted({f"{p['model_id']}@{p['model_version']}" for p in preds})
            ais = self.db.fetchall("SELECT role, provider, model, decision, summary FROM ai_assessments "
                                   "WHERE candidate_id=? ORDER BY created_at, assessment_id", (candidate_id,))
            if ais:
                snap["ai_assessments"] = ais
                snap["ai_models"] = sorted({f"{a['provider']}:{a['model']}" for a in ais})
            objs = self.db.fetchall("SELECT category, severity, text, source FROM ai_objections WHERE candidate_id=? "
                                    "ORDER BY id", (candidate_id,))
            if objs:
                snap["objections"] = objs
        dec = None
        if decision_id:
            dec = self.db.fetchone("SELECT * FROM decisions WHERE decision_id=?", (decision_id,))
        if dec:
            snap["decision"] = {"decision_id": dec["decision_id"], "decision": dec["decision"],
                                "reject_stage": dec["reject_stage"], "ai_decision": dec["ai_decision"],
                                "reasons": from_json(dec["reasons_json"], [])}
            if dec["ev_json"]:
                snap["ev"] = from_json(dec["ev_json"])
            if dec["sizing_json"]:
                snap["sizing"] = from_json(dec["sizing_json"])
            if dec["no_trade_json"]:
                snap["no_trade"] = from_json(dec["no_trade_json"])
            checks = self.db.fetchall("SELECT check_name, passed, severity, reason FROM risk_checks "
                                      "WHERE decision_id=? ORDER BY id", (decision_id,))
            if not checks and candidate_id:
                checks = self.db.fetchall("SELECT check_name, passed, severity, reason FROM risk_checks "
                                          "WHERE candidate_id=? ORDER BY id", (candidate_id,))
            if checks:
                snap["risk_checks"] = checks
        if human_decision_id:
            hd = self.db.fetchone("SELECT * FROM human_decisions WHERE decision_id=?", (human_decision_id,))
            if hd:
                snap["human_decision"] = {k: hd[k] for k in ("decision_id", "action", "conviction", "notes",
                                                            "decided_at", "ref_session_date", "ref_price",
                                                            "supersedes_id")}
                symbol = symbol or hd["symbol"]
        as_of = (cand or {}).get("as_of_date") or (snap.get("human_decision") or {}).get("ref_session_date")
        if as_of:
            reg = self.db.fetchone("SELECT label FROM regime_snapshots WHERE as_of_date<=? ORDER BY as_of_date DESC "
                                   "LIMIT 1", (as_of,))
            if reg and reg["label"]:
                snap["regime"] = reg["label"]
        if symbol:
            sec = self.db.fetchone("SELECT sector FROM security_master WHERE symbol=? AND sector IS NOT NULL "
                                   "ORDER BY retrieved_at DESC LIMIT 1", (symbol,))
            if sec:
                snap["sector"] = sec["sector"]
                snap["sector_pit_status"] = "ASSUMED_STATIC"
        return snap

    # ------------------------------------------------------------------------------------------
    # trade lifecycle
    # ------------------------------------------------------------------------------------------
    def open_trade(
        self,
        *,
        book: str,
        symbol: str,
        qty: float,
        entry_date: str,
        entry_price: float,
        direction: str = "LONG",
        candidate_id: str | None = None,
        decision_id: str | None = None,
        human_decision_id: str | None = None,
        strategy_id: str | None = None,
        strategy_version: str | None = None,
        model_version: str | None = None,
        ai_model: str | None = None,
        plan: TradePlan | dict | None = None,
        signal_date: str | None = None,
        regime: str | None = None,
        sector: str | None = None,
        journal: dict[str, Any] | None = None,
        trade_id: str | None = None,
    ) -> str:
        """Persist a new OPEN trade (+ its immutable plan) and return its id."""
        trade_id = trade_id or new_id("trade")
        snap = self.build_snapshot(candidate_id, decision_id, human_decision_id, symbol)
        snap.update(journal or {})
        pd_ = _plan_dict(plan)
        cand = self.db.fetchone("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)) if candidate_id else None
        if cand:
            for key, col in (("entry_ref_price", "entry_ref_price"), ("stop_price", "stop_price"),
                             ("target_price", "target_price"), ("holding_sessions", "holding_sessions"),
                             ("invalidation", "invalidation")):
                if pd_.get(key) is None and cand[col] is not None:
                    pd_[key] = cand[col]
        signal_date = (signal_date or pd_.get("signal_date") or (cand["as_of_date"] if cand else None)
                       or (snap.get("human_decision") or {}).get("ref_session_date"))
        strategy = snap.get("strategy") or {}
        strategy_id = strategy_id or strategy.get("id")
        strategy_version = strategy_version or strategy.get("version")
        model_version = model_version or (",".join(snap["model_versions"]) if snap.get("model_versions") else None)
        ai_model = ai_model or (",".join(snap["ai_models"]) if snap.get("ai_models") else None)
        regime = regime or snap.get("regime")
        sector = sector or snap.get("sector")
        snap["plan"] = {**pd_, "signal_date": signal_date}
        now = utcnow_iso()
        with self.db.transaction():
            self.db.insert("trades", {
                "trade_id": trade_id, "book": book, "candidate_id": candidate_id, "decision_id": decision_id,
                "human_decision_id": human_decision_id, "symbol": symbol, "direction": direction,
                "strategy_id": strategy_id, "strategy_version": strategy_version, "model_version": model_version,
                "ai_model": ai_model, "status": "OPEN", "qty": float(qty), "entry_date": entry_date,
                "entry_price": float(entry_price), "stop_price": _num(pd_.get("stop_price")),
                "target_price": _num(pd_.get("target_price")), "regime": regime, "sector": sector,
                "journal_json": to_json(snap), "created_at": now, "updated_at": now,
            })
            self.db.insert("trade_plans", {
                "trade_id": trade_id, "book": book, "signal_date": signal_date,
                "entry_convention": pd_.get("entry") or "next_open", "entry_ref_price": _num(pd_.get("entry_ref_price")),
                "stop_price": _num(pd_.get("stop_price")), "target_price": _num(pd_.get("target_price")),
                "holding_sessions": int(pd_["holding_sessions"]) if pd_.get("holding_sessions") is not None else None,
                "direction": direction, "invalidation": pd_.get("invalidation"), "created_at": now,
            })
            self.record_event(trade_id, "opened", {"qty": qty, "entry_price": entry_price, "entry_date": entry_date})
        log_event(log, "trade opened", book=book, trade_id=trade_id, symbol=symbol, qty=qty, entry_price=entry_price)
        return trade_id

    def add_entry_fill(self, trade_id: str, qty: float, entry_price: float) -> None:
        """A further (partial) entry fill: update size and average entry price."""
        self.db.execute("UPDATE trades SET qty=?, entry_price=?, updated_at=? WHERE trade_id=?",
                        (float(qty), float(entry_price), utcnow_iso(), trade_id))
        self.record_event(trade_id, "entry_fill", {"qty": qty, "entry_price": entry_price})

    def close_trade(self, trade_id: str, *, exit_date: str, exit_price: float, exit_reason: str,
                    gross_pnl: float, costs: float, net_pnl: float, ret: float | None,
                    holding_sessions: int | None = None, mae: float | None = None, mfe: float | None = None,
                    details: dict[str, Any] | None = None) -> None:
        row = self.get_trade(trade_id)
        if row is None:
            raise JournalError(f"unknown trade {trade_id}")
        if row["status"] != "OPEN":
            raise JournalError(f"trade {trade_id} is already {row['status']}")
        self.db.execute(
            "UPDATE trades SET status='CLOSED', exit_date=?, exit_price=?, exit_reason=?, gross_pnl=?, costs=?, "
            "net_pnl=?, ret=?, holding_sessions=?, mae=?, mfe=?, updated_at=? WHERE trade_id=?",
            (exit_date, exit_price, exit_reason, gross_pnl, costs, net_pnl, ret, holding_sessions, mae, mfe,
             utcnow_iso(), trade_id))
        self.record_event(trade_id, "closed", {"exit_date": exit_date, "exit_price": exit_price,
                                                "exit_reason": exit_reason, "net_pnl": net_pnl, "ret": ret,
                                                **(details or {})})
        log_event(log, "trade closed", trade_id=trade_id, exit_reason=exit_reason, net_pnl=net_pnl)

    def record_event(self, trade_id: str, event: str, details: dict[str, Any] | None = None) -> None:
        if self.db.fetchone("SELECT 1 FROM trades WHERE trade_id=?", (trade_id,)) is None:
            raise JournalError(f"unknown trade {trade_id}")
        self.db.insert("trade_events", {"trade_id": trade_id, "event": event, "at": utcnow_iso(),
                                        "details_json": to_json(details or {})})

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        return self.db.fetchone("SELECT * FROM trades WHERE trade_id=?", (trade_id,))

    def list_trades(self, book: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM trades WHERE 1=1", []
        if book:
            sql += " AND book=?"
            params.append(book)
        if status:
            sql += " AND status=?"
            params.append(status)
        return self.db.fetchall(sql + " ORDER BY created_at, trade_id", params)

    # ------------------------------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------------------------------
    def explain(self, trade_id: str) -> dict[str, Any]:
        """Trace one trade through every stage of the decision chain (see module docstring)."""
        t = self.get_trade(trade_id)
        if t is None:
            raise JournalError(f"unknown trade {trade_id}")
        cid, did, hid = t["candidate_id"], t["decision_id"], t["human_decision_id"]
        cand = self.db.fetchone("SELECT * FROM candidates WHERE candidate_id=?", (cid,)) if cid else None
        dec = self.db.fetchone("SELECT * FROM decisions WHERE decision_id=?", (did,)) if did else None
        hd = self.db.fetchone("SELECT * FROM human_decisions WHERE decision_id=?", (hid,)) if hid else None
        plan = self.db.fetchone("SELECT * FROM trade_plans WHERE trade_id=?", (trade_id,))
        orders = self.db.fetchall("SELECT * FROM orders WHERE trade_id=? AND book=? ORDER BY created_at, order_id",
                                  (trade_id, t["book"]))
        stages: dict[str, dict[str, Any]] = {}

        def put(stage: str, label: InfoKind, content: Any) -> None:
            empty = content in (None, [], {})
            stages[stage] = {"stage": stage, "available": not empty,
                             "label": (InfoKind.UNCERTAINTY if empty else label).value,
                             "content": content if not empty else "no record"}

        run = self.db.fetchone("SELECT run_id, kind, git_commit, git_dirty, config_hash FROM runs WHERE run_id=?",
                               (cand["run_id"],)) if cand and cand["run_id"] else None
        data: dict[str, Any] = {}
        if cand:
            data = {"as_of_date": cand["as_of_date"], "pit_status": cand["pit_status"],
                    "is_synthetic": bool(cand["is_synthetic"]), "run": run}
        elif hd:
            data = {"ref_session_date": hd["ref_session_date"], "ref_price": hd["ref_price"],
                    "fill_convention": hd["fill_convention"]}
        put("DATA", InfoKind.FACT, data)
        put("FEATURES", InfoKind.FACT, from_json(cand["features_json"], {}) if cand else {})
        strategy: dict[str, Any] = {}
        if cand or t["strategy_id"]:
            strategy = {"strategy_id": t["strategy_id"], "strategy_version": t["strategy_version"]}
            if cand:
                strategy.update(score=cand["score"], rank=cand["rank"], opportunity_score=cand["opportunity_score"],
                                reasons=from_json(cand["reasons_json"], []))
            if plan:
                strategy["plan"] = {k: plan[k] for k in ("signal_date", "entry_convention", "entry_ref_price",
                                                         "stop_price", "target_price", "holding_sessions",
                                                         "invalidation")}
        put("STRATEGY", InfoKind.MODEL_OUTPUT, strategy)
        put("MODEL", InfoKind.MODEL_OUTPUT, self.db.fetchall(
            "SELECT model_id, model_version, target, probability, prediction, as_of_date FROM ml_predictions "
            "WHERE candidate_id=? ORDER BY id", (cid,)) if cid else [])
        put("AI", InfoKind.AI_OPINION, self.db.fetchall(
            "SELECT role, provider, model, decision, summary, created_at FROM ai_assessments WHERE candidate_id=? "
            "ORDER BY created_at, assessment_id", (cid,)) if cid else [])
        objections = self.db.fetchall("SELECT category, severity, text, source FROM ai_objections WHERE candidate_id=? "
                                      "ORDER BY id", (cid,)) if cid else []
        put("OBJECTIONS", InfoKind.AI_OPINION, objections)
        put("EV", InfoKind.MODEL_OUTPUT, from_json(dec["ev_json"], {}) if dec and dec["ev_json"] else {})
        checks = []
        if did:
            checks = self.db.fetchall("SELECT check_name, passed, severity, reason FROM risk_checks WHERE decision_id=? "
                                      "ORDER BY id", (did,))
        if not checks and cid:
            checks = self.db.fetchall("SELECT check_name, passed, severity, reason FROM risk_checks WHERE candidate_id=? "
                                      "ORDER BY id", (cid,))
        put("RISK", InfoKind.FACT, checks)
        decision: dict[str, Any] = {}
        if dec:
            decision["bot"] = {"decision_id": dec["decision_id"], "decision": dec["decision"],
                               "reject_stage": dec["reject_stage"], "ai_decision": dec["ai_decision"],
                               "reasons": from_json(dec["reasons_json"], []),
                               "sizing": from_json(dec["sizing_json"], None)}
        if hd:
            decision["human"] = {k: hd[k] for k in ("decision_id", "action", "conviction", "notes", "decided_at",
                                                    "ref_session_date", "ref_price", "supersedes_id")}
        put("DECISION", InfoKind.FACT, decision)

        entry_orders = [o for o in orders if o["purpose"] == "entry"]
        exit_orders = [o for o in orders if o["purpose"] == "exit"]

        def order_view(o: dict[str, Any]) -> dict[str, Any]:
            intent = self.db.fetchone("SELECT session_date, intent_json FROM order_intents WHERE order_id=?",
                                      (o["order_id"],))
            events = self.db.fetchall("SELECT event, status, at FROM order_events WHERE order_id=? ORDER BY id",
                                      (o["order_id"],))
            return {k: o[k] for k in ("order_id", "client_order_id", "broker", "purpose", "side", "qty", "order_type",
                                      "time_in_force", "status", "filled_qty", "filled_avg_price", "created_at")} | {
                "decision_session": intent["session_date"] if intent else None,
                "intent": from_json(intent["intent_json"], {}) if intent else {},
                "events": events}

        put("ORDER", InfoKind.FACT, [order_view(o) for o in entry_orders])
        order_ids = [o["order_id"] for o in orders]
        fills = []
        if order_ids:
            q = ",".join("?" for _ in order_ids)
            fills = self.db.fetchall(f"SELECT fill_id, order_id, side, qty, price, commission, modeled_cost, filled_at, "
                                     f"session_date, source FROM fills WHERE order_id IN ({q}) AND book=? "
                                     f"ORDER BY filled_at, fill_id", [*order_ids, t["book"]])
        put("FILL", InfoKind.FACT, fills)
        exit_events = self.db.fetchall("SELECT event, at, details_json FROM trade_events WHERE trade_id=? "
                                       "AND event IN ('exit_signal', 'exit_submitted', 'closed') ORDER BY id", (trade_id,))
        exit_info: dict[str, Any] = {}
        if exit_orders or exit_events:
            exit_info = {"orders": [order_view(o) for o in exit_orders],
                         "events": [{"event": e["event"], "at": e["at"], "details": from_json(e["details_json"], {})}
                                    for e in exit_events],
                         "exit_reason": t["exit_reason"]}
        put("EXIT", InfoKind.FACT, exit_info)
        if t["status"] == "CLOSED":
            put("RESULT", InfoKind.FACT, {k: t[k] for k in ("entry_date", "entry_price", "exit_date", "exit_price",
                                                            "exit_reason", "qty", "gross_pnl", "costs", "net_pnl",
                                                            "ret", "holding_sessions", "mae", "mfe")})
        else:
            stages["RESULT"] = {"stage": "RESULT", "available": False, "label": InfoKind.UNCERTAINTY.value,
                                "content": "trade still open: result unknown"}
        chain = [stages[s] for s in EXPLAIN_STAGES]
        return {
            "trade_id": trade_id, "book": t["book"], "symbol": t["symbol"], "status": t["status"],
            "chain": chain, "stages": stages, "missing": [c["stage"] for c in chain if not c["available"]],
            "journal_snapshot": from_json(t["journal_json"], {}),
            "is_synthetic": bool(cand["is_synthetic"]) if cand else None,
        }
