"""OVERNIGHT / NEXT-SESSION discovery mode.

When the market is closed QuantLab asks: "what setups are developing that could matter for the
NEXT trading session?". The pipeline, in time order:

  session closed (D 16:00 ET, the information cutoff for price/volume)
  -> end-of-day scan          discovery on data through D (``run_discovery``; one run per session)
  -> post-close information   items with available_at in (D 16:00, D 20:00] ET
  -> overnight refresh        items in (D 20:00, D+1 04:00] ET, then pre-market (04:00, 09:30]:
                              catalysts are attached with their timestamps; a discovered setup with a
                              fresh catalyst can be PROMOTED to the next-session watchlist. That never
                              makes it paper eligible.
  -> pre-open validation      before D+1 09:30 ET: re-check data freshness, corporate actions, data
                              quality, kill switch, strategy status and the recorded decision. The
                              actual next-session open is NEVER an input (the recheck refuses to run
                              once it has occurred).
  -> next-session decision    the unchanged decision chain (pipeline + runner); orders only for
                              TRADE decisions inside the execution window.

Every item keeps its own timestamp and phase: REGULAR_SESSION, POST_CLOSE, OVERNIGHT, PRE_MARKET,
NEXT_SESSION. Information that arrived after a cutoff can never change an earlier record.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.core.calendar import to_session
from quantlab.db.database import from_json, to_json, utcnow_iso

ET = ZoneInfo("America/New_York")
PHASES = ("REGULAR_SESSION", "POST_CLOSE", "OVERNIGHT", "PRE_MARKET", "NEXT_SESSION")
CLOSE, POST_CLOSE_END, PREMARKET_START, OPEN = time(16, 0), time(20, 0), time(4, 0), time(9, 30)


def _et(d, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(pd.Timestamp(d).date(), t), tz=ET)


def _utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def sessions_between(start, end) -> list[pd.Timestamp]:
    from quantlab.data.audit import expected_sessions
    return list(expected_sessions(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))


def market_state(now: datetime | None = None) -> dict[str, Any]:
    """OPEN/CLOSED, the last completed session and the next session, from the rule-based NYSE
    calendar (holidays and special closures; early closes are NOT modelled: treated as 16:00)."""
    now = _utc(now or datetime.now(timezone.utc))
    et = now.tz_convert(ET)
    try:
        days = sessions_between(et - pd.Timedelta(days=14), et + pd.Timedelta(days=14))
    except Exception as exc:       # pragma: no cover - calendar failure must be visible
        return {"status": "UNKNOWN", "error": f"calendar unavailable: {exc}", "calendar_source": "MISSING"}
    today = pd.Timestamp(et.date())
    is_session = today in days
    is_open = is_session and _et(today, OPEN) <= et < _et(today, CLOSE)
    completed = [d for d in days if _et(d, CLOSE) <= et]
    upcoming = [d for d in days if _et(d, OPEN) > et]
    return {"status": "OPEN" if is_open else "CLOSED", "now": now.isoformat(),
            "current_session": str(today.date()) if is_open else None,
            "last_completed_session": str(completed[-1].date()) if completed else None,
            "next_session": str(upcoming[0].date()) if upcoming else None,
            "next_open_at": _et(upcoming[0], OPEN).tz_convert("UTC").isoformat() if upcoming else None,
            "calendar_source": "NYSE rules (data/audit.py; early closes not modelled)"}


def info_phase(ts, d, next_session) -> str:
    """Which phase an item with timestamp ``ts`` belongs to, relative to decision session ``d``."""
    t = _utc(ts)
    if t <= _et(d, CLOSE):
        return "REGULAR_SESSION"
    if t <= _et(d, POST_CLOSE_END):
        return "POST_CLOSE"
    if next_session is None:
        return "OVERNIGHT"
    if t <= _et(next_session, PREMARKET_START):
        return "OVERNIGHT"
    if t < _et(next_session, OPEN):
        return "PRE_MARKET"
    return "NEXT_SESSION"


def latest_run(db, run_id: str | None = None) -> dict[str, Any] | None:
    if run_id:
        return db.fetchone("SELECT * FROM discovery_runs WHERE discovery_run_id=?", (run_id,))
    return db.fetchone("SELECT * FROM discovery_runs WHERE is_synthetic IN (0,1) ORDER BY as_of_date DESC, "
                       "created_at DESC LIMIT 1")


def _window_check(run: dict[str, Any], now: pd.Timestamp) -> str | None:
    ns = run.get("next_session")
    if not ns:
        return "no next session recorded for this run (exchange calendar missing)"
    if now >= _et(ns, OPEN):
        return (f"the next session ({ns}) has already opened; overnight/pre-open work for decision session "
                f"{run['as_of_date']} is closed (the actual open is never an input)")
    if now <= _utc(run["info_cutoff_at"]):
        return "the decision session has not closed yet"
    return None


def _items(ctx, symbols: set[str], t0: pd.Timestamp, t1: pd.Timestamp, synthetic: bool) -> list[dict[str, Any]]:
    """News and earnings events for ``symbols`` that became available in (t0, t1]. Items without an
    availability timestamp are never used. Real and synthetic data are never mixed."""
    out: list[dict[str, Any]] = []
    syn = bool(synthetic)
    for kind, src in (("news", "news"), ("earnings_event", "events")):
        ids = ctx.store.dataset_ids(src, synthetic=syn)
        if not ids:
            continue
        df = ctx.store.load(src, ids)
        if df.empty or "available_at" not in df.columns:
            continue
        ts = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
        m = df["symbol"].isin(symbols) & (ts > t0) & (ts <= t1)
        for r, t in zip(df[m].itertuples(), ts[m]):
            out.append({"kind": kind, "symbol": r.symbol, "available_at": t.isoformat(),
                        "source_id": str(getattr(r, "news_id", None) or getattr(r, "source_id", "")),
                        "detail": {"headline": getattr(r, "headline", None), "event_type": getattr(r, "event_type", None),
                                   "pit_status": getattr(r, "pit_status", None)}})
    return out


def overnight_refresh(ctx, now=None, run_id: str | None = None) -> dict[str, Any]:
    """Attach post-close / overnight / pre-market catalysts (available_at <= now) to the latest
    next-session run. A discovered setup with a fresh catalyst is promoted to the watchlist; this
    never makes it paper eligible."""
    db = ctx.db
    now = _utc(now or datetime.now(timezone.utc))
    run = latest_run(db, run_id)
    if run is None:
        return {"ok": False, "reason": "no discovery run to refresh"}
    why = _window_check(run, now)
    if why:
        return {"ok": False, "reason": why, "discovery_run_id": run["discovery_run_id"]}
    cands = {r["symbol"]: r for r in db.fetchall(
        "SELECT discovery_id, symbol, status, on_watchlist, high_quality FROM discovery_candidates WHERE discovery_run_id=?",
        (run["discovery_run_id"],))}
    seen = {(r["symbol"], r["source_id"]) for r in db.fetchall(
        "SELECT symbol, source_id FROM overnight_updates WHERE discovery_run_id=?", (run["discovery_run_id"],))}
    t0 = _utc(run["info_cutoff_at"])
    items = _items(ctx, set(cands), t0, now, bool(run["is_synthetic"]))
    rows = []
    for it in items:
        if (it["symbol"], it["source_id"]) in seen:
            continue
        c = cands[it["symbol"]]
        effect = "CATALYST_ADDED"
        if c["status"] in ("DISCOVERED",) and not c["on_watchlist"]:
            effect = "PROMOTED_TO_WATCH"
        rows.append({"discovery_run_id": run["discovery_run_id"], "discovery_id": c["discovery_id"],
                     "symbol": it["symbol"], "refreshed_at": utcnow_iso(), "info_cutoff_at": now.isoformat(),
                     "phase": info_phase(it["available_at"], run["as_of_date"], run["next_session"]),
                     "kind": it["kind"], "available_at": it["available_at"], "source_id": it["source_id"],
                     "detail_json": to_json(it["detail"]), "effect": effect, "created_at": utcnow_iso()})
    if rows:
        db.insert_many("overnight_updates", rows)
    return {"ok": True, "discovery_run_id": run["discovery_run_id"], "info_cutoff_at": now.isoformat(),
            "items_found": len(items), "recorded": len(rows),
            "promoted": sum(1 for r in rows if r["effect"] == "PROMOTED_TO_WATCH")}


def preopen_recheck(ctx, now=None, run_id: str | None = None) -> dict[str, Any]:
    """Fresh validation pass BEFORE the next session opens. Uses only information available at
    ``now`` (< next open) and never the actual open. Transitions: -> PAPER_ELIGIBLE (only when the
    recorded decision chain already said TRADE and nothing below blocks it), REJECTED, UNKNOWN,
    INVALIDATED, or unchanged."""
    from quantlab.data.validation import quarantine_map
    from quantlab.discovery.engine import strategy_links
    from quantlab.discovery.status import strategy_research_status
    from quantlab.monitoring.killswitch import KillSwitch

    db = ctx.db
    now = _utc(now or datetime.now(timezone.utc))
    run = latest_run(db, run_id)
    if run is None:
        return {"ok": False, "reason": "no discovery run to recheck"}
    why = _window_check(run, now)
    if why:
        return {"ok": False, "reason": why, "discovery_run_id": run["discovery_run_id"]}
    d, ns = run["as_of_date"], run["next_session"]
    ms = market_state(now)
    stale = ms.get("last_completed_session") and ms["last_completed_session"] != d
    last_bar = db.fetchone("SELECT MAX(end_date) AS e FROM datasets WHERE kind='bars' AND is_synthetic=?",
                           (run["is_synthetic"],))["e"]
    bar_missing = not last_bar or str(last_bar)[:10] < d
    ca = db.fetchone("SELECT MAX(end_date) AS e, COUNT(*) AS n FROM datasets WHERE kind='corporate_actions' "
                     "AND is_synthetic=?", (run["is_synthetic"],))
    ca_missing = not ca or not ca["n"]
    actions_next = set()
    if not ca_missing:
        acts = ctx.store.load("corporate_actions", ctx.store.dataset_ids("corporate_actions",
                                                                       synthetic=bool(run["is_synthetic"])))
        if not acts.empty:
            av = pd.to_datetime(acts["available_at"], utc=True, errors="coerce")
            m = (pd.to_datetime(acts["ex_date"]).dt.strftime("%Y-%m-%d") == ns) & (av <= now)
            actions_next = set(acts.loc[m, "symbol"])
    paused, pause_reason, _ = KillSwitch(db).state()
    quarantine = quarantine_map(db)
    strategies = strategy_research_status(db, ctx.config)
    links = strategy_links(db, d, None, synthetic=bool(run["is_synthetic"])) or {}
    catalysts = {}
    for r in db.fetchall("SELECT symbol, phase, kind, available_at FROM overnight_updates WHERE discovery_run_id=? "
                         "AND available_at <= ?", (run["discovery_run_id"], now.isoformat())):
        catalysts.setdefault(r["symbol"], []).append(dict(r))
    out_rows, summary = [], {}
    for c in db.fetchall("SELECT discovery_id, symbol, status, on_watchlist FROM discovery_candidates "
                         "WHERE discovery_run_id=? AND status IN ('WATCH','VALIDATION_PENDING','PAPER_ELIGIBLE','DISCOVERED')",
                         (run["discovery_run_id"],)):
        s, before = c["symbol"], c["status"]
        checks = []

        def chk(name, ok, reason):
            checks.append({"name": name, "passed": bool(ok), "reason": reason})
        chk("data_fresh", not stale, f"run is for {d}; last completed session is {ms.get('last_completed_session')}")
        chk("latest_bar", not bar_missing, f"stored bars end {last_bar}")
        chk("corporate_actions_data", not ca_missing, "corporate-action data present" if not ca_missing
            else "corporate-action data unavailable")
        chk("corporate_action_next_open", s not in actions_next, f"no split/dividend ex-date on {ns} known by now"
            if s not in actions_next else f"corporate action with ex-date {ns}: levels change at the open")
        q = quarantine.get(s, "none") if s in quarantine else "none"
        quarantined = s in quarantine and (quarantine[s] is None or str(quarantine[s])[:10] <= ns)
        chk("data_quality", not quarantined, "not quarantined" if not quarantined else f"quarantined ({q})")
        chk("kill_switch", paused.value == "ACTIVE", f"system {paused.value}" + (f": {pause_reason}" if pause_reason else ""))
        lk = links.get(s, [])
        dec = next((x for x in lk if x.get("decision") == "TRADE"), None)
        sid = (dec or (lk[0] if lk else {})).get("strategy_id")
        st = strategies.get(sid, {}).get("status") if sid else None
        chk("strategy_status", (st == "PAPER_ELIGIBLE") if dec else True,
            f"{sid} is {st}" if sid else "no strategy covers this setup")
        chk("pre_market_price", True, "pre-market price UNKNOWN (no pre-market data source); the next-open "
            "execution rule decides at the open")
        cat = catalysts.get(s, [])
        chk("overnight_catalyst", True, (f"{len(cat)} item(s): " + ", ".join(f"{x['kind']} {x['phase']}" for x in cat))
            if cat else "none recorded (sources cover almost no symbols; UNKNOWN is not 'none')")
        after, reason = decide_transition(
            before, missing_inputs=[x["reason"] for x in checks if not x["passed"] and x["name"] in
                                    ("data_fresh", "latest_bar", "corporate_actions_data")],
            corporate_action=s in actions_next, blocked=[x["reason"] for x in checks if not x["passed"] and x["name"] in
                                                         ("data_quality", "kill_switch")],
            trade_decision=dec is not None, strategy_eligible=(st == "PAPER_ELIGIBLE"), next_session=ns)
        summary[after] = summary.get(after, 0) + 1
        out_rows.append({"discovery_run_id": run["discovery_run_id"], "discovery_id": c["discovery_id"], "symbol": s,
                         "checked_at": now.isoformat(), "next_session": ns,
                         "next_open_at": _et(ns, OPEN).tz_convert("UTC").isoformat(), "status_before": before,
                         "status_after": after, "reason": reason, "checks_json": to_json(checks),
                         "created_at": utcnow_iso()})
    if out_rows:
        db.insert_many("preopen_checks", out_rows)
    return {"ok": True, "discovery_run_id": run["discovery_run_id"], "checked_at": now.isoformat(),
            "next_session": ns, "checked": len(out_rows), "transitions": summary}


def decide_transition(before: str, *, missing_inputs: list[str], corporate_action: bool, blocked: list[str],
                      trade_decision: bool, strategy_eligible: bool, next_session: str | None) -> tuple[str, str]:
    """Pre-open status transition (pure). Order: missing critical inputs -> UNKNOWN; a corporate
    action at the next open -> INVALIDATED; data-quality/kill-switch block -> REJECTED; an eligible
    candidate stays PAPER_ELIGIBLE only while its TRADE decision and strategy eligibility still hold;
    nothing here can create eligibility for a candidate the decision chain did not approve."""
    if missing_inputs:
        return "UNKNOWN", "critical inputs missing or stale: " + "; ".join(missing_inputs)
    if corporate_action:
        return "INVALIDATED", f"corporate action at the {next_session} open: levels change"
    if blocked:
        return "REJECTED", "; ".join(blocked)
    if before == "PAPER_ELIGIBLE":
        if trade_decision and strategy_eligible:
            return "PAPER_ELIGIBLE", "TRADE decision stands; still requires the next-open execution window"
        return "REJECTED", "the TRADE decision or the strategy's eligibility no longer holds"
    return before, "no pre-open information changes the setup; still requires next-session confirmation"


def next_session_alerts(ctx, now=None) -> list[dict[str, Any]]:
    db = ctx.db
    now = _utc(now or datetime.now(timezone.utc))
    ms = market_state(now)
    out: list[dict[str, Any]] = []

    def add(level, code, msg):
        out.append({"level": level, "code": code, "message": msg})
    if ms.get("status") == "UNKNOWN" or not ms.get("next_session"):
        add("CRITICAL", "MISSING_CALENDAR", "exchange calendar unavailable: next session cannot be determined")
        return out
    last = ms["last_completed_session"]
    run = db.fetchone("SELECT * FROM discovery_runs WHERE is_synthetic=0 ORDER BY as_of_date DESC, created_at DESC LIMIT 1") \
        or latest_run(db)
    bars_end = db.fetchone("SELECT MAX(end_date) AS e FROM datasets WHERE kind='bars'")["e"]
    after_processing = now >= _et(last, time(20, 30)) if last else False
    if bars_end and str(bars_end)[:10] < last and after_processing:
        add("CRITICAL", "MISSING_LATEST_BAR", f"latest completed session is {last} but stored bars end {str(bars_end)[:10]}")
    if run is None:
        add("CRITICAL", "NO_OVERNIGHT_SCAN", "no discovery run exists")
    elif run["as_of_date"] < last:
        lvl = "CRITICAL" if after_processing else "WARN"
        add(lvl, "STALE_CANDIDATE_LIST", f"latest discovery run is for {run['as_of_date']}; last completed session is {last}"
            + ("" if after_processing else " (end-of-day scan not due yet)"))
        if after_processing:
            add("CRITICAL", "NO_OVERNIGHT_SCAN", f"no end-of-day scan for {last}")
    if run is not None and run.get("info_cutoff_at") and (now - _utc(run["info_cutoff_at"])) > pd.Timedelta(days=4):
        add("WARN", "CANDIDATE_DATA_TOO_OLD", f"candidate information cutoff {run['info_cutoff_at']} is older than 4 days")
    news = db.fetchone("SELECT MAX(end_date) AS e, COUNT(*) AS n FROM datasets WHERE kind='news'")
    if not news or not news["n"]:
        add("INFO", "NO_NEWS_SOURCE", "no news dataset: overnight news catalysts are UNKNOWN")
    elif news["e"] and (pd.Timestamp(now.date()) - pd.Timestamp(str(news["e"])[:10])).days > 3:
        add("WARN", "NEWS_INGESTION_STALE", f"news data ends {str(news['e'])[:10]}: overnight catalysts are UNKNOWN")
    ca = db.fetchone("SELECT MAX(end_date) AS e, COUNT(*) AS n FROM datasets WHERE kind='corporate_actions'")
    if not ca or not ca["n"]:
        add("WARN", "CORPORATE_ACTIONS_UNAVAILABLE", "no corporate-action data: pre-open recheck marks candidates UNKNOWN")
    if run is not None and ms["status"] == "CLOSED":
        ns_open = _et(ms["next_session"], OPEN)
        if now >= ns_open - pd.Timedelta(hours=5, minutes=30) and run["as_of_date"] == last:
            done = db.fetchone("SELECT COUNT(*) AS n FROM preopen_checks WHERE discovery_run_id=?",
                               (run["discovery_run_id"],))["n"]
            if not done:
                add("WARN", "PREOPEN_NOT_DONE", f"pre-open validation for the {ms['next_session']} session not completed")
    return out


def next_session_state(ctx, now=None, top: int = 10) -> dict[str, Any]:
    """Everything the dashboard's NEXT TRADING SESSION section shows (read-only)."""
    db = ctx.db
    now = _utc(now or datetime.now(timezone.utc))
    ms = market_state(now)
    run = latest_run(db)
    out: dict[str, Any] = {"market": ms, "run": None, "counts": {}, "top": [], "pipeline": [], "mismatch": None,
                           "alerts": next_session_alerts(ctx, now)}
    if run is None:
        return out
    rid = run["discovery_run_id"]
    out["run"] = {k: run.get(k) for k in ("discovery_run_id", "as_of_date", "next_session", "info_cutoff_at",
                                          "created_at", "is_synthetic", "calendar_source")}
    # the shown setups must be for the ACTUAL next session; otherwise say so loudly
    out["mismatch"] = (None if run.get("next_session") == ms.get("next_session") else
                       f"these setups were prepared on {run['as_of_date']} for the {run.get('next_session')} session; "
                       f"the next session is {ms.get('next_session')} and no end-of-day scan exists for "
                       f"{ms.get('last_completed_session')} yet")
    promoted = {r["symbol"] for r in db.fetchall(
        "SELECT symbol FROM overnight_updates WHERE discovery_run_id=? AND effect='PROMOTED_TO_WATCH'", (rid,))}
    rows = db.fetchall("SELECT * FROM discovery_candidates WHERE discovery_run_id=? ORDER BY discovery_score IS NULL, "
                       "discovery_score DESC", (rid,))
    pre = {}
    for r in db.fetchall("SELECT * FROM preopen_checks WHERE discovery_run_id=? ORDER BY id", (rid,)):
        pre[r["symbol"]] = r
    cats: dict[str, list] = {}
    for r in db.fetchall("SELECT symbol, phase, kind, available_at, detail_json FROM overnight_updates "
                         "WHERE discovery_run_id=? ORDER BY available_at", (rid,)):
        cats.setdefault(r["symbol"], []).append({**{k: r[k] for k in ("phase", "kind", "available_at")},
                                                 "detail": from_json(r["detail_json"], {})})
    out["counts"] = {
        "candidates": len(rows), "high_ranked": sum(1 for r in rows if r["high_quality"]),
        "watchlist": sum(1 for r in rows if r["on_watchlist"] or r["symbol"] in promoted),
        "validation_pending": sum(1 for r in rows if r["status"] == "VALIDATION_PENDING"),
        "paper_eligible": sum(1 for r in rows if (pre.get(r["symbol"]) or {}).get("status_after", r["status"])
                              in ("PAPER_ELIGIBLE",)) + sum(1 for r in rows if r["status"] == "TRADED"),
        "promoted_overnight": len(promoted),
    }
    order = {"TRADED": 0, "PAPER_ELIGIBLE": 1, "WATCH": 2, "VALIDATION_PENDING": 3, "DISCOVERED": 4, "REJECTED": 5}
    ranked = [r for r in rows if r["on_watchlist"] or r["symbol"] in promoted or r["status"] in
              ("PAPER_ELIGIBLE", "TRADED", "VALIDATION_PENDING")] or rows
    ranked = sorted(ranked, key=lambda r: (order.get((pre.get(r["symbol"]) or {}).get("status_after") or r["status"], 6),
                                           -(r["discovery_score"] if r["discovery_score"] is not None else -1)))
    for r in ranked[:top]:
        setup = from_json(r.get("setup_json"), {}) or {}
        p = pre.get(r["symbol"])
        out["top"].append({"symbol": r["symbol"], "score": r["discovery_score"], "origin": r.get("origin"),
                           "status": (p or {}).get("status_after") or r["status"],
                           "status_at_scan": r["status"], "block_reason": r["block_reason"], "setup": setup,
                           "preopen": {"status_after": p["status_after"], "reason": p["reason"], "checked_at": p["checked_at"]}
                           if p else None, "catalysts": cats.get(r["symbol"], []),
                           "promoted_overnight": r["symbol"] in promoted})
    last_upd = db.fetchone("SELECT MAX(refreshed_at) AS t, COUNT(*) AS n FROM overnight_updates WHERE discovery_run_id=?", (rid,))
    post = db.fetchone("SELECT COUNT(*) AS n FROM overnight_updates WHERE discovery_run_id=? AND phase='POST_CLOSE'", (rid,))
    last_pre = db.fetchone("SELECT MAX(checked_at) AS t, COUNT(*) AS n FROM preopen_checks WHERE discovery_run_id=?", (rid,))
    dec = db.fetchone("SELECT r.run_id, r.finished_at, r.status FROM runs r WHERE r.kind='pipeline' AND r.as_of_date=? "
                      "ORDER BY r.started_at DESC LIMIT 1", (run["as_of_date"],))
    out["pipeline"] = [
        {"stage": "Session closed", "at": run.get("info_cutoff_at"), "state": "done"},
        {"stage": "End-of-day scan", "at": run.get("created_at"), "state": "done"},
        {"stage": "Post-close information", "at": None, "state": f"{post['n']} item(s)" if post["n"] else "none recorded"},
        {"stage": "Overnight candidate refresh", "at": last_upd["t"], "state": "done" if last_upd["t"] else "not run"},
        {"stage": "Pre-open validation", "at": last_pre["t"], "state": f"done ({last_pre['n']} checked)" if last_pre["t"] else "not run"},
        {"stage": "Next-session decision", "at": (dec or {}).get("finished_at"),
         "state": (dec or {}).get("status") or "no pipeline run for this session"},
    ]
    return out


__all__ = ["PHASES", "decide_transition", "info_phase", "market_state", "next_session_alerts", "next_session_state", "overnight_refresh",
           "preopen_recheck"]
