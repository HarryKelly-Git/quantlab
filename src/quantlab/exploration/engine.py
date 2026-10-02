"""PAPER_EXPLORATION engine (paper only; there is no live path here or anywhere in QuantLab).

``paper.mode`` selects one of two clearly separated modes:

  STRICT       only TRADE decisions of the unchanged decision chain are traded (validated strategy,
               EV after costs, risk chain). Exploration candidates are still planned and TRACKED as
               SHADOW (never traded), so forward evidence accumulates either way.
  EXPLORATION  additionally, a small FIXED daily budget of the best-ranked discovery / strategy
               candidates is paper traded without proven edge, to learn forward what works.

Exploration does NOT require: strategy validation or promotion, statistically significant
out-of-sample results, positive historical expectancy (the EV gate).

Exploration ALWAYS requires (both modes, never relaxed): valid price/volume data at the decision
session, the research universe's liquidity/history/security-type rules plus exploration's own
minimum price and ADV, no data-quality quarantine, kill switch ACTIVE, long-only direction, no
existing position in the symbol, position / session / total exposure limits -- and at submission
every gate of the SAME execution service strict orders use (kill switch, daily order cap, per-order
notional cap, duplicates, the runner's order window, broker verified, reconciliation passed).

Decisions are append-only (``exploration_decisions``); lifecycle changes and outcomes are new rows
(``exploration_events``, ``exploration_outcomes``). A decision is never rewritten after the fact.
Ranking = the SELECTION score carried on each discovered candidate (discovery.families.selection_score:
12-1 momentum, low volatility, liquidity -- fixed from published priors; the descriptive discovery score
predicts nothing, see docs/SELECTION-EVIDENCE.md), UNKNOWN last, discovery rank as tie-break. N is fixed in
config and never fitted to results.
"""
from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.core.costs import CostModel
from quantlab.exploration.upside import upside_profile
from quantlab.core.types import TradePlan, new_id
from quantlab.db.database import from_json, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)
ET = ZoneInfo("America/New_York")
MODES = ("STRICT", "EXPLORATION")
STRATEGY_ID = "EXPLORATION"
STRATEGY_VERSION = "1"
HORIZONS = (1, 3, 5, 10, 20)
NOT_REQUIRED = ("strategy validation / promotion", "statistically significant out-of-sample results",
                "positive historical expectancy (EV gate)")
TERMINAL_EVENTS = ("SUBMITTED", "CANCELLED_PREOPEN", "REFUSED")
# REQUIRED inputs: missing or stale -> the candidate is SKIPPED (plan) or CANCELLED_PREOPEN (submit).
REQUIRED_AT_PLAN = ("valid_price_volume", "research_universe", "basic_liquidity", "data_quality", "kill_switch",
                    "long_only_direction", "no_open_position", "stop_computable", "not_pinned", "max_open_positions",
                    "industry_concentration", "position_size")
REQUIRED_AT_SUBMIT = ("before_pre_open_cutoff", "not_pinned", "after_decision_close", "kill_switch", "data_quality", "no_open_position",
                      "bars_for_decision_session", "corporate_action_data", "no_corporate_action_at_open",
                      "preopen_recheck_not_invalid", "plan_values", "execution_guard")
# OPTIONAL inputs: missing -> recorded as UNKNOWN, never zero, and never a reason to trade or not.
OPTIONAL_INPUTS = ("news", "earnings / guidance", "fundamentals", "industry / sector", "consensus surprise",
                   "pre-market price", "overnight catalysts")
OPEN_ORDER_STATES = ("pending_submit", "new", "accepted", "partially_filled", "unknown")


def paper_mode(config) -> str:
    m = str(config.get("paper.mode", "STRICT")).strip().upper()
    if m not in MODES:
        raise ValueError(f"paper.mode must be one of {MODES}, got {m!r}")
    return m


@dataclass(frozen=True)
class ExplorationPolicy:
    max_new_per_session: int = 5
    max_open_positions: int = 10
    max_concurrent_experiments: int = 10
    max_per_industry: int = 2
    max_position_pct: float = 0.02
    max_session_exposure_pct: float = 0.10
    max_total_exposure_pct: float = 0.20
    holding_sessions: int = 10            # used only when hold_arms is empty
    hold_arms: tuple = (5, 10, 20)        # time-exit arms, one per decision (see hold_for)
    stop_atr: float = 2.0
    min_price: float = 5.0
    min_adv: float = 5_000_000.0
    # a stock whose daily range is this small is pinned (typically a pending cash takeover): capped upside,
    # deal-break downside, and a 2xATR stop inside the noise. The selection score (low volatility + 12-1
    # momentum) finds such names by construction, e.g. HZO on 2026-10-01 (0.28%, being acquired).
    min_atr_pct: float = 0.0075
    watched_not_traded: int = 10
    submit_after_et: str = "08:30"
    min_observations_to_propose: int = 30
    max_considered: int = 60
    # research tracking only (see quantlab.exploration.tracked): recorded every session, NEVER traded or sized
    tracked_watchlist: tuple = ()

    @classmethod
    def from_config(cls, config) -> "ExplorationPolicy":
        c = dict(config.get("exploration", {}) or {})
        if "tracked_watchlist" in c:
            c["tracked_watchlist"] = normalize_symbols(c["tracked_watchlist"])
        return cls(**{k: type(getattr(cls, k))(v) for k, v in c.items() if hasattr(cls, k)})


def normalize_symbols(x: Any) -> tuple[str, ...]:
    """A configured symbol list as unique upper-case tickers, in order (None / '' -> empty)."""
    if x is None:
        return ()
    items = [x] if isinstance(x, str) else list(x)
    out: list[str] = []
    for s in items:
        t = str(s or "").strip().upper()
        if t and t not in out:
            out.append(t)
    return tuple(out)


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _latest_run(db, session: str | None = None, run_id: str | None = None, synthetic: bool | None = None):
    if run_id:
        return db.fetchone("SELECT * FROM discovery_runs WHERE discovery_run_id=?", (run_id,))
    sql, args = "SELECT * FROM discovery_runs WHERE 1=1", []
    if session:
        sql += " AND as_of_date=?"
        args.append(str(session)[:10])
    if synthetic is not None:
        sql += " AND is_synthetic=?"
        args.append(int(synthetic))
    return db.fetchone(sql + " ORDER BY as_of_date DESC, created_at DESC LIMIT 1", args)


def open_exposure(db, book: str) -> dict[str, Any]:
    rows = db.fetchall("SELECT symbol, qty, entry_price, strategy_id FROM trades WHERE book=? AND status='OPEN'", (book,))
    exp = [r for r in rows if r["strategy_id"] == STRATEGY_ID]
    return {"symbols": {r["symbol"] for r in rows}, "exploration_open": len(exp),
            "exploration_notional": float(sum((r["qty"] or 0) * (r["entry_price"] or 0) for r in exp))}


def industry_code(record: dict[str, Any] | None) -> str | None:
    """The point-in-time SIC industry group of a candidate (``{"industry": {"code": 283.0, ...}}``),
    or None when it is UNKNOWN."""
    ind = (record or {}).get("industry")
    code = ind.get("code") if isinstance(ind, dict) else None
    try:
        v = float(code)
    except (TypeError, ValueError):
        return None
    return str(int(v)) if math.isfinite(v) else None


def held_industries(db, book: str) -> collections.Counter:
    """Industry of every OPEN or still-working exploratory position, read from its own decision
    record (the industry known when it was chosen, never re-derived later)."""
    rows = db.fetchall(
        "SELECT d.pre_trade_json FROM exploration_decisions d WHERE d.selection='SELECTED' AND d.mode='EXPLORATION' AND ("
        " EXISTS (SELECT 1 FROM trades t WHERE t.decision_id=d.decision_id AND t.book=? AND t.status='OPEN')"
        " OR (EXISTS (SELECT 1 FROM orders o WHERE o.decision_id=d.decision_id AND o.purpose='entry'"
        f"   AND o.status IN ({','.join('?' for _ in OPEN_ORDER_STATES)}))"
        "  AND NOT EXISTS (SELECT 1 FROM trades t2 WHERE t2.decision_id=d.decision_id)))",
        (book, *OPEN_ORDER_STATES))
    out: collections.Counter = collections.Counter()
    for r in rows:
        code = industry_code(from_json(r["pre_trade_json"], {}) or {})
        if code is not None:
            out[code] += 1
    return out


def pending_entries(db) -> dict[str, Any]:
    """Exploratory entries submitted but not yet a trade (working orders): they count toward the
    open-position and exposure caps exactly like open positions."""
    rows = db.fetchall(
        "SELECT d.symbol, d.qty, d.ref_price FROM exploration_decisions d JOIN exploration_events e "
        "ON e.decision_id=d.decision_id AND e.event='SUBMITTED' JOIN orders o ON o.order_id=e.order_id "
        f"WHERE o.status IN ({','.join('?' for _ in OPEN_ORDER_STATES)}) "
        "AND NOT EXISTS (SELECT 1 FROM trades t WHERE t.decision_id=d.decision_id)", OPEN_ORDER_STATES)
    return {"n": len(rows), "notional": float(sum((r["qty"] or 0) * (r["ref_price"] or 0) for r in rows)),
            "symbols": {r["symbol"] for r in rows}}


def _pending_symbols(db, mode: str, *, now=None, cutoff_hhmm: str = "09:25") -> set[str]:
    """Selected exploratory entries that may still become a position: planned and not yet at a
    terminal pre-open event, or with an entry order still working (the opg order or its fallback).

    A decision whose entry orders all ended unfilled (expired / cancelled / rejected) is no longer
    pending, and neither is one whose trade exists (an open trade is caught by the open-position
    check; a closed one frees the symbol). Before this, every symbol ever selected stayed blocked
    for good, shrinking the tradeable pool with every session. A planned decision that was never
    submitted stops being pending once its own pre-open order cutoff has passed (the runner was down,
    say): it can no longer become a position."""
    states = ",".join("?" for _ in OPEN_ORDER_STATES)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    rows = db.fetchall(
        "SELECT d.symbol, d.next_session, "
        f"EXISTS (SELECT 1 FROM orders o WHERE o.decision_id=d.decision_id AND o.purpose='entry' AND o.status IN ({states})) AS working, "
        "EXISTS (SELECT 1 FROM exploration_events e WHERE e.decision_id=d.decision_id AND e.event IN "
        "('SUBMITTED','CANCELLED_PREOPEN','REFUSED')) AS terminal "
        "FROM exploration_decisions d WHERE d.selection='SELECTED' AND d.mode=? "
        "AND NOT EXISTS (SELECT 1 FROM trades t WHERE t.decision_id=d.decision_id)",
        (*OPEN_ORDER_STATES, mode))
    out: set[str] = set()
    for r in rows:
        if r["working"]:
            out.add(r["symbol"])
        elif not r["terminal"]:
            ns = r["next_session"]
            if ns is None or now < pd.Timestamp(f"{str(ns)[:10]} {cutoff_hhmm}", tz=ET).tz_convert("UTC"):
                out.add(r["symbol"])
    return out


def _catalyst_direction(cat: dict[str, Any]) -> str:
    pe, me = cat.get("post_earnings") or {}, cat.get("material_event") or {}
    if pe.get("state") == "FIRED":
        return pe.get("reaction_direction") or "UNKNOWN"
    if me.get("state") == "FIRED":
        return me.get("direction") or "UNKNOWN"
    return "NONE"


def hold_for(session: str, symbol: str, pol: "ExplorationPolicy") -> int:
    """Holding period for one decision, drawn from ``pol.hold_arms`` by a hash of (session, symbol).

    Reproducible, balanced in expectation, and independent of the selection score and rank, so that
    outcomes by holding period can be compared later without the choice leaking into them. The stop
    (stop_atr x ATR) is the same for every arm: the holding period is the only thing that differs."""
    arms = tuple(int(a) for a in (pol.hold_arms or ()) if int(a) > 0)
    if not arms:
        return int(pol.holding_sessions)
    h = int(hashlib.sha256(f"{str(session)[:10]}|{symbol}".encode()).hexdigest(), 16)
    return arms[h % len(arms)]


def _selection_of(r) -> float | None:
    """The candidate's selection score from its setup record (None = UNKNOWN)."""
    return _f((from_json(r["setup_json"], {}) or {}).get("selection_score"))


def _selection_key(r) -> tuple:
    """Highest selection score first; UNKNOWN after every scored candidate; discovery rank breaks ties."""
    s = _selection_of(r)
    return (0 if s is not None else 1, -(s or 0.0), r["rank"] if r["rank"] is not None else 10 ** 9, r["symbol"])


def plan_exploration(ctx, *, book: str = "BOT", session=None, run_id: str | None = None, equity: float | None = None,
                     mode: str | None = None, now=None, synthetic: bool | None = None, scan=None) -> dict[str, Any]:
    """Choose this session's exploratory candidates and record the COMPLETE pre-trade state.
    Idempotent per (session, mode). Never submits anything (see :func:`preopen_submit`).

    The regime throttle (``exploration.regime_throttle``, see exploration.regime_throttle) may lower the
    session's new-entry cap; the regime is recorded on every decision. Tracked-watchlist symbols
    (``exploration.tracked_watchlist``, see exploration.tracked) are recorded as TRACKED and never traded;
    ``scan`` is the session's discovery ScanResult, used only for them (features and universe percentile
    of tracked symbols outside the candidate pool)."""
    from quantlab.data.validation import quarantine_map
    from quantlab.exploration.regime_throttle import RegimeThrottle, throttle_state, throttle_text
    from quantlab.monitoring.killswitch import KillSwitch
    db, cfg = ctx.db, ctx.config
    mode = (mode or paper_mode(cfg)).upper()
    pol = ExplorationPolicy.from_config(cfg)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    run = _latest_run(db, session=session, run_id=run_id, synthetic=synthetic)
    if run is None:
        return {"ok": False, "reason": "no discovery run for the session"}
    d = run["as_of_date"]
    done = db.fetchall("SELECT selection, COUNT(*) AS n FROM exploration_decisions WHERE session_date=? AND mode=? "
                       "GROUP BY selection", (d, mode))
    if done:
        return {"ok": True, "session": d, "mode": mode, "already_planned": True,
                "counts": {r["selection"]: r["n"] for r in done}}
    state, pause_reason, _ = KillSwitch(db).state()
    quarantine = quarantine_map(db)
    ox = open_exposure(db, book)
    pend = pending_entries(db)
    ox = {**ox, "exploration_open": ox["exploration_open"] + pend["n"],
          "exploration_notional": ox["exploration_notional"] + pend["notional"]}
    pending = _pending_symbols(db, "EXPLORATION", now=now, cutoff_hhmm=str(cfg.get("paper.runner.order_cutoff_et", "09:25")))
    ind_held = held_industries(db, book)          # open + working positions, by PIT industry group
    if equity is None:
        from quantlab.execution.ledger import Ledger
        led = Ledger(db, book, config=cfg).state()
        # Ledger.positions() is a list of position rows (qty, avg_cost); positions at cost -- the
        # live pipeline passes mark-to-market equity and never reaches this fallback
        pos = led.get("positions") or []
        pos = list(pos.values()) if isinstance(pos, dict) else list(pos)
        equity = float(led.get("cash") or 0.0) + sum(float((p.get("qty") or 0) * (p.get("avg_cost") or 0))
                                                     for p in pos if isinstance(p, dict))
    equity = float(equity or 0.0)
    max_notional = float(cfg.get("risk.max_order_notional", 15000))
    rows = db.fetchall("SELECT * FROM discovery_candidates WHERE discovery_run_id=? AND status NOT IN ('TRADED','PAPER_ELIGIBLE')",
                       (run["discovery_run_id"],))
    rows = sorted(rows, key=_selection_key)
    order = {r["symbol"]: i + 1 for i, r in enumerate(rows)}
    dataset_ids = from_json(run["dataset_ids_json"], [])
    # regime throttle: the market metric of D (exact date, never later) may lower the new-entry cap
    regime = throttle_state(db, d, pol.max_new_per_session, RegimeThrottle.from_config(cfg))
    cap_new = int(regime["effective_max_new"])
    n_displace = max(pol.max_new_per_session - cap_new, 0) if regime["throttled"] else 0
    # tracked symbols leave the tradeable pool here (never traded, never sized, never counted) and get
    # their own TRACKED record below; an empty watchlist leaves the pool unchanged
    tracked = set(pol.tracked_watchlist)
    pool = [r for r in rows if r["symbol"] not in tracked] if tracked else rows
    selected = watched = displaced = 0
    session_notional = 0.0
    out = []
    for r in pool[:pol.max_considered]:
        s = r["symbol"]
        hold = hold_for(d, s, pol)
        setup = from_json(r["setup_json"], {}) or {}
        lv = setup.get("levels") or {}
        facts = (from_json(r["factors_json"], {}) or {}).get("features", {}) or {}
        cat = from_json(r["catalyst_record_json"], {}) or {}
        checks = from_json(r["checks_json"], []) or []
        links = from_json(r["strategy_links_json"], []) or []
        chain = from_json(r["evidence_chain_json"], []) or []
        close, atr = _f(lv.get("close")), _f(lv.get("atr"))
        adv = _f((facts.get("adv20") or {}).get("value"))
        ck: list[dict[str, Any]] = []

        def chk(name, ok, why):
            ck.append({"name": name, "passed": bool(ok), "reason": why})
        by = {c["name"]: c for c in checks}
        fq, uni = by.get("feature_quality"), by.get("research_universe")
        chk("valid_price_volume", close is not None and fq is not None and fq["passed"],
            "valid bar and features at the decision session" if (close is not None and fq and fq["passed"])
            else (fq or {}).get("reason", "not in the discovery scan (no validated price/volume data)"))
        chk("research_universe", uni is not None and uni["passed"], (uni or {}).get("reason", "universe not evaluated"))
        chk("basic_liquidity", close is not None and adv is not None and close >= pol.min_price and adv >= pol.min_adv,
            f"close {close if close is not None else 'UNKNOWN'} (min {pol.min_price:g}), ADV "
            f"{f'{adv:,.0f}' if adv is not None else 'UNKNOWN'} (min {pol.min_adv:,.0f})")
        q = s in quarantine and (quarantine[s] is None or str(quarantine[s])[:10] <= d)
        chk("data_quality", not q, "not quarantined" if not q else f"quarantined ({quarantine.get(s)})")
        chk("kill_switch", state.value == "ACTIVE", f"system {state.value}" + (f": {pause_reason}" if pause_reason else ""))
        cdir = _catalyst_direction(cat)
        long_ok = r["direction_bias"] != "BEARISH" and cdir != "NEGATIVE"
        chk("long_only_direction", long_ok, f"bias {r['direction_bias']}, catalyst {cdir.lower()}"
            + ("" if long_ok else ": QuantLab paper trading is long-only"))
        chk("no_open_position", s not in ox["symbols"] and s not in pending,
            "no open or pending position" if s not in ox["symbols"] and s not in pending else "already held or pending")
        chk("stop_computable", atr is not None and atr > 0, "ATR known" if atr else "ATR UNKNOWN: no invalidation stop")
        pinned = bool(atr and close and atr / close < pol.min_atr_pct)
        chk("not_pinned", not pinned,
            (f"daily range {atr / close:.2%} of price < {pol.min_atr_pct:.2%}: price pinned (typically a pending cash "
             "takeover): capped upside, deal-break downside" if pinned else
             (f"daily range {atr / close:.2%} of price" if atr and close else "ATR UNKNOWN")))
        eligible = all(c["passed"] for c in ck)
        qty = 0
        stop = None
        by_throttle = False
        if eligible and selected < cap_new:
            n_open = ox["exploration_open"] + selected
            chk("max_open_positions", n_open < min(pol.max_open_positions, pol.max_concurrent_experiments),
                f"{n_open} open exploratory position(s) (max {min(pol.max_open_positions, pol.max_concurrent_experiments)})")
            # more positions must mean more independent bets, not one sector bought five times
            ind = industry_code(cat)
            n_ind = ind_held[ind] if ind is not None else 0
            chk("industry_concentration", ind is None or n_ind < pol.max_per_industry,
                (f"industry {ind}: {n_ind} open/pending/selected (max {pol.max_per_industry})" if ind is not None
                 else "industry UNKNOWN: concentration cannot be measured, not a reason to skip"))
            cap = min(pol.max_position_pct * equity, pol.max_session_exposure_pct * equity - session_notional,
                      pol.max_total_exposure_pct * equity - ox["exploration_notional"] - session_notional, max_notional)
            qty = int(math.floor(max(cap, 0.0) / close)) if close else 0
            chk("position_size", qty >= 1, f"{qty} share(s) = {qty * (close or 0):,.2f} within caps (position "
                f"{pol.max_position_pct:.0%}, session {pol.max_session_exposure_pct:.0%}, total "
                f"{pol.max_total_exposure_pct:.0%} of equity {equity:,.0f}; order cap {max_notional:,.0f})")
            if all(c["passed"] for c in ck):
                selection = "SELECTED" if mode == "EXPLORATION" else "SHADOW"
                selected += 1
                session_notional += qty * close
                stop = round(close - pol.stop_atr * atr, 4)
                if ind is not None:
                    ind_held[ind] += 1
            else:
                selection, qty = "SKIPPED", 0
        elif eligible and displaced < n_displace:
            # within the normal budget but beyond the regime-throttled cap: the throttle's counterfactual,
            # recorded whatever the watched-not-traded allowance
            selection = "WATCHED_NOT_TRADED"
            displaced += 1
            by_throttle = True
        elif eligible and watched < pol.watched_not_traded:
            selection = "WATCHED_NOT_TRADED"
            watched += 1
        elif not eligible:
            selection = "SKIPPED"
        else:
            continue
        failed = [c for c in ck if not c["passed"]]
        strict_fail = [f"{x['stage']}: {x['text']}" for x in chain if x.get("state") in ("FAIL", "NOT_REACHED")
                       and x.get("stage") in ("VALIDATION", "RISK", "EV", "PAPER ELIGIBILITY")]
        best = next((x for x in links if x.get("decision") == "TRADE"), links[0] if links else None)
        sel_s = _selection_of(r)
        if selection in ("SELECTED", "SHADOW"):
            reason = (f"experimental setup #{selected} of up to {cap_new} for {d} "
                      f"({setup.get('setup_type')}; {r['setup_class'] or 'class UNKNOWN'}; selection score "
                      f"{f'{sel_s:.2f}' if sel_s is not None else 'UNKNOWN'}, order {order.get(s)}; discovery rank {r['rank']}; hold {hold} sessions)"
                      + (f"; {throttle_text(regime)}" if regime["throttled"] else ""))
        elif by_throttle:
            reason = (f"eligible and within the normal budget of {pol.max_new_per_session}, not traded because of the "
                      f"{throttle_text(regime)}: tracked for comparison")
        elif selection == "WATCHED_NOT_TRADED":
            reason = "eligible but beyond the session budget: tracked for comparison"
        else:
            reason = "skipped: " + "; ".join(f"{c['name']}: {c['reason']}" for c in failed)
        pre = {
            "ticker": s, "decided_at": now.isoformat(), "mode": mode, "selection": selection,
            "session_date": d, "next_session": run["next_session"], "information_cutoff_at": run["info_cutoff_at"],
            "discovery_run_id": run["discovery_run_id"], "discovery_id": r["discovery_id"],
            "candidate": {"origin": r["origin"], "discovery_score": r["discovery_score"], "rank": r["rank"],
                          "selection_score": _selection_of(r), "selection_order": order.get(r["symbol"]),
                          "status": r["status"], "on_watchlist": bool(r["on_watchlist"]),
                          "setup_type": setup.get("setup_type"), "setup_class": r["setup_class"],
                          "families": (from_json(r["families_json"], {}) or {}).get("fired", []),
                          "catalyst_families": r["catalyst_families"], "direction_bias": r["direction_bias"]},
            "strategy": [{k: x.get(k) for k in ("strategy_id", "strategy_version", "decision", "reject_stage", "ev_bps",
                                                "reasons")} for x in links],
            "validation_state": {"strict_status": r["status"], "block_stage": r["block_stage"],
                                 "block_reason": r["block_reason"]},
            "ev_estimate": ({"ev_bps_after_costs": best.get("ev_bps"), "source": "the unchanged chain's recorded decision"}
                            if best and best.get("ev_bps") is not None else
                            {"ev_bps_after_costs": None, "source": "UNKNOWN: no strategy EV for this setup"}),
            "price": {"close": close, "atr": atr, "levels": lv},
            "volume": {"adv20": adv, "rel_volume_1d": ((facts.get("rel_volume_1d") or {}).get("value"))},
            "catalyst": {"families": cat.get("families"), "direction": cdir,
                         "post_earnings": cat.get("post_earnings"), "material_event": cat.get("material_event")} if cat else None,
            "industry": cat.get("industry") if cat else None, "fundamentals": cat.get("fundamentals") if cat else None,
            "risk_checks": ck, "reason_for_entering": reason,
            "strict_would_reject_because": [r["block_reason"]] + strict_fail if r["block_reason"] else strict_fail,
            "not_required_in_exploration": list(NOT_REQUIRED),
            "confirmation": setup.get("confirm", []),
            "invalidation": setup.get("invalidate", []) + ([f"stop {stop:,.2f} = close - {pol.stop_atr:g} x ATR"] if stop else []),
            "expected_holding_sessions": hold,
            # base rates for this volatility profile at this hold: move SIZE, not direction (exploration.upside)
            "upside": upside_profile(facts.get("atr14_pct"), facts.get("range_contraction_20_60"), hold),
            "expected": "an experiment: no edge is assumed. Measured at 1/3/5/10/20 sessions vs SPY, vs watched-but-"
                        "not-traded and vs rejected candidates",
            "unknowns": setup.get("missing", []),
            "required_inputs": {"at_plan": ck, "at_submit": list(REQUIRED_AT_SUBMIT)},
            "optional_inputs": {"classes": list(OPTIONAL_INPUTS), "unknown_now": setup.get("missing", []),
                                "rule": "optional information may be UNKNOWN; it never blocks and never counts as zero"},
            "data_timestamps": {"information_cutoff_at": run["info_cutoff_at"], "decision_session_bar": d,
                                "dataset_ids": dataset_ids},
            "features": {"discovery": facts, "catalyst": (cat or {}).get("features")},
            "sizing": {"equity": equity, "qty": qty, "notional": qty * (close or 0.0), "stop": stop,
                       "limits": {k: getattr(pol, k) for k in ("max_position_pct", "max_session_exposure_pct",
                                                               "max_total_exposure_pct", "max_open_positions")}},
            "evidence_chain": chain,
            "regime": {**regime, "displaced_by_throttle": by_throttle},
        }
        out.append({"decision_id": new_id("expl"), "session_date": d, "next_session": run["next_session"], "symbol": s,
                    "mode": mode, "selection": selection, "rank": r["rank"], "discovery_run_id": run["discovery_run_id"],
                    "discovery_id": r["discovery_id"], "origin": r["origin"], "setup_type": setup.get("setup_type"),
                    "setup_class": r["setup_class"], "families": ",".join(pre["candidate"]["families"]) or None,
                    "catalyst_families": r["catalyst_families"], "discovery_score": r["discovery_score"],
                    "ref_price": close, "qty": float(qty), "stop_price": stop,
                    "holding_sessions": hold, "strict_blocker": "; ".join(pre["strict_would_reject_because"])[:2000] or None,
                    "reason": reason[:2000], "info_cutoff_at": run["info_cutoff_at"], "pre_trade_json": to_json(pre),
                    "is_synthetic": int(run["is_synthetic"]), "created_at": utcnow_iso()})
    tracked_absent: dict[str, str] = {}
    if tracked:
        from quantlab.exploration.tracked import tracked_decisions
        trk, tracked_absent = tracked_decisions(pol=pol, run=run, rows=rows, order=order, scan=scan, now=now, mode=mode,
                                                regime=regime, dataset_ids=dataset_ids)
        out.extend(trk)
    with db.transaction():
        if out:
            db.insert_many("exploration_decisions", out)
            db.insert_many("exploration_events", [{"decision_id": o["decision_id"], "event": "PLANNED", "at": now.isoformat(),
                                                   "order_id": None, "details_json": to_json({"mode": mode}),
                                                   "created_at": utcnow_iso()}
                                                  for o in out if o["selection"] == "SELECTED"])
    counts: dict[str, int] = {}
    for o in out:
        counts[o["selection"]] = counts.get(o["selection"], 0) + 1
    log_event(log, "exploration planned", session=d, mode=mode, regime=regime["state"], max_new=cap_new, **counts)
    res = {"ok": True, "session": d, "mode": mode, "counts": counts,
           "regime": {k: regime[k] for k in ("state", "label", "metric", "value", "throttled", "effective_max_new")},
           "selected": [{"symbol": o["symbol"], "qty": o["qty"], "stop": o["stop_price"], "setup": o["setup_type"],
                         "reason": o["reason"]} for o in out if o["selection"] in ("SELECTED", "SHADOW")]}
    if tracked:
        res["tracked"] = {"recorded": [o["symbol"] for o in out if o["selection"] == "TRACKED"],
                          "not_recorded": tracked_absent}
    return res


def preopen_submit(ctx, exec_service, *, now=None, session: str | None = None, mode: str | None = None) -> dict[str, Any]:
    """Revalidate each planned exploratory entry with the information available NOW (before the next
    open) and submit it through the paper execution service -- or cancel it. Never uses the open."""
    from quantlab.data.validation import quarantine_map
    from quantlab.monitoring.killswitch import KillSwitch
    db, cfg = ctx.db, ctx.config
    mode = (mode or paper_mode(cfg)).upper()
    if mode != "EXPLORATION":
        return {"ok": True, "mode": mode, "submitted": 0, "reason": "paper.mode is STRICT: exploratory entries are "
                                                                     "tracked as SHADOW, never submitted"}
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    sql = ("SELECT d.* FROM exploration_decisions d WHERE d.selection='SELECTED' AND d.mode='EXPLORATION' AND NOT EXISTS "
           "(SELECT 1 FROM exploration_events e WHERE e.decision_id=d.decision_id AND e.event IN "
           "('SUBMITTED','CANCELLED_PREOPEN','REFUSED'))")
    args: list[Any] = []
    if session:
        sql += " AND d.session_date=?"
        args.append(str(session)[:10])
    todo = db.fetchall(sql + " ORDER BY d.session_date, d.rank", args)
    if not todo:
        return {"ok": True, "mode": mode, "submitted": 0, "cancelled": 0, "refused": 0}
    state, pause_reason, _ = KillSwitch(db).state()
    quarantine = quarantine_map(db)
    acts = ctx.store.load("corporate_actions", ctx.store.dataset_ids("corporate_actions",
                                                                    synthetic=bool(todo[0]["is_synthetic"])))
    held = open_exposure(db, exec_service.book)["symbols"]
    syn = bool(todo[0]["is_synthetic"])
    ca_data = bool(ctx.store.dataset_ids("corporate_actions", synthetic=syn))
    last_bar = (db.fetchone("SELECT MAX(end_date) AS e FROM datasets WHERE kind='bars' AND is_synthetic=?",
                            (int(syn),)) or {}).get("e")
    cutoff_hhmm = str(cfg.get("paper.runner.order_cutoff_et", "09:25"))
    pol = ExplorationPolicy.from_config(cfg)
    rechecks: dict[tuple[str, str], dict[str, Any]] = {}
    for r in db.fetchall("SELECT discovery_run_id, symbol, status_before, status_after, reason, checked_at FROM preopen_checks "
                         "WHERE checked_at <= ? ORDER BY id", (now.isoformat(),)):
        rechecks[(r["discovery_run_id"], r["symbol"])] = r
    res = {"ok": True, "mode": mode, "submitted": 0, "cancelled": 0, "refused": 0, "details": []}
    for d in todo:
        ns = d["next_session"]
        why = []
        if ns is None:
            why.append("next session unknown (calendar)")
        else:
            cut_at = pd.Timestamp(f"{ns} {cutoff_hhmm}", tz=ET).tz_convert("UTC")
            if now >= cut_at:
                why.append(f"at/after the pre-open cutoff ({cutoff_hhmm} ET on {ns}): no order; the actual open is "
                           "never an input")
        if d["info_cutoff_at"] and now <= pd.Timestamp(d["info_cutoff_at"]):
            why.append("the decision session has not closed yet")
        if not last_bar or str(last_bar)[:10] < str(d["session_date"]):
            why.append(f"REQUIRED data missing: bars for {d['session_date']} are not stored (latest {last_bar})")
        if not ca_data:
            why.append("REQUIRED data missing: no corporate-action data (a split or dividend at the open cannot be ruled out)")
        if not (d["qty"] and d["qty"] >= 1 and d["ref_price"] and d["ref_price"] > 0 and d["stop_price"]
                and 0 < d["stop_price"] < d["ref_price"]):
            why.append("REQUIRED plan values missing or invalid (quantity / reference price / stop)")
        elif (d["ref_price"] - d["stop_price"]) / (pol.stop_atr * d["ref_price"]) < pol.min_atr_pct:
            why.append(f"price pinned: daily range {(d['ref_price'] - d['stop_price']) / (pol.stop_atr * d['ref_price']):.2%} "
                       f"of price < {pol.min_atr_pct:.2%} (typically a pending cash takeover)")
        rc = rechecks.get((d["discovery_run_id"], d["symbol"]))
        if rc and (rc["status_after"] in ("UNKNOWN", "INVALIDATED") or
                   (rc["status_after"] == "REJECTED" and rc["status_before"] != "REJECTED")):
            why.append(f"pre-open recheck {rc['checked_at'][:16]}: {rc['status_after']} ({rc['reason']})")
        if state.value != "ACTIVE":
            why.append(f"kill switch: system {state.value}" + (f" ({pause_reason})" if pause_reason else ""))
        s = d["symbol"]
        if s in quarantine and (quarantine[s] is None or str(quarantine[s])[:10] <= str(ns)):
            why.append("data-quality quarantine")
        if s in held:
            why.append("position already open in the book")
        if len(acts) and ns:
            m = (acts["symbol"] == s) & (pd.to_datetime(acts["ex_date"]).dt.strftime("%Y-%m-%d") == ns) & \
                (pd.to_datetime(acts["available_at"], utc=True) <= now)
            if bool(m.any()):
                why.append(f"corporate action with ex-date {ns} known by now: reference prices change at the open")
        if why:
            db.insert("exploration_events", {"decision_id": d["decision_id"], "event": "CANCELLED_PREOPEN", "at": now.isoformat(),
                                             "order_id": None, "details_json": to_json({"reasons": why}),
                                             "created_at": utcnow_iso()})
            res["cancelled"] += 1
            res["details"].append({"symbol": s, "event": "CANCELLED_PREOPEN", "reasons": why})
            continue
        db.insert("exploration_events", {"decision_id": d["decision_id"], "event": "REVALIDATED", "at": now.isoformat(),
                                         "order_id": None, "details_json": to_json({
                                             "required_checked": list(REQUIRED_AT_SUBMIT),
                                             "preopen_recheck": dict(rc) if rc else "none for this candidate",
                                             "latest_bar": str(last_bar), "optional_unknown": list(OPTIONAL_INPUTS)}),
                                         "created_at": utcnow_iso()})
        pre = from_json(d["pre_trade_json"], {}) or {}
        plan = TradePlan(entry_ref_price=d["ref_price"], stop_price=d["stop_price"], holding_sessions=int(d["holding_sessions"]),
                         invalidation="; ".join(pre.get("invalidation") or [])[:500])
        r = exec_service.submit_entry(s, float(d["qty"]), decision_id=d["decision_id"], plan=plan,
                                      journal={"mode": "PAPER_EXPLORATION", "decision_id": d["decision_id"],
                                               "reason": d["reason"], "strict_blocker": d["strict_blocker"],
                                               "unknowns": pre.get("unknowns"), "features": pre.get("features")},
                                      strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION,
                                      session_date=d["session_date"])
        ev = "REFUSED" if r.get("refused") else "SUBMITTED"
        db.insert("exploration_events", {"decision_id": d["decision_id"], "event": ev, "at": now.isoformat(),
                                         "order_id": r.get("order_id"), "details_json": to_json(r), "created_at": utcnow_iso()})
        res["refused" if ev == "REFUSED" else "submitted"] += 1
        res["details"].append({"symbol": s, "event": ev, "order_id": r.get("order_id"), "reason": r.get("reason")})
    return res


class ExplorationOutcomeTracker:
    """Forward outcomes of every exploration decision (traded or not): entry at the next open after
    the decision session, exit at the close h sessions later; cost-adjusted like the backtester."""

    def __init__(self, db, config):
        self.db = db
        self.costs = CostModel.from_config(config)

    def update(self, panel, as_of) -> int:
        from quantlab.discovery.research import _one_way
        as_of = pd.Timestamp(as_of)
        dates = panel.dates
        pos = {d: i for i, d in enumerate(dates)}
        last = int(dates.searchsorted(as_of, side="right")) - 1
        rows = self.db.fetchall("SELECT decision_id, session_date, symbol, stop_price, ref_price, pre_trade_json FROM "
                                "exploration_decisions WHERE session_date <= ?", (str(as_of.date()),))
        have = {(r["decision_id"], r["horizon_sessions"]) for r in self.db.fetchall(
            "SELECT decision_id, horizon_sessions FROM exploration_outcomes")}
        cols = {s: j for j, s in enumerate(panel.symbols)}
        ao, ac, ah, al = (panel.aopen.to_numpy(float), panel.aclose.to_numpy(float), panel.ahigh.to_numpy(float),
                          panel.alow.to_numpy(float))
        mkt = cols.get("SPY")
        out = []
        for r in rows:
            i = pos.get(pd.Timestamp(r["session_date"]))
            j = cols.get(r["symbol"])
            if i is None or j is None:
                continue
            pre = from_json(r["pre_trade_json"], {}) or {}
            adv = _f(((pre.get("volume") or {}).get("adv20")))
            ow = float(_one_way(np.array([adv if adv is not None else np.nan]), self.costs)[0])
            cdir = ((pre.get("catalyst") or {}).get("direction")) or "NONE"
            stop_ratio = (r["stop_price"] / r["ref_price"]) if (r["stop_price"] and r["ref_price"]) else None
            for h in HORIZONS:
                if (r["decision_id"], h) in have or i + h > last or i + 1 > last:
                    continue
                e, x = ao[i + 1, j], ac[i + h, j]
                if not (np.isfinite(e) and np.isfinite(x)) or e <= 0:
                    continue
                gross = x / e - 1
                cost = ow * (1 + x / e)
                spy = (ac[i + h, mkt] / ao[i + 1, mkt] - 1) if mkt is not None else None
                hi, lo = np.nanmax(ah[i + 1:i + h + 1, j]), np.nanmin(al[i + 1:i + h + 1, j])
                breached = None if stop_ratio is None else int(lo / ac[i, j] <= stop_ratio)
                persisted = None
                if cdir in ("POSITIVE", "NEGATIVE") and spy is not None and np.isfinite(spy):
                    persisted = int((gross - cost - spy > 0) == (cdir == "POSITIVE"))
                out.append({"decision_id": r["decision_id"], "horizon_sessions": h, "entry_date": str(dates[i + 1].date()),
                            "end_date": str(dates[i + h].date()), "entry_price": float(e), "gross_ret": float(gross),
                            "cost_ret": float(cost), "net_ret": float(gross - cost),
                            "spy_ret": float(spy) if spy is not None and np.isfinite(spy) else None,
                            "mfe": float(hi / e - 1), "mae": float(lo / e - 1), "stop_breached": breached,
                            "thesis_valid": None if breached is None else int(not breached),
                            "catalyst_persisted": persisted, "computed_at": utcnow_iso()})
        if out:
            self.db.insert_many("exploration_outcomes", out)
        return len(out)


def _stats(x: pd.Series, mfe: pd.Series | None = None, mae: pd.Series | None = None) -> dict[str, Any]:
    x = x.dropna()
    n = int(len(x))
    if not n:
        return {"n": 0}
    w, lo = x[x > 0], x[x <= 0]
    return {"n": n, "win_rate": float((x > 0).mean()), "mean": float(x.mean()), "median": float(x.median()),
            "avg_win": float(w.mean()) if len(w) else None, "avg_loss": float(lo.mean()) if len(lo) else None,
            "mfe": float(mfe.dropna().mean()) if mfe is not None and mfe.notna().any() else None,
            "mae": float(mae.dropna().mean()) if mae is not None and mae.notna().any() else None}


def experiment_results(db, horizon: int = 5) -> dict[str, Any]:
    """EXPLORATION vs STRICT vs SPY, and traded vs watched-not-traded vs rejected, with sample counts.
    Net returns from the next open, cost-adjusted. Small samples are shown as small samples."""
    q = db.query_df(
        "SELECT d.decision_id, d.mode, d.selection, d.symbol, d.setup_type, d.setup_class, d.families, d.catalyst_families, "
        "o.net_ret, o.spy_ret, o.mfe, o.mae, o.thesis_valid, o.catalyst_persisted, "
        "(SELECT e.event FROM exploration_events e WHERE e.decision_id=d.decision_id AND e.event IN "
        "('SUBMITTED','CANCELLED_PREOPEN','REFUSED') ORDER BY e.id DESC LIMIT 1) AS final_event "
        "FROM exploration_decisions d LEFT JOIN exploration_outcomes o ON o.decision_id=d.decision_id AND o.horizon_sessions=?",
        (horizon,))
    groups: dict[str, Any] = {}
    if len(q):
        traded = q[(q["selection"] == "SELECTED") & (q["final_event"] == "SUBMITTED")]
        groups["Exploratory paper trades"] = _stats(traded["net_ret"], traded["mfe"], traded["mae"])
        sh = q[q["selection"] == "SHADOW"]
        groups["Exploratory selections tracked in STRICT mode (not traded)"] = _stats(sh["net_ret"], sh["mfe"], sh["mae"])
        wn = q[q["selection"] == "WATCHED_NOT_TRADED"]
        groups["Watched but not traded"] = _stats(wn["net_ret"], wn["mfe"], wn["mae"])
        sk = q[q["selection"] == "SKIPPED"]
        groups["Skipped by exploration safety checks"] = _stats(sk["net_ret"], sk["mfe"], sk["mae"])
        sel = q[q["selection"].isin(["SELECTED", "SHADOW"])]
        groups["SPY over the same windows (selected)"] = _stats(sel["spy_ret"])
        tk = q[q["selection"] == "TRACKED"]
        if len(tk):
            groups["Tracked watchlist (research only, never traded)"] = _stats(tk["net_ret"], tk["mfe"], tk["mae"])
    dq = db.query_df("SELECT c.status, o.net_ret, o.mfe, o.mae FROM discovery_outcomes o JOIN discovery_candidates c "
                     "ON c.discovery_id=o.discovery_id WHERE o.horizon_sessions=?", (horizon,)) \
        if db.fetchone("SELECT 1 FROM sqlite_master WHERE name='discovery_outcomes'") else pd.DataFrame()
    if len(dq):
        rj = dq[dq["status"] == "REJECTED"]
        groups["Rejected discovery candidates"] = _stats(rj["net_ret"], rj["mfe"], rj["mae"])
        st = dq[dq["status"].isin(["PAPER_ELIGIBLE", "TRADED"])]
        groups["STRICT validated signals"] = _stats(st["net_ret"], st["mfe"], st["mae"])
    by: dict[str, dict[str, Any]] = {}
    if len(q):
        sel = q[q["selection"].isin(["SELECTED", "SHADOW"])]
        for key, col in (("discovery family", "families"), ("setup type", "setup_type"), ("catalyst type", "catalyst_families")):
            parts = {}
            for val, g in sel.assign(k=sel[col].fillna("none")).groupby("k"):
                parts[str(val)] = _stats(g["net_ret"])
            by[key] = parts
    trades = db.query_df("SELECT t.symbol, t.status, t.entry_date, t.entry_price, t.exit_date, t.exit_price, t.exit_reason, "
                         "t.qty, t.decision_id FROM trades t WHERE t.strategy_id=? ORDER BY t.entry_date DESC",
                         (STRATEGY_ID,))
    return {"horizon": horizon, "groups": groups, "by": by,
            "decisions": int(len(q)), "exploratory_trades": trades.to_dict("records") if len(trades) else [],
            "note": "net of modelled costs, entry at the next open; sample counts shown; no significance is implied"}


def _et_time(text: str) -> time:
    h, m = str(text).split(":")
    return time(int(h), int(m))


__all__ = ["ExplorationOutcomeTracker", "ExplorationPolicy", "MODES", "NOT_REQUIRED", "OPTIONAL_INPUTS",
           "REQUIRED_AT_PLAN", "REQUIRED_AT_SUBMIT", "STRATEGY_ID", "experiment_results", "open_exposure", "paper_mode",
           "pending_entries", "plan_exploration", "preopen_submit"]


def _r(x: Any, nd: int = 4) -> float | None:
    v = _f(x)
    return None if v is None else round(v, nd)


def learning_report(db, *, synthetic: bool = False, big_move: float = 0.10, limit: int = 25) -> dict[str, Any]:
    """Score every exploratory decision -- traded, watched AND skipped -- at the end of ITS OWN hold.

    * by selection x hold arm: net, vs SPY, how often +5/+8/+10% was touched, stop rate, MFE/MAE;
    * missed big winners: candidates NOT traded that touched +``big_move`` (with why they were passed over);
    * failed picks: traded candidates whose stop was breached;
    * calibration: the upside profile recorded at the decision vs what actually happened;
    * by_throttle / by_regime: the same outcomes split by the regime-throttle state and the regime label
      recorded on the decision (NOT_RECORDED = planned before the regime was recorded);
    * tracked_watchlist: the TRACKED records (never traded) pooled and per symbol, next to SELECTED and
      WATCHED_NOT_TRADED. TRACKED records are never "missed big winners" and stay out of the calibration.
    A decision appears only once its hold has fully elapsed (no partial windows)."""
    rows = db.fetchall(
        "SELECT d.symbol, d.session_date, d.selection, d.holding_sessions AS hold, d.setup_type, d.reason, "
        "d.pre_trade_json, o.net_ret, o.spy_ret, o.mfe, o.mae, o.stop_breached FROM exploration_decisions d "
        "JOIN exploration_outcomes o ON o.decision_id = d.decision_id AND o.horizon_sessions = d.holding_sessions "
        "WHERE d.is_synthetic = ? ORDER BY d.session_date", (int(bool(synthetic)),))
    if not rows:
        return {"matured": 0, "note": "no decision has reached the end of its holding period yet"}
    df = pd.DataFrame([dict(r) for r in rows])
    pre = df["pre_trade_json"].map(lambda s: from_json(s, {}) or {})
    df["selection_score"] = pre.map(lambda p: _f((p.get("candidate") or {}).get("selection_score")))
    ups = pre.map(lambda p: p.get("upside") or {})
    for t in (5, 10):
        df[f"pred_touch_{t}"] = ups.map(lambda u, t=t: _f(u.get(f"p_touch_{t}")) if u.get("state") == "KNOWN" else None)
    for c in ("net_ret", "spy_ret", "mfe", "mae"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["stopped"] = pd.to_numeric(df["stop_breached"], errors="coerce").fillna(0) >= 1
    reg = pre.map(lambda p: p.get("regime") if isinstance(p.get("regime"), dict) else None)
    df["regime_label"] = reg.map(lambda r: str(r.get("label") or "UNKNOWN") if r else "NOT_RECORDED")
    df["throttle_state"] = reg.map(lambda r: str(r.get("state") or "UNKNOWN") if r else "NOT_RECORDED")
    # throttle-displaced candidates (eligible, within the normal budget, not traded because of the throttle)
    # are the throttle's counterfactual: kept apart from the ordinary watched-not-traded
    df["arm"] = [("WATCHED_NOT_TRADED (throttle-displaced)" if (r and r.get("displaced_by_throttle")) else s)
                 for r, s in zip(reg, df["selection"])]

    def stats(g: pd.DataFrame) -> dict[str, Any]:
        return {"n": int(len(g)), "mean_net": _r(g["net_ret"].mean()),
                "median_net": _r(g["net_ret"].median()), "mean_vs_spy": _r((g["net_ret"] - g["spy_ret"]).mean()),
                "p_up_5": _r((g["mfe"] >= 0.05).mean()), "p_up_8": _r((g["mfe"] >= 0.08).mean()),
                "p_up_10": _r((g["mfe"] >= 0.10).mean()), "stop_rate": _r(g["stopped"].mean()),
                "mean_mfe": _r(g["mfe"].mean()), "mean_mae": _r(g["mae"].mean())}
    groups = []
    for (sel, hold), g in df.groupby(["selection", "hold"]):
        groups.append({"selection": sel, "hold": int(hold), **stats(g)})
    by_throttle = [{"throttle": t, "selection": a, **stats(g)} for (t, a), g in df.groupby(["throttle_state", "arm"])]
    by_regime = [{"regime": lab, "selection": a, **stats(g)} for (lab, a), g in df.groupby(["regime_label", "arm"])]
    tk = df[df["selection"] == "TRACKED"]
    tracked = None
    if len(tk):
        tracked = {
            "comparison": [{"selection": s, **stats(df[df["selection"] == s])}
                           for s in ("SELECTED", "WATCHED_NOT_TRADED", "TRACKED") if (df["selection"] == s).any()],
            "symbols": [{"symbol": sym, **stats(g), "last_session": str(g["session_date"].iloc[-1]),
                         "last_net": _r(g["net_ret"].iloc[-1]), "last_hold": int(g["hold"].iloc[-1])}
                        for sym, g in tk.groupby("symbol")],
            "note": "tracked symbols are recorded every session and never traded; each matures at its own hold arm"}

    def recs(x: pd.DataFrame) -> list[dict[str, Any]]:
        return [{"symbol": r["symbol"], "session": r["session_date"], "selection": r["selection"], "hold": int(r["hold"]),
                 "mfe": _r(r["mfe"]), "mae": _r(r["mae"]), "net": _r(r["net_ret"]),
                 "selection_score": _r(r["selection_score"], 3), "setup_type": r["setup_type"],
                 "why": str(r["reason"] or "")[:160]} for _, r in x.iterrows()]
    cand = df[df["selection"] != "TRACKED"]          # tracked records were never candidates passed over
    missed = cand[(cand["selection"] != "SELECTED") & (cand["mfe"] >= big_move)].sort_values("mfe", ascending=False)
    failed = df[(df["selection"] == "SELECTED") & df["stopped"]].sort_values("net_ret")
    calib = []
    for col, thr in (("pred_touch_5", 0.05), ("pred_touch_10", 0.10)):
        d = cand.dropna(subset=[col, "mfe"])
        if d.empty:
            continue
        d = d.assign(b=pd.cut(d[col], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True))
        for b, g in d.groupby("b", observed=True):
            calib.append({"target": f"+{int(thr * 100)}%", "predicted_bucket": str(b), "n": int(len(g)),
                          "mean_predicted": _r(g[col].mean()), "realized": _r((g["mfe"] >= thr).mean())})
    return {"matured": int(len(df)), "matured_tracked": int(len(tk)), "by_selection_and_hold": groups,
            "n_missed_big_winners": int(len(missed)), "missed_big_winners": recs(missed.head(limit)),
            "n_failed_picks": int(len(failed)), "failed_picks": recs(failed.head(limit)), "calibration": calib,
            "by_throttle": by_throttle, "by_regime": by_regime, "tracked_watchlist": tracked,
            "notes": [f"missed big winner = not traded, touched +{big_move:.0%} within its own hold",
                      "the upside profile forecasts move SIZE from volatility; calibration checks those base rates, "
                      "it is not a directional signal (docs/UPSIDE-EVIDENCE.md)",
                      "by_throttle / by_regime pool the hold arms; the regime throttle is unvalidated "
                      "(docs/REGIME-THROTTLE.md): these splits are its forward test"]}
