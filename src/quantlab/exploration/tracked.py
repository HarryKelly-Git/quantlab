"""TRACKED WATCHLIST: research tracking of owner-chosen symbols. NEVER traded, NEVER sized.

``exploration.tracked_watchlist`` (default empty; the owner's list belongs in the git-ignored
config/local.yaml). Each session, every tracked symbol present in the session's data gets one
``exploration_decisions`` row with ``selection = 'TRACKED'``: features, selection score and its
percentile in the scanned universe, the upside profile for its hold arm (``hold_for``), catalyst /
industry / context, and the regime -- like any other candidate -- but:

* qty 0, never SELECTED, no PLANNED event, so ``preopen_submit`` and ``_pending_symbols`` never see it;
* it is removed from the tradeable pool before selection and counts toward no cap (new entries,
  watched-not-traded, industry, exposure, considered); with an empty list planning is unchanged;
* its ``stop_price`` is a HYPOTHETICAL stop (close - stop_atr x ATR, the same rule as the traded picks)
  used only to score stop breaches in the outcome step; it is never an order.

Source of a tracked symbol's record, point-in-time at D: its discovery candidate row when it is in the
session's candidate pool (fired a family or is strategy-linked), else its row of the SAME discovery scan
(the basic-filter cross-section the selection score is computed over), else -- when it is in the data
but outside the scan (price / liquidity / history / benchmark) -- a record with UNKNOWN features and
the reason. Discovery scoring, ranks and statuses are not touched.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from quantlab.core.types import new_id
from quantlab.db.database import from_json, to_json, utcnow_iso
from quantlab.exploration.upside import upside_profile

TRACKED = "TRACKED"
NOT_TRADED_NOTE = "tracked watchlist: research only, never traded, never sized, outside every exploration cap"


def _f(x: Any) -> float | None:
    from quantlab.exploration.engine import _f as f
    return f(x)


def _universe_scores(scan) -> pd.Series:
    if scan is None or getattr(scan, "table", None) is None or "selection_score" not in scan.table.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(scan.table["selection_score"], errors="coerce").dropna()


def universe_percentile(scores: pd.Series, value: float | None) -> float | None:
    """Share of the scanned universe (symbols with a known selection score) at or below ``value``."""
    if value is None or not len(scores):
        return None
    return float((scores <= value + 1e-12).mean())


def tracked_decisions(*, pol, run, rows, order: dict[str, int], scan, now, mode: str, regime: dict[str, Any],
                      dataset_ids) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Decision rows for the tracked symbols of session ``run['as_of_date']`` and, for symbols that get
    none, why. ``rows`` = the session's discovery candidate rows, ``scan`` = the same session's
    discovery ScanResult (or None, e.g. a CLI plan: then only candidates in the pool are recorded)."""
    from quantlab.exploration.engine import hold_for, industry_code
    d = run["as_of_date"]
    if scan is not None and str(pd.Timestamp(scan.as_of).date()) != str(d)[:10]:
        scan = None                       # another session's scan is never used (point-in-time)
    scores = _universe_scores(scan)
    by_sym = {r["symbol"]: r for r in rows}
    table = scan.table if scan is not None else None
    not_scanned = (getattr(scan, "not_scanned", None) or {}) if scan is not None else {}
    out: list[dict[str, Any]] = []
    absent: dict[str, str] = {}
    for s in pol.tracked_watchlist:
        cand = by_sym.get(s)
        in_scan = table is not None and s in table.index
        if cand is not None:
            source = "discovery candidate (in the session's candidate pool)"
            setup = from_json(cand["setup_json"], {}) or {}
            lv = setup.get("levels") or {}
            facts = (from_json(cand["factors_json"], {}) or {}).get("features", {}) or {}
            cat = from_json(cand["catalyst_record_json"], {}) or {}
            context = from_json(cand["catalyst_json"], {}) or {}
            checks = {c["name"]: c for c in (from_json(cand["checks_json"], []) or [])}
            uni = (checks.get("research_universe") or {}).get("reason")
            sel = _f(setup.get("selection_score"))
            info = {"origin": cand["origin"], "rank": cand["rank"], "discovery_score": cand["discovery_score"],
                    "status": cand["status"], "setup_type": setup.get("setup_type"), "setup_class": cand["setup_class"],
                    "families": (from_json(cand["families_json"], {}) or {}).get("fired", []),
                    "catalyst_families": cand["catalyst_families"], "direction_bias": cand["direction_bias"]}
        elif in_scan:
            source = "discovery scan (basic-filter universe; not in the candidate pool: no family fired, no strategy link)"
            t = table.loc[s]
            lv = t["levels"] if isinstance(t["levels"], dict) else {}
            facts = t["factors"] if isinstance(t["factors"], dict) else {}
            cat = t["catalyst"] if isinstance(t.get("catalyst"), dict) else {}
            context = t["context"] if isinstance(t.get("context"), dict) else {}
            uni = str(t.get("universe_reason") or "") or None
            sel = _f(t["selection_score"])
            info = {"origin": TRACKED, "rank": None, "discovery_score": _f(t["score"]), "status": None,
                    "setup_type": None, "setup_class": None, "families": list(t["fired"] or []),
                    "catalyst_families": ",".join(t.get("catalyst_fired") or []) or None,
                    "direction_bias": t.get("bias")}
        elif scan is not None and s in not_scanned and not_scanned[s] != "no bar on session":
            source = f"in the data, outside the discovery scan ({not_scanned[s]}): features UNKNOWN"
            lv, facts, cat, context, uni, sel = {}, {}, {}, {}, f"not scanned: {not_scanned[s]}", None
            info = {"origin": TRACKED, "rank": None, "discovery_score": None, "status": None, "setup_type": None,
                    "setup_class": None, "families": [], "catalyst_families": None, "direction_bias": None}
        else:
            absent[s] = (f"not recorded: {not_scanned[s]}" if s in not_scanned else
                         "not recorded: not in the session's data" if scan is not None else
                         "not recorded: outside the candidate pool and no discovery scan was passed (the daily "
                         "pipeline passes it)")
            continue
        if in_scan and cand is not None:
            sel_u = _f(table.at[s, "selection_score"])
        else:
            sel_u = sel
        pct = universe_percentile(scores, sel_u)
        hold = hold_for(d, s, pol)
        close, atr = _f(lv.get("close")), _f(lv.get("atr"))
        stop = round(close - pol.stop_atr * atr, 4) if (close and atr and atr > 0 and close - pol.stop_atr * atr > 0) else None
        up = upside_profile(facts.get("atr14_pct"), facts.get("range_contraction_20_60"), hold)
        ind = cat.get("industry") if cat else None
        reason = (f"{NOT_TRADED_NOTE}; selection score {f'{sel:.2f}' if sel is not None else 'UNKNOWN'}"
                  f" (universe percentile {f'{pct:.0%}' if pct is not None else 'UNKNOWN'}); scored over a "
                  f"{hold}-session hold; source: {source}")
        pre = {
            "ticker": s, "decided_at": pd.Timestamp(now).isoformat(), "mode": mode, "selection": TRACKED,
            "tracked": {"never_traded": True, "source": source, "universe_percentile": pct,
                        "universe_n": int(len(scores)) or None,
                        "percentile_basis": "share of the session's scanned symbols with a known selection score at "
                                            "or below this one" if len(scores) else "UNKNOWN: no scan for the session"},
            "session_date": d, "next_session": run["next_session"], "information_cutoff_at": run["info_cutoff_at"],
            "discovery_run_id": run["discovery_run_id"], "discovery_id": cand["discovery_id"] if cand is not None else None,
            "candidate": {**info, "selection_score": sel, "selection_order": order.get(s),
                          "on_watchlist": bool(cand["on_watchlist"]) if cand is not None else False},
            "price": {"close": close, "atr": atr, "levels": lv},
            "volume": {"adv20": _f((facts.get("adv20") or {}).get("value")),
                       "rel_volume_1d": (facts.get("rel_volume_1d") or {}).get("value")},
            "catalyst": {"families": cat.get("families"), "post_earnings": cat.get("post_earnings"),
                         "material_event": cat.get("material_event")} if cat else None,
            "industry": ind, "industry_code": industry_code(cat) if cat else None,
            "fundamentals": cat.get("fundamentals") if cat else None, "context": context or None,
            "universe": uni, "reason_for_entering": reason,
            "expected_holding_sessions": hold, "upside": up,
            "hypothetical_stop": {"price": stop, "rule": f"close - {pol.stop_atr:g} x ATR",
                                  "use": "outcome scoring (stop breach) only: never an order"},
            "sizing": {"qty": 0, "notional": 0.0, "note": "tracked: never sized"},
            "expected": "research tracking: measured at 1/3/5/10/20 sessions vs SPY like every other decision",
            "data_timestamps": {"information_cutoff_at": run["info_cutoff_at"], "decision_session_bar": d,
                                "dataset_ids": dataset_ids},
            "features": {"discovery": facts, "catalyst": (cat or {}).get("features")},
            "regime": {**regime, "displaced_by_throttle": False},
        }
        out.append({"decision_id": new_id("expl"), "session_date": d, "next_session": run["next_session"], "symbol": s,
                    "mode": mode, "selection": TRACKED, "rank": info["rank"], "discovery_run_id": run["discovery_run_id"],
                    "discovery_id": cand["discovery_id"] if cand is not None else None, "origin": info["origin"],
                    "setup_type": info["setup_type"], "setup_class": info["setup_class"],
                    "families": ",".join(info["families"]) or None, "catalyst_families": info["catalyst_families"],
                    "discovery_score": _f(info["discovery_score"]), "ref_price": close, "qty": 0.0, "stop_price": stop,
                    "holding_sessions": hold, "strict_blocker": None, "reason": reason[:2000],
                    "info_cutoff_at": run["info_cutoff_at"], "pre_trade_json": to_json(pre),
                    "is_synthetic": int(run["is_synthetic"]), "created_at": utcnow_iso()})
    return out, absent


__all__ = ["NOT_TRADED_NOTE", "TRACKED", "tracked_decisions", "universe_percentile"]
