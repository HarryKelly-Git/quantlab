"""Evidence packets: the ONLY information an LLM role is allowed to see about a candidate.

Why a packet (instead of letting the model "look things up"):
* Point-in-time. Everything in the packet was knowable at ``information_cutoff`` (= cutoff of the
  candidate's session, ARCHITECTURE.md section 2). :func:`build_evidence_packet` walks every
  timestamp in the packet and raises :class:`EvidenceLeakError` if anything is later. It never
  silently drops a future item: a future item reaching this point is an upstream PIT bug.
* No invented numbers. Every number was computed by Python upstream. The hallucination guard in
  :mod:`quantlab.ai.roles` checks that LLM citations (``source_field`` paths) resolve with
  :func:`resolve_path` and that numbers in free text appear in :func:`packet_numbers`.
* Transparency. Each section carries an :class:`~quantlab.core.types.InfoKind` label in
  ``packet["info_kinds"]``; missing sections are the literal string ``"UNKNOWN"`` and labeled
  UNCERTAINTY.
* Reproducibility. The packet is canonical JSON with a deterministic ``packet_hash`` (identifiers
  such as ``candidate_id``/``created_at`` are deliberately excluded, so re-running the same session
  yields the same hash and the LLM cache can replay the stored answer instead of re-querying).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from datetime import date, datetime, time as dtime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.calendar import MARKET_CLOSE, MARKET_TZ, TradingCalendar, to_session
from quantlab.core.types import (
    HISTORICAL_RESEARCH_PIT,
    Candidate,
    CheckResult,
    CheckSeverity,
    InfoKind,
    PitStatus,
    new_id,
)
from quantlab.db.database import Database, utcnow_iso

PACKET_VERSION = "1.0"
UNKNOWN = "UNKNOWN"

CONTEXT_SECTIONS = ("features", "ml_predictions", "regime", "sector_context", "strategy_history",
                    "expected_value", "no_trade_checks", "events", "news", "fundamentals")

SECTION_KINDS: dict[str, InfoKind] = {
    "meta": InfoKind.FACT,
    "candidate": InfoKind.MODEL_OUTPUT,          # a strategy's opinion (score, reasons)
    "candidate.symbol": InfoKind.FACT,
    "candidate.as_of_date": InfoKind.FACT,
    "proposed_trade": InfoKind.MODEL_OUTPUT,
    "features": InfoKind.MODEL_OUTPUT,           # deterministic computations on FACT data
    "ml_predictions": InfoKind.MODEL_OUTPUT,
    "regime": InfoKind.MODEL_OUTPUT,
    "sector_context": InfoKind.MODEL_OUTPUT,
    "strategy_history": InfoKind.MODEL_OUTPUT,
    "expected_value": InfoKind.MODEL_OUTPUT,
    "no_trade_checks": InfoKind.MODEL_OUTPUT,
    "events": InfoKind.FACT,
    "news": InfoKind.FACT,
    "fundamentals": InfoKind.FACT,
    "data_quality": InfoKind.MODEL_OUTPUT,
    "data_quality.unknown_sections": InfoKind.UNCERTAINTY,
}

# Keys holding instants (tz-aware, compared with the cutoff) and session dates (compared with as_of).
TIMESTAMP_KEYS = frozenset({"time", "event_time", "created_at", "updated_at", "available_at",
                            "published_at", "timestamp", "filed_at", "accepted_at"})
DATE_KEYS = frozenset({"reaction_date", "filed_date", "data_as_of", "ex_date", "date"})
# Never put these in a packet: retrieval time reveals "today" to the model, payloads can carry
# synthetic ground truth, URLs invite the model to "browse".
DROP_KEYS = frozenset({"retrieved_at", "payload_json", "url", "candidate_id", "created_at_db"})

_DEFAULT_LIMITS = {"max_news_items": 20, "max_events": 20, "max_text_chars": 300, "max_string_chars": 2000}


class EvidenceError(ValueError):
    """The packet cannot be built safely (bad input); never silently repaired."""


class EvidenceLeakError(EvidenceError):
    """Something in the packet post-dates the information cutoff (point-in-time violation)."""


# ================================================================================================
# Building
# ================================================================================================
def build_evidence_packet(candidate: Candidate, context: dict[str, Any] | None = None,
                          config: Any = None) -> dict[str, Any]:
    """Assemble the evidence packet for one candidate.

    ``context`` keys (all optional; a missing key becomes ``"UNKNOWN"``): features, ml_predictions,
    regime, sector_context, strategy_history, expected_value, no_trade_checks, events, news,
    fundamentals, proposed_trade, data_as_of, information_cutoff, pit_status, is_synthetic.
    """
    ctx = dict(context or {})
    limits = _limits(config)
    as_of = to_session(candidate.as_of_date).date()
    cutoff = _information_cutoff(as_of, ctx.get("information_cutoff"), config)

    statuses = [PitStatus(candidate.pit_status)]
    if ctx.get("pit_status") is not None:
        statuses.append(PitStatus(ctx["pit_status"]))
    pit = PitStatus.weakest(statuses)

    features = ctx.get("features")
    if features is None and candidate.features:
        features = candidate.features
    plan = ctx.get("proposed_trade", candidate.plan)

    packet: dict[str, Any] = {
        "meta": {
            "packet_version": PACKET_VERSION,
            "as_of_date": as_of.isoformat(),
            "information_cutoff": cutoff.isoformat(),
            "data_as_of": _clean(ctx.get("data_as_of"), limits),
            "pit_status": pit.value,
            "is_synthetic": _clean(ctx.get("is_synthetic"), limits),
            "numbers_computed_by": "python",
        },
        "candidate": {
            "symbol": candidate.symbol,
            "as_of_date": as_of.isoformat(),
            "strategy_id": candidate.strategy_id,
            "strategy_version": candidate.strategy_version,
            "direction": _clean(candidate.direction, limits),
            "score": _clean(candidate.score, limits),
            "reasons": _clean(list(candidate.reasons), limits),
            "risk": _clean(dict(candidate.risk), limits) if candidate.risk else UNKNOWN,
            "pit_status": PitStatus(candidate.pit_status).value,
        },
        "proposed_trade": _clean(plan, limits),
        "features": _clean(features, limits),
        "ml_predictions": _clean(ctx.get("ml_predictions"), limits),
        "regime": _clean(ctx.get("regime"), limits),
        "sector_context": _clean(ctx.get("sector_context"), limits),
        "strategy_history": _clean(ctx.get("strategy_history"), limits),
        "expected_value": _clean(ctx.get("expected_value"), limits),
        "no_trade_checks": _clean(ctx.get("no_trade_checks"), limits),
        "events": _recent(_clean(ctx.get("events"), limits), ("time", "available_at"), limits["max_events"]),
        "news": _recent(_clean(_news_items(ctx.get("news"), limits), limits), ("created_at", "available_at"),
                        limits["max_news_items"]),
        "fundamentals": _clean(ctx.get("fundamentals"), limits),
    }
    for section in ("events", "news"):
        total = len(ctx[section]) if isinstance(ctx.get(section), (list, tuple)) else None
        packet["meta"][f"{section}_total_items"] = UNKNOWN if total is None else total

    _check_pit(packet, pd.Timestamp(cutoff), as_of, "packet")
    packet["data_quality"] = _data_quality(packet, pit)
    packet["info_kinds"] = _info_kinds(packet)
    packet["packet_hash"] = compute_packet_hash(packet)
    return packet


def _limits(config: Any) -> dict[str, int]:
    out = dict(_DEFAULT_LIMITS)
    if config is not None:
        for k in out:
            out[k] = int(config.get(f"ai.evidence.{k}", out[k]))
    return out


def _information_cutoff(as_of: date, given: Any, config: Any) -> datetime:
    """cutoff(D) per ARCHITECTURE.md section 2. A supplied cutoff may be EARLIER (more conservative)
    but never later than the rule allows."""
    cutoff_time = MARKET_CLOSE
    if config is not None:
        hh, mm = str(config.get("project.info_cutoff_local_time", "16:00")).split(":")
        cutoff_time = dtime(int(hh), int(mm))
    rule = TradingCalendar([as_of], cutoff_time=cutoff_time, tz=MARKET_TZ).cutoff(as_of)
    if given is None:
        return rule.to_pydatetime()
    ts = pd.Timestamp(given)
    if ts.tzinfo is None:
        raise EvidenceError("information_cutoff must be timezone-aware")
    ts = ts.tz_convert("UTC")
    if ts > rule:
        raise EvidenceLeakError(f"information_cutoff {ts.isoformat()} is later than cutoff({as_of}) "
                                f"{rule.isoformat()}")
    return ts.to_pydatetime()


def _news_items(news: Any, limits: dict[str, int]) -> Any:
    """Keep only headline-level fields; headlines are truncated (they are untrusted third-party text)."""
    if not isinstance(news, (list, tuple)):
        return news
    out = []
    n = limits["max_text_chars"]
    for item in news:
        if not isinstance(item, dict):
            raise EvidenceError("news items must be dicts with headline/created_at/source")
        row = {k: v for k, v in item.items() if k not in DROP_KEYS and k != "summary"}
        if isinstance(row.get("headline"), str) and len(row["headline"]) > n:
            row["headline"] = row["headline"][:n] + "..."
        out.append(row)
    return out


def _recent(items: Any, keys: tuple[str, ...], limit: int) -> Any:
    """Most recent first, capped. Counts of dropped items are recorded in meta.*_total_items."""
    if not isinstance(items, list):
        return items

    def sort_key(item: Any) -> str:
        if isinstance(item, dict):
            for k in keys:
                v = item.get(k)
                if isinstance(v, str) and v != UNKNOWN:
                    return v
        return ""

    return sorted(items, key=sort_key, reverse=True)[:limit]


def _clean(v: Any, limits: dict[str, int] | None = None) -> Any:
    """JSON-safe, deterministic representation. Missing / non-finite -> "UNKNOWN"."""
    limits = limits or _DEFAULT_LIMITS
    if v is None or v is pd.NaT or v is pd.NA:
        return UNKNOWN
    if isinstance(v, bool) or isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating, Decimal)):
        f = float(v)
        return f if math.isfinite(f) else UNKNOWN
    if isinstance(v, str):
        n = limits["max_string_chars"]
        return v if len(v) <= n else v[:n] + "..."
    if isinstance(v, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(v)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").isoformat()
        if ts == ts.normalize():
            return ts.date().isoformat()          # naive midnight = a session date
        return ts.isoformat()                     # naive instant: rejected by the PIT check if timed
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, CheckResult):
        return {"name": v.name, "passed": bool(v.passed), "severity": _clean(v.severity), "reason": v.reason,
                "details": _clean(v.details, limits), "blocking": bool(v.blocking)}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _clean({f.name: getattr(v, f.name) for f in dataclasses.fields(v)}, limits)
    if hasattr(v, "model_dump"):
        return _clean(v.model_dump(), limits)
    if isinstance(v, dict):
        return {str(k): _clean(val, limits) for k, val in v.items() if str(k) not in DROP_KEYS}
    if isinstance(v, (list, tuple, set, frozenset)):
        seq = sorted(v, key=str) if isinstance(v, (set, frozenset)) else v
        return [_clean(x, limits) for x in seq]
    if isinstance(v, (pd.Series,)):
        return _clean(v.to_dict(), limits)
    return str(v)


def _check_pit(node: Any, cutoff: pd.Timestamp, as_of: date, path: str) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}"
            if k in TIMESTAMP_KEYS or k in DATE_KEYS:
                _check_value(k, v, cutoff, as_of, p)
            _check_pit(v, cutoff, as_of, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_pit(v, cutoff, as_of, f"{path}[{i}]")


def _check_value(key: str, v: Any, cutoff: pd.Timestamp, as_of: date, path: str) -> None:
    if not isinstance(v, str) or v == UNKNOWN:
        return
    try:
        ts = pd.Timestamp(v)
    except (ValueError, TypeError):
        raise EvidenceError(f"{path}: unparseable timestamp {v!r}") from None
    if ts.tzinfo is not None:
        if ts.tz_convert("UTC") > cutoff:
            raise EvidenceLeakError(f"{path}={v} is after information_cutoff {cutoff.isoformat()}")
        return
    if len(v) <= 10 or ts == ts.normalize():
        if ts.date() > as_of:
            raise EvidenceLeakError(f"{path}={v} is after as_of_date {as_of.isoformat()}")
        return
    raise EvidenceError(f"{path}: naive timestamp {v!r} (availability timestamps must be timezone-aware)")


def _data_quality(packet: dict[str, Any], pit: PitStatus) -> dict[str, Any]:
    unknown_sections = sorted(k for k in (*CONTEXT_SECTIONS, "proposed_trade") if packet.get(k) == UNKNOWN)
    checks = packet.get("no_trade_checks")
    if isinstance(checks, list):
        critical_failed: Any = sorted(c.get("name", "?") for c in checks
                                      if isinstance(c, dict) and _is_blocking(c))
        liq = [c for c in checks if isinstance(c, dict) and _is_liquidity_check(c)]
        liquidity_ok: Any = UNKNOWN if not liq else all(bool(c.get("passed")) for c in liq)
    else:
        critical_failed = UNKNOWN
        liquidity_ok = UNKNOWN
    return {
        "pit_status": pit.value,
        "pit_usable_for_research": pit in HISTORICAL_RESEARCH_PIT,
        "unknown_sections": unknown_sections,
        "critical_checks_failed": critical_failed,
        "liquidity_ok": liquidity_ok,
    }


def _is_blocking(check: dict[str, Any]) -> bool:
    return check.get("passed") is False and str(check.get("severity", "")).upper() == CheckSeverity.CRITICAL.value


def _is_liquidity_check(check: dict[str, Any]) -> bool:
    name = str(check.get("name", "")).lower()
    return any(t in name for t in ("liquid", "dollar_volume", "adv"))


def _info_kinds(packet: dict[str, Any]) -> dict[str, str]:
    kinds = {k: v.value for k, v in SECTION_KINDS.items()}
    for section in (*CONTEXT_SECTIONS, "proposed_trade"):
        if packet.get(section) == UNKNOWN:
            kinds[section] = InfoKind.UNCERTAINTY.value
    return kinds


# ================================================================================================
# Hashing / persistence
# ================================================================================================
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compute_packet_hash(packet: dict[str, Any]) -> str:
    body = {k: v for k, v in packet.items() if k != "packet_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def verify_packet(packet: dict[str, Any]) -> str:
    """Return the hash, raising EvidenceError if the packet was modified after it was built."""
    h = compute_packet_hash(packet)
    if packet.get("packet_hash") != h:
        raise EvidenceError("packet_hash mismatch: packet was modified after build_evidence_packet()")
    return h


def packet_for_prompt(packet: dict[str, Any]) -> str:
    """Canonical JSON with '<', '>' and '&' escaped as JSON unicode escapes, so untrusted strings
    (news headlines) cannot close the <evidence_packet> tag and inject instructions."""
    s = canonical_json(packet)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def persist_packet(db: Database, candidate_id: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Insert into evidence_packets (immutable). Idempotent per (candidate_id, packet_hash)."""
    h = verify_packet(packet)
    existing = db.fetchone("SELECT * FROM evidence_packets WHERE candidate_id=? AND packet_hash=? "
                           "ORDER BY created_at LIMIT 1", (candidate_id, h))
    if existing is not None:
        return existing
    row = {"packet_id": new_id("pkt"), "candidate_id": candidate_id, "created_at": utcnow_iso(),
           "packet_hash": h, "packet_json": canonical_json(packet)}
    db.insert("evidence_packets", row)
    return row


# ================================================================================================
# Queries used by the hallucination guard / roles
# ================================================================================================
_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[\d+\]")


def resolve_path(packet: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve 'features.ret_20d' / 'news[0].headline' / 'news.0.headline' in the packet."""
    if not isinstance(path, str):
        return False, None
    p = path.strip()
    for prefix in ("packet.", "$."):
        if p.startswith(prefix):
            p = p[len(prefix):]
    if not p or p.startswith(".") or ".." in p:
        return False, None
    tokens = _PATH_TOKEN.findall(p)
    if "".join(tokens).replace("[", "").replace("]", "") != p.replace(".", "").replace("[", "").replace("]", ""):
        return False, None
    node: Any = packet
    for tok in tokens:
        if tok.startswith("["):
            tok = tok[1:-1]
        if isinstance(node, dict):
            if tok not in node:
                return False, None
            node = node[tok]
        elif isinstance(node, list):
            if not tok.isdigit() or int(tok) >= len(node):
                return False, None
            node = node[int(tok)]
        else:
            return False, None
    return True, node


def info_kind_of(packet: dict[str, Any], path: str) -> str:
    """InfoKind label of a path: longest matching prefix in packet['info_kinds']."""
    kinds = packet.get("info_kinds", {})
    p = path.replace("[", ".").replace("]", "")
    parts = p.split(".")
    for i in range(len(parts), 0, -1):
        k = ".".join(parts[:i])
        if k in kinds:
            return kinds[k]
    return InfoKind.UNCERTAINTY.value


_NUM_IN_STR = re.compile(r"\d+(?:\.\d+)?")
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})")
_SKIP_NUMBER_KEYS = frozenset({"packet_hash", "info_kinds"})


def packet_numbers(packet: Any) -> list[float]:
    """Every number that appears in the packet: numeric values, numbers inside string values
    (dates, versions, headlines) and numbers inside keys (e.g. 20 from 'ret_20d')."""
    found: set[float] = set()

    def add_str(s: str) -> None:
        for m in _NUM_IN_STR.findall(s):
            try:
                found.add(abs(float(m)))
            except ValueError:
                pass

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _SKIP_NUMBER_KEYS or k.endswith("_id"):
                    continue
                add_str(k)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            if math.isfinite(float(node)):
                found.add(abs(float(node)))
        elif isinstance(node, str):
            add_str(node)

    walk(packet)
    return sorted(found)


def packet_dates(packet: Any) -> set[str]:
    """All ISO dates (YYYY-MM-DD) appearing in string values of the packet."""
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            out.update(_ISO_DATE.findall(node))

    walk(packet)
    return out


def blocking_checks(packet: dict[str, Any]) -> list[str]:
    """Names of CRITICAL no-trade checks that failed (hard risk constraints the judge cannot override)."""
    checks = packet.get("no_trade_checks")
    if not isinstance(checks, list):
        return []
    return sorted(str(c.get("name", "?")) for c in checks if isinstance(c, dict) and _is_blocking(c))


# ================================================================================================
# PIT context from a DataBundle (events / news / fundamentals for one symbol)
# ================================================================================================
def bundle_context(bundle: Any, symbol: str, as_of: Any, max_news: int = 20, max_events: int = 20) -> dict[str, Any]:
    """Events, news and as-of fundamentals for ``symbol`` knowable at cutoff(as_of).

    Always works on ``bundle.truncate(as_of)`` so the result is identical whether it is given the
    full-history bundle or an already truncated one (truncation-invariance test in tests/ai).
    """
    d = to_session(as_of)
    view = bundle.truncate(d)
    cutoff = view.calendar.cutoff(d)

    ev = view.events
    events: list[dict[str, Any]] = []
    if not ev.empty:
        sub = ev[ev["symbol"] == symbol].sort_values(["available_at", "source_id"], ascending=[False, True])
        for r in sub.head(max_events).itertuples(index=False):
            events.append({"type": r.event_type, "time": r.event_time, "reaction_date": r.reaction_date,
                           "available_at": r.available_at, "pit_status": r.pit_status})

    nw = view.news
    news: list[dict[str, Any]] = []
    if not nw.empty:
        sub = nw[nw["symbol"] == symbol].sort_values(["available_at", "news_id"], ascending=[False, True])
        for r in sub.head(max_news).itertuples(index=False):
            news.append({"headline": r.headline, "created_at": r.created_at, "source": r.source,
                         "available_at": r.available_at, "pit_status": r.pit_status})

    fundamentals: dict[str, Any] = {}
    fu = view.fundamentals
    if not fu.empty:
        sub = fu[fu["symbol"] == symbol]
        # As-of rule (ARCHITECTURE.md 2.7): latest period, value from its most recent filing so far.
        sub = sub.sort_values(["concept", "period_end", "available_at", "accession"])
        for concept, grp in sub.groupby("concept", sort=True):
            r = grp.iloc[-1]
            fundamentals[str(concept)] = {
                "value": r["value"], "unit": r["unit"], "period_start": r["period_start"],
                "period_end": r["period_end"], "form": r["form"], "filed_date": r["filed_date"],
                "available_at": r["available_at"], "pit_status": r["pit_status"],
            }

    dates = view.panel.dates
    return {
        "events": events,
        "news": news,
        "fundamentals": fundamentals if fundamentals else None,
        "information_cutoff": cutoff,
        "data_as_of": dates[-1] if len(dates) else None,
        "is_synthetic": bool(bundle.is_synthetic),
    }
