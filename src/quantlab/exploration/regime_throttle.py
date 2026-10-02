"""Market-regime RISK THROTTLE for the exploration budget (paper only; a throttle, not a ban).

When the configured market metric of the decision session D (default ``market_trend_200`` = SPY vs its
200-session average, from ``regime_snapshots``) is below the threshold, the number of NEW exploratory
entries for D is capped at ``min(max_new_per_session, regime_throttle.max_new_per_session)``. Eligible
candidates beyond that cap, up to the normal budget, are recorded as WATCHED_NOT_TRADED with a reason
naming the throttle, so the forward evidence for and against the throttle accumulates.

Point-in-time: only the snapshot whose ``as_of_date`` is exactly D is read. The daily pipeline writes it
in the ``research`` step, before ``discover`` and ``explore`` (pipeline/daily.py STEPS), from the view
truncated to D; its inputs are trailing market features. A snapshot for a later date is never read and
an older one is never carried forward: no snapshot for D (or an unknown metric) = UNKNOWN = no
throttle, recorded as such. Evidence and status: docs/REGIME-THROTTLE.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from quantlab.db.database import from_json

STATES = ("THROTTLED", "NOT_THROTTLED", "UNKNOWN", "DISABLED")


@dataclass(frozen=True)
class RegimeThrottle:
    enabled: bool = False
    metric: str = "market_trend_200"
    below: float = 0.0
    max_new_per_session: int = 2

    @classmethod
    def from_config(cls, config) -> "RegimeThrottle":
        c = dict(config.get("exploration.regime_throttle", {}) or {})
        en = c.get("enabled", cls.enabled)
        enabled = en.strip().lower() in ("1", "true", "yes", "on") if isinstance(en, str) else bool(en)
        return cls(enabled=enabled, metric=str(c.get("metric", cls.metric)), below=float(c.get("below", cls.below)),
                   max_new_per_session=max(int(c.get("max_new_per_session", cls.max_new_per_session)), 0))


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def session_regime(db, session: str) -> dict[str, Any] | None:
    """The regime snapshot recorded FOR session ``session`` (exact date), or None. Never a later or an
    earlier session's snapshot."""
    row = db.fetchone("SELECT as_of_date, label, metrics_json, created_at FROM regime_snapshots WHERE as_of_date=?",
                      (str(session)[:10],))
    if row is None:
        return None
    return {"as_of_date": row["as_of_date"], "label": row["label"], "metrics": from_json(row["metrics_json"], {}) or {},
            "recorded_at": row["created_at"]}


def throttle_state(db, session: str, policy_max_new: int, throttle: RegimeThrottle) -> dict[str, Any]:
    """The regime context and the effective new-entry cap for session ``session``. Recorded on every
    exploration decision of that session (``pre_trade_json["regime"]``)."""
    snap = session_regime(db, session)
    metrics = (snap or {}).get("metrics") or {}
    value = _num(metrics.get(throttle.metric))
    out: dict[str, Any] = {
        "as_of_date": snap["as_of_date"] if snap else None,
        "label": (snap["label"] or "UNKNOWN") if snap else "UNKNOWN",
        "market_trend_200": _num(metrics.get("market_trend_200")),
        "metric": throttle.metric, "value": value, "below": throttle.below,
        "policy_max_new": int(policy_max_new), "throttle_max_new": int(throttle.max_new_per_session),
        "throttled": False, "effective_max_new": int(policy_max_new),
        "source": "regime_snapshots (as_of_date = decision session)" if snap else
                  f"no regime snapshot for {str(session)[:10]}: UNKNOWN (a later or earlier snapshot is never used)",
    }
    if not throttle.enabled:
        out["state"] = "DISABLED"
    elif value is None:
        out["state"] = "UNKNOWN"
    elif value < throttle.below:
        out.update(state="THROTTLED", throttled=True,
                   effective_max_new=int(min(int(policy_max_new), int(throttle.max_new_per_session))))
    else:
        out["state"] = "NOT_THROTTLED"
    return out


def throttle_text(reg: dict[str, Any]) -> str:
    """One line naming the throttle, for decision reasons."""
    v = reg.get("value")
    return (f"regime throttle: {reg.get('metric')} {f'{v:+.1%}' if v is not None else 'UNKNOWN'} < {reg.get('below'):+.1%} "
            f"on {reg.get('as_of_date')} ({reg.get('label')}): at most {reg.get('effective_max_new')} new instead of "
            f"{reg.get('policy_max_new')}")


__all__ = ["RegimeThrottle", "STATES", "session_regime", "throttle_state", "throttle_text"]
