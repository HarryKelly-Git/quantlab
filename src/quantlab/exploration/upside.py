"""Upside profile of a candidate: calibrated base rates for its volatility profile and hold.

For a candidate's ATR% and 20/60-session range contraction (both known at the decision close) and
its holding period, :func:`upside_profile` returns the historical frequencies of touching +5 / +8 /
+10 / +15 %, of doing so before the 2 x ATR stop, of the stop being hit, the median sessions to each
target and the median drawdown on the way, from ``data/upside_table.json`` (built by
scripts/research/build_upside_table.py from a 334,901-observation PIT replay, 2021-03..2024-11; the
2025+ holdout was never read).

What it is NOT: an edge. The 2026-10-01 study (docs/UPSIDE-EVIDENCE.md) found that the
characteristics associated with big moves -- volatility, range compression, time since earnings,
distance from the 52-week high, volume spikes -- raise the odds of a big move DOWN about as much as
UP, and do not improve net return. The profile says how far a stock is likely to travel, not which
way. It is recorded on every decision so forecasts can be checked against outcomes
(:func:`quantlab.exploration.engine.learning_report`) and so the options layer has an empirical
move distribution to price structures against (:func:`move_distribution`).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

TABLE = Path(__file__).with_name("data") / "upside_table.json"
_FIELDS = ("n", "p_stop", "mean_net",
           *(f"{k}_{t}" for t in (5, 8, 10, 15) for k in ("p_touch", "p_clean", "p_down_first", "median_days_to",
                                                           "median_dd_before")))


@lru_cache(maxsize=1)
def _table() -> dict[str, Any]:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def _num(x: Any) -> float | None:
    if isinstance(x, dict):
        x = x.get("value")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _cell(atr_pct: Any, range_contraction: Any, hold: Any) -> tuple[str | None, dict[str, Any] | None, str]:
    t = _table()
    m = t["meta"]
    a, rc = _num(atr_pct), _num(range_contraction)
    try:
        h = int(hold)
    except (TypeError, ValueError):
        h = None
    if a is None or a <= 0:
        return None, None, "ATR% UNKNOWN"
    if rc is None:
        return None, None, "range contraction UNKNOWN"
    if h not in m["holds"]:
        return None, None, f"hold {hold} not in the table ({m['holds']})"
    key = f"{int(np.searchsorted(m['atr_pct_edges'], a, side='right'))}|" \
          f"{int(np.searchsorted(m['range_contraction_edges'], rc, side='right'))}|{h}"
    cell = t["cells"].get(key)
    return (key, cell, "") if cell else (None, None, f"no historical cell {key}")


def upside_profile(atr_pct: Any, range_contraction: Any, hold: Any) -> dict[str, Any]:
    """Base rates for this volatility profile at this hold. ``state`` is KNOWN or UNKNOWN (never zeros)."""
    version = _table()["meta"]["version"]
    key, cell, why = _cell(atr_pct, range_contraction, hold)
    if cell is None:
        return {"state": "UNKNOWN", "why": why, "version": version}
    return {"state": "KNOWN", "version": version, "cell": key, "hold": int(hold),
            **{f: cell.get(f) for f in _FIELDS},
            "meaning": "historical base rates for this volatility profile -- move size, not direction"}


def move_distribution(atr_pct: Any, range_contraction: Any, hold: Any) -> np.ndarray | None:
    """Equal-weight empirical net-return quantiles (stop + time exit, after costs) for the options
    comparator; None when the profile is UNKNOWN."""
    _, cell, _ = _cell(atr_pct, range_contraction, hold)
    return None if cell is None else np.asarray(cell["net_quantiles"], dtype=float)


__all__ = ["TABLE", "move_distribution", "upside_profile"]
