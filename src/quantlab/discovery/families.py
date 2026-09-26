"""Discovery families: fixed, explainable descriptions of what looks interesting in today's market.

Discovery is NOT validation. Nothing in this package can permit, size or place a trade; it only
ranks and explains. Every number comes from the existing FeatureSet (features/, ARCHITECTURE.md
section 4) computed on the point-in-time view of the session.

SCORED families (price/volume data, full-universe real coverage). Each contributes 0..20 points,
from cross-sectional percentile ranks among the scanned symbols. Fixed definitions, fixed equal
weights, never fitted to results:
  momentum              ret_20d, ret_60d, ret_120d
  relative_strength     rs_spy_20, rs_spy_63                 (sector-relative is a CONTEXT family)
  volume_activity       rel_volume_1d, rel_volume_5d         (direction-agnostic)
  breakout_compression  breakout_55, -range_contraction_20_60
  mean_reversion        -ret_z_3d, -dist_ma20                (oversold magnitude)

CONTEXT families (earnings, news, fundamentals, sector). Recorded only for symbols the data source
actually covers; everything else is UNKNOWN, never 0. They NEVER enter the discovery score (the
real universe has almost no coverage for them; see the feature-coverage panel).

Feature states per symbol: VALID (finite value); UNKNOWN (not available: too little history for the
feature's lookback, or the source does not cover the symbol); INVALID (the inputs should exist but
the value is not finite: a data-quality issue, reported separately).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

SCORED = ("momentum", "relative_strength", "volume_activity", "breakout_compression", "mean_reversion")
CONTEXT = ("earnings", "news", "fundamentals", "sector")
POINTS = 20.0

LABELS = {
    "momentum": "Momentum", "relative_strength": "Relative strength", "volume_activity": "Volume/activity",
    "breakout_compression": "Breakout/compression", "mean_reversion": "Mean reversion", "earnings": "Earnings",
    "news": "News", "fundamentals": "Fundamentals", "sector": "Sector",
}

# (feature, sign) per scored family: sign -1 means "lower is more interesting" (ranked on -value)
SCORED_FEATURES: dict[str, tuple[tuple[str, int], ...]] = {
    "momentum": (("ret_20d", 1), ("ret_60d", 1), ("ret_120d", 1)),
    "relative_strength": (("rs_spy_20", 1), ("rs_spy_63", 1)),
    "volume_activity": (("rel_volume_1d", 1), ("rel_volume_5d", 1)),
    "breakout_compression": (("breakout_55", 1), ("range_contraction_20_60", -1)),
    "mean_reversion": (("ret_z_3d", -1), ("dist_ma20", -1)),
}
# extra features used by triggers/reasons (not scored)
AUX_FEATURES = ("ret_1d", "dist_ma50", "ma50_over_ma200", "ret_z_1d", "atr14_pct", "adv20")
CONTEXT_FEATURES = {
    "earnings": ("days_since_earnings", "ear_z", "event_rel_volume"),
    "news": ("news_count_1d", "news_count_z"),
    "fundamentals": ("rev_growth_yoy", "eps_growth_yoy", "ni_margin", "roe"),
    "sector": ("sector_rs_spy_63", "rs_sector_63"),
}
SOURCES = {
    "price_volume": "Alpaca SIP daily bars (raw + in-house split/dividend adjustment)",
    "momentum": "FeatureSet price features on Alpaca SIP bars",
    "relative_strength": "FeatureSet relative features vs SPY (Alpaca SIP bars)",
    "volume_activity": "FeatureSet dollar-volume features (Alpaca SIP bars)",
    "breakout_compression": "FeatureSet price features on Alpaca SIP bars",
    "mean_reversion": "FeatureSet price features on Alpaca SIP bars",
    "earnings": "SEC EDGAR 8-K item 2.02 timing (events dataset)",
    "news": "Alpaca news (Benzinga), available_at = created_at",
    "fundamentals": "SEC EDGAR companyfacts (as-of replay)",
    "sector": "reference sector/industry/SIC -> sector ETF map (ASSUMED_STATIC)",
}


@dataclass(frozen=True)
class Triggers:
    """Fixed thresholds deciding when a family 'fires' (a setup is discovered). From config."""

    mom_pct_60d: float = 0.90
    mom_pct_20d: float = 0.95
    rs_pct_63: float = 0.90
    vol_rel_1d: float = 3.0
    vol_rel_5d: float = 2.0
    brk_rel_volume: float = 1.5
    brk_max_prev_contraction: float = 0.5
    brk_min_range_expansion: float = 2.0
    mr_ret_z_3d: float = -2.5
    mr_ret_z_1d: float = -3.0
    earn_max_days: float = 3.0
    earn_abs_z: float = 1.5
    earn_rel_volume: float = 2.0
    news_count_1d: float = 2.0
    news_z: float = 2.5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Triggers":
        t = cfg.get("triggers", {}) or {}
        g = lambda fam, k, d: float((t.get(fam) or {}).get(k, d))   # noqa: E731
        return cls(g("momentum", "min_pct_60d", 0.90), g("momentum", "min_pct_20d", 0.95),
                   g("relative_strength", "min_pct_63", 0.90), g("volume_activity", "min_rel_volume_1d", 3.0),
                   g("volume_activity", "min_rel_volume_5d", 2.0), g("breakout_compression", "min_rel_volume", 1.5),
                   g("breakout_compression", "max_prev_contraction", 0.5),
                   g("breakout_compression", "min_range_expansion", 2.0), g("mean_reversion", "max_ret_z_3d", -2.5),
                   g("mean_reversion", "max_ret_z_1d", -3.0), g("earnings", "max_days_since", 3.0),
                   g("earnings", "min_abs_ear_z", 1.5), g("earnings", "min_event_rel_volume", 2.0),
                   g("news", "min_count_1d", 2.0), g("news", "min_count_z", 2.5))


def pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile (0..1] among finite values; NaN stays NaN (never ranked as 0)."""
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.rank(pct=True, method="average")


def flag(v: Any) -> bool:
    """True only for a KNOWN true flag (True / 1.0). NaN (UNKNOWN) is never truthy."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f) and f >= 0.5)


def _fmt_pct(x: float) -> str:
    return f"{x:+.1%}"


def _top(p: float) -> str:
    return f"top {max(1, round((1 - p) * 100))}%"


# ------------------------------------------------------------------------------------------------
# reasons + triggers per scored family (row-wise; the frame is small: one row per scanned symbol)
# ------------------------------------------------------------------------------------------------
def fire_momentum(r: pd.Series, p: pd.Series, t: Triggers) -> tuple[bool, list[str], str]:
    reasons = []
    up_trend = np.isfinite(r["dist_ma50"]) and r["dist_ma50"] > 0
    fired = bool(up_trend and ((np.isfinite(p["ret_60d"]) and p["ret_60d"] >= t.mom_pct_60d)
                               or (np.isfinite(p["ret_20d"]) and p["ret_20d"] >= t.mom_pct_20d)
                               or flag(r.get("new_high_50"))))
    if np.isfinite(r["ret_20d"]) and np.isfinite(p["ret_20d"]) and p["ret_20d"] >= 0.8:
        reasons.append(f"strong 20d return {_fmt_pct(r['ret_20d'])} ({_top(p['ret_20d'])})")
    if np.isfinite(r["ret_60d"]) and np.isfinite(p["ret_60d"]) and p["ret_60d"] >= 0.8:
        reasons.append(f"strong 60d return {_fmt_pct(r['ret_60d'])} ({_top(p['ret_60d'])})")
    if np.isfinite(r["ret_120d"]) and np.isfinite(p["ret_120d"]) and p["ret_120d"] >= 0.8:
        reasons.append(f"strong 120d return {_fmt_pct(r['ret_120d'])}")
    if flag(r.get("new_high_50")):
        reasons.append("new 50-day high")
    elif flag(r.get("new_high_20")):
        reasons.append("new 20-day high")
    if up_trend and np.isfinite(r["ma50_over_ma200"]) and r["ma50_over_ma200"] > 0:
        reasons.append("uptrend (above 50d MA, 50d MA above 200d MA)")
    if np.isfinite(r.get("accel", np.nan)) and r["accel"] > 0 and fired:
        reasons.append("momentum accelerating (20d pace above 60d pace)")
    return fired, reasons, "BULLISH"


def fire_relative_strength(r: pd.Series, p: pd.Series, t: Triggers) -> tuple[bool, list[str], str]:
    fired = bool(np.isfinite(r["rs_spy_63"]) and r["rs_spy_63"] > 0 and np.isfinite(p["rs_spy_63"])
                 and p["rs_spy_63"] >= t.rs_pct_63)
    reasons = []
    if np.isfinite(r["rs_spy_63"]) and np.isfinite(p["rs_spy_63"]) and p["rs_spy_63"] >= 0.8:
        reasons.append(f"outperforming SPY by {r['rs_spy_63'] * 100:+.1f} pp over 63 sessions ({_top(p['rs_spy_63'])})")
    if np.isfinite(r["rs_spy_20"]) and r["rs_spy_20"] > 0 and np.isfinite(p["rs_spy_20"]) and p["rs_spy_20"] >= 0.8:
        reasons.append(f"outperforming SPY by {r['rs_spy_20'] * 100:+.1f} pp over 20 sessions")
    return fired, reasons, "BULLISH"


def _day_bias(ret_1d: float) -> str:
    if not np.isfinite(ret_1d) or abs(ret_1d) < 0.005:
        return "NEUTRAL"
    return "BULLISH" if ret_1d > 0 else "BEARISH"


def fire_volume_activity(r: pd.Series, p: pd.Series, t: Triggers) -> tuple[bool, list[str], str]:
    rv1, rv5 = r["rel_volume_1d"], r["rel_volume_5d"]
    fired = bool((np.isfinite(rv1) and rv1 >= t.vol_rel_1d) or (np.isfinite(rv5) and rv5 >= t.vol_rel_5d))
    reasons, bias = [], _day_bias(r["ret_1d"])
    if np.isfinite(rv1) and rv1 >= 1.5:
        reasons.append(f"relative volume {rv1:.1f}x its 20-day average")
    if np.isfinite(rv5) and rv5 >= 1.5:
        reasons.append(f"5-day volume {rv5:.1f}x its 60-day average")
    if fired and np.isfinite(rv1) and rv1 >= 1.5:
        reasons.append({"BULLISH": "price-volume confirmation: up on heavy volume",
                        "BEARISH": "heavy volume on a down day",
                        "NEUTRAL": "unusual activity, price little changed"}[bias])
    return fired, reasons, bias


def fire_breakout_compression(r: pd.Series, p: pd.Series, t: Triggers) -> tuple[bool, list[str], str]:
    brk, rv1 = r["breakout_55"], r["rel_volume_1d"]
    new_high = bool(np.isfinite(brk) and brk > 0)
    breakout = bool(new_high and np.isfinite(rv1) and rv1 >= t.brk_rel_volume)
    squeeze = bool(np.isfinite(r.get("prev_contraction", np.nan)) and r["prev_contraction"] <= t.brk_max_prev_contraction
                   and np.isfinite(r.get("range_expansion", np.nan)) and r["range_expansion"] >= t.brk_min_range_expansion)
    reasons = []
    if breakout:
        reasons.append(f"recent range breakout: new 55-day high with {rv1:.1f}x volume")
    elif new_high:
        reasons.append("new 55-day high (volume not confirming)")
    if squeeze:
        reasons.append(f"range compressed (20d/60d range {r['prev_contraction']:.2f}) then expanded "
                       f"(today's range {r['range_expansion']:.1f}x ATR)")
    elif np.isfinite(r["range_contraction_20_60"]) and r["range_contraction_20_60"] <= t.brk_max_prev_contraction:
        reasons.append(f"volatility compression (20d/60d range {r['range_contraction_20_60']:.2f})")
    return breakout or squeeze, reasons, _day_bias(r["ret_1d"]) if (breakout or squeeze) else "NEUTRAL"


def fire_mean_reversion(r: pd.Series, p: pd.Series, t: Triggers) -> tuple[bool, list[str], str]:
    z3, z1 = r["ret_z_3d"], r["ret_z_1d"]
    fired = bool((np.isfinite(z3) and z3 <= t.mr_ret_z_3d) or (np.isfinite(z1) and z1 <= t.mr_ret_z_1d))
    reasons = []
    if np.isfinite(z3) and z3 <= -2.0:
        reasons.append(f"unusually large 3-day drop (z = {z3:.1f})")
    elif np.isfinite(z1) and z1 <= -2.5:
        reasons.append(f"unusually large 1-day drop (z = {z1:.1f})")
    if np.isfinite(r["dist_ma20"]) and r["dist_ma20"] <= -0.08:
        reasons.append(f"oversold: {r['dist_ma20']:.1%} below its 20-day average")
    if fired:
        stab = np.isfinite(r.get("close_location", np.nan)) and r["close_location"] >= 0.5 and \
            np.isfinite(r["ret_1d"]) and r["ret_1d"] >= 0
        reasons.append("stabilisation: closed up, in the upper half of the day's range" if stab
                       else "no stabilisation yet (still falling)")
    return fired, reasons, "BEARISH"


def fired_masks(xs: pd.DataFrame, pr: pd.DataFrame, comp: pd.DataFrame, t: Triggers) -> pd.DataFrame:
    """THE definition of when a scored family fires (vectorised; the row-wise functions above only
    write the reason text). ``xs`` = feature values, ``pr`` = unsigned percentile ranks, ``comp`` =
    component points (a family whose component is UNKNOWN never fires). NaN never satisfies a test."""
    f = lambda c: pd.to_numeric(xs[c], errors="coerce")    # noqa: E731
    up = f("dist_ma50") > 0
    nh50 = f("new_high_50") >= 0.5
    mom = up & ((pr["ret_60d"] >= t.mom_pct_60d) | (pr["ret_20d"] >= t.mom_pct_20d) | nh50)
    rs = (f("rs_spy_63") > 0) & (pr["rs_spy_63"] >= t.rs_pct_63)
    vol = (f("rel_volume_1d") >= t.vol_rel_1d) | (f("rel_volume_5d") >= t.vol_rel_5d)
    breakout = (f("breakout_55") > 0) & (f("rel_volume_1d") >= t.brk_rel_volume)
    squeeze = (f("prev_contraction") <= t.brk_max_prev_contraction) & (f("range_expansion") >= t.brk_min_range_expansion)
    mr = (f("ret_z_3d") <= t.mr_ret_z_3d) | (f("ret_z_1d") <= t.mr_ret_z_1d)
    out = pd.DataFrame({"momentum": mom, "relative_strength": rs, "volume_activity": vol,
                        "breakout_compression": breakout | squeeze, "mean_reversion": mr}, index=xs.index)
    out = out.fillna(False).astype(bool)
    for fam in SCORED:
        out[fam] &= comp[fam].notna()
    return out


def stabilising(xs: pd.DataFrame) -> pd.Series:
    """Mean-reversion stabilisation evidence at D: closed up, in the upper half of the day's range."""
    return (pd.to_numeric(xs["close_location"], errors="coerce") >= 0.5) & (pd.to_numeric(xs["ret_1d"], errors="coerce") >= 0)


FIRE: dict[str, Callable[[pd.Series, pd.Series, Triggers], tuple[bool, list[str], str]]] = {
    "momentum": fire_momentum, "relative_strength": fire_relative_strength,
    "volume_activity": fire_volume_activity, "breakout_compression": fire_breakout_compression,
    "mean_reversion": fire_mean_reversion,
}


def fire_context(fam: str, r: pd.Series, t: Triggers) -> tuple[bool, list[str]]:
    """Context families (only called for symbols the source covers)."""
    if fam == "earnings":
        d, z, rv = r.get("days_since_earnings"), r.get("ear_z"), r.get("event_rel_volume")
        recent = d is not None and np.isfinite(d) and d <= t.earn_max_days
        strong = (z is not None and np.isfinite(z) and abs(z) >= t.earn_abs_z) or \
                 (rv is not None and np.isfinite(rv) and rv >= t.earn_rel_volume)
        if recent and strong:
            return True, [f"post-earnings reaction {d:.0f} session(s) ago" + (f", abnormal return z = {z:+.1f}" if
                                                                               np.isfinite(z) else "")]
        return False, []
    if fam == "news":
        n1, nz = r.get("news_count_1d"), r.get("news_count_z")
        fired = (np.isfinite(n1) and n1 >= t.news_count_1d) or (np.isfinite(nz) and nz >= t.news_z)
        return bool(fired), [f"{n1:.0f} fresh news item(s) today (z = {nz:+.1f})"] if fired else []
    if fam == "fundamentals":
        g, e = r.get("rev_growth_yoy"), r.get("eps_growth_yoy")
        fired = np.isfinite(g) and g >= 0.25 and np.isfinite(e) and e > 0
        return bool(fired), [f"revenue growth {g:+.0%} y/y with EPS growth"] if fired else []
    if fam == "sector":
        s = r.get("sector_rs_spy_63")
        fired = np.isfinite(s) and s > 0.05
        return bool(fired), [f"sector ETF outperforming SPY by {s * 100:+.1f} pp (63 sessions)"] if fired else []
    raise ValueError(fam)


__all__ = ["AUX_FEATURES", "CONTEXT", "CONTEXT_FEATURES", "FIRE", "LABELS", "POINTS", "SCORED", "SCORED_FEATURES", "fired_masks", "stabilising",
           "SOURCES", "Triggers", "fire_context", "pct_rank"]
