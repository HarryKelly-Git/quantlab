"""Fundamental features built with the AS-OF rule (ARCHITECTURE.md section 2, rule 7).

For every symbol, facts are replayed in the order they became usable (first session whose cutoff is
>= ``available_at``). At each such session the latest-filed value for every (concept, period) is
kept, so a restatement replaces the old value only from its own filing onward. Derived metrics are
recomputed at each update and carried forward. Nothing filed after D can influence D.

Flow concepts (revenue, income, EPS) use standalone quarterly periods (``fiscal_period == 'Q'``);
balance-sheet concepts use the latest period end. Declared PIT_CONSERVATIVE because real filings
without an exact acceptance time are assumed available one session after the filing date.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import full_like_nan, memo, safe_div

_SRC_F = "bundle.fundamentals (as-of: latest filing with available_at <= cutoff(D))"
FLOW = ("Revenues", "NetIncomeLoss", "EarningsPerShareDiluted", "GrossProfit", "OperatingIncomeLoss")
STOCK = ("Assets", "StockholdersEquity", "LongTermDebt")
# full fiscal-year durations: cash-flow statements are YTD-only in 10-Qs, so FCF uses FY values
FYFLOW = ("OperatingCashFlow", "Capex", "Revenues")
METRICS = ("rev_growth_yoy", "ni_margin", "roe", "leverage", "eps_growth_yoy", "eps_ttm", "sue", "gross_margin",
           "op_margin", "fcf_margin")


def _near(d: pd.Timestamp, keys: list[pd.Timestamp], target_days: int, tol: int = 25) -> pd.Timestamp | None:
    for k in keys:
        if abs((d - k).days - target_days) <= tol:
            return k
    return None


def _ttm(q: dict[pd.Timestamp, float]) -> float:
    ends = sorted(q, reverse=True)[:4]
    if len(ends) < 4:
        return np.nan
    gaps = [(ends[i] - ends[i + 1]).days for i in range(3)]
    if not all(60 <= g <= 120 for g in gaps):
        return np.nan                     # a missing quarter: TTM is UNKNOWN, never guessed
    return float(sum(q[e] for e in ends))


def _yoy(q: dict[pd.Timestamp, float]) -> tuple[float, float] | None:
    if not q:
        return None
    ends = sorted(q, reverse=True)
    prior = _near(ends[0], ends[1:], 365)
    return (q[ends[0]], q[prior]) if prior is not None else None


def _sue(eps: dict[pd.Timestamp, float]) -> float:
    ends = sorted(eps, reverse=True)
    diffs = []
    for e in ends[:9]:
        prior = _near(e, [k for k in ends if k < e], 365)
        if prior is not None:
            diffs.append(eps[e] - eps[prior])
    if len(diffs) < 5:
        return np.nan
    sd = float(np.std(diffs[1:], ddof=1))    # dispersion of PREVIOUS seasonal changes
    return diffs[0] / sd if sd > 0 else np.nan


def _same_quarters_ratio(num: dict, den: dict) -> float:
    """TTM(num) / TTM(den) over the SAME four consecutive quarters, else NaN (UNKNOWN)."""
    n_t, d_t = _ttm(num), _ttm(den)
    if not (np.isfinite(n_t) and np.isfinite(d_t) and d_t > 0):
        return np.nan
    return n_t / d_t if sorted(num, reverse=True)[:4] == sorted(den, reverse=True)[:4] else np.nan


def _fcf_margin(fy: dict[str, dict]) -> float:
    """(OCF - capex) / revenue of the latest fiscal year for which all three are known, else NaN."""
    ocf, capex, rev = fy.get("OperatingCashFlow", {}), fy.get("Capex", {}), fy.get("Revenues", {})
    common = sorted(set(ocf) & set(capex) & set(rev), reverse=True)
    if not common or not rev[common[0]] > 0:
        return np.nan
    e = common[0]
    return (ocf[e] - capex[e]) / rev[e]


def _metrics(flow: dict[str, dict], stock: dict[str, tuple], fy: dict[str, dict] | None = None) -> dict[str, float]:
    out = dict.fromkeys(METRICS, np.nan)
    rev, ni, eps = flow["Revenues"], flow["NetIncomeLoss"], flow["EarningsPerShareDiluted"]
    y = _yoy(rev)
    if y and y[1] > 0:
        out["rev_growth_yoy"] = y[0] / y[1] - 1.0
    rev_t, ni_t = _ttm(rev), _ttm(ni)
    if np.isfinite(rev_t) and rev_t > 0 and np.isfinite(ni_t):
        # require the same four quarters for numerator and denominator
        if sorted(rev, reverse=True)[:4] == sorted(ni, reverse=True)[:4]:
            out["ni_margin"] = ni_t / rev_t
    eq = stock.get("StockholdersEquity")
    if eq and eq[1] > 0 and np.isfinite(ni_t):
        out["roe"] = ni_t / eq[1]
    debt, assets = stock.get("LongTermDebt"), stock.get("Assets")
    if debt and assets and assets[1] > 0:
        out["leverage"] = debt[1] / assets[1]
    ye = _yoy(eps)
    if ye and ye[1] > 0:
        out["eps_growth_yoy"] = ye[0] / ye[1] - 1.0
    out["eps_ttm"] = _ttm(eps)
    out["sue"] = _sue(eps)
    out["gross_margin"] = _same_quarters_ratio(flow.get("GrossProfit", {}), rev)
    out["op_margin"] = _same_quarters_ratio(flow.get("OperatingIncomeLoss", {}), rev)
    out["fcf_margin"] = _fcf_margin(fy or {})
    return out


def _asof_table(fs: FeatureSet) -> dict[str, pd.DataFrame]:
    """All fundamental metrics as wide frames (built once per FeatureSet)."""
    def build() -> dict[str, pd.DataFrame]:
        p = fs.panel
        frames = {m: full_like_nan(fs) for m in (*METRICS, "fundamental_age")}
        f = fs.bundle.fundamentals
        if f.empty:
            return frames
        f = f[f["symbol"].isin(p.symbols) & f["concept"].isin(FLOW + STOCK + FYFLOW)].copy()
        if f.empty:
            return frames
        f["usable"] = fs.bundle.calendar.first_usable_sessions(f["available_at"])
        f = f.dropna(subset=["usable", "value"]).sort_values(["symbol", "usable", "available_at"], kind="mergesort")
        T, sym_index = len(p.dates), {s: j for j, s in enumerate(p.symbols)}
        out = {m: np.full((T, len(p.symbols)), np.nan) for m in (*METRICS, "fundamental_age")}
        row_of = p.dates.get_indexer(pd.DatetimeIndex(f["usable"]))
        syms, concepts = f["symbol"].to_numpy(), f["concept"].to_numpy()
        fps = f["fiscal_period"].astype(str).str.upper().to_numpy()
        pes, vals = pd.DatetimeIndex(f["period_end"]), f["value"].to_numpy(dtype="float64")
        pss = pd.DatetimeIndex(f["period_start"]) if "period_start" in f.columns else pes
        n = len(f)
        i = 0
        # plain-Python replay (vectorizing an as-of state machine obscures it); O(rows) per symbol
        while i < n:
            sym, j = syms[i], sym_index[syms[i]]
            flow: dict[str, dict] = {c: {} for c in FLOW}
            fy: dict[str, dict] = {c: {} for c in FYFLOW}
            # fiscal Q4 = FY - YTD9 (same period start), derived from the versions known AS OF each
            # update: robust to restated YTD9 filings (never relies on a pre-derived Q4 row)
            fy_d: dict[str, dict] = {c: {} for c in FLOW}
            ytd9: dict[str, dict] = {c: {} for c in FLOW}
            stock: dict[str, tuple] = {}
            updates: list[tuple[int, dict[str, float]]] = []
            while i < n and syms[i] == sym:
                r = row_of[i]
                while i < n and syms[i] == sym and row_of[i] == r:
                    c, pe = concepts[i], pes[i]
                    if c in FYFLOW and fps[i] == "FY":
                        fy[c][pe] = vals[i]                           # later filing overwrites (as-of)
                    if c in FLOW:
                        # "Q4" = fiscal Q4 derived as FY - YTD9 by the SEC provider: a quarter too
                        if fps[i] in ("Q", "Q4"):
                            flow[c][pe] = vals[i]                     # later filing overwrites (as-of)
                        elif fps[i] == "FY":
                            fy_d[c][(pss[i], pe)] = vals[i]
                        elif fps[i] == "YTD9":
                            ytd9[c][(pss[i], pe)] = vals[i]
                    elif c in STOCK:
                        cur = stock.get(c)
                        if cur is None or pe >= cur[0]:
                            stock[c] = (pe, vals[i])
                    i += 1
                if r >= 0:
                    for c in FLOW:
                        for (ps, pe_fy), v_fy in fy_d[c].items():
                            for (ps9, pe9), v9 in ytd9[c].items():
                                if ps9 == ps and 80 <= (pe_fy - pe9).days <= 100:
                                    flow[c][pe_fy] = v_fy - v9
                    updates.append((r, _metrics(flow, stock, fy)))
            for k, (r, met) in enumerate(updates):
                end = updates[k + 1][0] if k + 1 < len(updates) else T
                for m in METRICS:
                    out[m][r:end, j] = met[m]
                out["fundamental_age"][r:end, j] = np.arange(end - r, dtype="float64")
        return {m: pd.DataFrame(a, index=p.dates, columns=p.symbols) for m, a in out.items()}
    return memo(fs, "fundamental_asof", build)  # type: ignore[return-value]


def _register(name: str, desc: str, key: str | None = None) -> None:
    @FEATURES.feature(name, "fundamental", desc, _SRC_F, PitStatus.PIT_CONSERVATIVE, lookback=0)
    def _f(fs: FeatureSet) -> pd.DataFrame:
        return _asof_table(fs)[key or name]


_register("rev_growth_yoy", "Revenues_q / Revenues_{same quarter a year earlier} - 1 (as-of)")
_register("ni_margin", "NetIncomeLoss_ttm / Revenues_ttm over the same 4 consecutive quarters (as-of)")
_register("roe", "NetIncomeLoss_ttm / latest StockholdersEquity (as-of; NaN if equity <= 0)")
_register("leverage", "LongTermDebt / Assets, latest period (as-of)")
_register("eps_growth_yoy", "EPS_q / EPS_{q-4} - 1 (NaN if EPS_{q-4} <= 0) (as-of)")
_register("fundamental_age", "sessions since the latest fundamental update became usable")
_register("gross_margin", "GrossProfit_ttm / Revenues_ttm over the same 4 consecutive quarters (as-of)")
_register("op_margin", "OperatingIncomeLoss_ttm / Revenues_ttm over the same 4 consecutive quarters (as-of)")
_register("fcf_margin", "(OperatingCashFlow_FY - Capex_FY) / Revenues_FY, latest fiscal year with all three (as-of)")


@FEATURES.feature("ep_ttm", "fundamental", "EPS_ttm (as-of) / RAW close", _SRC_F + "; panel close", PitStatus.PIT_CONSERVATIVE)
def ep_ttm(fs: FeatureSet) -> pd.DataFrame:
    return safe_div(_asof_table(fs)["eps_ttm"], fs.panel.close)


@FEATURES.feature("sue", "event", "seasonal random-walk SUE: (EPS_q - EPS_{q-4}) / std of the previous seasonal changes "
                  "(>= 4), valid from the filing's availability", _SRC_F, PitStatus.PIT_CONSERVATIVE)
def sue(fs: FeatureSet) -> pd.DataFrame:
    return _asof_table(fs)["sue"]
