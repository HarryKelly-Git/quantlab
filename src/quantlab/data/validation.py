"""Data integrity checks. A failed CRITICAL check must never be silently traded through: the caller
(health monitor / pipeline) pauses the system; symbol-level problems quarantine the symbol.

Every problem is written to ``data_quality_issues``; symbol quarantines go to
``symbol_quarantine`` (both append-only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.types import CheckResult, CheckSeverity
from quantlab.data.panel import DataBundle
from quantlab.db.database import Database, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

# Split ratios that raw prices can jump by when a split is missing from the corporate actions.
_SPLIT_RATIOS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 50, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10,
                          1 / 15, 1 / 20, 1 / 25, 1 / 30, 1 / 50, 3 / 2, 2 / 3])


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)
    quarantined: dict[str, str] = field(default_factory=dict)   # symbol -> reason

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.blocking]

    @property
    def ok(self) -> bool:
        return not self.critical_failures


class DataValidator:
    def __init__(self, config: Config, db: Database | None = None):
        self.config = config
        self.db = db
        self.max_stale = int(config.get("monitoring.max_data_staleness_sessions", 2))
        self.max_missing = float(config.get("monitoring.max_missing_bar_fraction", 0.02))
        self.extreme_ret = float(config.get("validation.data.extreme_return", 0.8))
        self.split_tol = float(config.get("validation.data.split_ratio_tolerance", 0.02))

    # -- raw bar checks -------------------------------------------------------------------------------
    def check_bars(self, bars: pd.DataFrame) -> list[CheckResult]:
        out: list[CheckResult] = []
        dup = bars.duplicated(["symbol", "date"], keep=False)
        out.append(CheckResult("duplicate_bars", not dup.any(), CheckSeverity.CRITICAL,
                               f"{int(dup.sum())} duplicate (symbol, date) rows" if dup.any() else "",
                               {"symbols": sorted(bars.loc[dup, "symbol"].unique().tolist())[:50]}))
        px = bars[["open", "high", "low", "close"]]
        bad_px = ~np.isfinite(px.to_numpy()).all(axis=1) | (px <= 0).any(axis=1).to_numpy()
        tol = 1e-6
        bad_ohlc = ((bars["low"] > bars[["open", "close"]].min(axis=1) * (1 + tol)) |
                    (bars["high"] < bars[["open", "close"]].max(axis=1) * (1 - tol))).to_numpy()
        bad_vol = ~(bars["volume"].fillna(-1) > 0).to_numpy()
        for name, mask, sev, msg in [
            ("non_positive_or_missing_price", bad_px, CheckSeverity.CRITICAL, "prices must be finite and > 0"),
            ("ohlc_inconsistent", bad_ohlc & ~bad_px, CheckSeverity.CRITICAL, "low/high do not bracket open/close"),
            ("non_positive_volume", bad_vol, CheckSeverity.WARNING, "volume missing or <= 0"),
        ]:
            syms = sorted(bars.loc[mask, "symbol"].unique().tolist())
            out.append(CheckResult(name, not mask.any(), sev, f"{int(mask.sum())} rows: {msg}" if mask.any() else "",
                                   {"symbols": syms[:50], "n_symbols": len(syms)}))
        return out

    # -- bundle checks --------------------------------------------------------------------------------
    def check_bundle(self, bundle: DataBundle, expected_last_session=None) -> ValidationReport:
        rep = ValidationReport()
        p = bundle.panel
        mkt = bundle.market_symbol
        if mkt not in p.symbols or p.close[mkt].notna().sum() == 0:
            rep.checks.append(CheckResult("benchmark_present", False, CheckSeverity.CRITICAL,
                                          f"market benchmark {mkt} has no data"))
        else:
            gaps = int(p.close[mkt].isna().sum())
            rep.checks.append(CheckResult("benchmark_present", gaps == 0, CheckSeverity.CRITICAL,
                                          f"{mkt} missing {gaps} sessions of the calendar" if gaps else ""))

        # staleness of the latest session vs what should exist
        if expected_last_session is not None:
            exp = pd.Timestamp(expected_last_session)
            last = p.dates[-1]
            behind = int(((p.dates > last) & (p.dates <= exp)).sum())
            if exp > last:
                behind = max(behind, int(np.busday_count(last.date(), exp.date())))
            rep.checks.append(CheckResult("data_freshness", behind <= self.max_stale, CheckSeverity.CRITICAL,
                                          f"latest bar {last.date()} is {behind} sessions behind {exp.date()}" if behind > self.max_stale else "",
                                          {"latest": str(last.date()), "expected": str(exp.date()), "behind": behind}))

        # missing bars inside each symbol's listed span
        close = p.close
        listed = close.notna()

        span = (listed.cumsum() > 0) & (listed[::-1].cumsum()[::-1] > 0)
        missing = (span & ~listed).sum()
        frac = (missing / span.sum().replace(0, np.nan)).fillna(0.0)
        gappy = frac[frac > self.max_missing]
        rep.checks.append(CheckResult("missing_bars", gappy.empty, CheckSeverity.WARNING,
                                      f"{len(gappy)} symbols miss > {self.max_missing:.0%} of sessions in their span" if len(gappy) else "",
                                      {"symbols": gappy.sort_values(ascending=False).head(50).round(4).to_dict()}))

        # unexplained split-like jumps (raw overnight ratio near a split ratio, no split recorded)
        prev_close = close.ffill().shift(1)
        ratio = (p.open / prev_close).where(p.split_ratio == 1.0)
        near = np.zeros(ratio.shape, dtype=bool)
        r = ratio.to_numpy()
        with np.errstate(invalid="ignore"):
            for k in _SPLIT_RATIOS:
                near |= np.abs(r * k - 1.0) <= self.split_tol
        big = np.abs(np.log(np.where(np.isfinite(r) & (r > 0), r, 1.0))) > np.log(1.4)
        suspect = near & big
        suspects: dict[str, str] = {}
        for i, j in zip(*np.nonzero(suspect)):
            sym = p.symbols[j]
            if sym not in suspects:
                suspects[sym] = f"overnight raw ratio {r[i, j]:.3f} on {p.dates[i].date()} looks like an unrecorded split"
        rep.quarantined.update(suspects)
        rep.checks.append(CheckResult("unexplained_split_jumps", not suspects, CheckSeverity.WARNING,
                                      f"{len(suspects)} symbols quarantined" if suspects else "",
                                      {"symbols": dict(list(suspects.items())[:50])}))

        # extreme returns without an event (warning only: real crashes/buyouts happen)
        ext = (p.ret.abs() > self.extreme_ret)
        if not bundle.events.empty:
            ev = bundle.events.dropna(subset=["reaction_date"])
            for sym, d in zip(ev["symbol"], ev["reaction_date"]):
                if sym in ext.columns and d in ext.index:
                    ext.at[d, sym] = False
        n_ext = int(ext.to_numpy().sum())
        rep.checks.append(CheckResult("extreme_returns", n_ext == 0, CheckSeverity.WARNING,
                                      f"{n_ext} daily |return| > {self.extreme_ret:.0%} without an earnings event" if n_ext else "",
                                      {"symbols": sorted(ext.columns[ext.any()].tolist())[:50]}))

        return rep

    # -- persistence ----------------------------------------------------------------------------------
    def record(self, report: ValidationReport, run_id: str | None = None, dataset_id: str | None = None) -> None:
        if self.db is None:
            return
        now = utcnow_iso()
        rows = []
        for c in report.checks:
            if c.passed:
                continue
            syms = c.details.get("symbols") or [None]
            syms = list(syms.keys()) if isinstance(syms, dict) else syms
            for s in syms[:50]:
                rows.append({"run_id": run_id, "dataset_id": dataset_id, "check_name": c.name, "severity": c.severity.value,
                             "symbol": s, "session_date": None, "detail": c.reason, "created_at": now})
        self.db.insert_many("data_quality_issues", rows)
        self.db.insert_many("symbol_quarantine", [
            {"symbol": s, "check_name": "unexplained_split_jumps", "reason": why, "from_date": None, "to_date": None,
             "run_id": run_id, "created_at": now} for s, why in report.quarantined.items()])
        if report.critical_failures:
            log_event(log, "data validation CRITICAL failures", level=40,
                      checks=[c.name for c in report.critical_failures])


def quarantined_symbols(db: Database) -> set[str]:
    """Symbols whose latest quarantine row has no end date."""
    rows = db.fetchall("SELECT symbol, to_date FROM symbol_quarantine ORDER BY id")
    state: dict[str, bool] = {}
    for r in rows:
        state[r["symbol"]] = r["to_date"] is None
    return {s for s, active in state.items() if active}
