"""System health checks and the bridge from CRITICAL failures to SYSTEM_PAUSED.

Every check returns a :class:`~quantlab.core.types.CheckResult` whose ``details['component']`` is
one of ``data | provider | broker | model | llm | reconciliation | system`` (the ``health_checks``
column). Severity semantics:
  * passed                  -> fine (severity INFO)
  * failed + WARNING        -> recorded and shown, trading continues
  * failed + CRITICAL       -> ``trip_if_critical`` pauses the system (ARCHITECTURE.md section 10)

FAIL SAFE: a check that cannot establish health (missing input, exception, unknown status) FAILS
as CRITICAL. ``run`` executes callables inside a guard so a crashing check becomes a CRITICAL
failure instead of disappearing. The LLM error-rate check is the one deliberate WARNING-only
check: the AI layer is an optional filter (``ai.unknown_policy``), so its outages must not stop
the quantitative system.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Iterable

import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import TradingCalendar, to_session
from quantlab.core.types import CheckResult, CheckSeverity, SystemState
from quantlab.db.database import MIGRATIONS_DIR, Database, to_json, utcnow_iso
from quantlab.logging_setup import current_run_id, get_logger, log_event
from quantlab.monitoring.killswitch import KillSwitch
from quantlab.monitoring.llm_monitor import LLMMonitor

log = get_logger(__name__)

COMPONENTS = ("data", "provider", "broker", "model", "llm", "reconciliation", "system")

CheckInput = CheckResult | Iterable[CheckResult] | Callable[[], "CheckResult | Iterable[CheckResult]"]


def _result(name: str, component: str, passed: bool, severity: CheckSeverity, reason: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, passed=passed, severity=CheckSeverity.INFO if passed else severity,
                       reason=reason, details={"component": component, **details})


def _iso(x: Any) -> str:
    if isinstance(x, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(x)
        return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).isoformat()
    if isinstance(x, date):
        return pd.Timestamp(x).tz_localize("UTC").isoformat()
    return _iso(pd.Timestamp(str(x)))


class HealthMonitor:
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config

    # =========================================================================================
    # Data
    # =========================================================================================
    def data_freshness(self, latest_bar_date: Any, expected_session: Any,
                       calendar: TradingCalendar | None = None) -> CheckResult:
        """How many sessions are missing between the latest stored bar and the session we are
        deciding for. Without a calendar, weekdays are used: exchange holidays then count as
        missing sessions, which can only raise a false alarm (the safe direction)."""
        name, comp = "data_freshness", "data"
        max_stale = int(self.config.get("monitoring.max_data_staleness_sessions", 2))
        if latest_bar_date is None or (isinstance(latest_bar_date, float) and pd.isna(latest_bar_date)):
            return _result(name, comp, False, CheckSeverity.CRITICAL, "no bars available: data freshness UNKNOWN",
                           expected_session=str(expected_session))
        try:
            latest, expected = to_session(latest_bar_date), to_session(expected_session)
        except (ValueError, TypeError) as exc:
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"unparseable dates: {exc}")
        if latest >= expected:
            return _result(name, comp, True, CheckSeverity.INFO, "bars are current", latest_bar_date=str(latest.date()),
                           expected_session=str(expected.date()), stale_sessions=0)
        start = latest + pd.Timedelta(days=1)
        stale = len(calendar.between(start, expected)) if calendar is not None else len(pd.bdate_range(start, expected))
        details = {"latest_bar_date": str(latest.date()), "expected_session": str(expected.date()),
                   "stale_sessions": stale, "max_stale_sessions": max_stale,
                   "calendar": "trading" if calendar is not None else "weekdays"}
        if stale == 0:
            return _result(name, comp, True, CheckSeverity.INFO, "bars are current", **details)
        if stale > max_stale:
            return _result(name, comp, False, CheckSeverity.CRITICAL,
                           f"data is {stale} sessions stale (max {max_stale})", **details)
        return _result(name, comp, False, CheckSeverity.WARNING,
                       f"data is {stale} session(s) stale (tolerated up to {max_stale})", **details)

    def missing_data(self, fraction: float | None, n_expected: int | None = None) -> CheckResult:
        """Fraction of expected (symbol, session) bars that are missing in the decision window."""
        name, comp = "missing_data", "data"
        max_frac = float(self.config.get("monitoring.max_missing_bar_fraction", 0.02))
        if fraction is None or pd.isna(fraction) or not (0.0 <= float(fraction) <= 1.0):
            return _result(name, comp, False, CheckSeverity.CRITICAL,
                           f"missing-bar fraction UNKNOWN or invalid ({fraction!r})", n_expected=n_expected)
        f = float(fraction)
        details = {"missing_fraction": f, "max_missing_fraction": max_frac, "n_expected": n_expected}
        if f > max_frac:
            return _result(name, comp, False, CheckSeverity.CRITICAL,
                           f"{100 * f:.2f}% of bars missing (max {100 * max_frac:.2f}%)", **details)
        return _result(name, comp, True, CheckSeverity.INFO, f"{100 * f:.2f}% of bars missing", **details)

    def ingest_failures(self, since: Any) -> CheckResult:
        """Failed ingest runs and dataset-level CRITICAL data-quality issues since ``since``.

        A failed ingest that was followed by a successful ingest is downgraded to WARNING (recovered).
        Symbol-level CRITICAL issues are normally handled by quarantine (WARNING), unless they are
        widespread (``monitoring.max_symbol_critical_issues`` distinct symbols), which suggests a
        provider/dataset problem rather than a few bad tickers.
        """
        name, comp = "ingest_failures", "provider"
        since_iso = _iso(since)
        runs = self.db.fetchall(
            "SELECT run_id, status, started_at, error FROM runs WHERE kind = 'ingest' AND started_at >= ? "
            "ORDER BY started_at", (since_iso,))
        failed = [r for r in runs if r["status"] in ("failed", "aborted")]
        last_fail = failed[-1]["started_at"] if failed else None
        recovered = bool(failed) and any(r["status"] == "succeeded" and r["started_at"] > last_fail for r in runs)
        issues = self.db.fetchall(
            "SELECT severity, symbol, check_name, detail FROM data_quality_issues WHERE created_at >= ?", (since_iso,))
        crit = [i for i in issues if i["severity"] == CheckSeverity.CRITICAL.value]
        dataset_crit = [i for i in crit if not i["symbol"]]
        symbol_crit = sorted({i["symbol"] for i in crit if i["symbol"]})
        warn = [i for i in issues if i["severity"] == CheckSeverity.WARNING.value]
        max_sym = int(self.config.get("monitoring.max_symbol_critical_issues", 25))
        details = {"since": since_iso, "n_ingest_runs": len(runs), "n_failed_runs": len(failed),
                   "recovered": recovered, "failed_run_ids": [r["run_id"] for r in failed][:20],
                   "n_dataset_critical": len(dataset_crit), "n_symbols_critical": len(symbol_crit),
                   "symbols_critical": symbol_crit[:50], "n_warnings": len(warn),
                   "dataset_critical_checks": sorted({i["check_name"] for i in dataset_crit})}
        problems: list[str] = []
        if failed and not recovered:
            problems.append(f"{len(failed)} failed ingest run(s) with no later success")
        if dataset_crit:
            problems.append(f"{len(dataset_crit)} dataset-level CRITICAL data-quality issue(s)")
        if len(symbol_crit) > max_sym:
            problems.append(f"CRITICAL data-quality issues on {len(symbol_crit)} symbols (max {max_sym})")
        if problems:
            return _result(name, comp, False, CheckSeverity.CRITICAL, "; ".join(problems), **details)
        if failed or symbol_crit or warn:
            bits = []
            if failed:
                bits.append(f"{len(failed)} failed ingest run(s), recovered")
            if symbol_crit:
                bits.append(f"CRITICAL issues on {len(symbol_crit)} symbol(s) (quarantine)")
            if warn:
                bits.append(f"{len(warn)} data-quality warning(s)")
            return _result(name, comp, False, CheckSeverity.WARNING, "; ".join(bits), **details)
        return _result(name, comp, True, CheckSeverity.INFO, "no ingest failures", **details)

    def validator_results(self, results: Iterable[CheckResult], component: str = "data") -> list[CheckResult]:
        """Pass through results from data/validation.py (or any validator), tagging a component."""
        out = []
        for r in results:
            if not isinstance(r, CheckResult):
                out.append(_result("validator_result_invalid", component, False, CheckSeverity.CRITICAL,
                                   f"validator returned a non-CheckResult: {type(r).__name__}"))
                continue
            details = dict(r.details or {})
            details.setdefault("component", component)
            out.append(CheckResult(name=r.name, passed=r.passed, severity=r.severity, reason=r.reason, details=details))
        return out

    # =========================================================================================
    # Broker / reconciliation
    # =========================================================================================
    def broker_health(self, probe: Callable[[], bool], name: str = "broker_health") -> CheckResult:
        """``probe`` returns True only when the broker is reachable and its state is known."""
        try:
            ok = probe()
        except Exception as exc:  # any failure of the probe = broker state UNKNOWN
            return _result(name, "broker", False, CheckSeverity.CRITICAL, f"broker probe raised: {type(exc).__name__}: {exc}")
        if ok is True:
            return _result(name, "broker", True, CheckSeverity.INFO, "broker healthy")
        return _result(name, "broker", False, CheckSeverity.CRITICAL, f"broker unhealthy or state unknown (probe returned {ok!r})")

    def reconciliation(self, book: str = "BOT", since: Any = None) -> CheckResult:
        """Latest reconciliation for ``book`` must be 'ok' (and newer than ``since`` when given)."""
        name, comp = f"reconciliation_{book.lower()}", "reconciliation"
        row = self.db.fetchone(
            "SELECT id, status, at, diffs_json FROM reconciliations WHERE book = ? ORDER BY at DESC, id DESC LIMIT 1", (book,))
        required = bool(self.config.get("risk.require_reconciliation", True))
        if row is None:
            sev = CheckSeverity.CRITICAL if required else CheckSeverity.WARNING
            return _result(name, comp, False, sev, f"no reconciliation on record for book {book}", book=book)
        details = {"book": book, "reconciliation_id": row["id"], "status": row["status"], "at": row["at"]}
        if since is not None and pd.Timestamp(_iso(row["at"])) < pd.Timestamp(_iso(since)):
            sev = CheckSeverity.CRITICAL if required else CheckSeverity.WARNING
            return _result(name, comp, False, sev, f"latest reconciliation ({row['at']}) predates {_iso(since)}", **details)
        if row["status"] == "ok":
            return _result(name, comp, True, CheckSeverity.INFO, "ledger matches broker", **details)
        if row["status"] == "mismatch":
            return _result(name, comp, False, CheckSeverity.CRITICAL, "reconciliation MISMATCH between ledger and broker",
                           diffs_json=row["diffs_json"], **details)
        if row["status"] == "broker_unavailable":
            return _result(name, comp, False, CheckSeverity.CRITICAL, "broker unavailable during reconciliation", **details)
        return _result(name, comp, False, CheckSeverity.CRITICAL, f"unknown reconciliation status {row['status']!r}", **details)

    # =========================================================================================
    # Models / LLM
    # =========================================================================================
    def models(self) -> CheckResult:
        """FAILED/ERROR models are CRITICAL (a model failure is a pause trigger). PAUSED models are
        a controlled state (WARNING). No models at all is fine (ML is an optional layer)."""
        name, comp = "models", "model"
        rows = self.db.fetchall("SELECT model_id, version, status FROM models")
        status = {(r["model_id"], r["version"]): (r["status"] or "").upper() for r in rows}
        failed = sorted(f"{m}@{v}" for (m, v), s in status.items() if s in ("FAILED", "ERROR"))
        paused = sorted(f"{m}@{v}" for (m, v), s in status.items() if s == "PAUSED")
        live = [k for k, s in status.items() if s in ("ACTIVE", "MONITORED")]
        details = {"n_models": len(rows), "failed": failed, "paused": paused, "n_live": len(live)}
        if failed:
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"model failure: {', '.join(failed)}", **details)
        if paused:
            return _result(name, comp, False, CheckSeverity.WARNING, f"paused model(s): {', '.join(paused)}", **details)
        return _result(name, comp, True, CheckSeverity.INFO, f"{len(live)} live model(s), none failed", **details)

    def llm_error_rate(self, start: Any = None, end: Any = None) -> CheckResult:
        """(error + invalid_schema) / attempted calls. WARNING only: AI is an optional filter."""
        name, comp = "llm_error_rate", "llm"
        max_rate = float(self.config.get("monitoring.llm_max_error_rate", 0.25))
        min_calls = int(self.config.get("monitoring.llm_min_calls_for_rate", 5))
        try:
            totals = LLMMonitor(self.db, self.config).call_stats(start, end)["totals"]
        except Exception as exc:
            return _result(name, comp, False, CheckSeverity.WARNING, f"LLM statistics unavailable: {exc}")
        n, rate = totals["n_attempted"], totals["failure_rate"]
        details = {"n_attempted": n, "n_error": totals["n_error"], "n_invalid_schema": totals["n_invalid_schema"],
                   "failure_rate": rate, "max_error_rate": max_rate}
        if n == 0:
            return _result(name, comp, True, CheckSeverity.INFO, "no LLM calls in window (AI disabled or idle)", **details)
        if n < min_calls:
            return _result(name, comp, True, CheckSeverity.INFO,
                           f"only {n} LLM call(s): too few to judge the error rate (min {min_calls})", **details)
        if rate is not None and rate > max_rate:
            return _result(name, comp, False, CheckSeverity.WARNING,
                           f"LLM failure rate {100 * rate:.1f}% > {100 * max_rate:.1f}% (AI decisions become UNKNOWN)", **details)
        return _result(name, comp, True, CheckSeverity.INFO, f"LLM failure rate {100 * (rate or 0):.1f}%", **details)

    # =========================================================================================
    # Database / state
    # =========================================================================================
    def db_integrity(self) -> CheckResult:
        """SQLite quick_check plus 'every migration file has been applied'."""
        name, comp = "db_integrity", "system"
        try:
            rows = self.db.execute("PRAGMA quick_check").fetchall()
            msgs = [str(r[0]) for r in rows]
        except Exception as exc:
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"quick_check failed to run: {exc}")
        if msgs != ["ok"]:
            return _result(name, comp, False, CheckSeverity.CRITICAL, "database integrity check FAILED", messages=msgs[:20])
        try:
            applied = {r["version"] for r in self.db.fetchall("SELECT version FROM schema_migrations")}
        except Exception as exc:
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"schema_migrations unreadable: {exc}")
        pending = sorted(f.stem for f in MIGRATIONS_DIR.glob("*.sql") if f.stem not in applied)
        if pending:
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"unapplied migrations: {', '.join(pending)}",
                           pending=pending)
        return _result(name, comp, True, CheckSeverity.INFO, "database ok", n_migrations=len(applied))

    def state_integrity(self) -> CheckResult:
        """The system_state row must exist, be valid and agree with the newest system_state_log row."""
        name, comp = "state_integrity", "system"
        try:
            row = self.db.fetchone("SELECT state, changed_at, changed_by FROM system_state WHERE id = 1")
            last = self.db.fetchone("SELECT state, changed_at, changed_by FROM system_state_log ORDER BY id DESC LIMIT 1")
        except Exception as exc:
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"system state unreadable: {exc}")
        if row is None:
            return _result(name, comp, False, CheckSeverity.CRITICAL, "system_state row missing")
        if row["state"] not in (SystemState.ACTIVE.value, SystemState.PAUSED.value):
            return _result(name, comp, False, CheckSeverity.CRITICAL, f"invalid system state {row['state']!r}")
        if last is None or last["state"] != row["state"]:
            return _result(name, comp, False, CheckSeverity.CRITICAL,
                           "system_state disagrees with system_state_log (unlogged change or corruption)",
                           state=row["state"], last_logged_state=None if last is None else last["state"])
        return _result(name, comp, True, CheckSeverity.INFO, f"system state {row['state']} consistent with log")

    # =========================================================================================
    # Running and enforcing
    # =========================================================================================
    def run(self, *checks: CheckInput, run_id: str | None = None) -> list[CheckResult]:
        """Evaluate checks (results, lists of results or zero-arg callables) and record every
        result in ``health_checks``. A callable that raises becomes a CRITICAL failure."""
        results: list[CheckResult] = []
        for c in checks:
            if callable(c) and not isinstance(c, CheckResult):
                label = getattr(c, "__name__", type(c).__name__)
                try:
                    c = c()
                except Exception as exc:
                    results.append(_result(f"{label}_crashed", "system", False, CheckSeverity.CRITICAL,
                                           f"health check crashed: {type(exc).__name__}: {exc}"))
                    continue
            if isinstance(c, CheckResult):
                results.append(c)
            elif c is None:
                results.append(_result("health_check_missing", "system", False, CheckSeverity.CRITICAL,
                                       "a health check returned nothing"))
            else:
                results.extend(self.validator_results(c, component="system"))
        rid = run_id if run_id is not None else current_run_id()
        now = utcnow_iso()
        self.db.insert_many("health_checks", [{
            "run_id": rid, "check_name": r.name,
            "component": (r.details or {}).get("component", "system"),
            "passed": int(bool(r.passed)), "severity": r.severity.value, "reason": r.reason,
            "details_json": to_json(r.details or {}), "created_at": now,
        } for r in results])
        for r in results:
            if not r.passed:
                log_event(log, f"health check {r.name} FAILED [{r.severity.value}]: {r.reason}",
                          level=logging.ERROR if r.blocking else logging.WARNING, check=r.name)
        return results

    @staticmethod
    def trip_if_critical(results: Iterable[CheckResult], killswitch: KillSwitch) -> list[CheckResult]:
        """Pause the system when any result is a CRITICAL failure. Returns the triggering checks.
        Warnings never pause."""
        blocking = [r for r in results if r.blocking]
        if blocking:
            reason = "CRITICAL health check failure: " + "; ".join(f"{r.name}: {r.reason}" for r in blocking)
            killswitch.pause(
                reason=reason[:2000],
                trigger="health:" + ",".join(sorted({r.name for r in blocking})),
                details={"checks": [{"name": r.name, "reason": r.reason, "details": r.details} for r in blocking]},
                actor="system:health",
            )
        return blocking

    def run_and_enforce(self, *checks: CheckInput, killswitch: KillSwitch | None = None,
                        run_id: str | None = None) -> tuple[list[CheckResult], list[CheckResult]]:
        """``run`` + ``trip_if_critical``. Returns (all results, triggering results)."""
        results = self.run(*checks, run_id=run_id)
        return results, self.trip_if_critical(results, killswitch or KillSwitch(self.db))
