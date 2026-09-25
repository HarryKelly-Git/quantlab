"""Real-data audit: is the price data we can actually obtain fit for the methodology?

Runs on a SMALL real sample (default SPY, XLK, AAPL, MSFT from 2020-01-01) and answers, with
evidence taken from the data itself rather than from provider labels:
  access, feed actually used, date coverage, freshness, missing sessions, duplicates, OHLC sanity,
  session/timezone handling (no weekend/holiday bars, one shared calendar), volume quality
  (consolidated vs single-venue), corporate-action behaviour (raw bars jump on split ex-dates while
  our in-house total return does not), and dividend coverage.

Verdicts are deliberately conservative:
  NOT_RUN                     credentials missing (nothing fetched, nothing assumed)
  FAILED                      access/ingest failed (incl. corporate actions) or a critical check failed
  INCONCLUSIVE                a decisive check could not be evaluated (UNKNOWN): not suitable
  DATA_LIMITATION             data usable only with stated limits (e.g. IEX-only volume)
  SUITABLE_SMALL_SAMPLE_ONLY  every check passed on the small sample. NOT a claim that the data
                              supports full-universe research (coverage, delistings and survivorship
                              remain UNKNOWN until audited at scale)
  SYNTHETIC_ONLY              the providers were synthetic (machinery check, not market evidence)
Every audit is stored append-only in ``data_audits`` and written to a JSON file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.core.types import new_id
from quantlab.data.panel import DataBundle
from quantlab.db.database import from_json, to_json, utcnow_iso
from quantlab.secrets import get_secret

ET = ZoneInfo("America/New_York")
DEFAULT_SYMBOLS = ("SPY", "XLK", "AAPL", "MSFT")
# SPY consolidated (SIP) volume has been tens of millions of shares per day every year since 2016;
# a single venue such as IEX carries a few percent of it. Used as an empirical feed check.
MIN_CONSOLIDATED_SPY_SHARES = 20_000_000
# Symbol-level data problems fail the WHOLE dataset only when systemic (share of symbols affected)
# or when a benchmark is affected; otherwise the affected symbols are quarantined / not adjusted.
SYSTEMIC_FRACTION = 0.05


@dataclass
class AuditCheck:
    name: str
    status: str               # PASS | WARN | FAIL | UNKNOWN
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAudit:
    status: str
    reason: str
    symbols: list[str]
    start: str
    end: str | None
    feed: str | None = None
    checks: list[AuditCheck] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    audit_id: str = field(default_factory=lambda: new_id("audit"))
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------
def required_credentials(config) -> dict[str, bool]:
    """Presence (never values) of the environment variables the real providers need."""
    names = [config.get("providers.alpaca.key_id_env"), config.get("providers.alpaca.secret_env"),
             config.get("providers.sec_edgar.user_agent_env")]
    return {n: get_secret(n) is not None for n in names if n}


# Unscheduled full-day closures since 2012 (not derivable from rules).
SPECIAL_CLOSURES = {pd.Timestamp(d) for d in ("2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09")}


def _easter(y: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous / Meeus algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, month, day)


def _nth_weekday(y: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    days = pd.date_range(pd.Timestamp(y, month, 1), periods=31, freq="D")
    days = days[(days.month == month) & (days.weekday == weekday)]
    return days[n] if n >= 0 else days[n]


def _observed(t: pd.Timestamp, saturday_to_friday: bool = True) -> pd.Timestamp | None:
    if t.weekday() == 5:
        return t - pd.Timedelta(days=1) if saturday_to_friday else None
    if t.weekday() == 6:
        return t + pd.Timedelta(days=1)
    return t


def nyse_holidays(years: range) -> set[pd.Timestamp]:
    """NYSE full-day holidays by the published rules (+ known special closures since 2012)."""
    out: set[pd.Timestamp] = set()
    for y in years:
        cands = [
            _observed(pd.Timestamp(y, 1, 1), saturday_to_friday=False),   # no Friday Dec 31 closure
            _nth_weekday(y, 1, 0, 2), _nth_weekday(y, 2, 0, 2),             # MLK, Washington's Birthday
            _easter(y) - pd.Timedelta(days=2),                              # Good Friday
            _nth_weekday(y, 5, 0, -1),                                      # Memorial Day
            _observed(pd.Timestamp(y, 6, 19)) if y >= 2022 else None,       # Juneteenth
            _observed(pd.Timestamp(y, 7, 4)), _nth_weekday(y, 9, 0, 0),    # Independence, Labor
            _nth_weekday(y, 11, 3, 3), _observed(pd.Timestamp(y, 12, 25)),  # Thanksgiving, Christmas
        ]
        out.update(c for c in cands if c is not None)
    out.update(d for d in SPECIAL_CLOSURES if d.year in years)
    return out


def _nyse_rule_holidays(years: range) -> set[pd.Timestamp]:   # backwards-compatible name
    return nyse_holidays(years)


def expected_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end] from the holiday rules (independent of any data provider)."""
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = pd.bdate_range(s, e)
    hol = nyse_holidays(range(s.year, e.year + 1))
    return days[~days.isin(list(hol))]


def expected_last_session(now: datetime | None = None) -> pd.Timestamp:
    """Most recent weekday whose regular session has closed (holidays ignored -> WARN, not FAIL)."""
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    d = pd.Timestamp(now_et.date())
    if now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 15):
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def audit_checks(bundle: DataBundle, bars: pd.DataFrame, actions: pd.DataFrame, feed: str | None,
                 symbols: list[str], now: datetime | None = None, requested_start=None,
                 requested_end=None) -> list[AuditCheck]:
    """Pure checks over already-fetched data (unit-testable offline)."""
    from quantlab.data.validation import DataValidator
    checks: list[AuditCheck] = []
    p = bundle.panel
    mkt = bundle.market_symbol

    # feed actually used (label) ------------------------------------------------------------------
    f = (feed or "").lower()
    if f == "sip":
        checks.append(AuditCheck("feed", "PASS", "consolidated SIP feed recorded"))
    elif f == "iex":
        checks.append(AuditCheck("feed", "FAIL", "DATA_LIMITATION: IEX-only feed; volume is single-venue (~2-3% of "
                                 "consolidated), history from ~2020; dollar-volume liquidity rules are NOT valid"))
    else:
        checks.append(AuditCheck("feed", "UNKNOWN", f"feed not recorded or unrecognized ({feed!r})"))

    # coverage ------------------------------------------------------------------------------------
    cov = {}
    for s in symbols:
        sb = bars[bars["symbol"] == s]
        cov[s] = {"bars": int(len(sb)), "first": str(sb["date"].min().date()) if len(sb) else None,
                  "last": str(sb["date"].max().date()) if len(sb) else None}
    missing_syms = [s for s, c in cov.items() if c["bars"] == 0]
    late = {}
    if requested_start is not None and len(bars):
        exp = expected_sessions(requested_start, max(pd.Timestamp(requested_start), bars["date"].max()))
        for s, c in cov.items():
            if c["first"] is not None and len(exp):
                lag = int((exp < pd.Timestamp(c["first"])).sum())
                if lag > 5:
                    late[s] = {"first_bar": c["first"], "sessions_late": lag}
    status = "FAIL" if missing_syms or late else "PASS"
    msg = ("no bars for " + str(missing_syms) + "; " if missing_syms else "") + \
          (f"history starts late vs requested {pd.Timestamp(requested_start).date()}: {late} (IEX history begins ~2020)" if late else "")
    checks.append(AuditCheck("coverage", status, msg or "every requested symbol covers the requested range",
                             {"per_symbol": cov, "late": late}))

    exp_last = expected_last_session(now)
    last = bars["date"].max() if len(bars) else None
    if last is None:
        checks.append(AuditCheck("freshness", "FAIL", "no bars at all"))
    else:
        behind = int(np.busday_count(pd.Timestamp(last).date(), exp_last.date())) if exp_last > last else 0
        checks.append(AuditCheck("freshness", "PASS" if behind <= 2 else "WARN",
                                 f"latest bar {pd.Timestamp(last).date()}, expected ~{exp_last.date()} ({behind} weekdays behind; "
                                 "holidays not modelled)", {"behind_weekdays": behind}))

    # integrity -----------------------------------------------------------------------------------
    for c in DataValidator(bundle_config_stub()).check_bars(bars):
        checks.append(AuditCheck(f"integrity:{c.name}", "PASS" if c.passed else ("FAIL" if c.blocking else "WARN"),
                                 c.reason or "ok", c.details))

    # sessions / timezone -------------------------------------------------------------------------
    dates = pd.DatetimeIndex(sorted(bars["date"].unique()))
    weekend = dates[dates.weekday >= 5]
    hol = nyse_holidays(range(dates.min().year, dates.max().year + 1)) if len(dates) else set()
    on_hol = [d for d in dates if d in hol]
    shared = True
    if mkt in set(bars["symbol"]):
        mkt_dates = set(bars.loc[bars["symbol"] == mkt, "date"])
        extra = sorted(set(bars["date"]) - mkt_dates)
        shared = not extra
    else:
        extra = []
    bad = len(weekend) > 0 or len(on_hol) > 0
    checks.append(AuditCheck("sessions_timezone", "FAIL" if bad else ("PASS" if shared else "WARN"),
                             "bars on weekends/holidays => session dates are shifted (timezone bug)" if bad else
                             ("session dates are weekdays, avoid fixed holidays, and match the benchmark calendar" if shared
                              else f"{len(extra)} dates exist for some symbols but not for {mkt}"),
                             {"weekend": [str(d.date()) for d in weekend[:10]], "holidays": [str(d.date()) for d in on_hol[:10]],
                              "dates_missing_from_benchmark": [str(pd.Timestamp(d).date()) for d in extra[:10]]}))

    # missing sessions vs an INDEPENDENT exchange calendar (not vs the data's own dates) -------------
    gaps, runs = {}, {}
    for s in symbols:
        sd = pd.DatetimeIndex(sorted(bars.loc[bars["symbol"] == s, "date"].unique()))
        if not len(sd):
            continue
        exp = expected_sessions(sd.min(), sd.max())
        miss = exp[~exp.isin(sd)]
        gaps[s] = int(len(miss))
        longest, cur, prev = 0, 0, None
        pos = {d: i for i, d in enumerate(exp)}
        for d in miss:
            cur = cur + 1 if prev is not None and pos[d] == pos[prev] + 1 else 1
            longest, prev = max(longest, cur), d
        runs[s] = longest
    if not gaps:
        checks.append(AuditCheck("missing_sessions", "UNKNOWN", "no bars to compare with the exchange calendar"))
    else:
        worst_run = max(runs.values())
        st = "FAIL" if worst_run >= 3 else ("WARN" if max(gaps.values()) > 0 else "PASS")
        checks.append(AuditCheck("missing_sessions", st,
                                 f"expected NYSE sessions missing per symbol {gaps}; longest consecutive hole {runs}",
                                 {"missing": gaps, "longest_run": runs}))
    # unrecorded split-like jumps (bars move by a split ratio with no split action) ---------------------
    vrep = DataValidator(bundle_config_stub()).check_bundle(bundle)
    jump = next((c for c in vrep.checks if c.name == "unexplained_split_jumps"), None)
    if jump is not None:
        n_q, n_sym = len(vrep.quarantined), max(len(p.symbols), 1)
        bench = {bundle.market_symbol, *bundle.sector_etfs.keys()}
        systemic = n_q / n_sym > SYSTEMIC_FRACTION or bool(set(vrep.quarantined) & bench)
        checks.append(AuditCheck("unexplained_split_jumps", "PASS" if jump.passed else ("FAIL" if systemic else "WARN"),
                                 (jump.reason or "no split-like raw jumps without a recorded split")
                                 + (f" ({n_q}/{n_sym} symbols; quarantined from confirmation)" if n_q else ""),
                                 jump.details))

    # volume quality (empirical, independent of the feed label) -----------------------------------
    if mkt in set(bars["symbol"]):
        med = float(bars.loc[bars["symbol"] == mkt, "volume"].median())
        ok = med >= MIN_CONSOLIDATED_SPY_SHARES
        checks.append(AuditCheck("volume_quality", "PASS" if ok else "FAIL",
                                 f"median {mkt} daily volume {med:,.0f} shares; "
                                 + ("consistent with consolidated volume" if ok else
                                    "far below consolidated levels => single-venue (e.g. IEX) volume: DATA_LIMITATION"),
                                 {"median_spy_volume": med}))
    else:
        checks.append(AuditCheck("volume_quality", "UNKNOWN", f"{mkt} not in sample"))

    # corporate actions: each recorded split must show up in the RAW bars (reconciliation) ------------
    from quantlab.data.panel import reconcile_splits
    splits = actions[(actions["action_type"] == "split") & actions["symbol"].isin(p.symbols)] if len(actions) else actions
    rec = reconcile_splits(p.open, p.close, splits) if len(splits) else splits
    if not len(rec):
        checks.append(AuditCheck("corporate_actions_splits", "UNKNOWN",
                                 "no split inside the sample: raw/adjusted handling not verifiable on this sample"))
    else:
        counts = rec["split_status"].value_counts().to_dict()
        bad = rec[rec["split_status"] == "unconfirmed"]
        testable = rec[~rec["split_status"].isin(["not_testable", "invalid"])]
        frac = len(bad) / max(len(testable), 1)
        bench = {bundle.market_symbol, *bundle.sector_etfs.keys()}
        systemic = frac > SYSTEMIC_FRACTION or bool(set(bad["symbol"]) & bench)
        st = "PASS" if bad.empty else ("FAIL" if systemic else "WARN")
        msg = (f"{len(rec)} recorded splits: {counts}. "
               + ("raw bars are UNADJUSTED and every split appears where recorded (or one session later)" if bad.empty else
                  f"{len(bad)} ({frac:.1%} of {len(testable)} testable) recorded splits never appear in the raw bars (bars pre-adjusted or record wrong) "
                  f"and are NOT applied" + ("; SYSTEMIC: data unusable" if systemic else "; symbol-level, not systemic")))
        checks.append(AuditCheck("corporate_actions_splits", st, msg,
                                 {"status_counts": counts, "unconfirmed": bad[["symbol", "ex_date", "ratio"]].astype(str)
                                  .head(50).to_dict("records")}))
    divs = actions[actions["action_type"] == "cash_dividend"] if len(actions) else actions
    per = {s: int((divs["symbol"] == s).sum()) for s in symbols} if len(divs) else {s: 0 for s in symbols}
    years = max((bars["date"].max() - bars["date"].min()).days / 365.25, 0) if len(bars) else 0
    low = [s for s in (mkt, "MSFT", "XLK") if s in per and years >= 1 and per[s] < 0.5 * years]
    checks.append(AuditCheck("corporate_actions_dividends", "WARN" if low else "PASS",
                             f"cash dividends per symbol {per} over {years:.1f} years"
                             + (f"; suspiciously few for {low}" if low else ""), per))
    return checks


def bundle_config_stub():
    """DataValidator only needs a couple of monitoring thresholds for check_bars()."""
    class _C:
        def get(self, key, default=None):
            return default
    return _C()


DECISIVE = ("feed", "volume_quality", "corporate_actions_splits", "sessions_timezone", "missing_sessions")


def verdict(checks: list[AuditCheck], feed: str | None, synthetic: bool,
            corporate_actions_ok: bool = True) -> tuple[str, str]:
    """FAIL -> FAILED (or DATA_LIMITATION for feed/volume only); UNKNOWN on a decisive check ->
    INCONCLUSIVE; WARNs never silently pass: they are listed in the reason."""
    if synthetic:
        return "SYNTHETIC_ONLY", "providers were synthetic: this validates the audit machinery, not market data"
    fails = [c for c in checks if c.status == "FAIL"]
    limitation = [c for c in fails if c.name in ("feed", "volume_quality")]
    hard = [c for c in fails if c not in limitation]
    if not corporate_actions_ok:
        hard.append(AuditCheck("corporate_actions", "FAIL", "corporate actions missing or failed: split/dividend "
                               "handling (and therefore every return) is unverifiable"))
    if hard:
        return "FAILED", "; ".join(f"{c.name}: {c.detail}" for c in hard)[:1500]
    unknown = [c for c in checks if c.status == "UNKNOWN" and c.name in DECISIVE and c.name not in ("feed",)]
    if limitation:
        return "DATA_LIMITATION", "; ".join(f"{c.name}: {c.detail}" for c in limitation + unknown)[:1500]
    if unknown or any(c.status == "UNKNOWN" and c.name == "feed" for c in checks):
        unk = unknown + [c for c in checks if c.status == "UNKNOWN" and c.name == "feed"]
        return "INCONCLUSIVE", "could not verify: " + "; ".join(f"{c.name}: {c.detail}" for c in unk)[:1500]
    warns = [c for c in checks if c.status == "WARN"]
    tail = ("; WARNINGS: " + "; ".join(f"{c.name}: {c.detail}" for c in warns)) if warns else ""
    return ("SUITABLE_SMALL_SAMPLE_ONLY",
            ("all decisive checks passed on the small sample; full-universe coverage, delistings and "
             "survivorship remain UNKNOWN" + tail)[:1500])


def run_data_audit(ctx, symbols=DEFAULT_SYMBOLS, start: str = "2020-01-01", end: str | None = None,
                   providers: dict | None = None, now: datetime | None = None) -> DataAudit:
    from quantlab.data.ingest import IngestionService
    symbols = [s.upper() for s in symbols]
    cfg = ctx.config
    audit = DataAudit("NOT_RUN", "", symbols, start, end)
    if providers is None:
        creds = required_credentials(cfg)
        missing = [n for n, ok in creds.items() if not ok]
        if missing:
            audit.reason = ("credentials not configured (environment variables unset): " + ", ".join(missing)
                            + ". Nothing was fetched; real-data suitability is UNKNOWN.")
            audit.checks.append(AuditCheck("credentials", "FAIL", audit.reason, creds))
            return _persist(ctx, audit)
    svc = IngestionService(cfg, ctx.store, ctx.db, providers=providers)
    rep = svc.ingest_all(start, end, symbols=symbols, run_id=audit.audit_id)
    out = rep.summary()
    access_ok = out.get("bars", {}).get("status") == "ok" and out.get("corporate_actions", {}).get("status") == "ok"
    audit.checks.append(AuditCheck("access", "PASS" if access_ok else "FAIL",
                                   f"ingest outcome per kind: { {k: v['status'] for k, v in out.items()} }"
                                   + ("" if access_ok else " (bars AND corporate actions are both required)"), out))
    bar_ids = rep.outcomes.get("bars").dataset_ids if rep.outcomes.get("bars") else []
    if not bar_ids:
        audit.status, audit.reason = "FAILED", f"no bars ingested: {out.get('bars')}"
        return _persist(ctx, audit)
    act_ids = rep.outcomes["corporate_actions"].dataset_ids if "corporate_actions" in rep.outcomes else []
    audit.dataset_ids = bar_ids + act_ids
    rows = ctx.db.fetchall(f"SELECT params_json, is_synthetic FROM datasets WHERE dataset_id IN "
                           f"({','.join('?' for _ in bar_ids)})", bar_ids)
    feeds = {str((from_json(r["params_json"], {}) or {}).get("feed")) for r in rows}
    synthetic = any(r["is_synthetic"] for r in rows)
    audit.feed = ",".join(sorted(feeds))
    bars = ctx.store.load("bars", bar_ids)
    actions = ctx.store.load("corporate_actions", act_ids)
    bundle = ctx.store.load_bundle(cfg.section("benchmarks"), snapshot={"bars": bar_ids, "corporate_actions": act_ids},
                                   synthetic=synthetic)
    audit.checks += audit_checks(bundle, bars, actions, next(iter(feeds)) if len(feeds) == 1 else None, symbols, now,
                                 requested_start=start, requested_end=end)
    audit.checks += _sec_checks(ctx, rep, symbols)
    audit.status, audit.reason = verdict(audit.checks, audit.feed, synthetic,
                                         corporate_actions_ok=out.get("corporate_actions", {}).get("status") == "ok")
    return _persist(ctx, audit)


def _sec_checks(ctx, rep, symbols: list[str]) -> list[AuditCheck]:
    """Earnings-event timing and fundamentals: counts and how availability was established.
    Events/facts without an exact acceptance time are PIT_CONSERVATIVE; if none arrived the
    event/fundamental strategies must treat those inputs as UNKNOWN (they produce no signal)."""
    out = []
    for kind, label in (("events", "sec_earnings_events"), ("fundamentals", "sec_fundamentals")):
        o = rep.outcomes.get(kind)
        if o is None or o.status != "ok":
            out.append(AuditCheck(label, "WARN", f"{kind}: {o.status if o else 'not attempted'} "
                                  f"({o.detail if o else ''}); dependent features will be UNKNOWN (no signal)"))
            continue
        df = ctx.store.load(kind, o.dataset_ids)
        pit = df["pit_status"].value_counts().to_dict()
        per = df["symbol"].value_counts().reindex([s for s in symbols], fill_value=0).to_dict()
        out.append(AuditCheck(label, "PASS", f"{len(df)} rows; per symbol {per}; availability status {pit}",
                              {"per_symbol": per, "pit_status": pit}))
    return out


def _persist(ctx, audit: DataAudit) -> DataAudit:
    ctx.db.insert("data_audits", {
        "audit_id": audit.audit_id, "created_at": audit.created_at, "status": audit.status, "reason": audit.reason,
        "symbols_json": to_json(audit.symbols), "start_date": audit.start, "end_date": audit.end, "feed": audit.feed,
        "dataset_ids_json": to_json(audit.dataset_ids), "checks_json": to_json([asdict(c) for c in audit.checks])})
    out = ctx.config.path("project.report_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"data_audit_{audit.audit_id}.json").write_text(json.dumps(audit.to_dict(), indent=2, default=str), encoding="utf-8")
    return audit


def latest_audit(db) -> dict[str, Any] | None:
    return db.fetchone("SELECT * FROM data_audits ORDER BY created_at DESC LIMIT 1")
