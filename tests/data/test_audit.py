"""Real-data audit machinery, exercised offline with crafted data (never presented as market data)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.core.calendar import TradingCalendar
from quantlab.data.audit import audit_checks, run_data_audit, verdict
from quantlab.data.panel import DataBundle, build_panel
from quantlab.data.providers.synthetic import SyntheticMarket, SyntheticSpec
from quantlab.testing.fixtures import SYNTHETIC_BENCHMARKS

NOW = datetime(2021, 1, 4, 23, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def mkt():
    return SyntheticMarket(SyntheticSpec(n_stocks=40, start="2019-01-02", end="2020-12-31", seed=42))


def _bundle(bars, actions):
    cal = TradingCalendar.from_dates(sorted(bars.loc[bars["symbol"] == "SPY", "date"].unique()))
    return DataBundle(build_panel(bars, actions, calendar=cal), cal, benchmarks=SYNTHETIC_BENCHMARKS)


def _split_symbols(mkt):
    a = mkt.world["corporate_actions"]
    return sorted(a.loc[a["action_type"] == "split", "symbol"].unique())


def _checks(bars, actions, feed, symbols):
    return {c.name: c for c in audit_checks(_bundle(bars, actions), bars, actions, feed, symbols, now=NOW)}


def _no_holiday_bars(bars):
    """The synthetic generator uses plain weekdays; drop the fixed NYSE holidays so session checks
    reflect what a correct provider would deliver."""
    from quantlab.data.audit import _nyse_rule_holidays
    hol = _nyse_rule_holidays(range(2019, 2021))
    return bars[~bars["date"].isin(hol)].reset_index(drop=True)


def test_clean_sip_like_sample_passes(mkt):
    syms = ["SPY"] + _split_symbols(mkt)[:3]
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(syms)]
    acts = mkt.world["corporate_actions"]
    c = _checks(bars, acts[acts["symbol"].isin(syms)], "sip", syms)
    assert c["feed"].status == "PASS"
    assert c["sessions_timezone"].status == "PASS", c["sessions_timezone"].detail
    assert c["integrity:duplicate_bars"].status == "PASS"
    assert c["volume_quality"].status == "PASS"
    assert c["corporate_actions_splits"].status == "PASS", c["corporate_actions_splits"].data
    st, _ = verdict(list(c.values()), "sip", synthetic=False)
    assert st == "SUITABLE_SMALL_SAMPLE_ONLY"


def test_iex_like_volume_is_a_data_limitation(mkt):
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(["SPY", "SYN001"])].copy()
    bars.loc[bars["symbol"] == "SPY", "volume"] = 1.5e6          # single-venue sized volume
    c = _checks(bars, mkt.world["corporate_actions"].iloc[0:0], "iex", ["SPY", "SYN001"])
    assert c["feed"].status == "FAIL" and c["volume_quality"].status == "FAIL"
    st, why = verdict(list(c.values()), "iex", synthetic=False)
    assert st == "DATA_LIMITATION" and "IEX" in why


def test_pre_adjusted_bars_are_detected(mkt):
    sym = _split_symbols(mkt)[0]
    acts = mkt.world["corporate_actions"]
    split = acts[(acts["symbol"] == sym) & (acts["action_type"] == "split")].iloc[0]
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(["SPY", sym])].copy()
    before = (bars["symbol"] == sym) & (bars["date"] < split["ex_date"])
    for col in ("open", "high", "low", "close"):
        bars.loc[before, col] = bars.loc[before, col] / split["ratio"]      # what split-ADJUSTED bars look like
    c = _checks(bars, acts[acts["symbol"] == sym], "sip", ["SPY", sym])
    assert c["corporate_actions_splits"].status == "FAIL"
    assert verdict(list(c.values()), "sip", synthetic=False)[0] == "FAILED"


def test_shifted_session_dates_are_detected(mkt):
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(["SPY", "SYN001"])].copy()
    bars["date"] = bars["date"] + pd.Timedelta(days=1)          # e.g. taking the UTC date of a late timestamp
    c = _checks(bars, mkt.world["corporate_actions"].iloc[0:0], "sip", ["SPY", "SYN001"])
    assert c["sessions_timezone"].status == "FAIL"


def test_duplicates_fail(mkt):
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(["SPY", "SYN001"])]
    dup = pd.concat([bars, bars.iloc[:3]], ignore_index=True)
    c = {x.name: x for x in audit_checks(_bundle(bars, mkt.world["corporate_actions"].iloc[0:0]), dup,
                                           mkt.world["corporate_actions"].iloc[0:0], "sip", ["SPY", "SYN001"], now=NOW)}
    assert c["integrity:duplicate_bars"].status == "FAIL"


def test_audit_without_credentials_records_not_run(ctx, monkeypatch):
    for n in ("ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY", "QUANTLAB_SEC_USER_AGENT"):
        monkeypatch.delenv(n, raising=False)
    a = run_data_audit(ctx)
    assert a.status == "NOT_RUN" and "ALPACA_PAPER_KEY_ID" in a.reason
    row = ctx.db.fetchone("SELECT status FROM data_audits WHERE audit_id=?", (a.audit_id,))
    assert row["status"] == "NOT_RUN"
    assert (ctx.config.path("project.report_dir") / f"data_audit_{a.audit_id}.json").is_file()
    assert ctx.db.fetchone("SELECT COUNT(*) AS n FROM datasets")["n"] == 0      # nothing fetched or faked


def test_audit_with_synthetic_providers_is_labelled_synthetic(ctx, mkt):
    providers = {k: mkt for k in ("prices", "corporate_actions", "reference", "events", "fundamentals", "news")}
    a = run_data_audit(ctx, symbols=["SPY", "SYN001", "SYN002"], start="2019-01-02", end="2020-12-31",
                       providers=providers, now=NOW)
    assert a.status == "SYNTHETIC_ONLY"
    names = {c.name for c in a.checks}
    assert {"access", "feed", "coverage", "sessions_timezone", "volume_quality", "sec_earnings_events"} <= names


# --- regressions from the adversarial review ---------------------------------------------------
def test_nyse_calendar_matches_known_session_counts():
    from quantlab.data.audit import expected_sessions
    for year, n in [(2019, 252), (2020, 253), (2021, 252), (2022, 251), (2023, 250), (2024, 252)]:
        assert len(expected_sessions(f"{year}-01-01", f"{year}-12-31")) == n, year


def test_missing_corporate_actions_fail_the_verdict(mkt):
    syms = ["SPY"] + _split_symbols(mkt)[:2]
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(syms)]
    c = _checks(bars, mkt.world["corporate_actions"].iloc[0:0], "sip", syms)
    st, why = verdict(list(c.values()), "sip", synthetic=False, corporate_actions_ok=False)
    assert st == "FAILED" and "corporate actions" in why


def test_unknown_decisive_check_is_inconclusive_not_suitable(mkt):
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(["SPY", "SYN001"])]
    c = _checks(bars, mkt.world["corporate_actions"].iloc[0:0], "sip", ["SPY", "SYN001"])
    assert c["corporate_actions_splits"].status == "UNKNOWN"          # no split in this sample
    assert verdict(list(c.values()), "sip", synthetic=False)[0] == "INCONCLUSIVE"


def test_late_history_and_shared_calendar_holes_are_caught(mkt):
    from quantlab.data.audit import audit_checks, expected_sessions
    syms = ["SPY", "SYN001"]
    bars = _no_holiday_bars(mkt.world["bars"])
    bars = bars[bars["symbol"].isin(syms)]
    late = bars[~((bars["symbol"] == "SYN001") & (bars["date"] < "2020-04-01"))]
    c = {x.name: x for x in audit_checks(_bundle(late, mkt.world["corporate_actions"].iloc[0:0]), late,
                                           mkt.world["corporate_actions"].iloc[0:0], "sip", syms, now=NOW,
                                           requested_start="2019-01-02")}
    assert c["coverage"].status == "FAIL" and "SYN001" in c["coverage"].data["late"]
    exp = expected_sessions("2019-06-03", "2019-07-31")
    holed = bars[~bars["date"].isin(exp[:40])]                         # 40 sessions gone for EVERY symbol
    c2 = {x.name: x for x in audit_checks(_bundle(holed, mkt.world["corporate_actions"].iloc[0:0]), holed,
                                            mkt.world["corporate_actions"].iloc[0:0], "sip", syms, now=NOW)}
    assert c2["missing_sessions"].status == "FAIL"
