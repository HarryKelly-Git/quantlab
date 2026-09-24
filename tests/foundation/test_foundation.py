"""Foundation contracts: config safety, DB immutability, store versioning, PIT panel construction."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from quantlab.config import ConfigError, load_config
from quantlab.core.calendar import TradingCalendar
from quantlab.core.types import PitStatus
from quantlab.data import schemas
from quantlab.data.panel import build_panel
from quantlab.data.store import DataStoreError, MarketDataStore
from quantlab.secrets import Secret, redact
from quantlab.testing.pit import LookaheadError, assert_truncation_invariant

from ..conftest import ROOT


# --- config ---------------------------------------------------------------------------------
def test_config_loads_and_hashes(config):
    assert config.get("safety.paper_only") is True
    assert len(config.hash) == 16
    assert config.with_overrides({"backtest": {"max_positions": 3}}).hash != config.hash


def test_config_rejects_live_broker_url():
    with pytest.raises(ConfigError):
        load_config(root=ROOT, overrides={"safety": {"allowed_broker_base_urls": ["https://api.alpaca.markets"]}})


def test_config_rejects_paper_only_false():
    with pytest.raises(ConfigError):
        load_config(root=ROOT, overrides={"safety": {"paper_only": False}})


def test_secret_never_reveals_in_repr(monkeypatch):
    s = Secret("X_KEY", "supersecretvalue123")
    assert "supersecret" not in repr(s) and "supersecret" not in str(s)
    monkeypatch.setenv("SOME_API_KEY", "abcdefgh12345678")
    assert "abcdefgh12345678" not in redact("token=abcdefgh12345678")


# --- database ---------------------------------------------------------------------------------
def test_migrations_idempotent(db):
    assert db.migrate() == []
    assert "human_decisions" in db.tables()


@pytest.mark.parametrize("table,row", [
    ("human_decisions", {"decision_id": "h1", "symbol": "AAA", "action": "BUY", "decided_at": "2024-01-02T00:00:00+00:00",
                         "ref_session_date": "2024-01-01", "created_at": "2024-01-02T00:00:00+00:00"}),
    ("research_ledger", {"entry_date": "2024-01-01", "entry_type": "idea", "text": "x", "author": "human",
                         "created_at": "2024-01-01T00:00:00+00:00"}),
])
def test_audit_tables_are_append_only(db, table, row):
    db.insert(table, row)
    first_col = list(row)[0]
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(f"UPDATE {table} SET {first_col}={first_col}")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(f"DELETE FROM {table}")


def test_transaction_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.insert("hypotheses", {"hypothesis_id": "x", "created_at": "t", "source": "human", "title": "t",
                                     "hypothesis": "h", "updated_at": "t"})
            raise RuntimeError("boom")
    assert db.fetchone("SELECT * FROM hypotheses WHERE hypothesis_id='x'") is None


# --- calendar -------------------------------------------------------------------------------
def test_cutoff_and_availability():
    cal = TradingCalendar.business_days("2024-01-01", "2024-01-31")
    d = pd.Timestamp("2024-01-10")
    # 15:59 ET on D -> usable on D; 16:01 ET on D -> usable on D+1
    before = pd.Timestamp("2024-01-10 15:59", tz="America/New_York")
    after = pd.Timestamp("2024-01-10 16:01", tz="America/New_York")
    assert cal.first_usable_session(before) == d
    assert cal.first_usable_session(after) == pd.Timestamp("2024-01-11")
    # pre-market release reacts same day, after-close release reacts next session
    assert cal.reaction_session(pd.Timestamp("2024-01-10 07:00", tz="America/New_York")) == d
    assert cal.reaction_session(pd.Timestamp("2024-01-10 16:05", tz="America/New_York")) == pd.Timestamp("2024-01-11")
    with pytest.raises(ValueError):
        cal.first_usable_session(pd.Timestamp("2024-01-10 10:00"))  # naive timestamps rejected


# --- panel: PIT-exact returns -----------------------------------------------------------------
def test_panel_returns_match_truth_through_splits_and_dividends(synthetic_market, bundle):
    truth = synthetic_market.world["_truth_returns"]
    ret = bundle.panel.ret[truth.columns]
    both = ret.notna() & truth.reindex(ret.index).notna()
    diff = (ret - truth.reindex(ret.index)).abs()[both]
    assert both.values.sum() > 1000
    assert float(np.nanmax(diff.values)) < 1e-9
    acts = synthetic_market.world["corporate_actions"]
    assert (acts["action_type"] == "split").any() and (acts["action_type"] == "cash_dividend").any()


def test_reverse_split_does_not_leak_into_raw_levels(bundle, synthetic_market):
    acts = synthetic_market.world["corporate_actions"]
    rs = acts[(acts["action_type"] == "split") & (acts["ratio"] < 1)]
    if rs.empty:
        pytest.skip("no reverse split in this synthetic seed")
    row = rs.iloc[0]
    sym, ex = row["symbol"], row["ex_date"]
    close = bundle.panel.close[sym].dropna()
    before = close[close.index < ex].iloc[-1]
    assert before < 1.5  # raw pre-split price stays a penny price historically


def test_panel_truncation_equals_rebuild(synthetic_market, bundle):
    w = synthetic_market.world
    d = bundle.panel.dates[400]
    bars = w["bars"][w["bars"]["date"] <= d]
    acts = w["corporate_actions"][w["corporate_actions"]["ex_date"] <= d]
    cal = TradingCalendar.from_dates(bundle.panel.dates[bundle.panel.dates <= d])
    rebuilt = build_panel(bars, acts, calendar=cal)
    trunc = bundle.panel.truncate(d)
    for f in ["ret", "close", "volume", "dollar_volume"]:
        pd.testing.assert_frame_equal(rebuilt.fields[f], trunc.fields[f][rebuilt.fields[f].columns], check_freq=False)
    # tri-scaled fields: ratios must match (scale is arbitrary but identical from the same start)
    r1 = rebuilt.aclose / rebuilt.aclose.shift(20)
    r2 = trunc.aclose[rebuilt.aclose.columns] / trunc.aclose[rebuilt.aclose.columns].shift(20)
    pd.testing.assert_frame_equal(r1, r2, check_freq=False)


def test_truncation_invariance_detects_lookahead(bundle):
    good = lambda b: b.panel.aclose / b.panel.aclose.rolling(20).mean() - 1
    assert_truncation_invariant(good, bundle, name="ma_distance")
    leaky_centered = lambda b: b.panel.aclose / b.panel.aclose.rolling(21, center=True).mean() - 1
    with pytest.raises(LookaheadError):
        assert_truncation_invariant(leaky_centered, bundle, name="centered_ma")
    leaky_norm = lambda b: (b.panel.ret - b.panel.ret.mean()) / b.panel.ret.std()
    with pytest.raises(LookaheadError):
        assert_truncation_invariant(leaky_norm, bundle, name="full_sample_zscore")


def test_bundle_truncate_filters_events_by_availability(bundle):
    d = bundle.panel.dates[300]
    t = bundle.truncate(d)
    cutoff = bundle.calendar.cutoff(d)
    assert (pd.to_datetime(t.events["available_at"], utc=True) <= cutoff).all()
    assert (pd.to_datetime(t.fundamentals["available_at"], utc=True) <= cutoff).all()
    assert len(t.events) < len(bundle.events)
    # restated fundamentals: later filing is invisible before it was filed
    f = bundle.fundamentals
    restated = f[f["accession"].str.contains("restate")]
    if len(restated):
        r = restated.iloc[0]
        early = bundle.truncate(bundle.calendar.last_session_on_or_before(r["filed_date"] - pd.Timedelta(days=1)))
        assert r["accession"] not in set(early.fundamentals["accession"])


# --- store ----------------------------------------------------------------------------------
def test_store_versioning_and_synthetic_isolation(db, tmp_path, synthetic_market):
    store = MarketDataStore(tmp_path / "data", db)
    bars = synthetic_market.get_daily_bars(["SYN001", "SPY"], "2018-01-01", "2018-12-31")
    a = store.write("bars", bars, "synthetic", is_synthetic=True)
    assert store.write("bars", bars, "synthetic", is_synthetic=True) == a  # content-addressed
    loaded = store.load("bars", [a])
    assert len(loaded) == len(bars)
    # a "real" dataset cannot be mixed with synthetic
    real = bars.assign(provider="alpaca")
    b = store.write("bars", real.assign(close=real["close"] * 1.0001), "alpaca", is_synthetic=False)
    with pytest.raises(DataStoreError):
        store.load("bars", [a, b])


def test_schema_conform_rejects_naive_timestamps(synthetic_market):
    ev = synthetic_market.world["events"].copy()
    ev["available_at"] = ev["available_at"].dt.tz_localize(None)
    with pytest.raises(schemas.SchemaError):
        schemas.conform("events", ev)


def test_pit_status_ordering():
    assert PitStatus.weakest([PitStatus.PIT, PitStatus.UNKNOWN]) is PitStatus.UNKNOWN
    assert PitStatus.weakest([PitStatus.PIT, PitStatus.ASSUMED_STATIC]) is PitStatus.ASSUMED_STATIC


def test_first_usable_sessions_vectorized_matches_scalar():
    cal = TradingCalendar.business_days("2024-01-01", "2024-01-31")
    ts = pd.Series(pd.to_datetime(["2024-01-10 15:59", "2024-01-10 16:01", "2024-01-12 23:00"]).tz_localize("America/New_York"))
    got = cal.first_usable_sessions(ts)
    assert list(got) == [cal.first_usable_session(t) for t in ts]
