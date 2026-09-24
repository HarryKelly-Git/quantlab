from __future__ import annotations

from quantlab.features.base import FeatureSet
from quantlab.regime import RegimeEngine
from quantlab.testing.pit import assert_truncation_invariant


def test_regime_labels_and_pit(bundle, config):
    eng = RegimeEngine(config)
    df = eng.compute(FeatureSet(bundle))
    assert set(df["label"].unique()) <= {"bull_calm", "bull_volatile", "bear_calm", "bear_volatile", "unknown"}
    assert (df["label"].iloc[:150] == "unknown").all()          # 200-session trend not known yet
    assert df["label"].iloc[-1] != "unknown"
    numeric = lambda b: eng.compute(FeatureSet(b)).drop(columns="label")
    assert_truncation_invariant(numeric, bundle, min_history=300, name="regime")


def test_regime_snapshot_persists(bundle, config, db):
    snap = RegimeEngine(config).persist(db, FeatureSet(bundle), bundle.panel.dates[-1], run_id="r1")
    assert snap["info_kind"] == "MODEL_OUTPUT"
    row = db.fetchone("SELECT label FROM regime_snapshots")
    assert row["label"] == snap["label"]
