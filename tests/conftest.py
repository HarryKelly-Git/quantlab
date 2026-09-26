"""Shared pytest fixtures. All market data in tests is SYNTHETIC (see quantlab.data.providers.synthetic)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.data.providers.synthetic import SyntheticMarket, SyntheticSpec
from quantlab.db.database import open_db
from quantlab.testing.fixtures import SYNTHETIC_BENCHMARKS, make_synthetic_bundle

os.environ["QUANTLAB_SKIP_LOCAL_CONFIG"] = "1"   # a developer's config/local.yaml never changes test outcomes

ROOT = Path(__file__).resolve().parents[1]

SMALL_SPEC = SyntheticSpec(n_stocks=40, start="2018-01-02", end="2021-12-31", seed=42)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QUANTLAB_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test (set QUANTLAB_NETWORK_TESTS=1 to run)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def config(tmp_path):
    """Default config with all runtime paths redirected into tmp_path."""
    var = tmp_path / "var"
    return load_config(root=ROOT, overrides={
        "project": {
            "var_dir": str(var), "db_path": str(var / "quantlab.db"), "data_dir": str(var / "data"),
            "log_dir": str(var / "logs"), "report_dir": str(var / "reports"), "model_dir": str(var / "models"),
        },
        "benchmarks": SYNTHETIC_BENCHMARKS,
        "providers": {"prices": "synthetic", "corporate_actions": "synthetic", "reference": "synthetic",
                      "fundamentals": "synthetic", "events": "synthetic", "news": "synthetic"},
    })


@pytest.fixture
def db(tmp_path):
    d = open_db(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def ctx(config):
    c = AppContext.create(config, init_logging=False)
    yield c
    c.close()


@pytest.fixture(scope="session")
def synthetic_market() -> SyntheticMarket:
    return SyntheticMarket(SMALL_SPEC)


@pytest.fixture(scope="session")
def bundle(synthetic_market):
    """Full-history synthetic DataBundle (null world: no planted edge). Treat as read-only."""
    return make_synthetic_bundle(market=synthetic_market)
