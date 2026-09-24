"""registry.build_providers: kind->backend dispatch, sharing, and no-keys-at-construction contract."""
from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.providers.alpaca_data import AlpacaDataProvider
from quantlab.data.providers.base import ProviderError, ProviderNotConfigured
from quantlab.data.providers.nasdaq_symbols import NasdaqTraderReferenceProvider
from quantlab.data.providers.registry import KINDS, build_providers
from quantlab.data.providers.sec_edgar import SecEdgarProvider
from quantlab.data.providers.synthetic import SyntheticMarket
from tests.data.fakes import FakeConfig

ALL_SYNTHETIC = {"providers": {k: "synthetic" for k in KINDS}}
ALL_NONE = {"providers": {k: "none" for k in KINDS}}
REAL_BACKENDS = {"providers": {
    "prices": "alpaca", "corporate_actions": "alpaca", "news": "alpaca",
    "reference": "nasdaq_trader", "fundamentals": "sec_edgar", "events": "sec_edgar",
    "alpaca": {"key_id_env": "TEST_ALPACA_KEY", "secret_env": "TEST_ALPACA_SECRET"},
    "sec_edgar": {"user_agent_env": "TEST_SEC_UA"},
}}


def test_builds_synthetic_for_every_kind_as_one_shared_instance():
    out = build_providers(FakeConfig(ALL_SYNTHETIC))
    assert set(KINDS) <= set(out)
    assert all(isinstance(v, SyntheticMarket) for v in out.values())
    first = next(iter(out.values()))
    assert all(v is first for v in out.values())  # one shared world, not six-or-seven separate ones


def test_bars_alias_matches_prices_for_ingest_compatibility():
    # quantlab.data.ingest.IngestionService looks up the price provider by the dataset-kind name
    # "bars" (schemas.py), not the config-key name "prices" (config.providers.*). Both must resolve
    # to the identical provider instance.
    out = build_providers(FakeConfig(ALL_SYNTHETIC))
    assert out["bars"] is out["prices"]


def test_builds_none_for_every_kind():
    out = build_providers(FakeConfig(ALL_NONE))
    assert all(v is None for v in out.values())


def test_mixed_none_and_synthetic():
    cfg = FakeConfig({"providers": {"prices": "synthetic", "corporate_actions": "none", "reference": "synthetic",
                                    "fundamentals": "none", "events": "none", "news": "synthetic"}})
    out = build_providers(cfg)
    assert isinstance(out["prices"], SyntheticMarket)
    assert out["corporate_actions"] is None
    assert out["prices"] is out["reference"] is out["news"]


def test_real_backends_construct_without_any_credentials_present(monkeypatch):
    for var in ("TEST_ALPACA_KEY", "TEST_ALPACA_SECRET", "TEST_SEC_UA"):
        monkeypatch.delenv(var, raising=False)
    out = build_providers(FakeConfig(REAL_BACKENDS))
    assert isinstance(out["prices"], AlpacaDataProvider)
    assert isinstance(out["reference"], NasdaqTraderReferenceProvider)
    assert isinstance(out["fundamentals"], SecEdgarProvider)
    # Same backend name shared across kinds that use it.
    assert out["prices"] is out["corporate_actions"] is out["news"]
    assert out["fundamentals"] is out["events"]
    # ProviderNotConfigured only surfaces once a fetch is attempted, not from build_providers itself.
    with pytest.raises(ProviderNotConfigured):
        out["prices"].get_daily_bars(["AAPL"], pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date())
    with pytest.raises(ProviderNotConfigured):
        out["fundamentals"].get_fundamentals(["AAPL"])


def test_unknown_backend_name_raises_provider_error():
    cfg = FakeConfig({"providers": {**{k: "synthetic" for k in KINDS}, "prices": "bloomberg"}})
    with pytest.raises(ProviderError):
        build_providers(cfg)


def test_kind_backend_interface_mismatch_raises_provider_error():
    # "reference" needs a ReferenceProvider; "alpaca" does not implement get_securities.
    cfg = FakeConfig({"providers": {
        "prices": "alpaca", "corporate_actions": "alpaca", "news": "alpaca",
        "reference": "alpaca",  # mismatch
        "fundamentals": "sec_edgar", "events": "sec_edgar",
        "alpaca": {"key_id_env": "TEST_ALPACA_KEY", "secret_env": "TEST_ALPACA_SECRET"},
        "sec_edgar": {"user_agent_env": "TEST_SEC_UA"},
    }})
    with pytest.raises(ProviderError):
        build_providers(cfg)


def test_separate_build_providers_calls_do_not_share_instances():
    cfg = FakeConfig(ALL_SYNTHETIC)
    out1 = build_providers(cfg)
    out2 = build_providers(cfg)
    assert out1["prices"] is not out2["prices"]
