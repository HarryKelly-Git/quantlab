from __future__ import annotations

import pytest

from tests.data.fakes import FakeClock


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Every test starts with no provider credentials set, so ProviderNotConfigured tests are
    reliable regardless of the developer's real shell environment."""
    for var in ("ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY", "QUANTLAB_SEC_USER_AGENT",
                "TEST_ALPACA_KEY", "TEST_ALPACA_SECRET", "TEST_SEC_UA"):
        monkeypatch.delenv(var, raising=False)
