"""Provider interfaces. QuantLab never depends on one vendor: each dataset kind has an abstract
provider, and config (``providers.*``) selects the implementation.

Every implementation must:
  * return frames that pass :func:`quantlab.data.schemas.conform` for its kind;
  * set ``available_at`` / ``pit_status`` honestly (UNKNOWN when availability cannot be established);
  * set ``provider`` and ``retrieved_at``;
  * raise :class:`ProviderError` on failure — never return partial data silently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class ProviderError(RuntimeError):
    """A provider failed to deliver complete, valid data."""


class ProviderNotConfigured(ProviderError):
    """Required credentials / settings are missing (e.g. no API key in the environment)."""


class Provider(ABC):
    name: str = "abstract"
    is_synthetic: bool = False


class PriceProvider(Provider):
    @abstractmethod
    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """RAW (unadjusted) daily OHLCV, schema ``bars``."""


class CorporateActionProvider(Provider):
    @abstractmethod
    def get_corporate_actions(self, symbols: list[str] | None, start: date, end: date) -> pd.DataFrame:
        """Splits and cash dividends, schema ``corporate_actions``."""


class ReferenceProvider(Provider):
    @abstractmethod
    def get_securities(self) -> pd.DataFrame:
        """Security master (current snapshot), schema ``reference``. pit_status ASSUMED_STATIC."""


class FundamentalsProvider(Provider):
    @abstractmethod
    def get_fundamentals(self, symbols: list[str], concepts: list[str] | None = None) -> pd.DataFrame:
        """Reported financial facts with filing availability, schema ``fundamentals``."""


class EventProvider(Provider):
    @abstractmethod
    def get_earnings_events(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Earnings-release events with exact public timestamps, schema ``events``."""


class NewsProvider(Provider):
    @abstractmethod
    def get_news(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """News items with creation timestamps, schema ``news``."""
