"""Builds the six dataset-kind provider instances from ``config.providers.*``.

Construction never requires credentials: an unconfigured provider (e.g. Alpaca with no keys set,
or SEC EDGAR with no declared User-Agent) can always be built here, and only raises
:class:`quantlab.data.providers.base.ProviderNotConfigured` the first time a fetch method actually
runs. This lets startup code build the whole registry unconditionally (e.g. to inspect which
kinds are wired up) without needing every secret to be present yet.

The same backend, when selected for more than one kind (e.g. ``alpaca`` for both ``prices`` and
``news``, or ``synthetic`` for everything), is constructed exactly ONCE and shared across those
kinds -- most importantly for ``synthetic``, where every kind must draw from the identical
in-memory :class:`~quantlab.data.providers.synthetic.SyntheticMarket` world.
"""
from __future__ import annotations

from typing import Any, Callable

from quantlab.core.calendar import TradingCalendar
from quantlab.data.providers.alpaca_data import AlpacaDataProvider
from quantlab.data.providers.base import (
    CorporateActionProvider,
    EventProvider,
    FundamentalsProvider,
    NewsProvider,
    Provider,
    PriceProvider,
    ProviderError,
    ReferenceProvider,
)
from quantlab.data.providers.nasdaq_symbols import NasdaqTraderReferenceProvider
from quantlab.data.providers.sec_edgar import SecEdgarProvider
from quantlab.data.providers.synthetic import SyntheticMarket

KINDS = ["prices", "corporate_actions", "reference", "fundamentals", "events", "news"]

# Interface each kind's provider must implement (defensive: catches a kind/backend mismatch in
# config, e.g. ``events: alpaca``, at registry-build time instead of an AttributeError at fetch time).
_REQUIRED_BASE: dict[str, type[Provider]] = {
    "prices": PriceProvider,
    "corporate_actions": CorporateActionProvider,
    "reference": ReferenceProvider,
    "fundamentals": FundamentalsProvider,
    "events": EventProvider,
    "news": NewsProvider,
}

# Backend name -> constructor. "synthetic" and "none" are handled specially in build_providers.
_CONSTRUCTORS: dict[str, Callable[[Any, TradingCalendar | None], Provider]] = {
    "alpaca": lambda config, calendar: AlpacaDataProvider(config),
    "nasdaq_trader": lambda config, calendar: NasdaqTraderReferenceProvider(config),
    "sec_edgar": lambda config, calendar: SecEdgarProvider(config, calendar=calendar),
}

KNOWN_BACKENDS = sorted({*_CONSTRUCTORS, "synthetic", "none"})


def build_providers(config: Any, calendar: TradingCalendar | None = None) -> dict[str, Provider | None]:
    """{kind: provider instance or None}, one entry per :data:`KINDS`, per ``config.providers.<kind>``."""
    built: dict[str, Provider] = {}
    out: dict[str, Provider | None] = {}
    for kind in KINDS:
        name = str(config.get(f"providers.{kind}")).strip().lower()
        if name == "none":
            out[kind] = None
            continue
        if name not in built:
            if name == "synthetic":
                built[name] = SyntheticMarket()
            elif name in _CONSTRUCTORS:
                built[name] = _CONSTRUCTORS[name](config, calendar)
            else:
                raise ProviderError(
                    f"config providers.{kind} = {name!r} is not a known backend "
                    f"(expected one of {KNOWN_BACKENDS})"
                )
        provider = built[name]
        required = _REQUIRED_BASE[kind]
        if not isinstance(provider, required):
            raise ProviderError(
                f"config providers.{kind} = {name!r} builds a {type(provider).__name__}, which does "
                f"not implement {required.__name__}"
            )
        out[kind] = provider
    # Compatibility alias: quantlab.data.ingest.IngestionService (a sibling module this task does
    # not own/modify) looks up the price provider as ``self.providers.get("bars")`` -- the dataset
    # KIND name from data/schemas.py -- while config.providers.* (and this registry's own KINDS,
    # per the operating instructions) name the same slot "prices". Both keys point at the one
    # provider instance so ingestion of bars (the one dataset ingest.py treats as load-bearing)
    # works regardless of which vocabulary the caller uses.
    out["bars"] = out["prices"]
    return out
