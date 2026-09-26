"""Configuration loading, validation and hashing.

Load order (later wins): config/default.yaml -> config/local.yaml (git-ignored) -> $QUANTLAB_CONFIG
-> explicit ``overrides`` dict. Secrets never live in config; config only names the environment
variables that hold them (see :mod:`quantlab.secrets`).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets"


class ConfigError(ValueError):
    pass


def find_project_root(start: Path | None = None) -> Path:
    """Directory containing ``config/default.yaml``. Honors $QUANTLAB_ROOT."""
    env_root = os.environ.get("QUANTLAB_ROOT")
    if env_root:
        return Path(env_root).resolve()
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parents[2])  # src/quantlab/config.py -> repo root
    for base in candidates:
        for p in [base, *base.parents]:
            if (p / "config" / "default.yaml").is_file():
                return p
    raise ConfigError("Could not locate project root (config/default.yaml). Set QUANTLAB_ROOT.")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --- validation of safety/research-critical sections -------------------------------------------

class _Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class SafetySection(_Section):
    paper_only: bool
    allowed_broker_base_urls: list[str]

    @field_validator("paper_only")
    @classmethod
    def _must_be_paper(cls, v: bool) -> bool:
        if v is not True:
            raise ValueError("safety.paper_only must be true: QuantLab never trades live money")
        return v

    @field_validator("allowed_broker_base_urls")
    @classmethod
    def _only_paper_urls(cls, v: list[str]) -> list[str]:
        for url in v:
            if url.rstrip("/") != PAPER_TRADING_BASE_URL:
                raise ValueError(f"non-paper broker URL not allowed in config: {url!r}")
        return v


class HoldoutSection(_Section):
    start: date


class ValidationSection(_Section):
    holdout: HoldoutSection


class CostsSection(_Section):
    half_spread_bps_tiers: list[tuple[float, float]]
    slippage_bps: float
    commission_per_share: float
    delisting_return: float

    @field_validator("slippage_bps", "commission_per_share")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cost parameters must be non-negative")
        return v


class PortfolioSection(_Section):
    max_positions: int
    max_gross_exposure: float
    max_position_weight: float
    max_sector_weight: float
    risk_per_trade: float

    @field_validator("max_gross_exposure")
    @classmethod
    def _no_leverage_by_default(cls, v: float) -> float:
        if v <= 0 or v > 2.0:
            raise ValueError("portfolio.max_gross_exposure must be in (0, 2]")
        return v


_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "safety": SafetySection,
    "validation": ValidationSection,
    "costs": CostsSection,
    "portfolio": PortfolioSection,
}


class Config:
    """Immutable-ish view over the merged configuration dictionary."""

    def __init__(self, data: dict[str, Any], root: Path, sources: list[str]):
        self._data = data
        self.root = root
        self.sources = sources
        self._validate()

    # -- access -----------------------------------------------------------------------------
    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is ...:
                    raise KeyError(f"config key not found: {dotted}")
                return default
        return copy.deepcopy(node)

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"config section {name!r} is not a mapping")
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def path(self, dotted: str) -> Path:
        """Resolve a path-valued key relative to the project root."""
        p = Path(self.get(dotted))
        return p if p.is_absolute() else (self.root / p)

    @property
    def hash(self) -> str:
        return config_hash(self._data)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        return Config(_deep_merge(self._data, overrides), self.root, [*self.sources, "<overrides>"])

    # -- validation -------------------------------------------------------------------------
    def _validate(self) -> None:
        for name, model in _SECTION_MODELS.items():
            if name not in self._data:
                raise ConfigError(f"missing required config section: {name}")
            try:
                model.model_validate(self._data[name])
            except Exception as exc:  # pydantic.ValidationError
                raise ConfigError(f"invalid config section {name!r}: {exc}") from exc


def config_hash(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> Config:
    root_path = Path(root).resolve() if root else find_project_root()
    sources: list[str] = []
    default_path = root_path / "config" / "default.yaml"
    data = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    sources.append(str(default_path))

    # QUANTLAB_SKIP_LOCAL_CONFIG=1 (set by the test-suite): a developer's git-ignored local.yaml
    # must never change test outcomes
    layers: list[Path] = [] if os.environ.get("QUANTLAB_SKIP_LOCAL_CONFIG") == "1" else [root_path / "config" / "local.yaml"]
    env_path = os.environ.get("QUANTLAB_CONFIG")
    if env_path:
        layers.append(Path(env_path))
    if path:
        layers.append(Path(path))
    for layer in layers:
        if layer.is_file():
            data = _deep_merge(data, yaml.safe_load(layer.read_text(encoding="utf-8")) or {})
            sources.append(str(layer))
    if overrides:
        data = _deep_merge(data, overrides)
        sources.append("<overrides>")
    return Config(data, root_path, sources)
