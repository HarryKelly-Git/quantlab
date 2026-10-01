"""The ``options:`` config block as one validated, immutable object (every tunable lives in
config/default.yaml; this module only reads and checks it). Recorded with every evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class OptionsConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PaperSettings:
    allocation: float = 2000.0
    max_open_positions: int = 3
    max_contracts_per_trade: int = 5
    max_premium_per_trade: float = 500.0
    max_premium_total: float = 1500.0
    entry_mid_fraction: float = 0.0
    exit_mid_fraction: float = 0.0
    close_before_expiry_sessions: int = 2


@dataclass(frozen=True)
class OptionsSettings:
    enabled: bool = True
    paper_trading: bool = False
    feed: str = "indicative"
    underlying_feed: str = "iex"
    max_spread_pct: float = 0.10
    min_bid: float = 0.10
    max_quote_age_minutes: float = 30.0
    min_open_interest: float = 100.0
    unknown_open_interest_fails: bool = False
    expiry_buffer_sessions: int = 5
    max_expiry_sessions: int = 120
    grid_deltas: tuple[float, ...] = (0.70, 0.50, 0.30)
    grid_delta_tolerance: float = 0.10
    grid_include_atm: bool = True
    grid_include_expected_move: bool = True
    grid_spreads: bool = True
    risk_free_rate: float = 0.04
    exit_iv_multiplier: float = 1.0
    exit_at_executable_side: bool = True
    model_fee_per_contract: float = 0.0
    stock_round_trip_cost_bps: float = 10.0
    min_expected_pnl_per_risk: float = 0.0
    min_option_advantage_per_risk: float = 0.10
    distribution_lookback_sessions: int = 756
    default_stop_atr: float = 2.0
    paper: PaperSettings = field(default_factory=PaperSettings)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, config: Any, overrides: dict[str, Any] | None = None) -> "OptionsSettings":
        raw = dict(config.get("options", {}) or {})
        raw.update(overrides or {})
        grid = dict(raw.pop("strike_grid", {}) or {})
        paper = dict(raw.pop("paper", {}) or {})
        dist = dict(raw.pop("distribution", {}) or {})
        kw: dict[str, Any] = {}
        for k in ("enabled", "paper_trading", "unknown_open_interest_fails", "exit_at_executable_side"):
            if k in raw:
                kw[k] = _bool(raw[k], f"options.{k}")
        for k in ("feed", "underlying_feed"):
            if k in raw:
                kw[k] = str(raw[k]).lower()
        for k in ("max_spread_pct", "min_bid", "max_quote_age_minutes", "min_open_interest", "risk_free_rate",
                  "exit_iv_multiplier", "model_fee_per_contract", "stock_round_trip_cost_bps",
                  "min_expected_pnl_per_risk", "min_option_advantage_per_risk"):
            if k in raw:
                kw[k] = float(raw[k])
        for k in ("expiry_buffer_sessions", "max_expiry_sessions"):
            if k in raw:
                kw[k] = int(raw[k])
        if "deltas" in grid:
            kw["grid_deltas"] = tuple(float(x) for x in grid["deltas"])
        if "delta_tolerance" in grid:
            kw["grid_delta_tolerance"] = float(grid["delta_tolerance"])
        for src, dst in (("include_atm", "grid_include_atm"), ("include_expected_move", "grid_include_expected_move"),
                         ("spreads", "grid_spreads")):
            if src in grid:
                kw[dst] = _bool(grid[src], f"options.strike_grid.{src}")
        if "lookback_sessions" in dist:
            kw["distribution_lookback_sessions"] = int(dist["lookback_sessions"])
        if "default_stop_atr" in dist:
            kw["default_stop_atr"] = float(dist["default_stop_atr"])
        pkw: dict[str, Any] = {}
        for k in ("allocation", "max_premium_per_trade", "max_premium_total", "entry_mid_fraction", "exit_mid_fraction"):
            if k in paper:
                pkw[k] = float(paper[k])
        for k in ("max_open_positions", "max_contracts_per_trade", "close_before_expiry_sessions"):
            if k in paper:
                pkw[k] = int(paper[k])
        s = cls(**kw, paper=PaperSettings(**pkw))
        s.validate()
        return s

    def validate(self) -> None:
        errs = []
        if self.feed != "indicative":
            # opra returns HTTP 403 on this account (OPRA agreement not signed); see docs/OPTIONS.md
            errs.append(f"options.feed must be 'indicative' on this account, got {self.feed!r}")
        if not 0 < self.max_spread_pct <= 1:
            errs.append("options.max_spread_pct must be in (0, 1]")
        if self.min_bid < 0 or self.max_quote_age_minutes <= 0 or self.min_open_interest < 0:
            errs.append("options.min_bid / max_quote_age_minutes / min_open_interest must be non-negative (age > 0)")
        if self.expiry_buffer_sessions < 0 or self.max_expiry_sessions <= 0:
            errs.append("options.expiry_buffer_sessions must be >= 0 and max_expiry_sessions > 0")
        if any(not 0 < d < 1 for d in self.grid_deltas):
            errs.append("options.strike_grid.deltas must be in (0, 1)")
        if self.exit_iv_multiplier <= 0:
            errs.append("options.exit_iv_multiplier must be > 0")
        p = self.paper
        for name in ("entry_mid_fraction", "exit_mid_fraction"):
            v = getattr(p, name)
            if not 0.0 <= v <= 1.0:
                # > 1 would price a buy below the midpoint: never allowed
                errs.append(f"options.paper.{name} must be in [0, 1] (never better than mid)")
        if p.max_open_positions < 0 or p.max_contracts_per_trade < 1:
            errs.append("options.paper.max_open_positions >= 0 and max_contracts_per_trade >= 1")
        if p.max_premium_per_trade <= 0 or p.max_premium_total <= 0 or p.allocation <= 0:
            errs.append("options.paper premium limits and allocation must be > 0")
        if p.max_premium_total > p.allocation + 1e-9:
            errs.append("options.paper.max_premium_total must not exceed options.paper.allocation")
        if p.close_before_expiry_sessions < 1:
            errs.append("options.paper.close_before_expiry_sessions must be >= 1 (never hold into expiry)")
        if errs:
            raise OptionsConfigError("; ".join(errs))


def _bool(v: Any, name: str) -> bool:
    if isinstance(v, bool):
        return v
    raise OptionsConfigError(f"{name} must be true/false, got {v!r}")
