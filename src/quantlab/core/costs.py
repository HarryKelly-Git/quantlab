"""Transaction-cost model shared by the backtester, the simulated paper broker, the EV engine and
the shadow/counterfactual outcome evaluators — so every layer charges identical costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.config import Config
from quantlab.core.types import Side


@dataclass(frozen=True)
class CostModel:
    half_spread_tiers: tuple[tuple[float, float], ...]   # (min median dollar volume, half-spread bps), descending
    slippage_bps: float
    commission_per_share: float = 0.0
    commission_min_per_order: float = 0.0
    delisting_return: float = -0.30
    # A held symbol counts as DELISTED only after this many consecutive sessions without a bar while
    # the market keeps trading; the exit is booked on that session. Point-in-time: never decided by
    # looking ahead for bars that may or may not come back. Shared with the paper SimBroker.
    delisting_missing_sessions: int = 5
    # Suspicious-open rule (core.tradesim.suspicious_open_mask): a fill on a session whose open is
    # flagged uses the worse of open and close. Shared with data validation and the paper SimBroker.
    suspicious_open_threshold: float = 0.25
    suspicious_open_min_reversion: float = 0.5
    # How core.tradesim fills a stop: "close" (checked on the close, exit next open) or "intraday"
    # (a broker-held stop-market order: gap through -> the open, touch -> the stop price). Shared by
    # every caller of simulate_plan so backtests, shadow outcomes and the paper runner agree.
    stop_model: str = "close"
    # Where the broker-held stop rests, as a multiple of the plan's stop distance below the entry
    # reference (intraday model only): 1.0 = AT the plan stop; 1.75 = a wider DISASTER stop (3.5 ATR
    # for a 2-ATR plan stop) while the plan stop itself stays close-based (exit next open).
    broker_stop_distance: float = 1.0
    # A DELISTED symbol with a known MERGER record (panel field ``merger``: cash / stock /
    # stock-and-cash merger) is an acquisition: holders are paid about the last close (the deal
    # price), not a -30% haircut. The record counts when it takes effect on a session in
    # [last bar - delisting_merger_lookback_sessions, delisting session]. See delisting_exit_return.
    delisting_return_merger: float = 0.0
    delisting_merger_lookback_sessions: int = 5

    def __post_init__(self) -> None:
        if self.stop_model not in ("close", "intraday"):
            raise ValueError(f"stop_model must be 'close' or 'intraday', got {self.stop_model!r}")
        if not (np.isfinite(self.broker_stop_distance) and self.broker_stop_distance >= 1.0):
            raise ValueError(f"broker_stop_distance must be >= 1.0, got {self.broker_stop_distance!r}")

    def broker_stop(self, ref: float, stop: float) -> float:
        """The broker-held stop level for a plan with entry reference ``ref`` and stop ``stop`` (same
        price space; below ref for a long, above for a short). Distance 1.0 returns ``stop`` itself."""
        return ref - self.broker_stop_distance * (ref - stop)

    @classmethod
    def from_config(cls, config: Config) -> "CostModel":
        c = config.section("costs")
        tiers = tuple(sorted(((float(a), float(b)) for a, b in c["half_spread_bps_tiers"]), key=lambda x: -x[0]))
        return cls(tiers, float(c["slippage_bps"]), float(c.get("commission_per_share", 0.0)),
                   float(c.get("commission_min_per_order", 0.0)), float(c.get("delisting_return", -0.30)),
                   int(config.get("execution.delisting_missing_sessions", 5)),
                   float(config.get("validation.data.suspicious_open.threshold", 0.25)),
                   float(config.get("validation.data.suspicious_open.min_reversion", 0.5)),
                   str(config.get("execution.stop_model", "close")),
                   float(config.get("execution.protective_stop.distance", 1.0)),
                   delisting_return_merger=float(c.get("delisting_return_merger", 0.0)),
                   delisting_merger_lookback_sessions=int(c.get("delisting_merger_lookback_sessions", 5)))

    def merger_delisting(self, merger_flags: np.ndarray | None, last_valid: int, i: int) -> bool:
        """Is a DELISTED exit booked on session index ``i`` (symbol's last bar at ``last_valid``) an
        acquisition? ``merger_flags`` is that symbol's column of the panel's ``merger`` field (None =
        no merger data). True when a merger record takes effect on a session in
        [last_valid - delisting_merger_lookback_sessions, i]. Reads rows <= i only, so a merger
        dated after the delisting session is never used (point-in-time)."""
        if merger_flags is None:
            return False
        lo = max(0, last_valid - self.delisting_merger_lookback_sessions)
        return bool(np.any(merger_flags[lo:i + 1]))

    def delisting_exit_return(self, merger_flags: np.ndarray | None, last_valid: int, i: int) -> float:
        """Haircut for that DELISTED exit: ``delisting_return_merger`` for an acquisition
        (:meth:`merger_delisting`), otherwise the conservative ``delisting_return``."""
        return (self.delisting_return_merger if self.merger_delisting(merger_flags, last_valid, i)
                else self.delisting_return)

    def half_spread_bps(self, median_dollar_volume: float | None) -> float:
        """Unknown liquidity is charged the WORST tier (conservative)."""
        if median_dollar_volume is None or not np.isfinite(median_dollar_volume):
            return self.half_spread_tiers[-1][1]
        for threshold, bps in self.half_spread_tiers:
            if median_dollar_volume >= threshold:
                return bps
        return self.half_spread_tiers[-1][1]

    def one_way_cost_frac(self, median_dollar_volume: float | None) -> float:
        return (self.half_spread_bps(median_dollar_volume) + self.slippage_bps) / 1e4

    def round_trip_cost_frac(self, median_dollar_volume: float | None) -> float:
        return 2 * self.one_way_cost_frac(median_dollar_volume)

    def fill_price(self, side: Side, ref_price: float, median_dollar_volume: float | None) -> float:
        c = self.one_way_cost_frac(median_dollar_volume)
        return ref_price * (1 + c) if side is Side.BUY else ref_price * (1 - c)

    def commission(self, qty: float) -> float:
        if qty == 0:
            return 0.0
        return max(abs(qty) * self.commission_per_share, self.commission_min_per_order)
