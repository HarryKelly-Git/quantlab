"""Portfolio construction: equal-risk sizing plus exposure/sector/correlation/positions caps.

Turns approved candidates (already past no-trade + EV) into whole-share order intents for one
book, applied in the ORDER given (the caller's priority order — normally ranking's
``opportunity_score`` descending). Two kinds of constraint:

  * CONTINUOUS caps (``max_position_weight``, ``cash_buffer``, ``max_gross_exposure``,
    ``max_sector_weight``, ``max_open_risk``) CLIP the equal-risk share count down to whatever
    still fits, rather than rejecting outright — a smaller position that still earns its keep is
    better than throwing away a name over one binding constraint. Every clip is named in the
    intent's ``sizing["binding_caps"]`` so it is auditable.
  * DISCRETE constraints (no stop, ``max_positions`` already full, correlation too high with a
    held/already-selected symbol, symbol already held/duplicated, qty rounds to zero after
    clipping) REJECT the candidate outright: there is no partial way to satisfy them.

Equal-risk sizing: ``qty = risk_per_trade * equity / |entry_ref - stop|``. A candidate with no
(or zero-distance) stop cannot be sized and is rejected ('no stop') — the no-trade engine's
execution-uncertainty rule is a softer, informational cousin of this hard requirement.

Correlation: ``max_pairwise_correlation`` compares the SIGNED correlation (over the trailing
``correlation_window_sessions`` sessions of ``panel.ret`` at/before ``as_of``, PIT-safe) of a
candidate's returns against every already-held or already-selected-this-call symbol. Only
POSITIVE correlation above the cap blocks — an anti-correlated pair reduces concentration risk,
it does not add to it. Symbols without enough overlapping return history are treated as UNKNOWN
and the check is skipped for that pair (never guessed), noted in the rejection/sizing detail.

State (cash, gross exposure, sector exposure, open risk, position count, selected symbols) is
threaded through the candidate list so caps see the cumulative effect of earlier accepted orders
in the SAME call — sizing is inherently sequential, not independent per candidate.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.core.types import Candidate, Direction
from quantlab.data.panel import Panel

_MIN_CORR_OBS = 10


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class HeldPosition:
    """One currently-held position of a :class:`BookState`."""

    qty: float
    price: float
    sector: str | None = None
    stop: float | None = None

    @property
    def market_value(self) -> float:
        """Gross (absolute) dollar exposure, long or short."""
        return abs(self.qty) * self.price


@dataclass
class BookState:
    """Everything :class:`PortfolioConstructor` needs to know about the book being sized into.

    ``positions`` accepts either :class:`HeldPosition` values or plain mappings
    ``{"qty": ..., "price": ..., "sector": ..., "stop": ...}`` (normalized in ``__post_init__``).
    ``open_risk`` is the CALLER-supplied sum over held positions of
    ``(entry - stop) * qty / equity`` (the constructor has no entry price for existing positions
    and does not try to reconstruct it); it only accumulates new orders' risk on top.
    """

    equity: float
    cash: float
    positions: dict[str, HeldPosition | Mapping[str, Any]] = field(default_factory=dict)
    open_risk: float = 0.0

    def __post_init__(self) -> None:
        norm: dict[str, HeldPosition] = {}
        for symbol, pos in self.positions.items():
            if isinstance(pos, HeldPosition):
                norm[symbol] = pos
            elif isinstance(pos, Mapping):
                norm[symbol] = HeldPosition(qty=float(pos["qty"]), price=float(pos["price"]),
                                            sector=pos.get("sector"), stop=_num(pos.get("stop")))
            else:
                raise TypeError(f"positions[{symbol!r}] must be a HeldPosition or mapping, got {type(pos)!r}")
        self.positions = norm


@dataclass
class SizedOrderIntent:
    """One candidate turned into a whole-share order, with a fully explainable sizing trail."""

    candidate_id: str
    symbol: str
    strategy_id: str
    strategy_version: str
    direction: Direction
    qty: int
    entry_ref_price: float
    stop_price: float
    target_price: float | None
    sector: str | None
    sizing: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.qty * self.entry_ref_price


@dataclass
class PortfolioRejection:
    """Why a candidate did NOT get an order this call."""

    candidate_id: str
    symbol: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class PortfolioConstructor:
    def __init__(self, config: Config):
        self.config = config
        p = lambda k, d: config.get(f"portfolio.{k}", d)  # noqa: E731
        self.risk_per_trade = float(p("risk_per_trade", 0.005))
        self.max_position_weight = float(p("max_position_weight", 0.10))
        self.cash_buffer = float(p("cash_buffer", 0.05))
        self.max_gross_exposure = float(p("max_gross_exposure", 1.00))
        self.max_positions = int(p("max_positions", 10))
        self.max_sector_weight = float(p("max_sector_weight", 0.30))
        self.max_open_risk = float(p("max_open_risk", 0.04))
        self.max_pairwise_correlation = float(p("max_pairwise_correlation", 0.85))
        self.correlation_window = int(p("correlation_window_sessions", 60))
        for name, v in (("risk_per_trade", self.risk_per_trade), ("max_position_weight", self.max_position_weight),
                        ("max_gross_exposure", self.max_gross_exposure), ("max_sector_weight", self.max_sector_weight),
                        ("max_open_risk", self.max_open_risk), ("correlation_window_sessions", self.correlation_window)):
            if v <= 0:
                raise ValueError(f"portfolio.{name} must be > 0, got {v!r}")
        if not 0.0 <= self.cash_buffer < 1.0:
            raise ValueError(f"portfolio.cash_buffer must be in [0, 1), got {self.cash_buffer!r}")
        if self.max_positions <= 0:
            raise ValueError("portfolio.max_positions must be > 0")
        if not 0.0 < self.max_pairwise_correlation <= 1.0:
            raise ValueError("portfolio.max_pairwise_correlation must be in (0, 1]")

    # -- correlation --------------------------------------------------------------------------------
    def _correlation_window(self, panel: Panel, as_of: Any):
        ret = panel.ret
        end = int(ret.index.searchsorted(to_session(as_of), side="right"))
        return ret.iloc[max(0, end - self.correlation_window):end]

    def _worst_correlation(self, window, symbol: str, others: set[str]) -> tuple[str, float] | None:
        """(other_symbol, correlation) with the HIGHEST signed correlation to ``symbol`` among
        ``others``, or None if ``symbol`` or none of ``others`` have enough overlapping history."""
        if symbol not in window.columns:
            return None
        sa = window[symbol]
        best: tuple[str, float] | None = None
        for other in sorted(others):
            if other == symbol or other not in window.columns:
                continue
            sb = window[other]
            mask = sa.notna() & sb.notna()
            if int(mask.sum()) < _MIN_CORR_OBS:
                continue
            c = float(sa[mask].corr(sb[mask]))
            if not math.isfinite(c):
                continue
            if best is None or c > best[1]:
                best = (other, c)
        return best

    # -- main -----------------------------------------------------------------------------------
    def build(self, book: BookState, approved: list[Candidate], panel: Panel, as_of: Any,
              sector_map: Mapping[str, str] | None = None) -> tuple[list[SizedOrderIntent], list[PortfolioRejection]]:
        sector_map = dict(sector_map or {})
        intents: list[SizedOrderIntent] = []
        rejections: list[PortfolioRejection] = []

        equity = float(book.equity)
        if not (equity > 0):
            return [], [PortfolioRejection(c.candidate_id, c.symbol.upper(), "book equity is not positive",
                                           {"equity": equity}) for c in approved]

        cash = float(book.cash)
        gross = sum(pos.market_value for pos in book.positions.values())
        open_risk = float(book.open_risk or 0.0)
        n_positions = len(book.positions)
        sector_exposure: dict[str, float] = defaultdict(float)
        for symbol, pos in book.positions.items():
            sec = pos.sector or sector_map.get(symbol)
            if sec:
                sector_exposure[sec] += pos.market_value

        window = self._correlation_window(panel, as_of)
        held_symbols = set(book.positions)
        selected_symbols: set[str] = set()
        claimed_symbols: set[str] = set(held_symbols)

        for cand in approved:
            symbol = cand.symbol.upper()
            cid = cand.candidate_id

            if symbol in claimed_symbols:
                reason = "symbol already held in book" if symbol in held_symbols else \
                    "duplicate symbol among approved candidates this call"
                rejections.append(PortfolioRejection(cid, symbol, reason))
                continue
            if n_positions >= self.max_positions:
                rejections.append(PortfolioRejection(cid, symbol, f"max_positions reached ({self.max_positions})"))
                continue

            entry = _num(cand.plan.entry_ref_price)
            if entry is None or entry <= 0:
                rejections.append(PortfolioRejection(cid, symbol, "no entry_ref_price"))
                continue
            stop = _num(cand.plan.stop_price)
            if stop is None or abs(entry - stop) <= 0:
                rejections.append(PortfolioRejection(cid, symbol, "no stop"))
                continue
            per_share_risk = abs(entry - stop)

            hit = self._worst_correlation(window, symbol, held_symbols | selected_symbols)
            if hit is not None and hit[1] > self.max_pairwise_correlation:
                other, rho = hit
                rejections.append(PortfolioRejection(
                    cid, symbol, f"pairwise correlation {rho:.2f} with {other} exceeds "
                    f"max {self.max_pairwise_correlation:.2f}", {"other": other, "correlation": rho}))
                continue

            raw_qty = self.risk_per_trade * equity / per_share_risk
            sector = sector_map.get(symbol)
            caps: list[tuple[str, float]] = [
                ("max_position_weight", self.max_position_weight * equity / entry),
                ("cash_buffer", max(0.0, cash - self.cash_buffer * equity) / entry),
                ("max_gross_exposure", max(0.0, self.max_gross_exposure * equity - gross) / entry),
                ("max_open_risk", max(0.0, (self.max_open_risk - open_risk) * equity) / per_share_risk),
            ]
            if sector:
                caps.append(("max_sector_weight",
                            max(0.0, self.max_sector_weight * equity - sector_exposure.get(sector, 0.0)) / entry))

            qty = raw_qty
            binding: list[str] = []
            for name, cap_qty in caps:
                if cap_qty < qty - 1e-12:
                    qty, binding = cap_qty, [name]
                elif abs(cap_qty - qty) <= 1e-12:
                    binding.append(name)
            qty_whole = math.floor(qty + 1e-9)

            if qty_whole < 1:
                rejections.append(PortfolioRejection(
                    cid, symbol, "sizing rounds to zero shares" + (f" (binding: {', '.join(binding)})" if binding else ""),
                    {"raw_qty": raw_qty, "binding_caps": binding}))
                continue

            notional = qty_whole * entry
            risk_frac = qty_whole * per_share_risk / equity
            sizing = {
                "risk_per_trade": self.risk_per_trade, "equity": equity, "entry_ref_price": entry, "stop_price": stop,
                "per_share_risk": per_share_risk, "raw_qty": raw_qty, "qty": qty_whole, "notional": notional,
                "position_weight": notional / equity, "sector": sector,
                "sector_weight_after": ((sector_exposure.get(sector, 0.0) + notional) / equity) if sector else None,
                "gross_exposure_after": (gross + notional) / equity, "open_risk_after": open_risk + risk_frac,
                "cash_after": cash - notional, "correlation_checked_against": sorted(held_symbols | selected_symbols),
                "binding_caps": binding,
            }
            intents.append(SizedOrderIntent(
                candidate_id=cid, symbol=symbol, strategy_id=cand.strategy_id, strategy_version=cand.strategy_version,
                direction=cand.direction, qty=qty_whole, entry_ref_price=entry, stop_price=stop,
                target_price=_num(cand.plan.target_price), sector=sector, sizing=sizing))

            cash -= notional
            gross += notional
            if sector:
                sector_exposure[sector] += notional
            open_risk += risk_frac
            n_positions += 1
            selected_symbols.add(symbol)
            claimed_symbols.add(symbol)

        return intents, rejections


__all__ = ["BookState", "HeldPosition", "PortfolioConstructor", "PortfolioRejection", "SizedOrderIntent"]
