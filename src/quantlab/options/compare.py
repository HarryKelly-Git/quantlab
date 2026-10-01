"""Stock vs option comparator: for one thesis and one move distribution, how does each expression
(the stock with its stop, and a small FIXED grid of defined-risk option structures) pay per dollar
at risk, and which one (or NO TRADE) does that distribution favour?

Everything here is MODEL_OUTPUT (core.types.InfoKind): the distribution is an input assumption,
option values before expiry are Black-Scholes repricings at the entry implied volatility x
``exit_iv_multiplier``, and quotes are INDICATIVE (not the OPRA NBBO). Entry is always at the
executable side (ask / bid); never at the midpoint.

Fixed rules (never optimised on data):
  * expiration: the first listed expiry at least ``horizon_sessions + expiry_buffer_sessions``
    NYSE sessions after today (and within ``max_expiry_sessions``);
  * strikes: single legs nearest |delta| 0.70 / 0.50 / 0.30 (within a tolerance), the ATM strike
    and the +1 expected-move strike (expected move = ATM IV x sqrt(horizon / 252)); one debit
    spread long ATM / short +1 expected move. Calls for LONG theses, puts for SHORT;
  * every contract must pass the liquidity filter; every rejection is recorded;
  * recommendation: the stock if its expected P&L per $ at risk exceeds
    ``min_expected_pnl_per_risk``; an option structure only if it beats BOTH that threshold and the
    stock by ``min_option_advantage_per_risk`` (model risk must be paid for); otherwise NO_TRADE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.audit import expected_sessions
from quantlab.options.data import OptionContract, OptionQuote
from quantlab.options.distribution import MoveDistribution, as_distribution
from quantlab.options.liquidity import LiquidityCheck, LiquidityRules, check_liquidity
from quantlab.options.pricing import MODEL_LABEL, bs_greeks, implied_vol
from quantlab.options.settings import OptionsSettings
from quantlab.options.structures import OptionStructure, StructureError, debit_spread, long_call, long_put

NO_TRADE = "NO_TRADE"
STOCK = "STOCK"
NY = "America/New_York"


class ThesisError(ValueError):
    pass


@dataclass(frozen=True)
class Thesis:
    symbol: str
    direction: str                 # LONG | SHORT
    horizon_sessions: int
    spot: float
    stop_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "direction", str(self.direction).strip().upper())
        if self.direction not in ("LONG", "SHORT"):
            raise ThesisError("direction must be LONG or SHORT")
        if int(self.horizon_sessions) < 1:
            raise ThesisError("horizon_sessions must be >= 1")
        if not (math.isfinite(self.spot) and self.spot > 0 and math.isfinite(self.stop_price) and self.stop_price > 0):
            raise ThesisError("spot and stop_price must be positive")
        if self.direction == "LONG" and not self.stop_price < self.spot:
            raise ThesisError("a LONG thesis needs a stop below spot")
        if self.direction == "SHORT" and not self.stop_price > self.spot:
            raise ThesisError("a SHORT thesis needs a stop above spot")

    @property
    def sign(self) -> int:
        return 1 if self.direction == "LONG" else -1

    @property
    def option_type(self) -> str:
        return "call" if self.direction == "LONG" else "put"

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "direction": self.direction, "horizon_sessions": int(self.horizon_sessions),
                "spot": self.spot, "stop_price": self.stop_price}


@dataclass
class ExpressionResult:
    expression_id: str
    kind: str
    eligible: bool
    ineligible_reason: str | None = None
    entry_cost: float | None = None          # USD per unit (1 share / 1 structure)
    risk_usd: float | None = None            # USD per unit: the loss each result is normalised by
    expected_pnl_usd: float | None = None
    expected_pnl_per_risk: float | None = None
    p_profit: float | None = None
    p_breakeven: float | None = None
    max_loss_usd: float | None = None        # contractual worst case per unit (inf = unbounded)
    max_loss_per_risk: float | None = None
    expected_loss_given_loss: float | None = None   # per $ at risk (<= 0); None if no losing scenario
    breakeven: float | None = None
    max_gain_usd: float | None = None        # None = unbounded
    structure: OptionStructure | None = None
    notes: list[str] = field(default_factory=list)
    chosen: bool = False
    label: str = "MODEL_OUTPUT"

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "structure"}
        d["structure"] = self.structure.to_dict() if self.structure is not None else None
        return d


@dataclass
class Comparison:
    thesis: Thesis
    distribution: MoveDistribution
    now: pd.Timestamp
    session_date: date
    horizon_date: date
    expiration: date | None
    expiry_reason: str
    t_remaining_years: float | None
    results: list[ExpressionResult]
    choice: str
    reason: str
    liquidity: list[LiquidityCheck]
    grid_notes: list[dict[str, Any]]
    settings: OptionsSettings
    data_notes: list[str]

    @property
    def chosen(self) -> ExpressionResult | None:
        return next((r for r in self.results if r.expression_id == self.choice), None)

    def summary(self) -> dict[str, Any]:
        return {"symbol": self.thesis.symbol, "direction": self.thesis.direction,
                "horizon_sessions": self.thesis.horizon_sessions, "session_date": str(self.session_date),
                "horizon_date": str(self.horizon_date), "expiration": str(self.expiration) if self.expiration else None,
                "expiry_reason": self.expiry_reason, "spot": self.thesis.spot, "stop": self.thesis.stop_price,
                "distribution": {"label": self.distribution.label, "n": self.distribution.n,
                                 "path_based": self.distribution.path_based},
                "choice": self.choice, "reason": self.reason,
                "contracts_checked": len(self.liquidity),
                "contracts_passed": sum(1 for c in self.liquidity if c.passed),
                "data_notes": self.data_notes, "info_kind": "MODEL_OUTPUT"}


# -- calendar helpers (NYSE rules, data/audit.py) ---------------------------------------------------
def sessions_between(start: date, end: date) -> int:
    """NYSE sessions in (start, end]."""
    if end <= start:
        return 0
    return int(len(expected_sessions(pd.Timestamp(start) + pd.Timedelta(days=1), pd.Timestamp(end))))


def session_offset(start: date, n: int) -> date:
    """The n-th NYSE session after ``start`` (n >= 1)."""
    days = expected_sessions(pd.Timestamp(start) + pd.Timedelta(days=1), pd.Timestamp(start) + pd.Timedelta(days=7 * n + 30))
    return days[n - 1].date()


def select_expiration(expirations: list[date], today: date, horizon_sessions: int, buffer_sessions: int,
                      max_sessions: int) -> tuple[date | None, str]:
    need = horizon_sessions + buffer_sessions
    for e in sorted(set(expirations)):
        n = sessions_between(today, e)
        if n < need:
            continue
        if n > max_sessions:
            return None, f"first expiry with >= {need} sessions is {e} ({n} sessions) > max_expiry_sessions {max_sessions}"
        return e, f"first expiry >= horizon {horizon_sessions} + buffer {buffer_sessions} sessions: {e} ({n} sessions)"
    return None, f"no listed expiry >= {need} sessions after {today}"


# -- per-expression evaluation --------------------------------------------------------------------
def _stats(pnl: np.ndarray, w: np.ndarray, risk: float) -> dict[str, float | None]:
    exp = float(np.dot(w, pnl))
    loss = pnl < 0
    pl = float(w[loss].sum())
    elgl = float(np.dot(w[loss], pnl[loss]) / pl / risk) if pl > 0 else None
    return {"expected_pnl_usd": exp, "expected_pnl_per_risk": exp / risk, "p_profit": float(w[pnl > 0].sum()),
            "expected_loss_given_loss": elgl}


def evaluate_stock(thesis: Thesis, dist: MoveDistribution, round_trip_cost_bps: float) -> ExpressionResult:
    spot, sgn = thesis.spot, thesis.sign
    cost = spot * round_trip_cost_bps / 1e4
    stop_dist = abs(spot - thesis.stop_price)
    risk = stop_dist + cost
    notes = [f"round-trip cost {round_trip_cost_bps:g} bps", "stop assumed to fill AT the stop price (gaps ignored)"]
    move = sgn * spot * dist.end_returns
    if dist.max_adverse is not None:
        stopped = dist.max_adverse >= stop_dist / spot - 1e-12
        pnl = np.where(stopped, -stop_dist, move) - cost
        notes.append("stop evaluated on the path (max_adverse)")
    else:
        pnl = np.maximum(move, -stop_dist) - cost
        notes.append("path unknown: stop applied at the horizon only (optimistic for the stock: stop-outs that "
                     "recover are not charged)")
    need = cost / spot
    if dist.max_favourable is not None:
        p_be = float(dist.weights[dist.max_favourable >= need].sum())
    else:
        p_be = float(dist.weights[sgn * dist.end_returns >= need].sum())
    st = _stats(pnl, dist.weights, risk)
    if thesis.direction == "LONG":
        kind, max_loss, eligible, why = STOCK, spot + cost, True, None
        breakeven = spot + cost
    else:
        kind, max_loss, eligible = "SHORT_STOCK", math.inf, False
        why = "short stock is not supported by QuantLab's paper books (long-only); shown for comparison"
        breakeven = spot - cost
    return ExpressionResult(
        expression_id=STOCK, kind=kind, eligible=eligible, ineligible_reason=why, entry_cost=spot, risk_usd=risk,
        p_breakeven=p_be, max_loss_usd=max_loss, max_loss_per_risk=max_loss / risk, breakeven=breakeven,
        max_gain_usd=None if thesis.direction == "LONG" else spot - cost, notes=notes, **st)


def evaluate_structure(structure: OptionStructure, thesis: Thesis, dist: MoveDistribution, t_remaining_years: float,
                       s: OptionsSettings) -> ExpressionResult:
    sid = structure.structure_id
    base = dict(expression_id=sid, kind=structure.kind.value, structure=structure, entry_cost=structure.entry_cost,
                breakeven=structure.breakeven, max_gain_usd=structure.max_gain)
    if structure.option_type != thesis.option_type:
        return ExpressionResult(eligible=False, ineligible_reason=f"{structure.option_type}s do not express a "
                                f"{thesis.direction} thesis", **base)
    fees = 2.0 * s.model_fee_per_contract * len(structure.legs)
    risk = structure.max_loss + fees
    S_h = thesis.spot * (1.0 + dist.end_returns)
    try:
        pnl = structure.pnl_at(S_h, t_remaining_years, s.risk_free_rate, s.exit_iv_multiplier,
                               s.exit_at_executable_side, s.model_fee_per_contract)
    except StructureError as exc:
        return ExpressionResult(eligible=False, ineligible_reason=str(exc), risk_usd=risk, max_loss_usd=risk,
                                max_loss_per_risk=1.0, **base)
    sgn = 1 if structure.option_type == "call" else -1
    need = sgn * (structure.breakeven / thesis.spot - 1.0)
    if dist.max_favourable is not None:
        p_be = float(dist.weights[dist.max_favourable >= need].sum())
    else:
        p_be = float(dist.weights[sgn * dist.end_returns >= need].sum())
    notes = [MODEL_LABEL if t_remaining_years > 0 else "held to expiry: intrinsic value (no model)",
             f"exit at horizon with {t_remaining_years * 365:.0f} calendar days left; IV x {s.exit_iv_multiplier:g}",
             "entry at the executable side (ask/bid) of INDICATIVE quotes, not the OPRA NBBO",
             "no stop: the loss is bounded by the premium"]
    if s.exit_at_executable_side and t_remaining_years > 0:
        notes.append("exit charged the entry half-spread per leg")
    return ExpressionResult(eligible=True, risk_usd=risk, p_breakeven=p_be, max_loss_usd=risk, max_loss_per_risk=1.0,
                            notes=notes, **base, **_stats(pnl, dist.weights, risk))


def recommend(results: list[ExpressionResult], s: OptionsSettings) -> tuple[str, str]:
    floor, margin = s.min_expected_pnl_per_risk, s.min_option_advantage_per_risk
    stock = next((r for r in results if r.expression_id == STOCK), None)
    stock_ok = bool(stock and stock.eligible and stock.expected_pnl_per_risk is not None
                    and stock.expected_pnl_per_risk > floor)
    hurdle = max(floor, stock.expected_pnl_per_risk) if stock_ok else floor
    opts = [r for r in results if r.expression_id != STOCK and r.eligible and r.expected_pnl_per_risk is not None]
    best = max(opts, key=lambda r: (r.expected_pnl_per_risk, r.expression_id)) if opts else None
    if best is not None and best.expected_pnl_per_risk > hurdle + margin:
        vs = (f"stock {stock.expected_pnl_per_risk:+.3f}" if stock_ok else "stock not tradeable/positive")
        return best.expression_id, (f"{best.kind} has the highest expected P&L per $ at risk "
                                    f"({best.expected_pnl_per_risk:+.3f}) and clears the option hurdle "
                                    f"{hurdle:+.3f} + margin {margin:.2f} ({vs})")
    if stock_ok:
        b = f"best structure {best.expected_pnl_per_risk:+.3f}" if best else "no eligible structure"
        return STOCK, (f"stock expected P&L per $ at risk {stock.expected_pnl_per_risk:+.3f} > {floor:+.3f}; "
                       f"no structure beats it by the {margin:.2f} margin ({b})")
    allr = [r for r in results if r.expected_pnl_per_risk is not None]
    top = max(allr, key=lambda r: r.expected_pnl_per_risk) if allr else None
    detail = (f"best: {top.kind} {top.expected_pnl_per_risk:+.3f}" + ("" if top.eligible else
              f" (ineligible: {top.ineligible_reason})")) if top else "nothing evaluable"
    return NO_TRADE, (f"no eligible expression clears its hurdle (stock > {floor:+.3f}; options > max(stock, "
                      f"{floor:+.3f}) + {margin:.2f}); {detail}")


# -- strike grid ----------------------------------------------------------------------------------
def contract_iv(c: OptionContract, q: OptionQuote | None, spot: float, today: date, r: float) -> tuple[float | None, str]:
    """Vendor IV when the snapshot has one, else Black-Scholes IV from the quote MID (description
    only), else UNKNOWN."""
    if q is not None and q.vendor_iv is not None and q.vendor_iv > 0:
        return float(q.vendor_iv), "VENDOR"
    if q is not None and q.mid is not None:
        t = max((c.expiration - today).days, 1) / 365.0
        iv = implied_vol(q.mid, spot, c.strike, t, r, c.type)
        if iv is not None:
            return iv, "MODEL_FROM_MID"
    return None, "UNKNOWN"


def contract_delta(c: OptionContract, q: OptionQuote | None, iv: float | None, spot: float, today: date,
                   r: float) -> float | None:
    if q is not None and q.vendor_delta is not None:
        return float(q.vendor_delta)
    if iv is None:
        return None
    return bs_greeks(spot, c.strike, max((c.expiration - today).days, 1) / 365.0, iv, r, c.type)["delta"]


def build_candidates(thesis: Thesis, contracts: list[OptionContract], quotes: dict[str, OptionQuote],
                     checks: dict[str, LiquidityCheck], today: date, s: OptionsSettings,
                     dist: MoveDistribution) -> tuple[list[OptionStructure], list[dict[str, Any]]]:
    typ, spot, r = thesis.option_type, thesis.spot, s.risk_free_rate
    notes: list[dict[str, Any]] = []
    liquid = sorted((c for c in contracts if c.type == typ and checks.get(c.symbol) and checks[c.symbol].passed),
                    key=lambda c: c.strike)
    if not liquid:
        notes.append({"grid": "ALL", "status": "NO_LIQUID_CONTRACT",
                      "detail": f"no {typ} in the chosen expiry passed the liquidity filter"})
        return [], notes
    ivs = {c.symbol: contract_iv(c, quotes.get(c.symbol), spot, today, r) for c in liquid}
    deltas = {c.symbol: contract_delta(c, quotes.get(c.symbol), ivs[c.symbol][0], spot, today, r) for c in liquid}
    picks: dict[str, OptionContract] = {}
    for target in s.grid_deltas:
        cands = [(abs(abs(deltas[c.symbol]) - target), c.strike, c) for c in liquid if deltas[c.symbol] is not None]
        best = min(cands, key=lambda x: (x[0], x[1])) if cands else None
        if best is None or best[0] > s.grid_delta_tolerance:
            notes.append({"grid": f"delta {target:.2f}", "status": "GRID_POINT_UNFILLED",
                          "detail": "no liquid contract with a known delta within tolerance"})
        else:
            picks[f"delta {target:.2f}"] = best[2]
    atm = min(liquid, key=lambda c: (abs(c.strike - spot), c.strike))
    if s.grid_include_atm:
        picks["ATM"] = atm
    em_contract = None
    if s.grid_include_expected_move or s.grid_spreads:
        atm_iv, src = ivs[atm.symbol]
        if atm_iv is not None:
            em_pct, em_src = atm_iv * math.sqrt(thesis.horizon_sessions / 252.0), f"ATM IV ({src})"
        else:
            em_pct, em_src = dist.std(), "distribution std (ATM IV UNKNOWN)"
        target_k = spot * (1 + em_pct) if typ == "call" else spot * (1 - em_pct)
        beyond = [c for c in liquid if (c.strike > atm.strike if typ == "call" else c.strike < atm.strike)]
        if beyond:
            em_contract = min(beyond, key=lambda c: (abs(c.strike - target_k), c.strike))
            notes.append({"grid": "+1 expected move", "status": "OK", "detail":
                          f"expected move {em_pct:.2%} from {em_src}; target strike {target_k:.2f} -> {em_contract.strike:g}"})
            if s.grid_include_expected_move:
                picks["+1 EM"] = em_contract
        else:
            notes.append({"grid": "+1 expected move", "status": "GRID_POINT_UNFILLED",
                          "detail": "no liquid strike beyond ATM in the thesis direction"})
    structures: dict[str, OptionStructure] = {}
    builder = long_call if typ == "call" else long_put
    for name, c in picks.items():
        try:
            st = builder(c, quotes[c.symbol], *ivs[c.symbol])
            structures.setdefault(st.structure_id, st)
        except (StructureError, KeyError) as exc:
            notes.append({"grid": name, "status": "STRUCTURE_REJECTED", "detail": str(exc)})
    if s.grid_spreads and em_contract is not None:
        try:
            st = debit_spread(atm, quotes[atm.symbol], em_contract, quotes[em_contract.symbol],
                              ivs[atm.symbol], ivs[em_contract.symbol])
            structures.setdefault(st.structure_id, st)
        except (StructureError, KeyError) as exc:
            notes.append({"grid": "spread ATM/+1 EM", "status": "STRUCTURE_REJECTED", "detail": str(exc)})
    return list(structures.values()), notes


# -- top level ------------------------------------------------------------------------------------
def compare(thesis: Thesis, dist: MoveDistribution | np.ndarray, contracts: list[OptionContract],
            quotes: dict[str, OptionQuote], now: pd.Timestamp, s: OptionsSettings,
            session_date: date | None = None) -> Comparison:
    """``dist`` may be a MoveDistribution or a plain array of end returns (equal weights)."""
    dist = as_distribution(dist)
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    today = session_date or now.tz_convert(NY).date()
    horizon = session_offset(today, int(thesis.horizon_sessions))
    data_notes = [f"quotes: {s.feed} feed (Alpaca indicative, NOT the OPRA NBBO): approximate",
                  f"distribution: {dist.label} (n={dist.n}, path_based={dist.path_based})",
                  "option exits before expiry are MODEL repricings; entries at ask/bid, never mid"]
    results = [evaluate_stock(thesis, dist, s.stock_round_trip_cost_bps)]
    own = [c for c in contracts if c.underlying == thesis.symbol]
    expiry, why = select_expiration([c.expiration for c in own], today, int(thesis.horizon_sessions),
                                    s.expiry_buffer_sessions, s.max_expiry_sessions)
    checks: list[LiquidityCheck] = []
    notes: list[dict[str, Any]] = []
    t_rem = None
    if expiry is not None:
        t_rem = max((expiry - horizon).days, 0) / 365.0
        chain = [c for c in own if c.expiration == expiry and c.type == thesis.option_type]
        rules = LiquidityRules.from_settings(s)
        by_sym = {c.symbol: check_liquidity(c, quotes.get(c.symbol), now, rules) for c in chain}
        checks = list(by_sym.values())
        structures, notes = build_candidates(thesis, chain, quotes, by_sym, today, s, dist)
        results += [evaluate_structure(st, thesis, dist, t_rem, s) for st in structures]
    else:
        notes.append({"grid": "ALL", "status": "NO_EXPIRY", "detail": why})
    choice, reason = recommend(results, s)
    for r in results:
        r.chosen = r.expression_id == choice
    return Comparison(thesis=thesis, distribution=dist, now=now, session_date=today, horizon_date=horizon,
                      expiration=expiry, expiry_reason=why, t_remaining_years=t_rem, results=results, choice=choice,
                      reason=reason, liquidity=checks, grid_notes=notes, settings=s, data_notes=data_notes)
