"""Prop-firm lifecycle simulator: evaluation -> funded -> payouts -> survival, with fees.

Research only. The objective it measures is EXPECTED LONG-TERM NET PAYOUT (payouts received minus every
fee paid), not the probability of passing an evaluation.

Rules are data (``PropRules``), loaded from YAML that must cite the firm's OFFICIAL documentation. Any
field written as ``UNKNOWN`` is recorded as unknown and the rules refuse to simulate until it is filled:
a rule is never invented.

Day model
---------
The simulator steps one trading day at a time. Each day is ``(pnl, low, high)`` in dollars:

* ``pnl``  the day's net closing P&L (after costs),
* ``low``  the worst intraday cumulative P&L (``<= min(0, pnl)``),
* ``high`` the best intraday cumulative P&L (``>= max(0, pnl)``).

Conservative ordering, always: within a day the HIGH is assumed to come before the LOW. That raises an
intraday-trailing floor first and then tests the low against it, which is the unfavourable order. A
breach of the drawdown floor anywhere in the day fails the account, whatever the close. A daily-loss-limit
hit flattens the day at exactly ``-daily_loss_limit`` (``dll_action: stop_day``) or fails the account
(``dll_action: fail``); a floor breach is checked first.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

UNKNOWN = "UNKNOWN"
DRAWDOWN_TYPES = ("static", "trailing_eod", "trailing_intraday")
DLL_ACTIONS = ("stop_day", "fail")


class RulesError(ValueError):
    pass


@dataclass(frozen=True)
class PropRules:
    name: str
    start_balance: float
    profit_target: float                       # evaluation passes at balance >= start + target
    max_drawdown: float                        # floor distance below the high-water mark (or start if static)
    drawdown_type: str = "trailing_eod"
    trail_lock_profit: float | None = None     # floor stops rising once it reaches start + this (None: never locks)
    daily_loss_limit: float | None = None      # None: no daily loss limit
    dll_action: str = "stop_day"
    max_contracts: int | None = None           # informational: sizing is the day stream's job
    eval_consistency: float | None = None      # best day <= this share of total profit to pass (None: no rule)
    funded_consistency: float | None = None    # same rule for payout eligibility
    min_eval_days: int = 0
    min_days_between_payouts: int = 0          # trading days since funding / the previous payout
    payout_min_profit: float = 0.0             # profit above start required before any payout
    payout_keep: float = 0.0                   # profit that must stay in the account after a payout
    payout_min_amount: float = 0.0
    payout_cap: float | None = None            # max withdrawal per payout (None: no cap)
    payout_split: float = 1.0                  # trader's share of a withdrawal
    max_payouts: int | None = None             # account closes (success) after this many payouts
    eval_fee: float = 0.0                      # charged when an evaluation attempt starts
    eval_fee_period_days: int | None = None    # recurring: charged again every N trading days in evaluation
    reset_fee: float | None = None             # retry after a failed evaluation (None: same as eval_fee)
    activation_fee: float = 0.0                # charged on passing, before funded trading
    restart_after_funded_fail: bool = True     # buy a new evaluation after losing a funded account
    verified: bool = False
    sources: tuple[str, ...] = ()
    notes: str = ""
    unknown: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.drawdown_type not in DRAWDOWN_TYPES:
            raise RulesError(f"drawdown_type must be one of {DRAWDOWN_TYPES}")
        if self.dll_action not in DLL_ACTIONS:
            raise RulesError(f"dll_action must be one of {DLL_ACTIONS}")
        if self.max_drawdown <= 0 or self.profit_target <= 0 or self.start_balance <= 0:
            raise RulesError("start_balance, profit_target and max_drawdown must be > 0")
        if not 0 < self.payout_split <= 1:
            raise RulesError("payout_split must be in (0, 1]")

    def require_complete(self) -> None:
        if self.unknown:
            raise RulesError(f"{self.name}: rules contain UNKNOWN fields {list(self.unknown)}: fill them from the "
                             "firm's official documentation before simulating")


def load_rules(path: str | Path) -> PropRules:
    """YAML -> PropRules. ``UNKNOWN`` values are kept out of the dataclass and listed in ``unknown``."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {f.name for f in fields(PropRules)} - {"unknown"}
    extra = set(raw) - allowed
    if extra:
        raise RulesError(f"unknown rule fields {sorted(extra)}")
    unknown = tuple(sorted(k for k, v in raw.items() if isinstance(v, str) and v.strip().upper() == UNKNOWN))
    vals = {k: v for k, v in raw.items() if k not in unknown}
    if "sources" in vals:
        vals["sources"] = tuple(vals["sources"] or ())
    for k in ("start_balance", "profit_target", "max_drawdown"):     # required numbers: placeholder if unknown
        if k in unknown:
            vals[k] = 1.0
    return PropRules(**vals, unknown=unknown)


# ------------------------------------------------------------------------------------------------
# one account
# ------------------------------------------------------------------------------------------------

class Account:
    """One evaluation or funded account. ``step`` returns 'ok' | 'failed'."""

    def __init__(self, rules: PropRules):
        self.r = rules
        self.balance = rules.start_balance
        self.hwm = rules.start_balance
        self.floor = rules.start_balance - rules.max_drawdown
        self.days = 0
        self.best_day = 0.0
        self.profit_days: list[float] = []

    def _lock(self, floor: float) -> float:
        lock = self.r.trail_lock_profit
        return min(floor, self.r.start_balance + lock) if lock is not None else floor

    def step(self, pnl: float, low: float, high: float) -> str:
        r = self.r
        low, high = min(low, pnl, 0.0), max(high, pnl, 0.0)
        if r.drawdown_type == "trailing_intraday":                  # high first (unfavourable order)
            self.hwm = max(self.hwm, self.balance + high)
            self.floor = max(self.floor, self._lock(self.hwm - r.max_drawdown))
        dll = r.daily_loss_limit
        worst = low if dll is None else max(low, -dll)               # trading stops at the limit
        if self.balance + worst <= self.floor + 1e-9:
            self.days += 1
            return "failed"
        if dll is not None and low <= -dll + 1e-9:
            self.days += 1
            if r.dll_action == "fail":
                return "failed"
            pnl = -dll
        self.balance += pnl
        self.days += 1
        self.best_day = max(self.best_day, pnl)
        if r.drawdown_type == "trailing_eod":
            self.hwm = max(self.hwm, self.balance)
            self.floor = max(self.floor, self._lock(self.hwm - r.max_drawdown))
        return "ok"

    @property
    def profit(self) -> float:
        return self.balance - self.r.start_balance

    def consistent(self, share: float | None) -> bool:
        if share is None:
            return True
        return self.profit > 0 and self.best_day <= share * self.profit + 1e-9


# ------------------------------------------------------------------------------------------------
# lifecycle
# ------------------------------------------------------------------------------------------------

@dataclass
class PathResult:
    fees: float = 0.0
    payouts: float = 0.0
    n_payouts: int = 0
    evaluations: int = 0
    passes: int = 0
    funded_failures: int = 0
    day_first_payout: int | None = None
    day_second_payout: int | None = None
    first_pass_day: int | None = None
    funded_survived_to_end: bool = False
    first_attempt_passed: bool = False

    @property
    def net(self) -> float:
        return self.payouts - self.fees


DayFn = Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray, np.ndarray]]


def simulate_path(rules: PropRules, pnl: np.ndarray, low: np.ndarray, high: np.ndarray,
                  max_evaluations: int | None = None) -> PathResult:
    """Run ONE lifecycle over a fixed sequence of trading days (deterministic for a given sequence)."""
    rules.require_complete()
    res = PathResult()
    n = len(pnl)
    t = 0
    reset = rules.eval_fee if rules.reset_fee is None else rules.reset_fee
    last = None                                   # None | "eval_fail" | "funded_fail"
    while t < n:
        if max_evaluations is not None and res.evaluations >= max_evaluations:
            break
        res.fees += reset if last == "eval_fail" else rules.eval_fee   # retry vs a new purchase
        res.evaluations += 1
        acct, passed, since_fee = Account(rules), False, 0
        while t < n:
            if rules.eval_fee_period_days and since_fee >= rules.eval_fee_period_days:
                res.fees += rules.eval_fee
                since_fee = 0
            st = acct.step(pnl[t], low[t], high[t])
            t += 1
            since_fee += 1
            if st == "failed":
                break
            if (acct.profit >= rules.profit_target - 1e-9 and acct.days >= rules.min_eval_days
                    and acct.consistent(rules.eval_consistency)):
                passed = True
                break
        if not passed:
            last = "eval_fail"
            continue                                                 # failed (or ran out of days): retry
        res.first_attempt_passed = res.first_attempt_passed or res.evaluations == 1
        res.passes += 1
        res.first_pass_day = res.first_pass_day if res.first_pass_day is not None else t
        res.fees += rules.activation_fee
        fund, since_payout, failed = Account(rules), 0, False
        while t < n:
            st = fund.step(pnl[t], low[t], high[t])
            t += 1
            since_payout += 1
            if st == "failed":
                failed = True
                break
            withdrawable = fund.profit - rules.payout_keep
            if (since_payout >= max(rules.min_days_between_payouts, 1) and fund.profit >= rules.payout_min_profit - 1e-9
                    and withdrawable >= max(rules.payout_min_amount, 1e-9)
                    and fund.consistent(rules.funded_consistency)):
                amount = withdrawable if rules.payout_cap is None else min(withdrawable, rules.payout_cap)
                fund.balance -= amount                               # the floor never moves down
                fund.best_day = 0.0                                  # consistency restarts per payout cycle
                res.payouts += amount * rules.payout_split
                res.n_payouts += 1
                since_payout = 0
                if res.n_payouts == 1:
                    res.day_first_payout = t
                elif res.n_payouts == 2:
                    res.day_second_payout = t
                if rules.max_payouts is not None and res.n_payouts >= rules.max_payouts:
                    return res
        if failed:
            last = "funded_fail"
            res.funded_failures += 1
            if not rules.restart_after_funded_fail:
                return res
            continue
        res.funded_survived_to_end = True
    return res


def simulate(rules: PropRules, days: DayFn, *, n_paths: int = 2000, horizon_days: int = 252, seed: int = 0,
             max_evaluations: int | None = None) -> dict[str, Any]:
    """Monte Carlo over ``n_paths`` independent lifecycles of ``horizon_days`` trading days each."""
    rules.require_complete()
    rng = np.random.default_rng(seed)
    out = [simulate_path(rules, *days(rng, horizon_days), max_evaluations=max_evaluations) for _ in range(n_paths)]
    net = np.array([r.net for r in out])
    fees = np.array([r.fees for r in out])
    pays = np.array([r.payouts for r in out])
    first = [r.day_first_payout for r in out if r.day_first_payout is not None]
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {
        "rules": rules.name, "verified": rules.verified, "n_paths": n_paths, "horizon_days": horizon_days,
        "seed": seed,
        "p_pass_any_eval": float(np.mean([r.passes > 0 for r in out])),
        "p_pass_first_eval": float(np.mean([r.first_attempt_passed for r in out])),
        "p_first_payout": float(np.mean([r.n_payouts >= 1 for r in out])),
        "p_second_payout": float(np.mean([r.n_payouts >= 2 for r in out])),
        "p_funded_alive_at_end": float(np.mean([r.funded_survived_to_end for r in out])),
        "expected_evaluations": float(np.mean([r.evaluations for r in out])),
        "expected_fees": float(fees.mean()),
        "expected_payouts": float(pays.mean()),
        "expected_net": float(net.mean()),
        "net_se": float(net.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan"),
        "p_net_loss": float(np.mean(net < 0)),
        "net_p5": q(net, 5), "net_p25": q(net, 25), "net_median": q(net, 50), "net_p75": q(net, 75),
        "net_p95": q(net, 95),
        "median_days_to_first_payout": float(np.median(first)) if first else None,
    }



# ------------------------------------------------------------------------------------------------
# day streams
# ------------------------------------------------------------------------------------------------

def parametric_days(mean: float, sd: float, *, excursion: float = 0.5, df: float | None = 4.0) -> DayFn:
    """SYNTHETIC days: pnl ~ mean + sd * (Student-t with ``df`` scaled to unit variance, or normal if None);
    intraday low/high extend beyond the close by |N(0, excursion * sd)| each (an assumption, labelled)."""
    def draw(rng: np.random.Generator, n: int):
        if df is None:
            z = rng.standard_normal(n)
        else:
            z = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
        pnl = mean + sd * z
        low = np.minimum(pnl, 0.0) - np.abs(rng.normal(0, excursion * sd, n))
        high = np.maximum(pnl, 0.0) + np.abs(rng.normal(0, excursion * sd, n))
        return pnl, low, high
    return draw


def bootstrap_days(pnl: np.ndarray, low: np.ndarray, high: np.ndarray, *, block: int = 5) -> DayFn:
    """Stationary-style block bootstrap of HISTORICAL days (keeps short runs of related days together)."""
    pnl, low, high = (np.asarray(a, float) for a in (pnl, low, high))
    m = len(pnl)
    if m < block:
        raise ValueError("need at least one block of history")

    def draw(rng: np.random.Generator, n: int):
        idx = np.concatenate([np.arange(s, s + block) % m for s in rng.integers(0, m, -(-n // block))])[:n]
        return pnl[idx], low[idx], high[idx]
    return draw
