"""Expected value of a candidate trade, from EMPIRICAL matured outcomes only.

Method (pooled pseudo-trades, a Bayesian-style shrinkage that is exact and easy to audit):

  * Evidence: the strategy version's matured trades (same trade semantics as core.tradesim) from
    :class:`~quantlab.decision.stats_provider.StrategyStatsProvider`: n trades with win rate p_hat,
    mean GROSS win W and mean GROSS loss L. In-sample-only evidence counts as
    ``n_eff = n * expected_value.in_sample_weight`` (it is optimistic). Out-of-sample and forward
    evidence counts in full.
  * Prior: ``k = expected_value.prior_trades`` pseudo-trades with NO edge, meaning win rate p0
    (``expected_value.prior_win_rate``, default 0.5) and payoffs W0 > 0, L0 = -p0/(1-p0) * W0, so
    that p0*W0 + (1-p0)*L0 = 0 (gross EV of the prior is exactly zero). W0 takes the empirical
    win size (or the plan's stop distance when there is no history). It only sets the scale of
    the displayed payoffs, never the EV.
  * Posterior (pool the real and pseudo trades):
        p   = (n_eff*p_hat + k*p0) / (n_eff + k)
        W   = (n_w*W_hat + k*p0*W0) / (n_w + k*p0)          n_w = n_eff*p_hat
        L   = (n_l*L_hat + k*(1-p0)*L0) / (n_l + k*(1-p0))  n_l = n_eff*(1-p_hat)
        gross = p*W + (1-p)*L  ==  n_eff/(n_eff+k) * mean_gross     (identity checked in tests)
  * Net: ``ev = gross - CostModel.round_trip_cost_frac(adv20)``. Unknown liquidity is charged the
    worst cost tier.

Consequences: n = 0 gives the prior only, so gross = 0 and ev = -cost < 0. An unproven strategy
therefore can never pass the EV gate (``expected_value.min_ev_bps_after_costs``). A small n stays
close to the prior and a large n approaches the empirical mean.

LLM output NEVER enters this computation. There is deliberately no parameter through which an AI
review, or its "qualitative_confidence", could reach it.
"""
from __future__ import annotations

import math

from quantlab.config import Config
from quantlab.core.costs import CostModel
from quantlab.core.types import Candidate, ExpectedValue, PitStatus
from quantlab.decision.stats_provider import StrategyStats
from quantlab.logging_setup import get_logger

log = get_logger(__name__)


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


class EVEngine:
    def __init__(self, config: Config, cost_model: CostModel | None = None):
        self.config = config
        self.cost_model = cost_model or CostModel.from_config(config)
        self.prior_trades = float(config.get("expected_value.prior_trades", 50))
        self.prior_win_rate = float(config.get("expected_value.prior_win_rate", 0.5))
        self.in_sample_weight = float(config.get("expected_value.in_sample_weight", 0.25))
        self.min_ev_bps = float(config.get("expected_value.min_ev_bps_after_costs", 10))
        if self.prior_trades <= 0:
            raise ValueError("expected_value.prior_trades must be > 0 (a no-edge prior is mandatory)")
        if not 0.0 < self.prior_win_rate < 1.0:
            raise ValueError("expected_value.prior_win_rate must be in (0, 1)")
        if not 0.0 <= self.in_sample_weight <= 1.0:
            raise ValueError("expected_value.in_sample_weight must be in [0, 1]")

    # ------------------------------------------------------------------------------------------
    def effective_n(self, stats: StrategyStats | None) -> float:
        if stats is None or stats.n <= 0:
            return 0.0
        return float(stats.n) if stats.validated else float(stats.n) * self.in_sample_weight

    @staticmethod
    def _plan_scale(candidate: Candidate) -> float | None:
        """Stop distance as a fraction of the entry reference: the natural loss scale of the plan."""
        e, s = candidate.plan.entry_ref_price, candidate.plan.stop_price
        if _finite(e) and _finite(s) and e > 0 and abs(e - s) > 0:
            return abs(e - s) / e
        return None

    def estimate(self, candidate: Candidate, stats: StrategyStats | None, adv: float | None) -> ExpectedValue:
        cost = self.cost_model.round_trip_cost_frac(adv)
        k, p0 = self.prior_trades, self.prior_win_rate
        n_eff = self.effective_n(stats)
        notes: list[str] = [f"round-trip cost {cost * 1e4:.1f} bps (CostModel, adv20="
                            f"{'UNKNOWN -> worst tier' if not _finite(adv) else f'{adv:,.0f}'})"]
        pit = candidate.pit_status

        has_payoffs = stats is not None and stats.n > 0 and stats.win_rate is not None
        # Scale of the no-edge prior's payoffs. Display only: the EV does not depend on it.
        w0: float | None = None
        if has_payoffs and _finite(stats.avg_win) and stats.avg_win > 0:
            w0 = stats.avg_win
        elif has_payoffs and _finite(stats.avg_loss) and stats.avg_loss < 0:
            w0 = -stats.avg_loss * (1 - p0) / p0
        else:
            w0 = self._plan_scale(candidate)
        l0 = -p0 / (1 - p0) * w0 if w0 is not None else None

        if not has_payoffs or n_eff <= 0:
            reason = "no matured trades" if stats is None or stats.n <= 0 else "evidence weight is zero"
            notes.append(f"prior only ({reason}): gross EV = 0 by construction, so EV = -cost")
            return ExpectedValue(p_win=p0, avg_win=w0, avg_loss=l0, cost=cost, ev=-cost, n_obs=0,
                                 method=f"prior_only(k={k:g})", pit_status=pit, notes=notes)

        p_hat = float(stats.win_rate)
        n_w, n_l = n_eff * p_hat, n_eff * (1 - p_hat)
        p_post = (n_w + k * p0) / (n_eff + k)
        w_hat = stats.avg_win if _finite(stats.avg_win) else 0.0
        l_hat = stats.avg_loss if _finite(stats.avg_loss) else 0.0
        w_post = (n_w * w_hat + k * p0 * w0) / (n_w + k * p0) if w0 is not None else None
        l_post = (n_l * l_hat + k * (1 - p0) * l0) / (n_l + k * (1 - p0)) if l0 is not None else None
        # Identity: pooled mean = n_eff/(n_eff+k) * empirical gross mean (prior mean is 0).
        emp_gross = p_hat * w_hat + (1 - p_hat) * l_hat
        gross = n_eff / (n_eff + k) * emp_gross
        ev = gross - cost

        if not stats.validated:
            notes.append(f"evidence is {stats.source} (not validated): weighted {self.in_sample_weight:g} "
                         f"-> n_eff={n_eff:.1f}")
        if stats.uses_synthetic:
            notes.append("SYNTHETIC evidence: not market evidence")
        notes.append(f"shrinkage weight n_eff/(n_eff+k) = {n_eff / (n_eff + k):.3f}; "
                     f"empirical gross mean {emp_gross * 1e4:.1f} bps -> shrunk {gross * 1e4:.1f} bps")
        if stats.pit_status.rank > pit.rank:
            pit = stats.pit_status
        return ExpectedValue(
            p_win=p_post, avg_win=w_post, avg_loss=l_post, cost=cost, ev=ev, n_obs=int(stats.n),
            method=f"empirical_shrunk(source={stats.source}, n={stats.n}, n_eff={n_eff:.1f}, k={k:g}, p0={p0:g})",
            pit_status=pit if isinstance(pit, PitStatus) else PitStatus(pit), notes=notes,
        )

    def passes_gate(self, ev: ExpectedValue | None) -> bool:
        """EV gate: strictly above ``expected_value.min_ev_bps_after_costs``. Unknown EV fails."""
        return ev is not None and _finite(ev.ev) and ev.ev * 1e4 > self.min_ev_bps
