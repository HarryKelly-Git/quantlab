"""How large must a daily edge be for a prop-firm lifecycle to pay, net of fees?

HYPOTHETICAL rules (typical SHAPE of a $50k futures evaluation; NOT any firm's verified rules). SYNTHETIC
day streams (Student-t, df 4). Output: expected net payout over one year per (daily sd, mean/sd) cell.

    set PYTHONPATH=<worktree>\\src
    python scripts\\research\\prop_required_edge.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time

from quantlab.futures.propsim import PropRules, parametric_days, simulate

HYPO = PropRules(
    name="HYPOTHETICAL_50K (shape only, unverified, not any firm)",
    start_balance=50_000, profit_target=3_000, max_drawdown=2_000, drawdown_type="trailing_eod",
    trail_lock_profit=0, daily_loss_limit=1_000, dll_action="stop_day",
    eval_fee=100, eval_fee_period_days=21, reset_fee=80, activation_fee=100,
    funded_consistency=0.4, min_days_between_payouts=5, payout_min_amount=500, payout_cap=2_000,
    payout_split=0.9, restart_after_funded_fail=True, verified=False)

SDS = (250, 500, 1000)
EDGES = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def main() -> None:
    t0 = time.time()
    rows = []
    for sd in SDS:
        for e in EDGES:
            r = simulate(HYPO, parametric_days(e * sd, sd), n_paths=2000, horizon_days=252, seed=11)
            rows.append({"daily_sd": sd, "mean_over_sd": e, "daily_mean": e * sd, "annual_sharpe": e * 252 ** 0.5,
                         **{k: r[k] for k in ("p_pass_any_eval", "p_pass_first_eval", "p_first_payout",
                                              "p_second_payout", "expected_evaluations", "expected_fees",
                                              "expected_payouts", "expected_net", "net_se", "p_net_loss",
                                              "net_p5", "net_median", "net_p95", "median_days_to_first_payout")}})
            x = rows[-1]
            print(f"sd {sd:5d} edge {e:+.2f} | pass {x['p_pass_any_eval']:.0%} pay1 {x['p_first_payout']:.0%} "
                  f"fees {x['expected_fees']:6.0f} payouts {x['expected_payouts']:7.0f} net {x['expected_net']:+7.0f} "
                  f"(se {x['net_se']:.0f}) P(loss) {x['p_net_loss']:.0%}", flush=True)
    json.dump({"rules": HYPO.name, "rows": rows}, open(sys.argv[1], "w"), indent=1)
    print(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
