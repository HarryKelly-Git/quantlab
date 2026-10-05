# How big an edge does a prop-firm lifecycle need? (2026-10-05)

Research only. Rules are **HYPOTHETICAL** (the typical *shape* of a $50k futures evaluation: $3k target,
$2k trailing end-of-day drawdown locking at the start balance, $1k daily loss limit, $100/month evaluation,
$80 reset, $100 activation, 90% split, $2k payout cap, 40% funded consistency). They are **not any firm's
verified rules**. Day P&L streams are **SYNTHETIC** (Student-t, df 4, intraday excursions assumed).
Simulator: `src/quantlab/futures/propsim.py` (19 rule tests). Script: `scripts/research/prop_required_edge.py`.
2,000 one-year lifecycles per cell, seed 11. Edge = mean daily net P&L / daily P&L standard deviation.

## Expected net payout over one year (payouts received minus all fees)

| daily sd | edge -0.10 | -0.05 | 0.00 | +0.05 | +0.10 | +0.15 | +0.20 |
|---|---|---|---|---|---|---|---|
| $250 | -1,316 | -1,139 | -714 | +48 | +1,169 | +2,529 | +4,168 |
| $500 | -1,280 | -686 | +309 | +1,856 | +4,061 | +6,640 | +9,610 |
| $1,000 | -2,120 | -924 | +1,217 | +4,427 | +8,863 | +14,978 | +22,350 |

With a payout buffer (payouts only above $2.5k profit, $2k must stay in the account), zero edge at $250 sd
is -920 (90% of traders lose money) and +0.05 is still -131. Full rows in the script output.

## What it means
1. **Negative edge always loses**, and most day-trading after costs is negative edge.
2. **The account is an option.** The trader's loss is capped at fees, payouts are not, so more variance
   raises payouts even with no edge. Under these rules a zero-edge trader at $500-1,000 daily sd shows a
   small positive mean, but still loses money 52-65% of the time. Whether that convexity survives depends
   entirely on the real rules (buffers, caps, consistency, contract limits, activation and monthly fees).
   That is why only VERIFIED firm rules may be used for decisions.
3. **The bar for a real, repeatable result:** about **+0.10 daily edge** (annualised Sharpe ~1.6 after all
   costs) gets the chance of a net loss below ~20-40% and a first payout in ~70-90% of years. +0.05
   (Sharpe ~0.8) is marginal. Any futures strategy should be judged against this bar, using its own daily
   P&L stream through `bootstrap_days`, not synthetic streams.

## Next
* Load one or two real firms' rules from official documentation (template: `config/propfirms/TEMPLATE.yaml`).
* Synthetic futures test harness (entries, no look-ahead, stops/targets, same-bar resolution, costs,
  sessions/DST, rolls, determinism), before any real data.
