# What exploration trades by, and why

Exploration used to paper-trade the top of the **discovery score**. It now trades the top of the
**selection score**. This document is the evidence and the protocol behind that change.

## Data and protocol (fixed before looking at results)

- PIT replay of the live discovery scan, real data, 2021-03-12 → 2024-11-27, every 5th session:
  188 sessions, 334,904 research-universe observations, restricted to ADV ≥ $5M (the exploration
  liquidity floor). The locked 2025+ holdout was not touched.
- Entry at the next session's open, exit at the close of D+h, costs from the backtester's
  CostModel (half-spread by ADV tier + slippage, both legs). Target: net return minus the
  same-date universe mean (market removed, costs kept).
- **Half A** (before 2023-01-24) = discovery; **half B** = confirmation.
- Per characteristic, per date: quintiles; top-minus-bottom spread; date-clustered t. h=5 is
  non-overlapping and primary.
- Pre-registered bar, applied to half A only: |t| ≥ 3 and monotonic quintiles.

Scripts and outputs are in the session scratchpad (`feat_replay.py`, `feat_analysis.py`,
`policy_eval.py`, `feat_obs.parquet`); they are research tools, not part of the package.

## Result 1: nothing clears the bar

27 characteristics tested, including every discovery-family input, trend, volatility,
liquidity, industry strength, earnings reaction, news and the discovery score itself. **None**
reaches |t| ≥ 3 in half A.

| characteristic | 5d spread A (bps) | t A | 5d spread B (bps) | t B | read |
|---|---|---|---|---|---|
| discovery score | +13 | 0.55 | +9 | 0.64 | **no predictive power**; quintiles not monotonic (0.0 / 0.3) |
| adv20 (liquidity) | +27 | 2.25 | +29 | 1.93 | monotonic in both halves; about half is cost |
| mom_12_1 | +14 | 0.47 | +24 | 1.33 | monotonic in B; 20-day t 0.65 / 1.52 |
| atr14_pct (volatility) | −59 | −1.27 | −13 | −0.51 | lower volatility better in both halves |
| ret_20d (1-month) | −11 | −0.43 | +6 | 0.39 | leans negative: the short-term reversal |

Liquidity decomposed (half A, 5d, bps, least → most liquid quintile): gross excess −7 → +5, cost
30 → 14. Liquid names win on both counts, and the cost half is certain money.

## Result 2: what the bot would have earned per trade

Top 5 per day, 5-session hold, net of costs, 470 trades per half:

| rule | net A | net B | vs universe A / B | win % A / B | worst decile A / B |
|---|---|---|---|---|---|
| old: top discovery score | −29 | +9 | −8 / +10 | 46 / 52 | −749 / −697 |
| top momentum points | −26 | −95 | −5 / −94 | 49 / 47 | −1,495 / −1,797 |
| **new: top selection score** | **+10** | **+23** | **+31 / +24** | **51 / 55** | **−365 / −351** |
| 12-1 momentum alone | −79 | +59 | −58 / +60 | 47 / 50 | −1,306 / −1,189 |

The selection score beats the old rule in both halves and roughly halves the worst-decile
loss and the median adverse excursion (−368 → −174 bps in A). Momentum alone is unstable and
fat-tailed; the volatility and liquidity terms are what tame it.

## The selection score

Equal-weight mean of per-session percentile ranks of `mom_12_1` (+), `atr14_pct` (−) and
`adv20` (+). The components and signs come from published work, not from this data: 12-1
momentum (Jegadeesh & Titman 1993), the low-volatility effect (Ang et al. 2006; Frazzini &
Pedersen 2014), and liquidity as the cost of trading. Weights are equal and were never tuned.
Any unknown input makes the score UNKNOWN, and such a candidate sorts after every scored one.
The candidate pool is unchanged: exploration still chooses only among discovered setups, through
every existing gate.

## What this does and does not show

- It is **not** statistically significant against the universe (t 1.13 / 1.18).
- It is **still below SPY** after costs (−3 / −16 bps per 5-day trade).
- Half B is **not** a clean out-of-sample test for the composite: the per-characteristic half-B
  results were seen before it was assembled. The clean tests are forward paper trading, which is
  what exploration is for, and the locked 2025+ holdout, which needs a human-issued token.
- The downside reduction is the most robust part. Calmer, more liquid names move less; that is
  mechanics, not luck.

The change moves the bot from a ranking with measured zero skill and fat tails to one with
literature support, sign-consistent evidence in both halves, and half the tail risk. It is a
hypothesis under forward test, recorded on every decision (`selection_score`, `selection_order`
in each decision's pre-trade record) so accepted and rejected candidates can be compared later.
