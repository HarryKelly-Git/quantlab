"""Alpha discovery research engine (docs/ALPHA-DISCOVERY-PLAN.md). PAPER research only.

Separate from the trading stack on purpose: nothing in ``quantlab.alpha`` can place an order or touch
the runner. It answers one question per hypothesis: does the market pay for this, after costs,
out of sample, after accounting for how many things were tried?

Layout:
  store.py       survivorship-free daily data store (Alpaca SIP, active + delisted, 2016-2024 only)
  universe.py    point-in-time research universes (common-stock rules, liquidity, history)
  splits.py      immutable TRAIN / VALIDATION / OOS / HOLDOUT splits and the OOS-once rule
  engine.py      cross-sectional portfolio backtester (weights -> returns, costs, turnover, delistings)
  metrics.py     the full metric suite (returns, risk, tails, trade stats, benchmark-relative)
  mht.py         multiple-testing statistics (White RC, Hansen SPA, PBO/CSCV, permutation, FDR)
  registry.py    persistent hypothesis queue + append-only results ledger + A-G classification
"""
