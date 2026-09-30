# Discovery score defect: momentum counted twice

## Root cause

`relative_strength` scored `rs_spy_20` and `rs_spy_63`, defined as `ret_n(stock) - ret_n(SPY)`.
The composite score normalises every feature by **cross-sectional percentile rank within a
session**, and within a session `ret_n(SPY)` is one scalar. A rank is invariant to subtracting a
constant, so:

- `pct_rank(rs_spy_20)` orders the universe exactly as `pct_rank(ret_20d)` — measured on real
  2026-09-29 data: `ret_20d - rs_spy_20` is constant to 5.7e-09 (float32 rounding), Spearman
  0.99999999
- `pct_rank(rs_spy_63)` is exactly `pct_rank(ret_63)`, which momentum approximates with `ret_60d`
  (Spearman 0.95-0.98 on six real sessions)

The features are computed correctly; the defect is in **normalisation + composition**: rank
normalisation annihilates the market adjustment, which is the family's entire content, and
equal-weight composition then gave the one momentum signal ~40% of the score instead of 20%.
Per-date Spearman between the two families' point totals: 0.90-0.96.

## Fix (commit with this document)

`families.SCORED_POINTS` = momentum, volume_activity, breakout_compression, mean_reversion. The
composite is the mean of those. `relative_strength` still fires (its trigger `rs_spy_63 > 0` is a
level test, genuinely market-relative), still explains itself and still reports its points. No
feature definition and no weight changed.

Locked by `tests/discovery/test_score_independence.py` (rank invariance, rs==ret ordering on the
crafted world, score excludes relative_strength, family still fires).

## Measured effect on rankings (six real sessions, 2023-12 .. 2026-09)

| session | symbols | top-20 unchanged |
|---|---|---|
| 2026-09-29 | 3,317 | 11 / 20 |
| 2026-07-07 | 3,294 | 8 / 20 |
| 2026-02-25 | 3,151 | 8 / 20 |
| 2025-07-22 | 2,967 | 14 / 20 |
| 2024-10-01 | 2,681 | 12 / 20 |
| 2023-12-13 | 2,554 | 12 / 20 |

Between 30% and 60% of the top 20 changes. For the live 2026-09-29 plan: HALO stays rank 1
(83.3 -> 81.5), TBBB stays rank 4 (80.4 -> 78.0).

All discovery research recorded before this commit (including DISCOVERY-RESEARCH-2021-2024.md)
was computed on the old composite.
