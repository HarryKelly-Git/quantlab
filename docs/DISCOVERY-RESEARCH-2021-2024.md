# Discovery research: does the score predict anything?

First properly-powered test of the discovery layer. Run 2026-09-30 on a scratch copy of the live
database, real data, point-in-time replay of the live scoring code. The 2025+ locked holdout was not
touched.

```
quantlab discovery-research --start 2021-01-01 --end 2024-12-31 --every 5 --data real
```

- Period 2021-03-12 → 2024-11-27, **188 sample sessions, 334,904 observations**
- Entry at the next session's open, exit at the close of D+h, costs from the backtester's CostModel
  (half-spread tier + slippage, both legs)
- A group must beat the same-date baseline with ≥200 obs on ≥30 dates, consistency in both halves
  (split 2023-01-24) and a date-clustered t ≥ 2.94 (Bonferroni across 15 groups)
- Caveats: currently-listed stocks only (survivorship flatters absolute returns); 10d/20d windows
  overlap across sampled dates

`discovery_research_runs` had zero rows before this — the discovery layer had never been evaluated
against its own baseline.

## Result: all 16 groups FLAT

Not one cohort beat the baseline. Every t-statistic lands between **−1.71 and +0.98**, against a
threshold of 2.94. This is not "almost significant" — there is nothing there.

| group | obs | 5d mean net | vs baseline | halves | t |
|---|---|---|---|---|---|
| Baseline (all research-universe symbols) | 334,904 | −0.11% | — | — | — |
| Any discovery family | 67,950 | −0.21% | **−0.07%** | −0.12% / −0.01% | −1.04 |
| High-ranked (score ≥ 70) | 9,077 | −0.17% | **−0.14%** | −0.21% / −0.07% | −0.98 |
| Momentum | 43,801 | −0.30% | −0.13% | −0.25% / −0.02% | −1.46 |
| Relative strength | 29,528 | −0.28% | −0.19% | −0.36% / −0.01% | −1.55 |
| **Momentum + relative strength** | 24,105 | **−0.35%** | **−0.21%** | −0.37% / −0.05% | **−1.71** |
| Volume/activity | 16,547 | −0.04% | −0.16% | −0.21% / −0.10% | −1.63 |
| Breakout/compression | 10,520 | −0.25% | −0.08% | −0.10% / −0.06% | −0.77 |
| Mean reversion | 6,831 | −0.21% | +0.21% | +0.54% / −0.11% | +0.98 |
| High momentum + abnormal volume | 524 | +0.44% | −0.10% | +0.34% / −0.49% | −0.14 |

(Remaining six combination groups all FLAT, between −0.34% and −0.17% mean net 5d.)

## What this actually says

**1. The score has no demonstrated skill, and its point estimate is mildly negative.**
"Any discovery family" underperforms the baseline by 7 bps over 5 days. "High-ranked (score ≥ 70)"
underperforms by **14 bps** — i.e. the higher the score, the slightly *worse* the outcome. The
ranking that selects which setups get paper-traded is, on this evidence, not better than picking from
the research universe at random.

**2. The setup class currently being traded is the worst cohort in the study.**
Exploration selects the top-ranked setups, which in practice are labelled "Momentum + Relative
strength". That group is last: −0.35% mean net at 5 days, −0.21% versus baseline, t = −1.71, and
negative in *both* halves. This is the one group where the point estimate is consistently against us.

**3. Longer holds make momentum worse, not better.**
Momentum mean net by horizon: 1d −0.24%, 3d −0.31%, 5d −0.30%, 10d −0.34%, **20d −0.43%**. The
10-session holding period sits on the wrong side of that curve. Mean reversion runs the other way
(10d +0.41%, 20d +0.80%) — one horizon rule cannot suit both.

**4. Momentum and relative strength are the same signal counted twice.**
Per-date Spearman between the two families is **0.94**. The report names the mechanism: within one
date, `rs_spy_n` is `ret_n` minus a constant (SPY's return), so their cross-sectional *ranks* are
identical. The two highest-firing families share two of their inputs, so "Momentum + Relative
strength" is largely "momentum, scored twice" — which is precisely why it dominates the top of the
ranking, and it is a defect in the scorer rather than a property of the market.

**5. The two things that look interesting both fail the consistency test.**
- *High momentum + abnormal volume*: +0.44% at 5d, +0.99% at 10d, 52–53% win rate — but n=524 and the
  halves are +0.34% / **−0.49%**. It changes sign.
- *Mean reversion at 10–20d*: +0.41% / +0.80% mean net, +0.21% vs baseline, n=6,831 — halves
  +0.54% / **−0.11%**. Also changes sign.

Neither is evidence. They are the only two places where a *pre-registered* follow-up test would be
worth the compute, and that test must be specified before looking at more data.

## What follows from it

**Justified by mechanism, do it:** fix the momentum / relative-strength rank identity. This is a
scoring defect with a named cause, not a fitted choice — either drop one family, or redefine relative
strength so it is not a rank-preserving transform of momentum within a date (e.g. versus industry
peers rather than versus SPY). Until it is fixed, the composite score is mis-weighted by construction
and every ranking built on it is suspect.

**Not justified, do not do it:** re-pointing exploration at "High momentum + abnormal volume" because
it topped this table. That is selection on noise — the cohort flips sign between halves and has 524
observations. Choosing it now would be exactly the overfitting this study exists to prevent.

**Open question the study cannot answer:** whether any daily price/volume ranking on liquid US
equities can clear costs. Combined with the walk-forward results (6 strategies, all NOT_SIGNIFICANT)
and the catalyst study (3 flat, 1 negative), the tally is now 8 strategies + 16 discovery cohorts +
4 catalyst hypotheses, all flat or negative after costs, across independent methodologies. That
consistency is itself the finding: the search space, not the implementation, is the binding
constraint.

## Status of live exploration

Unchanged by this document. EXPLORATION mode exists to trade candidates *without* proven edge and
learn forward, which is a deliberate choice and stays a deliberate choice. What changes is the
expectation attached to it: forward exploration results should be read as a test of the machinery,
not as a search for profit, and the prior on the current setup class is now measured and mildly
negative rather than merely unknown.
