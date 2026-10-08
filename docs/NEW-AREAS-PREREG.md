# New areas: pre-registration (2026-10-08)

Committed BEFORE any result was computed. PAPER research. Nothing is used from 2025 on (sealed holdout).

**Why these three.** Stock-picking rules, events, news and options have all failed after costs. These
areas instead:
- (A) use QuantLab's one proven strength, volatility forecasting, at the level of a whole-market ETF
  holding, which is where Harry's real money is;
- (C) test a well-known market-timing effect on SPY;
- (F) apply the "high predicted volatility underperforms" finding to the bot's own entries.

## A. Market overlays on 60 years of US market data

**Data:** Ken French daily factors, July 1963 .. Dec 2024.
- Market = Mkt-RF + RF, value-weighted US stocks.
- Cash = RF (one-month T-bill).

**Splits:**

| Split | Years |
|---|---|
| TRAIN | 1963-07 .. 1999 |
| VAL | 2000 .. 2012 |
| OOS | 2013 .. 2024 |

**Strategies.** All rebalance MONTHLY at month end using data up to that day.

| ID | Rule |
|---|---|
| BH | 100% market |
| VM1 | Volatility-managed, no leverage. Weight = min(1, target / last month's realised vol). Target = the TRAIN-period market vol. |
| VM15 | The same, capped at 1.5x. Borrowing costs RF + 1%/yr. |
| TR10 | Trend. 100% market if the market's total-return index is above its 10-month average at month end, else 100% cash. |
| VM1+TR10 | VM1 weight while the trend is up, 0 otherwise |

**Costs:** 5 bps per unit of weight traded.

**Metrics:**
- CAGR, volatility, Sharpe (excess of cash), max drawdown, worst calendar year;
- average market exposure;
- alpha vs BH: the intercept of monthly excess returns on BH's, Newey-West 6 lags;
- Sharpe difference vs BH with a 90% 12-month-block bootstrap CI.

**Verdicts per rule:**

| Verdict | Condition |
|---|---|
| RISK-ADJUSTED IMPROVEMENT | Sharpe above BH in TRAIN, VAL and OOS, and the OOS CI of the difference above 0 |
| ABSOLUTE IMPROVEMENT | The above, plus OOS CAGR >= BH |
| RISK REDUCTION ONLY | Lower max drawdown, without either improvement |
| NO IMPROVEMENT | Otherwise |

**Also reported:**
- the post-publication periods (TR10 2007-2024; VM 2017-2024);
- the equity curves.

## C. SPY overnight vs intraday (2016-2024, Alpaca SIP, survivorship-free store)

| ID | Holding |
|---|---|
| ON | Hold SPY from each close to the next open only, flat intraday. 1 bp per side, i.e. 2 bps per day. |
| ID | Hold SPY open to close only, same cost |
| BH | Hold SPY |

**Splits:** TRAIN 2016-19, VAL 2020-21, OOS 2022-24.

**Verdict:** ON beats BH if its net Sharpe is above BH's in all three splits, with the OOS CI above 0.

## F. Low-volatility screen on the bot's own strategies (replay, corrected delisting convention)

**Filter:** drop a candidate when QuantLab's 20-day volatility forecast for it (P2 model QL_OLS, daily
forecasts) is above the 80th percentile of that day's liquid universe. A missing forecast keeps the
candidate; the number kept this way is reported.

**Comparison:** the filtered book against the unfiltered S1 book.

**Rule:** the E4 IMPROVEMENT rule.
- OOS Sharpe +0.10 or more, with the 90% monthly-block CI above 0;
- CAGR not lower;
- the TRAIN and VAL Sharpe differences both above 0.

**Variants:** one variant per area as listed. No thresholds are tuned.

---

## Results (added 2026-10-08, after the runs; the rules above were not changed)

Results: `research/alpha/results/new_areas/`. Ledger: `NA_A_market_overlays`, `NA_C_spy_overnight`, `NA_F_lowvol_screen`.

**A. Market overlays: every rule RISK REDUCTION ONLY (corrected 2026-10-08).**

> **Correction.** The first run applied each month-end decision one month late: on 95% of days it used the
> decision from two month ends earlier. The cause was a code bug (`shift(1)` before a forward-filled
> reindex), not a rule change. That run reported every rule as NO IMPROVEMENT; its file is kept as
> `A_market_overlays__lagged_bug.json` and its ledger row stays. The corrected re-run uses the same
> pre-registered rules, with ledger override `A-TIMING-FIX`.

| Rule | 2013-24 return/yr | Sharpe | Worst fall | 2000-12 worst fall |
|---|---|---|---|---|
| BH | 14.5% | 0.79 | −34% | −55% |
| VM1 | 12.4% | 0.85 | −24% | −36% |
| VM15 | 12.8% | 0.76 | −26% | −37% |
| TR10 | 11.3% | 0.81 | −22% | −26% |
| VM1+TR10 | 10.6% | 0.82 | −18% | −18% |

**Reading it**
- Every rule cut the worst falls by a third or more.
- Each one returned 1.7-3.9 points a year less than holding in 2013-24.
- None beat holding's Sharpe in all three periods: each was below BH in 1963-99. So none passes as an
  improvement.

These rules are insurance against long bear markets, paid for with return. Whether that is worth it is a
risk choice for real money, outside QuantLab's paper scope.

**C. SPY overnight / intraday: both DO NOT BEAT buy-and-hold.** Overnight-only made −0.7%/yr after costs in
2022-24 (buy-and-hold +8.9%); intraday-only −0.8%.

**F. Low-volatility screen on the bot's book: NO IMPROVEMENT** (corrected delisting convention).

| Period | Sharpe, current book | Sharpe, screened | Difference (90% CI) |
|---|---|---|---|
| 2016-19 | +0.25 | −0.02 | −0.27 (−1.01, +0.47) |
| 2020-21 | +0.06 | +1.17 | +1.10 (−0.01, +2.13) |
| 2022-24 | +0.01 | −0.35 | −0.34 (−0.95, +0.34) |

- 87,537 of 330,660 candidate-days were dropped; 448 had no forecast and were kept.
- Descriptive: 46% of the current book's trades sit in the most volatile fifth of the liquid universe
  (momentum_trend and relative_strength about 90%). Those trades averaged −0.8% against +0.4% for the rest,
  2016-24. Removing them still made the book worse in two of three periods: the slots they free are taken by
  other candidates that did not do better.
- Under master's −30% delisting rule the screen is worse in every period (2022-24 difference −1.04).
