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
