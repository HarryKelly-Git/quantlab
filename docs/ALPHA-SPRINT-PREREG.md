# Alpha sprint: pre-registration (2026-10-07)

Committed BEFORE any sprint evaluation was run. Baseline truth: [ALPHA-DISCOVERY-REPORT-2026-10.md](ALPHA-DISCOVERY-REPORT-2026-10.md).
Question: can QuantLab discover and validate at least one economically meaningful edge after realistic
costs? If one survives, it is integrated into PAPER trading. PAPER ONLY; no live path is added.

Every value below is fixed now. Changing one after seeing a result creates a new, labelled variant. It
is logged in the ledger, and its result is reported next to the original.

## 0. Splits, data, and what "out of sample" means here

| Area | TRAIN (fit) | VAL (choose among pre-listed variants) | OOS (one look per item) | Sealed |
|---|---|---|---|---|
| Equity, vol forecasts, master replay | 2016-01 .. 2019-12 | 2020-01 .. 2021-12 | 2022-01 .. 2024-12 | 2025+ |
| Options (DoltHub end-of-day chains) | 2019-02 .. 2021-12 | 2022 | 2023 .. 2024 | 2025+ |

- **2022-24 is research history, not a pristine test set.** Earlier hypotheses were evaluated on these
  years. No forecast model, sizing rule, option rule or regime filter in this file has been. Each
  item gets one OOS look, recorded in `research/alpha/ledger.jsonl`.
- **The 2025+ holdout stays sealed** (`HOLDOUT_LOCK`). Nothing in this file uses it, ranks on it or
  tunes on it. This file is not a holdout pre-registration.
- **Data:**
  - **Prices:** the survivorship-free Alpaca SIP store (panel v3, entity-by-date resolution,
    delisted companies included).
  - **Options:** DoltHub chains (OPTIONS_DATA_VERSION audit-b; wrong-company months removed).
  - **The bot's forward records:** a copy of the paper database, when Harry exports it.
- **Options data limits that bind the conclusions:** end-of-day snapshots only (weekly in 2019,
  Mon/Wed/Fri 2020-24). No option volume, no open interest, no quote timestamps.
- **Alpaca PAPER fills options at the NBBO.** "Limit orders fill only when marketable: buy limit >=
  best ask" (docs/EXTERNAL-SERVICES.md). In the paper account a mid fill is therefore impossible, and
  the PESSIMISTIC level below is the realistic one for paper.

## 1. P1: missed opportunities and filter ablation

**Bot database** (`alpha/botdb.ablation`, coded and tested before any bot data arrived):

| Scenario | What changes |
|---|---|
| A | current decisions |
| B | remove the liquidity filter (`no_trade.liquidity`) |
| C | remove the volatility filter (`no_trade.volatility`) |
| D | remove the score threshold: **not identifiable from the database** (sub-threshold names are never recorded); tested on history below |
| E | remove each other blocking check one at a time (every check name observed failing) |
| F | relaxed thresholds: ADV floor x0.5, volatility cap x1.25, EV floor 0 bps instead of 10 |

- A rejected candidate is re-admitted only if ALL its blocking failures are removed. EV is re-checked
  from `decisions.ev_json` when the risk chain stopped before its EV stage.
- Re-admitted sets are an upper bound: portfolio capacity and sizing are not replayed.
- **Status rule:** EXPLORATORY below 20 distinct as-of dates. No winner is declared, and no filter is
  changed on this sample.

**History (master replay, section 3 harness):** every replayable strategy is run with its own score
threshold relaxed one step:

| Strategy | Parameter | Live | Relaxed |
|---|---|---|---|
| momentum_trend | min_score_pct | 0.90 | 0.80 |
| relative_strength | vs_market_min_pct | 0.80 | 0.70 |
| mean_reversion | shock_z | -2.0 | -1.5 |
| breakout | volume_mult | 1.5 | 1.25 |
| extreme_reversal | extreme_return_z | -4.0 | -3.0 |
| sector_rotation | stock_rank_pct | 0.80 | 0.70 |

Reported per strategy on VAL and OOS: trades, net return per trade, Sharpe, CAGR.

## 2. P2: volatility and move-magnitude forecast engine (`alpha/volforecast.py`)

**Sample:**
- **Origins:** every week's last session at the close, 2016-2024 (TRAIN/VAL/OOS by origin date).
- **Universe:** the research liquid universe at the origin: price >= $5, MDV20 >= $5M,
  >= 252 sessions, common stock.
- **Horizons:** h = 1, 3, 5, 10, 20 sessions.

**Targets** (outcomes only; never features):
- RV_h = sqrt(252/h x sum of squared daily log total returns over t+1..t+h).
- Absolute move: |P_{t+h}/P_t - 1| on total-return prices.
- P(|move| > 2%), P(|move| > 5%), P(|move| > 10%).
- Forecast quantiles q90, q95 and q99 of |move|.
- A delisting inside the window books the panel's delisting return on the next session (floored at
  -99% for logs).

**Baselines** (trailing only, at the origin close):

| Name | Definition |
|---|---|
| HV21 | 21-session close-to-close vol of daily log returns, annualised (historical RV) |
| HV63 | the same over 63 sessions (rolling vol) |
| EWMA | RiskMetrics, lambda 0.94, seeded with the first 21 sessions |
| GARCH | GARCH(1,1) with variance targeting. omega_i = (1-a-b) x trailing-252 variance of stock i. (a, b) by pooled QMLE on TRAIN (500 randomly drawn stocks, seed 7). Multi-step variance by the GARCH mean-reversion formula. |
| HAR | log RV_h on log RV over the trailing 1, 5 and 22 sessions; pooled OLS per horizon on TRAIN |
| IV | ATM implied vol, option subset only (section 4) |

**QuantLab model:**
- **Target:** log RV_h.
- **Features:**
  - HAR terms;
  - Parkinson range vol (21);
  - overnight and intraday vol (21);
  - SPY HV21 and SPY HV21/HV252;
  - log MDV20;
  - 21- and 252-session return;
  - distance from the 252-session high;
  - largest |return| in 21 sessions;
  - each stock's HV21 relative to the universe median.
- **Earnings:** the earnings-in-window flag is PIT_ASSUMED (the 2020-24 calendar is backfilled). It
  is used only in a labelled variant ("QL+earn", option-era rows); the primary model excludes it.
- **Two fixed specifications, fit on TRAIN:**
  - OLS;
  - HistGradientBoostingRegressor(max_iter 300, learning_rate 0.05, max_leaf_nodes 31,
    min_samples_leaf 200, random_state 7).
- **Selection:** the one with the lower VAL QLIKE goes to OOS.

**Conversion to the RV scale:** regression models use exp(pred + s^2/2), where s^2 is the TRAIN
residual variance. Scale baselines are reported raw AND with one scalar per horizon: the TRAIN median
of realised/forecast. This compares information content, not level bias.

**Distribution:**
- |move_h| = sigma_h x sqrt(h/252) x |Z|, with Z a unit-variance Student-t.
- nu is fit per horizon and model on TRAIN (grid 3..30).
- Probabilities and quantiles follow from that distribution.

**Metrics** (VAL and OOS, every horizon):
- MAE and RMSE of RV;
- QLIKE = RV^2/f^2 - ln(RV^2/f^2) - 1 (primary loss);
- bias: mean ln(f/RV);
- Mincer-Zarnowitz slope and intercept;
- mean cross-sectional Spearman IC;
- decile calibration (forecast decile -> mean realised RV);
- for P(>2/5/10%): Brier, log-loss, AUC and a 10-bin reliability table;
- tail coverage of q90/q95/q99, with date-block bootstrap CIs;
- directional independence: AUC of f for sign(return_h) and Spearman(f, signed return), expected
  about 0.5 and 0;
- regime stability: QLIKE relative to HV21 by year, by SPY-vol regime and by liquidity tercile;
- Diebold-Mariano t on weekly mean QLIKE differences, Newey-West with 4 lags.

**Decision rule:**
- The QuantLab model "beats the baselines" only if its OOS QLIKE is lower than every non-IV baseline
  at >= 3 of 5 horizons with DM t <= -2.
- Otherwise the simplest baseline within 1% of the best QLIKE is the selected forecast.
- Statistical superiority is not economic value (section 7).

## 3. P3: volatility-adjusted sizing on the existing paper strategies (`alpha/master_replay.py`)

**Harness:** master's own code runs unchanged: UniverseEngine, FeatureSet, StrategyArena and
BacktestEngine, with `config/default.yaml` (identical to master's, apart from the separate
momentum_breakout section).

**Input data:** a DataBundle built from the survivorship-free store:
- raw OHLCV;
- split and dividend factors derived from the raw and total-return closes;
- security types from the security master.

**Strategies:**
- **Replayed:** momentum_trend, mean_reversion, breakout, relative_strength, extreme_reversal,
  sector_rotation.
- **NOT REPLAYED** (their evidence comes from the bot database only):
  - pead_ear: needs a PIT earnings-event feed the store lacks before 2020;
  - quality_momentum: needs fundamentals.
- Master's own conventions apply: delisting return -30%, cost tiers, next-open entries, stops.

**Sizing rules.** Signals, entries, exits, stops, costs and capacity are identical across rules
(max_positions 10). Only the quantity changes.

| Rule | Quantity |
|---|---|
| S0 fixed | equal_weight, 10% per position (engine-native) |
| S1 current | equal_risk: 0.5% of equity at risk to a stop of stop_atr x ATR14; cap 10%. Engine-native, and what the bot does today. |
| S2 inverse-vol | w = 10% x m / HV21_i, where m = the TRAIN median HV21 of entered candidates; cap 25% (sanity only) |
| S3 capped inverse-vol | S2 with a 10% cap |
| S4 forecast-vol | S3 with the section 2 selected forecast of RV over the strategy's holding period, in place of HV21 |

- Implemented by subclassing BacktestEngine in `alpha/` (master's engine file is not edited).
- **Metrics** (VAL and OOS, net of costs):
  - CAGR, total return, volatility, Sharpe, Sortino, max drawdown, Calmar, worst month, skew;
  - average gross exposure; return per unit of average exposure;
  - turnover, number of trades, net return per trade.
- **Classification** (OOS, against S1):
  - **IMPROVEMENT:** Sharpe +0.10 or more, with the 90% monthly-block bootstrap CI of the
    difference above 0, AND CAGR not lower.
  - **RISK REDUCTION ONLY:** vol or drawdown lower, without both of those.
  - **WORSE:** otherwise.
- Lower volatility or drawdown alone is never called success. Every result also reports whether
  each strategy has ANY net edge on survivorship-free data under S1.

## 4. P4: implied vs forecast volatility across maturities

**Sample:** DoltHub features at every snapshot and expiry.
- **Liquid subset** (primary): underlying MDV20 >= $20M and straddle spread <= 10% of mid. All rows
  are reported too.
- **DTE buckets** (calendar days): 5-12, 13-25, 26-45, 46-75.
- **Forecast:** each section 2 model at the snapshot, mapped to the sessions to expiry by log-variance
  interpolation between horizons (flat beyond 20).

**Tests:**
- (a) QLIKE of IV and of each model against RV to expiry, by bucket.
- (b) Encompassing: ln RV = a + b1 ln IV + b2 ln E[RV], fit on TRAIN. Report the VAL/OOS
  out-of-sample R^2 gain over an IV-only fit, and b2 with Driscoll-Kraay t (6 weekly lags).
- (c) Residuals:
  - G = ln E[RV] - ln IV;
  - by decile of G: realised ln(RV/IV), and straddle returns at the four fill levels of section 5;
  - the "E[RV] - IV > 0" side is a buy, the "IV - E[RV] > 0" side a sale.

**Rule:** QuantLab adds information beyond IV only if the OOS R^2 gain > 0 AND the OOS b2 DK t >= 3.
IV is the baseline to beat; it is not assumed beaten.

## 5. P5: options execution engine (`alpha/opt_exec.py`)

**Fill levels** for a buy, with h = (ask - bid)/2. A sale mirrors around mid.

| Level | Fill |
|---|---|
| MID | mid |
| CONSERVATIVE | mid + 0.5h |
| PESSIMISTIC | ask (the Alpaca paper fill) |
| WORST_REASONABLE | ask + 0.25h (thin or stale quote, size) |

**Fees and settlement:**
- Regulatory and clearing fees: $0.03 per contract per fill (ASSUMED; Alpaca charges no options
  commission).
- Positions are held to expiry and settle at intrinsic value on the expiry close.
- An ITM settlement pays the equity cost model on the intrinsic notional.

**Tracked per trade:** bid, ask, spread $, spread % of mid, underlying MDV20, DTE, delta, slippage $
at each level, and the following, all UNKNOWN in this data and marked so:
- option volume;
- open interest;
- quote freshness.

**Classification** (OOS, liquid subset):
- **ROBUST:** mean net return > 0 at PESSIMISTIC, with the date-clustered 95% CI lower bound > 0.
- **EXECUTION-SENSITIVE:** mean > 0 at MID or CONSERVATIVE, but not ROBUST.
- **NONVIABLE:** mean <= 0 at MID.

A midpoint-only result is never reported as alpha.

## 6. P6: four option hypotheses, no others

**Entries:** one per (underlying, week), from the week's first snapshot. Expiry: the one nearest 30
DTE within 21-45 calendar days.

| # | Rule | Precondition / variant |
|---|---|---|
| O1 | Long ATM straddle when (forecast E\|move\| to expiry) / (implied move = straddle mid / spot) >= 1 + k | k chosen on VAL from {0.10, 0.20, 0.30} by mean net return at CONSERVATIVE; then one OOS look. Part 49 tested a related rule (-2.0%). The new reason is the section 2 model and the four fill levels. If that model does not beat the Part-49 model on QLIKE, O1 is labelled a replication. |
| O2 | Long 25-delta strangle (call and put with delta nearest +/-0.25, same expiry, prices from the chains) when model E[payoff at expiry] - cost at CONSERVATIVE > 0 | E[payoff] from the section 2 distribution |
| O3 | Defined-risk debit spread | Needs magnitude AND direction evidence: a directional model with OOS AUC >= 0.55 at the holding horizon. None exists (H12 0.51-0.52; every directional family E/D). **Not run by rule.** |
| O4 | Calendar (sell the front ATM, buy the back ATM, same strike) | Needs a term-structure edge on TRAIN+VAL: the IV slope (back - front) must predict front-vs-back realised vol beyond the IVs themselves (DK t >= 3). If not, **not run**. If it is, enter when the forecast front RV < front IV x 0.9, and exit at the front expiry at the back leg's bid. |

**Recorded for every trade:**
- expected move, implied move, confidence (TRAIN sd of the log forecast error);
- cost, spread cost and slippage per fill level;
- breakeven move, model P(profit), model EV;
- max loss, max profit, holding period;
- realised outcome.

## 7. P7: model competition (economic value decides)

**Rows:**
- HV21, HV63, EWMA, GARCH, HAR;
- QuantLab (the selected specification);
- IV (option subset);
- IV+QuantLab ensemble: log-linear weights fit on TRAIN, kept only if VAL improves;
- no forecast: fixed sizing, no option trade.

**Columns:**
- OOS QLIKE, IC and calibration error;
- OOS Sharpe and CAGR of the replayed master book sized with that forecast (S3 structure);
- OOS O1 net return per trade at CONSERVATIVE and PESSIMISTIC.

**Winner:** best OOS economic value. If no row creates positive OOS economic value after costs, the
leaderboard says so and there is no winner. Rows within noise of each other: the simpler one wins.

## 8. P8: regimes (PIT labels, trailing data only)

**Labels:**

| Label | Definition |
|---|---|
| trend | SPY > its 200-session SMA |
| volatility | SPY HV20 trailing-504 percentile terciles |
| breadth | share of liquid names above their own 50-session SMA; trailing-504 terciles |
| correlation | var(equal-weight return) / mean var(stock returns) over 63 sessions; trailing-504 terciles |
| momentum | SPY 126-session return > 0 |
| risk-on/off | HYG/LQD 63-session change > 0 |

**Reported:** mean net return per trade and trade counts per strategy and label, on TRAIN+VAL and on
OOS.

**Disable rule:**
- Fixed on TRAIN+VAL: disable a strategy in a label value when its mean net trade return there is < 0
  with date-clustered t <= -2 and >= 100 trades.
- Applied once to OOS: does total net P&L and Sharpe improve?
- Holm correction across the (strategy, label) cells tested.

## 9. P9: paper integration gate

A candidate is integrated only if ALL of these hold:
1. OOS economic value > 0 after realistic costs (for options: ROBUST at PESSIMISTIC);
2. it beats its simple baseline OOS;
3. it is positive in >= 2 of the 3 OOS years;
4. it is not merely risk reduction.

**If one qualifies:**
- A separate PAPER pre-registration is committed first. It covers entry/exit, sizing, maximum
  exposure, maximum daily loss, liquidity limits, execution assumptions and kill conditions.
- Master's safety controls are unchanged.
- No live path.

**If none qualifies:** nothing is integrated, and docs/ALPHA-SPRINT-FINAL.md says so plainly.

## 10. Reproducibility

- Every run is appended to `research/alpha/ledger.jsonl`.
- Results go to `research/alpha/results/sprint/`.
- Scripts are `scripts/research/alpha/sprint_*.py`.
- The seed is 7, and the git commit is recorded in every result file.
