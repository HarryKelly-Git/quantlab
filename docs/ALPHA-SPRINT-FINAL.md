# Alpha sprint: final report (2026-10-07)

**Question:** can QuantLab discover and validate at least one economically meaningful edge after realistic
costs, and if so, integrate it into paper trading?

**Answer: no edge survives.** Nothing is integrated.
- **Strongest surviving hypothesis:** O1, buying ATM straddles when QuantLab's forecast move exceeds the
  implied move by at least 20%.
  - It is positive out of sample at every fill level.
  - It is not statistically robust, all of its profit comes from 2023, and its stock selection does not
    beat buying every straddle on the same dates.
- **Vol forecasting:** QuantLab's forecast is the best volatility forecaster tested. That is statistically
  useful, but it is not economically tradeable: sizing with it does not rescue the bot's strategies, and
  option prices already contain almost all of its information.
- **The bot's own strategies:** they have no net edge on survivorship-free 2016-24 data.

Pre-registration: [ALPHA-SPRINT-PREREG.md](ALPHA-SPRINT-PREREG.md), committed before any evaluation (b5f107f).
Results: `research/alpha/results/sprint/`. Ledger: `research/alpha/ledger.jsonl` (hypotheses S2, S3, S6, S7, S8).
Code: `src/quantlab/alpha/{volforecast,master_replay,opt_exec,ca_fixes,botdb}.py`, `scripts/research/alpha/sprint_*.py`.
PAPER ONLY. No live path was added. Master's code and safety controls were not changed.

---

## Two findings that changed the numbers (read first)

**1. The data had fake corporate-action moves. Fixed in patch "sprint-c", `alpha/ca_fixes.py`.**

Replaying the bot's strategies exposed impossible losses. The worst was BHVN, "−94% in a week". Pfizer
bought Biohaven for about $148.50 cash plus spin-off shares, and the spin-off took over the ticker. A scan
found more fakes in Alpaca's adjusted bars:

| Stock | Event | Adjusted move | Raw move |
|---|---|---|---|
| NVS 2019-04-09 | Alcon spin-off | −82% | −12% |
| CNX 2017-11 | | −90% | |
| SITC 2018-07 | | −92% | |
| BTX 2022-10 | | −95% | flat |
| RTX 2020-04 | | −71% | |
| EQT 2018-11 | | −57% | |
| UHAL 2022-11 | 9-for-1 share distribution, never adjusted | −90% | −90% |

UHAL holders were actually whole.

Alpaca's corporate-action list has only 120 spin-offs for 2016-24, so a price rule does the flagging. It
flags days where the adjustment AMPLIFIED the move; a correct adjustment only ever shrinks one.
- **Flagged:** 94 entity-days (25 inside the liquid universe), set to UNKNOWN, with adjusted prices
  re-chained.
- **Truncated:** 2 cash-merger ticker reuses (BHVN, VIVO).
- **Manual:** 1 entry with its source (UHAL).

Every sprint result below was re-run on the corrected panel. The first-run ledger entries are kept and
marked by the `SPRINT-C` override.

Effect of the fix:
- The P2 forecast verdict flipped from HAR to QuantLab.
- The bot's book went from OOS Sharpe −1.46 to −0.77.

The equity families in the October report (H01-H23) used the uncorrected panel. With about 25 fake
in-universe days in nine years, their decile results are unlikely to flip, but this is unmeasured. See
section 11.

**2. Master's delisting rule decides the sign of the bot's results.**

Master's backtester books −30% whenever a held stock stops trading. On survivorship-free data, every one
of the 13 delisted trades in the bot's book was a cash acquisition at or above the last price:

> Corium, American Railcar, Dova, Casper, Turning Point, Biohaven, Twitter, ChemoCentryx, Akouos, Albireo,
> Provention, BELLUS, National Western.

Momentum-type strategies naturally hold takeover targets, so the rule turns their wins into losses. Both
conventions are reported:
- **Pre-registered:** master's −30%.
- **Corrected:** the research panel's rule, −30% only for distressed delistings and 0% for acquisitions.
  None of the 13 was distressed.

The corrected one answers "is there an edge".

---

## 1. What was tested

| # | Item | Data | Design |
|---|---|---|---|
| P1 | Missed opportunities and filter ablation A-F on the bot's own forward records | Bot database export | **Blocked: no export arrived.** The ablation code is ready and tested (`botdb.ablation`). D (score threshold) was run on 2016-24 history instead. |
| P2 | Volatility and move-magnitude forecasts at 1, 3, 5, 10 and 20 sessions | 898k stock-weeks, 2016-24 | 6 baselines (HV21, HV63, EWMA, GARCH, HAR; IV in P4) against QuantLab OLS and gradient boosting. Fit on TRAIN 2016-19, chosen on VAL 2020-21, one OOS look 2022-24. |
| P3 | Vol-adjusted sizing of the bot's own strategies | Survivorship-free replay of 6 of the 8 enabled strategies through master's own engine | Rules: S0 fixed, S1 current (inverse-ATR), S2 inverse-vol, S3 capped inverse-vol, S4 forecast-vol |
| P4 | Implied vs forecast volatility across 4 maturity buckets | 2.37M option rows, 2019-24; liquid subset 609k | Accuracy, encompassing regressions, residual deciles |
| P5 | Execution engine | Same | 4 fill levels: mid, conservative, pessimistic (= the Alpaca paper fill), worst-reasonable. Classes: ROBUST, EXECUTION-SENSITIVE, NONVIABLE. |
| P6 | O1 long straddle, O2 long 25-delta strangle, O3 debit spread, O4 calendar | Same | O3 and O4 were gated by pre-registered preconditions |
| P7 | Model competition by OOS economic value | P3 and P6 | Every forecast was used for sizing and for O1 |
| P8 | Regimes | P3 trades | Trend, volatility, breadth, correlation, momentum and risk-on/off labels (all point-in-time), plus a pre-registered disable rule |

## 2. What failed

| Hypothesis | Result | Verdict |
|---|---|---|
| Bot's combined book (S1, current sizing) | OOS Sharpe −0.77 (master rule) / **+0.01 (corrected)**. TRAIN +0.25, VAL +0.06 (corrected). | No edge |
| Each replayed strategy alone (corrected rule, OOS Sharpe) | momentum_trend −0.09, mean_reversion −0.51, breakout −0.55, relative_strength +0.07, extreme_reversal −0.75, sector_rotation −0.35 | No edge |
| Vol-adjusted sizing (P3, master rule) | OOS Sharpe: S0 −0.62, S2 −1.29, S3 −0.60, S4 −0.61, against S1 −0.77. All "WORSE": more exposure, lower CAGR. | Fails |
| Vol-adjusted sizing with each forecast (P7, corrected rule) | HV21 −0.12, HV63 −0.01, EWMA −0.09, GARCH −0.24, HAR −0.22, QuantLab −0.07, against S1 +0.01 | None beats current sizing |
| Relaxed score thresholds (P1-D, history) | OOS Sharpe −0.74 against −0.77; VAL −0.61 against +0.06 | Fails |
| Regime disable rule (P8) | 1 of 90 cells qualified (Holm p = 1.0). OOS Sharpe −0.07 against +0.01 without it. DEV-vs-OOS sign agreement of regime cells: 54%. | Fails |
| O3 debit spread | Not run by rule: no directional model reaches AUC 0.55 (best 0.51-0.52) | Gated |
| O4 calendar | Not run: term-structure precondition failed (slope t = −1.13; |t| >= 3 needed) | Gated |
| Selling straddles (all, or the "IV too high" decile) | OOS −6.2% (all) and −7.3% (bottom decile) per trade at the bid | NONVIABLE |
| IV + QuantLab ensemble (P7) | VAL QLIKE 1% better than calibrated IV (kept), OOS ±0. Too few O1 signals (< 30 on VAL) for an economic test. | No economic value |

## 3. What remains uncertain

- **The bot's real filters.** P1 needs Harry's database export. The bot's live no-trade, EV and portfolio
  gates were not replayed: the backtester runs the strategy layer only. Live decisions may differ from
  this replay.
- **pead_ear and quality_momentum** were not replayed: they need a point-in-time earnings feed and
  fundamentals the store lacks. Their only evidence is the bot's own forward record.
- **Whether QuantLab's forecast plus an earnings flag would select straddles better.** In the O1 sample,
  the older Part-49 model with an earnings-in-window flag forecasts option-period vol better (VAL QLIKE
  0.20 against 0.29). The pre-registered P2 model excluded earnings because the 2020-24 calendar is
  backfilled (PIT_ASSUMED).
- **Earlier equity results on the corrected panel.** About 25 fake in-universe days in nine years;
  unmeasured.
- **Mid-price option fills.** There are no intraday quotes, and Alpaca paper cannot fill at mid. No option
  conclusion depends on the fill level:
  - the rules that fail are also about zero or negative at mid;
  - O1 and O2 are positive even at the ask.

  Better fills would not change any verdict.

## 4. Strongest surviving hypothesis

**O1: buy the ATM straddle (about 30 DTE, liquid underlyings) when QuantLab's expected absolute move to
expiry is at least 1.2 times the implied move.** The 1.2 threshold was chosen on VAL 2022 from
{1.1, 1.2, 1.3}.

| Split | Trades | Mid | Conservative | Pessimistic (ask) | Worst |
|---|---|---|---|---|---|
| TRAIN 2019-21 (descriptive) | 679 | +29.5% | +27.2% | +25.1% | +24.0% |
| VAL 2022 (k chosen here) | 36 | +17.6% | +15.7% | +13.9% | +13.0% |
| **OOS 2023-24** | **770** | **+11.8%** | **+9.9%** | **+8.2%** | **+7.3%** |

Net return per straddle premium, held to expiry, including fees and exercise costs.

Why it is not an edge:
- **CI includes 0.** The OOS 95% date-clustered CI at the ask is **[−3.6%, +21.6%]**. The median trade
  loses 10.8%, and the hit rate is 45%.
- **One year carries it.**

  | Year | Mean per trade |
  |---|---|
  | 2023 | **+25.6%** |
  | 2024 | **−11.7%** |

  Quarterly, 2023 Q3 made +38% and Q4 +55%, and every 2024 quarter with meaningful trade count lost.
- **Concentrated.** The top 10 trades produce 54% of the profit; without the top 1%, the mean falls to +5.0%.
- **Timing, not selection.** Buying every liquid straddle on the same dates returned +5.5%. O1's excess
  over that is +4.2% per trade, t = 0.99 across 102 dates (2023 +13.4%, 2024 −6.4%).
- **Not new.** By the pre-registered test it is a replication of Part 49, whose model forecasts better in
  this sample.

O2 (25-delta strangles when model EV > cost) is weaker:
- OOS mean +6.4% at the ask (n = 7,151), against −10.4% for all liquid strangles.
- CI [−11.7%, +23.9%].
- The median trade loses 98%, and without the top 1% the mean is −2.2%.
- 2023 +30.3%, 2024 −23.5%.

## 5. Exact evidence

| Claim | Evidence (file) |
|---|---|
| QuantLab forecasts volatility best | P2_vol_forecast.json (table below) |
| Option prices contain almost all of it | P4_P6_options.json `P4.encompassing` |
| The bot has no net edge | P3_sizing.json; P7_leaderboard.json `corrected_delisting` |
| Sizing does not help | P3 `classification_vs_S1`; P7 `sizing_runs`, `corrected_delisting.sizing_runs`, `winner_per_prereg` |
| O1/O2 are regime bets | diagnostics.json; P4_P6_options.json `O1_long_straddle`, `O2_long_strangle` |
| Data fixes | ca_fixes_report.json; diagnostics.json `S1_delisted_trades_master_convention` |

**P2, OOS 2022-24 (about 347k stock-weeks per horizon). QLIKE, lower is better; QuantLab = OLS.**

| Horizon | HV21 | HV63 | EWMA | GARCH | HAR | **QuantLab** | DM t, QuantLab vs HAR | IC, QuantLab vs GARCH |
|---|---|---|---|---|---|---|---|---|
| 1d | 3.06 | 3.02 | 2.88 | 2.85 | 1.87 | **1.78** | −7.2 | 0.386 / 0.361 |
| 5d | 0.99 | 0.88 | 0.90 | 0.84 | 0.79 | **0.70** | −8.0 | 0.661 / 0.631 |
| 20d | 0.82 | 0.67 | 0.69 | 0.56 | 0.49 | **0.46** | −1.2 | 0.783 / 0.757 |

- **Selection:** by the pre-registered rule, QuantLab beat every non-IV baseline (QLIKE lower, DM t <= −2)
  at 4 of 5 horizons and was selected.
- **Stability:** its QLIKE relative to HV21 is 0.69-0.73 in each OOS year and in calm, normal and stressed
  markets.
- **Large moves:** P(|5d move| > 10%) AUC 0.792, against 0.785 for GARCH and 0.760 for HV21.
- **Calibration fails out of sample.** Predicted 6.0% against 8.9% realised. The q95 band is exceeded 8.9%
  of the time (5% nominal), q99 2.4% (1% nominal). TRAIN 2017-19 was calmer than 2022-24.
- **Direction:** AUC 0.47-0.50. Higher forecast vol goes with slightly worse returns (IC −0.03 to −0.05).

**P4, liquid options, OOS 2023-24:**
- **Accuracy:** IV QLIKE 0.16 against QuantLab 0.32 at 26-45 DTE. IV is twice as accurate.
- **Encompassing:** QuantLab's coefficient beside IV is 0.16-0.18 (DK t 4.4-5.2) in every bucket.
- **Out-of-sample R² gain over IV alone:** +0.15% to +0.48%.

## 6. Performance after realistic costs

- **Equity costs:** master's tiers, 2-25 bps half-spread plus 5 bps slippage, next-open entries, stops.
- **Options:** quoted spreads at four fill levels, $0.03 per contract (assumed), exercise costs on the full
  stock notional.

**The bot's book (S1), OOS 2022-24:**

| Delisting rule | CAGR | Sharpe | Max drawdown | Net per trade |
|---|---|---|---|---|
| Master's −30% | −8.0% | −0.77 | −29% | −0.9% |
| Corrected | −0.3% | +0.01 | −11% | −0.2% |

Average exposure 35%, hit rate 44%.

**Options (OOS, net per premium):**
- Buying every liquid 30-DTE straddle: −2.6% at the ask (+0.5% at mid).
- O1: +8.2% at the ask.
- O2: +6.4% at the ask.
- Both are not robust (section 4).

## 7. Out-of-sample performance

- **Equity:** OOS = 2022-24. Earlier hypotheses used these years too, so this is research history, not a
  pristine test. Each sprint item had one look.
- **Options:** OOS = 2023-24.
- **Verdict:** nothing passes the pre-registered P9 gate. The gate needs OOS value > 0 after costs, ROBUST
  options, a win against the simple baseline, positive results in most OOS years, and more than risk
  reduction.

| Item | OOS result | Gate |
|---|---|---|
| Bot book, current sizing | Sharpe +0.01 (corrected) / −0.77 (master) | Fails (value ≈ 0) |
| Best sizing rule | No rule beats current sizing under the corrected convention | Fails |
| O1 | +8.2% per trade at the ask, CI includes 0, 1 of 2 years positive, selection t = 0.99 | Fails (not ROBUST, not consistent) |
| O2 | +6.4%, CI includes 0, 1 of 2 years positive | Fails |
| Regime filter | Lowers OOS Sharpe | Fails |

## 8. Holdout status

| Item | Status |
|---|---|
| 2025+ historical holdout | **Sealed and untouched**: `HOLDOUT_LOCK.json` shows 0 of 1 evaluations used, and no access attempt was logged |
| Holdout data | Not in the research store (it ends 2024-12-31) |
| Bot's own forward records | Out of scope for the lock; not received (P1) |

**Caution for any future holdout use.** 2025 contains the April 2025 volatility spike. Any long-volatility
rule will look good in 2025 regardless of skill. A holdout test of O1 must therefore be pre-registered on
its EXCESS over buying every straddle on the same dates, not on its raw return.

## 9. Execution sensitivity

Options, OOS, net per premium. Pessimistic = the ask (the only fill Alpaca paper allows).

| Rule | Mid | Conservative | Pessimistic | Worst | Class (pre-registered rule) |
|---|---|---|---|---|---|
| Buy all liquid straddles | +0.5% | −1.1% | −2.6% | −3.3% | EXECUTION-SENSITIVE |
| Sell all liquid straddles | −2.9% | −4.5% | −6.2% | −7.1% | NONVIABLE |
| O1 | +11.8% | +9.9% | +8.2% | +7.3% | EXECUTION-SENSITIVE\* |
| O2 | +14.7% | +10.4% | +6.4% | +4.6% | EXECUTION-SENSITIVE\* |

- **Spreads:** liquid straddles have a median spread of 6.4% of mid; 25-delta strangles 13.8%.
- **\*Class note:** O1 and O2 are positive even at the ask. The binding problem is statistical (CI includes
  0) and regime concentration, not execution. Under the literal class definitions they land in
  EXECUTION-SENSITIVE because they are not ROBUST.
- **Better fills would not change the verdict.** Paying for intraday quotes to chase mid fills is not
  justified.
- **Equity:** the bot's book costs about 0.2% per round trip. Cost is not what separates it from an edge.

## 10. Required data

Ranked by information per dollar:

1. **The bot's database export.** Free, needs Harry, takes 5 minutes; steps in
   [BOT-DB-IMPORT.md](BOT-DB-IMPORT.md). It unblocks P1, the only forward evidence on the bot's own filters.
   The analysis runs the moment the file lands:
   `scripts/research/alpha/run_botdb.py var/quantlab_export.db`.
2. **A point-in-time earnings calendar for 2019-24.** The DoltHub one is backfilled. Earnings are the
   biggest driver of option-period vol that QuantLab's pre-registered model leaves out.
3. **Complete corporate-action history** (spin-offs, distributions, merger terms). Alpaca's list misses
   most spin-offs. The sprint-c price rule is a patch, not a source of truth.
4. **Intraday option quotes, OI and volume (ThetaData): not justified.** The option results do not hinge
   on execution, so no purchase.

## 11. Recommended next experiment

Ranked by expected information per unit of cost:

1. **Run P1 on the bot export** as soon as it exists. No cost.
2. **Re-run the October equity families on the sprint-c panel.** This confirms the baseline report
   survives the data fix: about 1-2 hours of compute, no new hypotheses.
3. **Fix the delisting convention in master's backtester.** Use the research panel's rule: acquisitions
   0%, distressed −30%. The −30%-on-everything rule makes any survivorship-free test of the bot
   meaningless. This is a backtest convention, not a safety control. It needs Harry's approval because it
   touches master.
4. **Only then decide O1's fate.**
   - Run it as a forward paper SHADOW: record signals, the four fill levels and the same-date-baseline
     excess every week, with no orders. The pre-registered decision metric is excess over the baseline.
   - Do not spend the one-time 2025+ holdout on a hypothesis whose OOS selection t is 0.99.

## 12. Does any strategy deserve paper deployment?

**No.**
- Nothing passed the pre-registered integration gate (section 7), so no paper integration was built. The
  bot's runner and pipeline are unchanged.
- **Fair description of the bot today:**
  - It is a working paper-trading machine with no demonstrated edge.
  - Its strategies earn about zero on survivorship-free data even under the generous corrected delisting
    rule, at about 35% average exposure.
  - Vol-forecast sizing does not change that.
- **O1:** a candidate for record-only shadow tracking, not deployment.

---

### Deviations from the pre-registration (all logged)

- **GARCH grid widened after the first TRAIN-only fit hit the b >= 0.70 bound.** The run was stopped
  before any VAL/OOS metric. The final fit is interior: a 0.14, b 0.49.
- **Sprint-c data patch applied after the first replay exposed the fake moves.** P2-P8 were all re-run;
  the superseded ledger entries remain, marked `SPRINT-C`.
- **The replay's split/dividend recovery was corrected twice before any result was read:** a 2-bp
  rounding tolerance, then signed residual cash flows. Crediting only positive residuals had biased
  returns upward.
- **Delisting convention:** master's −30% stays the pre-registered primary. The corrected convention is
  reported alongside, because all 13 delisted trades were verified acquisitions.
- **P7 winner:** the first implementation picked the "best improvement over current sizing" even when
  every book lost money. Re-scored with the pre-registered text (positive absolute value under both
  delisting conventions), without re-running: **no winner**.
- **O1 labelled a replication of Part 49:** the pre-registered QLIKE test went against the new model.
