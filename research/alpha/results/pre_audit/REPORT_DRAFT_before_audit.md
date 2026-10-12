<!-- First draft of the report, written BEFORE the independent audit. Superseded by docs/ALPHA-DISCOVERY-REPORT-2026-10.md. Kept for the record: several claims below (the H26 placebo, the full-universe option decile spreads, the H33 t-statistics) were found to be wrong. -->
# Alpha discovery report (2026-10-06/07)

Branch `alpha-discovery`. Plan: [ALPHA-DISCOVERY-PLAN.md](ALPHA-DISCOVERY-PLAN.md). Code: `src/quantlab/alpha/`.
Records:
- every run: `research/alpha/ledger.jsonl` (append-only, 298 runs, 260 distinct configurations);
- per-family results: `research/alpha/results/*.json`;
- current status of each hypothesis: `research/alpha/queue.json`;
- literature register (44 entries): `research/alpha/literature.json`;
- dashboard page: `/alpha`.

PAPER RESEARCH ONLY. Nothing here is advice to trade real money.

---------------------------------------------------------------------------------------------------

## Executive summary

**The question was whether the market gives QuantLab anything worth exploiting. On the data available
in 2016-2024, the answer is no, and that answer is now much better supported than before.**

1. **No class-A or class-B result.** About 310 configurations were tested across 30 hypotheses:
   reversal, momentum, residual momentum, lottery, idiosyncratic volatility, intraday gaps,
   intraday momentum, seasonality, earnings premium, earnings surprise, revisions, pairs, nine
   option-volatility sorts, pre-earnings straddles, index variance premium, ML and portfolio
   combinations. Every family is class E (no edge) or class D (the effect is real but trading it
   destroys it). Every multiple-testing statistic agrees: Hansen SPA p ≥ 0.32, Deflated Sharpe ≈ 0.
2. **QuantLab can predict how much a stock will move, not which way.** Out of sample (2022-2024,
   905k stock-weeks):
   - Next-month volatility is highly predictable: rank IC 0.80, against 0.69 for trailing volatility.
   - The size of next week's move: rank IC 0.36. The odds of a 10% move: AUC 0.70.
   - Direction (AUC 0.515) and timing (AUC 0.50) are close to a coin flip.
3. **The answer to Part 49.** QuantLab's volatility forecast carries information beyond option
   implied volatility. This holds even among liquid options out of sample: encompassing-regression
   t = 12.7. But it does **not** forecast better than liquid options' IV on its own (2023-24 log-variance
   MSE 0.123 vs 0.091). **No realistic option book makes money after the bid-ask spread.** The median
   ATM straddle spread is 19% of its value; the variance risk premium is about 2-6% at mid.
4. **The impressive option results were an artefact.** Sorting straddles by HV−IV,
   forecast−IV, option momentum or IV rank gives decile spreads of +13% to +23% a month with
   t = 5-10. Most of that is illiquidity: in options with spreads ≤ 10% the pattern disappears out
   of sample.
5. **Survivorship bias was real and large.** It is now measured and removed. A current-listings-only
   universe overstated long-only CAGR by:
   - +1.2 pp/yr for the whole liquid market;
   - +1.3 pp for momentum;
   - +2.4 pp for recent losers;
   - **+5.3 pp for the most volatile stocks.**

   Every earlier long-only QuantLab result was flattered by roughly this much.
6. **Several published effects did not reproduce net of costs in 2016-2024:**
   - pre-earnings straddles (gain at mid equals the non-earnings placebo; −30% after spreads);
   - weekly reversal (gross ≈ 0 in 2016-19);
   - post-earnings drift (carried by the top 5% of trades);
   - intraday market momentum;
   - pre-FOMC drift (+14 bp, t 0.7);
   - pairs trading.

   The literature register already documented the decay of most of these.

**Recommendation.**
- Stop searching daily US-equity price, volume, calendar and earnings patterns for a stand-alone edge.
- Keep QuantLab as a risk and volatility instrument: size positions by predicted magnitude.
- Treat the remaining option ideas as **execution problems** (mid-price fills, longer-dated or
  lower-turnover structures), not signal problems.
- The next evidence has to come from **forward paper data** and the **2025+ holdout**, which is still
  untouched for every hypothesis here.

---------------------------------------------------------------------------------------------------

## What QuantLab had already disproven (before this phase)

About 30 daily price/volume rules on liquid US stocks were flat or negative after costs (8 strategies,
16 discovery cohorts, 4 catalyst hypotheses, 15 new-data tests; docs/REAL-MONEY-READINESS.md).
Price-reaction PEAD, insider and House disclosures, and the discovery score had no predictive power.
The best ranking (12-1 momentum, low vol, liquidity) was not significant and still below SPY.

## What this phase learned

| Learning | Evidence |
|---|---|
| Survivorship bias is 1-5 pp/yr for long-only books, largest for volatile names | H02 |
| Movement size and volatility are predictable; direction is not | H12, H33, and earlier UPSIDE-EVIDENCE |
| QuantLab's volatility forecast adds information beyond IV but does not beat liquid IV alone | H33 |
| Single-stock option spreads (median 19% of an ATM straddle) exceed the variance premium | H24-H33 baselines |
| Option "anomaly" decile spreads in the full universe are mostly illiquidity | H24/H27/H30/H33 liquid re-test |
| Index ATM straddle selling earned ≈0 in 2019-24 even on SPY (0.4-1.2% spreads) | H35 |
| Weak equity streams are not independent (3.1 effective of 5) and do not combine into an edge | H36 |
| Complex ML overfits: tree models reach train AUC 0.70-0.82 and give no OOS gain over linear | H12 |

## New data acquired (all free)

| Dataset | Coverage | Notes |
|---|---|---|
| Survivorship-free daily bars | 2016-2024, 14,160 entities incl. 1,905 delisted common stocks | Alpaca SIP, raw and adjusted. Delisted tickers recovered from Alpaca's inactive list and a historical Nasdaq Trader directory (DoltHub `stocks.symbol`) re-queried with `asof`. Coverage is weak for delistings before 2017-10. |
| End-of-day option chains | 2019-02 .. 2024-12, ~77M contracts, ~2,000 underlyings | DoltHub `post-no-preference/options`: bid/ask/IV/Greeks, 3 expiries, ~20 strikes; no volume or open interest. Put-call parity confirms snapshots are **same-session end-of-day** in every period sampled. Data before 2021-05 was bulk-imported from the publisher's archive; later data was committed in near real time. |
| Earnings calendar and estimates | calendar 2020-2024; estimates 2017-2024 | Calendar history was **backfilled** (rows first appear after the events), so 2020-24 dates are PIT_ASSUMED realised dates. True point-in-time only from late 2024 (2025+ is holdout). |
| Fama-French factors, FOMC dates | 2016-2024 | Ken French library; Federal Reserve calendar pages. |
| SPY/QQQ 1-minute bars | 2016-2024 | For intraday momentum. |

## Method (fixed before any outcome)

- **Splits** (`alpha/splits.py`, pinned by tests):

  | Data | TRAIN | VALIDATION | OOS |
  |---|---|---|---|
  | equity | 2016-19 | 2020-21 | 2022-24 |
  | options | 2019-02..2021 | 2022 | 2023-24 |
  | earnings calendar | 2020-21 | 2022 | 2023-24 |

  The 2025+ holdout was **not downloaded** into the research store.
- **Pre-registration:** each family's variant grid was committed to git before its first run.
- **Selection** on TRAIN only. **One OOS evaluation** per family, enforced by `registry.append_run`. The
  guard refused a second H20 OOS run; that override is logged with its reason.
- **Costs:** QuantLab's CostModel tiers plus slippage. Shorts pay 50 bps/yr borrow. Intraday books pay
  a full round trip daily. Options pay the ask in and the bid out (or intrinsic at expiry).
- **Statistics:** Newey-West t, Deflated Sharpe with family trial counts, Hansen SPA and White's Reality
  Check across variants, PBO via CSCV, effective-trial counts, Benjamini-Hochberg, top-1/5/10% trade
  removal, 2x costs, delisting-return sensitivity, and regime/liquidity/volatility splits.
- **Point-in-time tests:** truncation-invariance tests for every equity signal. Method validation: a
  planted reversal is found (t > 3); the null-world average |t| < 1.25; planted SPA/RC/PBO detection
  passes.

## Strategies tested and results

### Equity cross-section: development (TRAIN 2016-19, VALIDATION 2020-21) and the single OOS look (2022-24)

| Family | Variants (effective) | Selected on TRAIN | TRAIN net t | VAL net t | SPA p | PBO | OOS Sharpe (t) | OOS beta | Class |
|---|---|---|---|---|---|---|---|---|---|
| H01 short-term reversal | 160 (2.0) | 10d raw, EW, L/S, 5d hold | −0.18 | −0.23 | 1.00 | 0.15 | −0.55 (−1.11) | 0.24 | E |
| H03 momentum matrix | 40 (4.1) | 12-1 when breadth strong | 1.89 | −0.09 | 0.32 | 0.16 | 0.58 (0.99) | −0.07 | E |
| H04 residual momentum | 16 (1.2) | 6-1 two-factor | 0.93 | −0.08 | 0.49 | 0.55 | 0.31 (0.62) | −0.37 | E |
| H16 MAX / lottery | 4 (1.0) | MAX5 EW | 0.05 | −0.62 | 1.00 | 0.21 | 0.13 (0.26) | −1.19 | E |
| H17 idiosyncratic vol | 8 (1.0) | IVOL 63d two-factor | 0.20 | −0.72 | 1.00 | 0.28 | 0.27 (0.54) | −1.11 | E |
| H23 revision momentum | 2 (1.2) | 63d consensus change | −0.89 | −0.05 | 1.00 | 0.60 | 0.27 (0.45) | −0.13 | E |
| H07 intraday gap reversal | 4 (1.2) | large caps EW | gross t 2.93 / net −44 bp/day | — | 1.00 | 0.00 | −5.5 (−10.0) | 0.10 | **D** |

- **Beta-adjusted OOS alpha t-stats:** H17 1.52, H16 1.29, H04 1.11, H03 1.09. None reaches 2. The
  low-vol/lottery alpha appears only in 2022-24 and was absent or negative in 2016-21.
- **Short-term reversal:** gross returns were ≈0 in 2016-19 (−0.5 to +0.1 bp/day by lookback) and
  positive only in the volatile 2020-21 window. Daily turnover of 0.2-1.5 costs 5-30 bp/day. Not one
  of 160 variants was positive net in TRAIN.

### Event, calendar and intraday (one OOS look each)

| Family | Result | Class |
|---|---|---|
| H09 seasonality (13 SPY effects) | No effect survives Benjamini-Hochberg (min q 0.82). Pre-FOMC +14 bp and NFP-Friday +13 bp positive in every split but t < 0.7; a round trip costs ~14 bp. | E |
| H08 SPY/QQQ intraday momentum | \|t\| < 1.3 in every split; slope ≈ 0 except in 2020-21 | E |
| H20 earnings announcement premium (calendar splits) | Best: 5 days before, exit before the release. TRAIN +1.6 bp (t −0.36), VAL −10.7 bp, OOS −11.9 bp (t −0.5) | E |
| H21 EPS-surprise drift | TRAIN +100-200 bp/event (t 1.4-2.2) but **negative without the top 5% of trades**. OOS t −0.45. *Disclosure:* a split bug showed the OOS of one variant once before the fix. | E |
| H26 pre-earnings straddle (Gao-Xing-Zhang) | At mid +1.7-2.4%, but the non-earnings placebo earns the same. Ask→bid −30% per trade, hit rate 4-6%. | E |
| H15 pairs (GGR distance) | ±1%/yr, \|t\| < 1.5 in every split, both variants | E |

### Options lab (2019-24 end-of-day chains; one 1-month ATM straddle per stock per week)

| Measure | TRAIN 2019-21 | VAL 2022 | OOS 2023-24 |
|---|---|---|---|
| IV > subsequent realised vol | 69% | 63% | 69% |
| Long straddle at **mid**, held to expiry | +5.5% | −4.2% | −2.5% |
| Long straddle at the **ask**, per $ mid | −11.4% | −25.8% | −23.0% |
| Short straddle at the **bid**, per $ mid | −22.4% | −17.3% | −18.1% |
| Same, liquid only (spread ≤ 10%): long at ask / short at bid | +10.1% / −17.0% | −4.3% / −2.6% | −1.5% / −4.8% |

**Decile sorts.** Long the top decile at the ask, short the bottom at the bid, per $ mid premium.

| Sort | Full universe, OOS L/S (t) | Liquid only, OOS L/S (t) | Class |
|---|---|---|---|
| H24 HV − IV (Goyal-Saretto) | −20.0% (−11.8) | −4.0% (−2.2) | D |
| H33 forecast − IV (Part 49) | −21.2% (−14.9) | −2.7% (−1.7) | D |
| H27 option momentum 12m | −16.0% (−10.1) | −4.1% (−3.0) | D |
| H30 IV percentile (low) | −18.2% (−11.0) | −1.6% (−1.3) | E |
| H28 term slope / H29 skew | no OOS edge after costs; skew does not predict 21-day stock returns (t 0.8) | | E |

The ask-to-ask decile spreads (+13% to +23%/month, t 5-10) look tradeable but are not. They credit the
short side with the ask, and the monotonic pattern comes from illiquid names: among liquid options the
OOS deciles are flat (−3.1% to +1.7%, no ordering).

**H33, the forecast against IV.** Log realised-variance MSE (lower is better):

| Subset | Period | QuantLab forecast | IV raw | IV debiased | Encompassing (t) |
|---|---|---|---|---|---|
| All options | VAL | 0.096 | 0.119 | 0.109 | forecast 0.74 (t 18.9), IV 0.36 (t 13.5) |
| All options | OOS | 0.123 | 0.197 | 0.162 | forecast 0.77 (t 36.1), IV 0.27 (t 14.9) |
| Liquid (spread ≤ 10%) | VAL | **0.089** | 0.131 | 0.125 | forecast 0.85 (t 18.8), IV 0.29 (t 9.1) |
| Liquid | OOS | 0.123 | 0.099 | **0.091** | forecast 0.41 (t 12.7), IV 0.66 (t 22.7) |

**H25, expected vs realised move.** Realised/implied move averages 0.82-1.05 (median 0.64-0.85).
Earnings windows are priced about right (0.93-1.00). High-IV non-earnings names are the most overpriced
(0.82), but selling them at the bid loses after spreads.

**H35, the index variance premium.**
- SPY (spreads 0.4-1.2%): IV > RV 67-77% of weeks. Short ATM straddle at the bid: −0.5% / −11.3% /
  +0.9% per month (t ≈ 0). The iron fly: +4.3% / −12.4% / −4.6%.
- DIA is similar.
- XLK/XLF quotes in this data are wide (14-51%).

Class E for the ATM straddle expression. OTM put-writing (the Cboe PUT index structure) was **not**
tested.

### Machine learning with separated targets (H12; fit 2016-19, one OOS look)

| Target | Linear OOS | Random forest OOS (train) | Gradient boosting OOS (train) | Trailing-vol baseline OOS |
|---|---|---|---|---|
| Direction, 5d (AUC) | 0.518 | 0.514 (0.704) | 0.515 (0.681) | 0.495 |
| Timing, +5% before −5% (AUC) | 0.518 | 0.499 (0.796) | 0.500 (0.821) | 0.497 |
| Magnitude, \|5d\| (weekly rank IC) | 0.245 | 0.358 | 0.357 | 0.318 |
| Volatility, 21d (weekly rank IC) | 0.791 | 0.781 | 0.798 | 0.693 |
| Tail, \|21d\| > 10% (AUC) | 0.704 | 0.707 | 0.698 | 0.679 |

### Portfolio construction (H36)

The five selected equity streams amount to 3.1 effective independent bets (MAX and IVOL correlate
0.96; momentum and residual momentum 0.72). With weights fitted on TRAIN, every method is negative in
VALIDATION. OOS Sharpe is 0.13-0.62 (best t 1.05, capped Kelly); yearly walk-forward OOS Sharpe is
0.20-0.51. Diversification reduces drawdowns (15-20%) but produces no significant return. Class E.

## Statistical significance and multiple testing

- **Families:** SPA p-values 0.32-1.00 and White RC p-values 0.50-0.98. Deflated Sharpe probabilities
  ≤ 0.48. Benjamini-Hochberg rejections: 0 in every family.
- **Effective trials** are far fewer than nominal: 160 reversal variants behave like 2.0 independent
  bets. Correlated variants inflate apparent search less than feared, but also add no information.
- **Event families:** none survives top-5% trade removal.

## Out-of-sample and walk-forward

- **11 OOS evaluations**, one per family. None has OOS t ≥ 2 net of costs.
- **Positive OOS Sharpe but not significant:** H03 (0.58), H04 (0.31), H17 (0.27), H23 (0.27). All
  were negative or flat in the 2020-21 validation window.
- **Walk-forward portfolio:** OOS Sharpe ≤ 0.51 (t ≤ 0.9).

## Best and failed signals

**Best, in the sense of information rather than tradeable return:**
1. Volatility and magnitude forecasts (IC 0.8 / 0.36).
2. The forecast's information beyond IV (encompassing t 12.7 in liquid options).
3. 12-1 momentum, positive in two of three periods.

**Failed:** everything else in the tables above, kept in the ledger for good.

## Regime dependence

Every selected variant has results split by SPY trend, SPY volatility tercile, credit (HYG/LQD), rates
(TLT) and dollar (UUP) (`results/*.json`, `by_*` keys). The clearest pattern: reversal and long-volatility
books did well only in 2020's volatility and badly otherwise. No family qualified as class C, because no
regime split was pre-registered as its own hypothesis, and doing so now, after seeing OOS, would be
data snooping. Candidates for a future pre-registered test: reversal in high-volatility regimes, and
momentum when breadth is strong.

## Data limitations

- Delisted-stock coverage is good from 2018 and weak for 2016-17. True survivorship bias is somewhat
  larger than H02 shows.
- Security types are name-based, and recycled tickers keep only their latest name.
- Option data:
  - 3 expiries and about 20 strikes per underlying; no volume or open interest.
  - Before 2021-05 it is archive-imported.
  - Mon/Wed/Fri snapshots before 2025.
  - Wings for capped-risk structures are priced by Black-Scholes at the 25-delta IV (MODEL).
- The 2020-24 earnings calendar is realised dates (PIT_ASSUMED).
- Market cap is unknown: dollar volume stands in for size.
- No SEC fundamentals in this cloud (no user-agent configured). No bot database here: the missed-opportunity
  engine (H13) and execution alpha (H39) are DATA-LIMITED. Exporting the PC database was blocked by the
  permission system and needs Harry.

## Overfitting risks

- The PBO for residual momentum (0.55) and revision momentum (0.60) flags overfitting risk in their
  selection.
- Tree models overfit badly: train AUC 0.70-0.82 against ~0.50 OOS.
- Option decile sorts are the largest potential false discovery here (t 5-10 that vanishes in liquid
  names). This is exactly what Part 52 warns about.
- Earlier QuantLab work used 2021-24 for price/volume ideas, so equity OOS is not virgin for those
  families.

## Transaction-cost impact

Costs are the binding constraint almost everywhere:

| Book | Gross | Costs |
|---|---|---|
| Reversal | 0-4 bp/day | 5-30 bp/day |
| Gap reversal | 8-10 bp/day | ~50+ bp/day |
| Option straddles | variance premium 2-6% | median spread 19% (~9% each side) |
| Momentum and low-vol books (low turnover) | | 2x costs moves t by < 0.5 |

## Execution impact

- Next-open execution against next-close (MOC): recorded per family in the robustness section. It
  does not change any class.
- Paper fills on Alpaca are optimistic for options (fills against indicative quotes). Real option
  execution would be worse than the bid/ask modelled here, not better.

## Portfolio impact

There is nothing to add to the bot's book. The equity streams are weak and mostly negatively exposed
to the market. Their combination is not significant.

## Recommended next steps (ranked by expected information value)

1. **Use the forward/holdout gate as designed.** Freeze the two least-bad, low-turnover candidates
   (12-1 momentum when breadth is strong; residual momentum 6-1) as SHADOW forward books and compare
   them with SPY for 3-6 months. Do not trade them.
2. **Volatility forecasting as a risk tool.** Use the magnitude/volatility model (IC 0.8) for position
   sizing and stop placement in the existing bot. This reduces risk; it does not add return.
3. **Options, only as an execution problem.**
   - Test longer-dated or OTM structures (put-writing, calendars) where spread per unit of premium is
     smaller.
   - Re-test with NBBO quotes **only if mid-price execution is realistic**.
   - ThetaData Standard is worth one month **only** for that (open interest, intraday quotes for
     timing).
4. **Run the bot-database studies** (H13 missed opportunities, H39 execution) once Harry exports the
   tables.
5. **Do not** keep mining daily US-equity price, volume or calendar patterns. About 340 configurations
   (30 earlier, 310 now) have failed, the literature documents their decay, and the next "winner" from
   this space is most likely a false positive.

## Per-batch summaries (Part 53 format)

| Batch | Hypothesis | Data | Method | Result | OOS | Robustness | Confidence | Next |
|---|---|---|---|---|---|---|---|---|
| 1 equity cross-section | reversal, momentum, residual momentum, MAX, IVOL, gap | survivorship-free daily 2016-24 | 232 pre-registered variants, TRAIN selection | no net edge | one look each: no t ≥ 2 | SPA/RC/DSR/PBO fail; costs dominate | high | retire; shadow-track two momentum books |
| 2 options | VRP, HV−IV, forecast−IV, momentum, term, skew, IV rank, earnings straddle, index VRP | DoltHub chains 2019-24 | deciles, realistic ask/bid economics, liquidity split | information yes, money no | negative after spreads | full-universe signal = illiquidity | high | execution-led re-test only |
| 3 events/calendar | EAP, SUE, revisions, seasonality, intraday momentum | calendar 2020-24, minute bars | event studies, BH | no edge | one look each | outlier-driven where positive | medium-high | retire |
| 4 ML/portfolio | target separation, ensembles, pairs | 905k stock-weeks | linear/RF/GBM; 7 construction methods | magnitude/vol predictable; direction not; no combined edge | one look | trees overfit | high | use vol forecasts for risk only |

---------------------------------------------------------------------------------------------------
*An independent adversarial code audit was run before this report was finalised. Its findings and the
fixes are in the section below.*
