# Alpha discovery report (2026-10)

Branch `alpha-discovery`. Plan and every pre-registration: [ALPHA-DISCOVERY-PLAN.md](ALPHA-DISCOVERY-PLAN.md).
Code: `src/quantlab/alpha/`. Records:
- every run: `research/alpha/ledger.jsonl`, append-only;
- results: `research/alpha/results/`;
- first-run results, kept and superseded: `results/pre_audit/`;
- hypothesis status: `research/alpha/queue.json`;
- literature register (44 entries): `research/alpha/literature.json`;
- dashboard: `/alpha`.

PAPER RESEARCH ONLY. Nothing here is advice to trade real money.

---------------------------------------------------------------------------------------------------

## Executive summary

**Does the market give QuantLab anything worth exploiting? On 2016-2024 US data, after realistic
costs, with a survivorship-free universe and multiple-testing control: no. That answer has now
survived two independent audits and a full corrected re-run.**

1. **No family reaches class A, B or C.** Every family is class E (no edge) or class D (the effect
   exists, but trading it destroys it). In total, 266 distinct configurations across 32 hypotheses
   were tested (818 ledger runs):
   - reversal, momentum, residual momentum, lottery, idiosyncratic volatility;
   - gaps, intraday momentum, seasonality;
   - the earnings premium, post-earnings drift, analyst revisions;
   - pairs;
   - ten option-volatility sorts, pre-earnings straddles, index variance selling;
   - machine learning and portfolio combinations.

   Every multiple-testing statistic agrees: Hansen SPA p ≥ 0.40 and Deflated Sharpe ≤ 0.50.
2. **The question in Part 49, "Can QuantLab predict realised movement better than the options
   market?", has a clear answer: no, not where it could be traded.**
   - In liquid options (spread ≤ 10% of mid), implied volatility forecasts next-month realised
     volatility better than QuantLab's model. The 2023-24 log-variance error is 0.085 for IV against
     0.120 for the model; in 2022 it is 0.069 against 0.089.
   - QuantLab's forecast adds only a sliver of information beyond IV: encompassing coefficient 0.13,
     Driscoll-Kraay t 3.9 in 2023-24 and t 1.4 in 2022.
   - The model beats IV only for illiquid options, whose quotes are noisy and cost about 20% of
     premium to cross.
   - Both trades Part 49 proposed lose money in liquid options in 2023-24:
     - **high predicted move + cheap IV**, long straddle: −2.0% per trade (t −0.5);
     - **high predicted move + expensive IV**, capped iron fly: −4.8% (t −2.4).
3. **What QuantLab *can* predict is how much a stock will move, not which way.** Out of sample
   (2022-24, 899k stock-weeks):
   - next-month volatility: rank IC 0.80, against 0.69 for trailing volatility alone;
   - the size of next week's move: rank IC 0.36;
   - the odds of a 10% move within a month: AUC 0.70;
   - direction: AUC 0.51-0.52; timing (up 5% before down 5%): 0.50-0.52. Both are a coin flip.

   This is useful for risk control. As finding 2 shows, it is not an options edge, because the options
   market already prices most of it.
4. **The independent audits changed the story in options, not in equities.** The first run had real
   bugs:
   - option chains joined to the wrong company for reused tickers;
   - a "placebo" that was itself the previous earnings date;
   - companies that later delisted were dropped from every earnings and options join;
   - duplicated companies;
   - zero-volume filler bars.

   Two claims in the first draft were wrong: "QuantLab's forecast adds information beyond IV even in
   liquid options (t 12.7)" and "the pre-earnings straddle placebo earns the same". After the fixes,
   every equity family kept the same selected variant and class.
5. **Survivorship bias is real and measurable.** About 18% of the liquid universe on any day consists
   of companies that later delisted. A universe of today's survivors overstates long-only CAGR by:
   - +1.2 pp/yr for the whole liquid market;
   - +1.5 pp for 12-1 momentum;
   - +2.4 pp for recent losers;
   - **+5.5 pp for the most volatile stocks.**

   Every earlier QuantLab long-only result was flattered by roughly this much, most of all for
   speculative names.
6. **Costs are the binding constraint almost everywhere.**
   - Single-stock option spreads (median 9-11% of premium per side) are larger than the variance
     risk premium (realised volatility is below implied in 63-70% of stock-months).
   - Reversal books turn over 20-150% of capital a day, and the gap book ~200%. Gross edges of
     0-10 bp/day cannot pay that.

**Recommendation.**
- Stop searching daily US-equity price, volume, calendar and earnings patterns for a stand-alone edge:
  ~300 configurations (30 earlier, 266 now) have failed, and the literature documents their decay.
- Use QuantLab's movement and volatility forecasts for **risk** (position sizing, stop distance,
  avoiding names about to move a lot), not as a trading signal and not to buy or sell options.
- The next real evidence will come from the **2025+ holdout** (still untouched) and the bot's
  **forward shadow outcomes**, not from more historical searching.

---------------------------------------------------------------------------------------------------

## Independent audit: what was wrong, what was fixed, what changed

Two independent review rounds ran after the first full set of results. Each reviewer read the code and
spot-checked the data with instructions to break the results, not to confirm them.

1. **Adversarial audit of the first run.** It found 2 critical and 6 major defects plus 11 minor ones,
   and verified 12 parts of the stack as correct:
   - engine timing;
   - delisting returns;
   - raw vs adjusted prices;
   - sector and beta windows;
   - gap alignment;
   - option outcomes;
   - H21 percentiles;
   - SPA, White RC and PBO;
   - H26 timing;
   - the event studies.
2. **Verification review of the fixes.** The first fixes were not complete:
   - reused tickers still in use were still matched to the wrong company;
   - the per-row options filter mostly removed legitimate high-volatility names;
   - the OOS guard had gaps;
   - the de-duplication ranked padded keys first.

   It also confirmed the H26 schedule rewrite (0 mismatches on 20,000 events), the entity merges and
   the absence of new look-ahead.

Every fix was committed, and the re-run protocol pre-registered (ALPHA-DISCOVERY-PLAN.md §10), before
any corrected number was computed: commits 5a31c53, 7b099a1 and b18f0af. Every first-run result is kept
in `research/alpha/results/pre_audit/` and every intermediate one in `results/audit_v2_panel/`. Nothing
was deleted from the ledger.

| ID | Severity | Defect | Effect on the first report | Fix |
|---|---|---|---|---|
| C1 | critical | H26 placebo = same company 63 sessions earlier = one fiscal quarter: 47% of placebos sat within ±3 sessions of the previous real earnings | "the placebo earns the same" compared earnings with earnings: invalid | placebo 31 sessions earlier (mid-quarter); kept only with no real event within 15 sessions or before expiry |
| C2 | critical | option chains joined to the company holding the ticker **today** (e.g. old-Caesars CZR options vs Eldorado; 2023 Paramount PARA options vs Banzai's SPAC). 2,049 obs had strike/spot off by > 20%; 164 straddles showed > +1,000% | full-universe option decile spreads inflated (H27 TRAIN +31% → +5% on removing the bad rows) | ticker → company resolved by date through Alpaca's symbol-change history (3,679 changes); data for a company missing from the store is dropped, never re-assigned; (ticker, month) chains whose put-call-parity spot misses the stock's close by > 5% are dropped |
| M1 | major | calendar, estimate and option joins kept only today's tickers: companies that later delisted were silently dropped (14% of calendar events, 8.4% of option rows) | earnings and options results were survivors-only; H23's signal was missing exactly for future delisters | the same date-aware company mapping |
| M2 | major | the same security under two keys on some days (renames, mergers, SPAC splices) | about 13 liquid names a day counted twice; 97 renames booked as delistings | twins de-duplicated on unbroken runs of identical trading days; a key that continued under its twin is "renamed", not delisted |
| M3 | major | H33 volatility model trained on 2019 rows with no earnings calendar | biased earnings coefficient | fit only on calendar-covered rows, with an embargo (no TRAIN label runs into VALIDATION) |
| M4 | major | H33 t-statistics clustered by week; one-month outcomes overlap 4-6 weeks | t-stats overstated ~1.7x | Driscoll-Kraay, 6 weekly lags (12 as a check) |
| M5 | major | H07 gap reversal fills at the same opening print that defines the gap (bid-ask bounce in signal and fill) | class D not established | pre-registered 09:35-entry test on a 40-stock minute-bar sample |
| M6 | major | OOS-once guard switched off for H20/H21 by hard-coded reasons | a re-selection would have spent OOS silently | one override per tag per hypothesis; every look at OOS-period data counted (`registry.oos_looks`) |
| V1 | major (review 2) | zero-volume "filler" bars from the entity mapping: 4.9% of all rows, with 268 places where a filler stretch ends in a jump joining two different securities (×130, ×2,500) | a few names per year carried absurd trailing returns into momentum/volatility signals (the first quality report counted 862k zero-volume bars but nothing excluded them) | panel v3 drops zero-volume bars |
| V2 | major (review 2) | first C2 filter dropped rows with no near-the-money strike: mostly cheap, high-IV names (26% of the top IV decile in March 2020) | would have mixed the C2 correction with a cut of the high-IV tail | per-(ticker, month) wrong-company test; strict-ATM kept as a labelled sensitivity |

**What the corrections changed:**

| Area | First run | Corrected |
|---|---|---|
| Equity families (6), selected variant and class | as reported | **identical** on panel v2 and v3; OOS t within ±0.05 |
| H27 option momentum 12m, TRAIN top-bottom (ask-to-ask) | +31% (t 4.6) | +4% (t 1.9) |
| H33 forecast information beyond IV, liquid options OOS | coefficient 0.41, t 12.7 (week-clustered) | 0.13, Driscoll-Kraay t 3.9; IV is the better forecast |
| H26 event vs placebo at mid | "the placebo earns the same" (placebo = previous earnings) | event +1.5-2 pp over a true placebo in 2020-21, not after |
| H21 post-earnings drift, TRAIN t | 1.4-2.2 | 0.3-0.8 (future delisters included) |
| Delistings in the panel | 1,793 (97 were renames) | 1,871 (filler bars had hidden 175 real delistings) |
| Clean calendar events matched to a company | 60,554 (companies that later delisted dropped) | 75,508 |

**How often each hypothesis has looked at OOS-period data** (`registry.oos_looks`; every look is in the
ledger):

| Hypothesis | OOS (selected-variant evaluations) | ALL (descriptive, incl. OOS years) | Walk-forward | Total |
|---|---|---|---|---|
| H01 | 3 | 0 | 1 | 4 |
| H02 | 0 | 2 | 0 | 2 |
| H03 | 3 | 0 | 1 | 4 |
| H04 | 3 | 0 | 1 | 4 |
| H06 | 0 | 1 | 0 | 1 |
| H07 | 3 | 0 | 1 | 4 |
| H08 | 0 | 1 | 0 | 1 |
| H09 | 0 | 2 | 0 | 2 |
| H12 | 0 | 2 | 0 | 2 |
| H15 | 0 | 2 | 0 | 2 |
| H16 | 3 | 0 | 1 | 4 |
| H17 | 3 | 0 | 1 | 4 |
| H20 | 3 | 0 | 0 | 3 |
| H21 | 3 | 0 | 0 | 3 |
| H23 | 2 | 0 | 1 | 3 |
| H24 | 0 | 3 | 0 | 3 |
| H26 | 0 | 6 | 0 | 6 |
| H27 | 0 | 4 | 0 | 4 |
| H28 | 0 | 2 | 0 | 2 |
| H29 | 0 | 2 | 0 | 2 |
| H30 | 0 | 5 | 0 | 5 |
| H33 | 0 | 5 | 0 | 5 |
| H36 | 0 | 2 | 0 | 2 |

Each row counts ledger entries that include OOS-period data: one per family evaluation, per variant for H26, and the realistic-economics and Part 49 tables for the option hypotheses. The equity families' selected variants were identical in every look.

## What QuantLab had already disproven (before this phase)

These results come from earlier QuantLab work (docs/REAL-MONEY-READINESS.md, SELECTION-EVIDENCE.md,
UPSIDE-EVIDENCE.md):
- About 30 daily price/volume rules on liquid US stocks were flat or negative after costs: 8 strategies, 16 discovery cohorts, 4 catalyst hypotheses and 15 new-data tests.
- Price-reaction PEAD, insider and House-member disclosures, and the discovery score had no predictive power.
- The best ranking (12-1 momentum, low volatility, liquidity) was not significant and still below SPY.
- The one durable finding was that some variables predict the SIZE of the next move, not its direction.

## What this phase learned

| Learning | Evidence |
|---|---|
| No daily US-equity cross-sectional, event or calendar rule tested has a net edge in 2016-24 | H01, H03, H04, H07, H08, H09, H15-H17, H20, H21, H23 (all E or D) |
| Survivorship bias is 1.2-5.5 pp/yr for long-only books, largest for volatile names | H02 |
| Movement size and volatility are predictable; direction and timing are not | H12 |
| In liquid options, implied volatility is a better forecast than QuantLab's model; the model adds only marginal information | H33 (corrected) |
| No option sort (IV-RV, forecast-IV, option momentum, IV rank, term structure, skew) makes money after crossing the spread | options realistic economics, every sort, every subset |
| Single-stock option spreads (median ~10% of premium per side) are larger than the variance premium | baselines |
| The published pre-earnings straddle effect is small at mid (+2-4%) and costs ~30% round trip | H26 (corrected placebo) |
| Selling index straddles earned ≈ 0 in 2019-24, even on SPY with 0.4-1.2% spreads | H35 |
| Weak equity streams are not independent and do not combine into an edge | H36 |
| Data integrity beats method sophistication: the audit's corrections mattered more than any model choice | audit section |

## New data acquired in this phase (all free)

| Dataset | Coverage | Notes |
|---|---|---|
| Survivorship-free daily bars | 2016-2024; 14,160 entities including ~1,900 delisted common stocks | Alpaca SIP, raw and adjusted. Delisted tickers recovered from Alpaca's inactive list and from a historical Nasdaq Trader directory (DoltHub `stocks.symbol`) re-queried with `asof`. |
| Symbol-change history | 3,679 ticker changes, 2017-04 → 2026-10 | Alpaca corporate actions. Identity metadata only: which company traded under a ticker on a past date. |
| End-of-day option chains | 2019-02 → 2024-12; ~73M quotes; ~2,000 underlyings | DoltHub `post-no-preference/options`: bid/ask/IV/Greeks, 3 expiries, ~20 strikes. No volume or open interest. Put-call parity confirms same-session end-of-day snapshots. Archive-imported before 2021-05. |
| Earnings calendar, EPS history, consensus estimates | calendar 2020-2024; estimates 2017-2024 | Calendar history was backfilled, so 2020-24 dates are realised dates (PIT_ASSUMED). |
| Fama-French factors, FOMC dates, SPY/QQQ 1-minute bars, 40-stock minute sample 2020-21 | 2016-2024 | Ken French library; Federal Reserve; Alpaca SIP. |

## New data required (ranked by information value per dollar)

1. **The bot's own database (free; needs Harry).** The 570 shadow opportunities, decisions and fills
   are needed for the missed-opportunity engine (Part 32) and for execution alpha. The cross-session
   export request was blocked by the permission system; it has to be exported on the PC. Their
   outcomes only mature between October and December 2026 anyway.
2. **Delisted companies whose ticker was later reused by a listed company** (e.g. old Caesars, old
   SunPower, old Barnes Group). They are missing from the store, so H02 understates survivorship bias.
   They can be recovered free: re-query Alpaca with `asof` set to the day before each reuse (from the
   symbol-change history).
3. **Intraday option quotes plus open interest (ThetaData Standard, ~$80/month)**, only for a specific
   execution hypothesis, "can QuantLab fill near mid?". It is not worth buying for more signal mining:
   the end-of-day chains already show that no tested signal survives the spread.
4. **A point-in-time earnings calendar with announcement times.** True PIT only from late 2024 in the
   free source; 2025+ is the holdout.
5. Point-in-time fundamentals (SEC; needs a configured user agent) for value or quality ideas. These
   were not tested in this phase.

## Method (fixed before any outcome; revisions dated in plan §10)

- **Splits** (`alpha/splits.py`, pinned by tests):

  | Data | TRAIN | VALIDATION | OOS |
  |---|---|---|---|
  | equity | 2016-19 | 2020-21 | 2022-24 |
  | options | 2019-02..2021 | 2022 | 2023-24 |
  | earnings calendar | 2020-21 | 2022 | 2023-24 |

  The 2025+ holdout was **not downloaded** into the research store. Code refuses to load it.
- **Pre-registration:** every family's variant grid, and every audit-driven revision, was committed to
  git before the run it governs.
- **Selection** on TRAIN only (net Sharpe; event studies: TRAIN mean). VALIDATION confirms.
- **OOS looks:** each family has one OOS evaluation per locked spec and data version. Re-evaluations
  after documented bug fixes go through a one-time tagged override, and every look at OOS-period data
  is counted (`registry.oos_looks`; table in the audit section).
- **Costs:**
  - stocks: QuantLab's CostModel tiers (2-25 bps by liquidity) plus 5 bps slippage; shorts pay 50 bps/yr borrow; intraday books pay a full round trip every day;
  - options: pay the ask in and the bid out, or intrinsic at expiry;
  - capped-risk wings priced by Black-Scholes at the 25-delta IV plus 10%.
- **Statistics:**
  - Newey-West t (≥ 5 lags for weekly one-month option series);
  - Driscoll-Kraay for panel regressions;
  - Deflated Sharpe with the family trial count;
  - Hansen SPA and White's Reality Check across variants;
  - PBO via CSCV, effective-trial counts, Benjamini-Hochberg;
  - removal of the top 1/5/10% of trades, 2x costs, delisting-return sensitivity (−30% / −100% / 0%);
  - regime, liquidity and volatility splits.
- **Point-in-time tests:** a truncation-invariance test for every signal: 17 audit tests plus the
  original suite (56 alpha tests). Method validation: a planted reversal is found (t > 3) and a null
  world is not.

## Strategies tested and results

### Equity cross-section

Final data: panel v3 (survivorship-free; twins de-duplicated; zero-volume filler removed). TRAIN
2016-19 (from the first active day), VALIDATION 2020-21, OOS 2022-24.

| Family | Variants (effective) | Selected on TRAIN | TRAIN net t | VAL net t | SPA p | PBO | DSR | OOS Sharpe (t) | OOS beta | Class |
|---|---|---|---|---|---|---|---|---|---|---|
| H01 short-term reversal (1-10d; raw/market/sector/residual; 5 weightings) | 160 (2.0) | 10d raw, EW, L/S, 5d hold | −0.17 | −0.24 | 1.00 | 0.20 | 0.00 | −0.55 (−1.12) | 0.24 | E |
| H03 momentum matrix (6 horizons × 6 definitions + 4 conditions) | 40 (4.1) | 12-1 when breadth strong | 1.86 | −0.11 | 0.40 | 0.34 | 0.00 | 0.59 (1.01) | −0.07 | E |
| H04 residual momentum | 16 (1.2) | 6-1 two-factor | 0.90 | −0.02 | 0.49 | 0.61 | 0.50 | 0.32 (0.65) | −0.37 | E |
| H16 MAX / lottery | 4 (1.0) | MAX5 EW | 0.01 | −0.64 | 1.00 | 0.26 | 0.25 | 0.13 (0.25) | −1.20 | E |
| H17 idiosyncratic volatility | 8 (1.0) | IVOL 63d two-factor | 0.13 | −0.82 | 1.00 | 0.34 | 0.19 | 0.27 (0.53) | −1.13 | E |
| H23 analyst-revision momentum | 2 (1.2) | 63d consensus change | −0.39 | −0.02 | 1.00 | 0.04 | 0.33 | 0.21 (0.35) | −0.14 | E |
| H07 intraday gap reversal | 4 (1.2) | large caps EW | gross t 2.9; net −59 bp/day | −7.65 | 1.00 | 0.00 | 0.00 | −5.5 (−10.1) | 0.10 | **D** |

- **Beta-adjusted OOS alpha t-stats:** H17 1.52, H16 1.27, H04 1.14, H03 1.11. None reaches 2. The
  low-volatility and lottery alpha appears only in 2022-24, and was absent or negative in 2016-21.
- **Short-term reversal** (the brief's first priority): gross returns of 0-5 bp/day are smaller than
  turnover costs of 5-30 bp/day. Not one of 160 variants is positive net in TRAIN.
- **H07 bounce test (M5, pre-registered plan §10.6):** 40 large caps, 2020-21, 3,591 positions, entry price re-measured from 1-minute bars.
  - Gross per day: opening print +4.8 bp (t 1.0); last trade before 09:35 +3.0 bp (t 0.7); 09:30-09:34 VWAP +5.2 bp (t 1.1).
  - Bounce share 38%, so by the pre-registered rule **class D stands**: most of the gross return survives a later entry.
  - The sample's gross returns are themselves not significant, so this test neither confirms nor refutes the full-universe gross signal (dev t 2.9). Either way the book loses ~59 bp/day net.
- **Overnight vs intraday (H06, descriptive; gross; the selected variant of each family):**

  | Book | Leg | TRAIN | VAL | OOS |
  |---|---|---|---|---|
  | 12-1 momentum (H03) | overnight | **+3.9 bp/day (t 5.8)** | +4.5 (t 1.2) | **+4.5 (t 3.4)** |
  | | intraday | +0.9 (t 0.5) | −4.4 (t −0.8) | −0.9 (t −0.4) |
  | Residual momentum (H04) | overnight | **+3.5 (t 3.7)** | +5.6 (t 1.6) | **+6.5 (t 4.6)** |
  | | intraday | +0.7 (t 0.2) | −4.3 (t −0.7) | −3.2 (t −0.9) |
  | Low MAX / low IVOL (H16/H17) | overnight | −3.1 / −3.2 (t −2.3 / −2.2) | −11.8 / −13.6 (t −2.6 / −2.9) | +0.6 / −0.2 |
  | | intraday | +4.8 / +4.6 | +8.4 / +7.1 | +2.7 / +4.4 |

  **Momentum is an overnight phenomenon in this sample, and the low-risk anomalies are an intraday
  one.** This replicates Lou, Polk & Skouras (2019).
  - It is not tradeable stand-alone: holding only overnight means a full round trip every day, about
    14 bp against a 4-6 bp gross.
  - It does mean rebalance timing (market-on-close vs next open) matters for any momentum book.

### Event, calendar and intraday

| Family | Result (corrected data) | Class |
|---|---|---|
| H20 earnings-announcement premium (calendar splits) | Every variant ≤ 0 in 2020-21. Selected: 5 days before, exit before the release. OOS −12 bp per event (t −0.2); −68 bp without the top 5% of events. | E |
| H21 EPS-surprise drift | TRAIN t 0.3-0.8; the first run's t 1.4-2.2 fell once future delisters were included. OOS +29 bp mean but t −0.6, and −542 bp without the top 5%. | E |
| H09 seasonality (13 SPY effects: day of week, month-end, turn of month, pre-holiday, OPEX, FOMC, pre-FOMC, NFP) | **None survives Benjamini-Hochberg** (min q 0.82; best raw p 0.06 is OPEX Friday, a *negative* effect). Pre-FOMC (+14 bp) and NFP Friday (+13-18 bp) are positive in every split but t < 0.8; a round trip costs ~14 bp. | E |
| H08 SPY/QQQ intraday momentum | \|t\| < 1.3 in every split; slope ≈ 0 except in 2020-21. Uses minute data only, unaffected by the audit. | E |
| H15 pairs (Gatev-Goetzmann-Rouwenhorst distance method; 20 pairs, 2σ, next-open) | Sharpe TRAIN/VAL/OOS: within sector −0.62 / +0.23 / −0.18; any sector −0.16 / +0.57 / −0.41. Returns ±1-2% a year, \|t\| < 1.2 everywhere. | E |
| H26 pre-earnings straddle (Gao-Xing-Zhang replication; entry 1-3 snapshots ≈ 2-7 sessions before) | **Mid-to-mid:** +1.9% to +3.9% per event, against +0.6% to +7.6% for the corrected non-earnings placebo. The event beat the placebo by 1.5-2 pp in 2020-21; they were level in 2022; in 2023-24 the placebo was higher. **Ask in, bid out:** −28% to −35% per trade, hit rate 4-10%. | E |

### Options lab (2019-24 end-of-day chains; one 1-month ATM straddle per stock per week; 390,637 observations, 1,911 companies, 9% from companies that later delisted)

| Measure | TRAIN 2019-21 | VAL 2022 | OOS 2023-24 |
|---|---|---|---|
| IV above the subsequent realised volatility | 69% | 63% | 70% |
| Long straddle at **mid**, held to expiry | +2.0% | −5.6% | −3.1% |
| Long straddle at the **ask**, per $ mid | −10.4% | −19.7% | −16.6% |
| Iron fly at the bid (capped, wings priced) | −9.0% | −13.1% | −16.8% |
| Median spread, % of mid per side | 9.6% | 11.3% | 9.3% |

**Every pre-registered sort, traded realistically.** Long the top decile at the ask, short the bottom
decile at the bid, P&L per $ of mid premium, OOS 2023-24, weekly NW t:

| Sort | All names | Liquid (spread ≤ 10%) | Strict ATM |
|---|---|---|---|
| H24 HV − IV (Goyal-Saretto) | −22.0% (−12.6) | −4.6% (−2.6) | −21.2% (−12.2) |
| H33 forecast − IV (Part 49) | −22.0% (−14.3) | −3.3% (−2.3) | −21.6% (−14.0) |
| H33 forecast (with IV) − IV | −20.7% (−14.4) | −4.9% (−4.1) | −20.0% (−13.7) |
| H27 option momentum 3m / 12m | −20.5% / −19.9% | −5.9% / −4.6% | −20.2% / −19.7% |
| H30 IV percentile low (pre-registered 52w) / first-run 150w | −19.2% / −19.0% | −2.8% / −1.4% | −18.9% / −18.6% |
| H30 IV change low | −31.8% | −4.4% | −31.5% |
| H28 term slope | −17.9% | −2.6% | −17.2% |
| H29 skew low | −30.4% | −4.1% | −29.8% |

- Ask-to-ask decile spreads (top-decile minus bottom-decile long-straddle return) still look large in
  2023-24: +6% to +22% a month, t 3-9 (skew excepted). They are not tradeable: they credit the short
  side with the ask, and among liquid options the OOS deciles have no ordering (H33: between −2.4%
  and +0.1% across the ten deciles).
- The C2 correction removed most of the TRAIN-period spread. Top-minus-bottom, first run → corrected:
  - H27 12m: +31% (t 4.6) → +4% (t 1.9);
  - H33: +29% (t 3.3) → +12% (t 1.5);
  - H24: +22% (t 3.9) → +9% (t 2.1).

**H33 / Part 49: QuantLab's volatility forecast against implied volatility.** Lower log-variance error
is better; encompassing t-statistics are Driscoll-Kraay with 6 weekly lags.

| Subset | Period | QuantLab forecast | IV raw | IV debiased (TRAIN bias) | Encompassing coefficient (t) |
|---|---|---|---|---|---|
| All options | VAL | 0.098 | 0.099 | **0.089** | forecast 0.46 (8.0), IV 0.59 (15.6) |
| All options | OOS | **0.127** | 0.204 | 0.177 | forecast 0.74 (15.0), IV 0.31 (8.2) |
| Liquid (spread ≤ 10%) | VAL | 0.089 | 0.069 | **0.066** | forecast 0.10 (1.4), IV 0.88 (20.9) |
| Liquid | OOS | 0.120 | 0.085 | **0.078** | forecast 0.13 (3.9), IV 0.88 (33.5) |
| Liquid, earnings in window | OOS | 0.136 | 0.104 | **0.096** | forecast 0.09 (1.9), IV 0.93 (24.4) |

The model: OLS of log realised vol on log 5/21/63-day realised vol, log range-vol and an earnings
flag. It was fitted on 130k calendar-covered TRAIN rows with an embargo. The first run reported
forecast t 36 (all) and 12.7 (liquid); those used week clustering and contaminated joins.

**Part 49, both directions** (plan §10.7). Within the top third of predicted movement each week:

| Trade | Subset | TRAIN | VAL | OOS |
|---|---|---|---|---|
| Predicted move ≫ IV: long straddle at the ask | liquid | −1.1% (−0.1) | −6.8% (−1.5) | −2.0% (−0.5) |
| | all | −13.1% (−1.4) | −21.7% (−6.0) | −16.3% (−4.2) |
| IV ≫ even a high predicted move: capped iron fly at the bid | liquid | −1.9% (−0.6) | +1.1% (+0.4) | −4.8% (−2.4) |
| | all | −9.3% (−3.9) | −7.9% (−5.6) | −19.7% (−11.1) |

**H25, expected vs realised move.** OOS realised/implied move averages 0.83-0.99 (median 0.64-0.84).
Earnings windows are priced about right (0.94-0.99). High-IV non-earnings names are the most
overpriced (0.83), but selling them at the bid loses after spreads (the iron fly rows above).

**H35, the index variance premium.**
- **SPY** (spreads 0.4-1.2%): IV > RV in 67% of months (2019-21), 48% (2022) and 77% (2023-24). Short ATM straddle at the bid, per month: −0.5% / −11.3% / +0.9% (t ≈ 0). Iron fly: +4.3% / −12.4% / −4.6%.
- **DIA:** similar.
- **XLK and XLF** quotes in this dataset are wide (14-51%) and every short structure loses.

Class E for the ATM straddle expression. OTM put-writing (the Cboe PUT index structure) was **not**
tested.

### Machine learning with separated targets (H12; fit 2016-19, one OOS look)

899k stock-weeks; features known at the close; separate model per target. OOS figures, with the
training figure in brackets for the tree models.

| Target | Linear | Random forest (train) | Gradient boosting (train) | Trailing-vol baseline |
|---|---|---|---|---|
| Direction, 5-day (AUC) | 0.518 | 0.514 (0.704) | 0.513 (0.680) | 0.495 |
| Timing, +5% before −5% (AUC) | 0.518 | 0.500 (0.795) | 0.497 (0.822) | 0.497 |
| Magnitude, \|5-day return\| (weekly rank IC) | 0.244 | 0.358 | 0.355 | 0.317 |
| Volatility, 21-day (weekly rank IC) | 0.792 | 0.781 | 0.798 | 0.694 |
| Tail, \|21-day\| > 10% (AUC) | 0.704 | 0.707 | 0.698 | 0.679 |

The tree models overfit badly (train AUC 0.70-0.82 for direction and timing, ~0.50 OOS). Complexity
helps only for magnitude (0.36 against 0.24 for linear).

### Portfolio construction (H36)

The selected streams of the five cross-sectional families (H01, H03, H04, H16, H17) amount to **3.1
effective independent bets**: MAX and IVOL correlate 0.96; momentum and residual momentum 0.72.

| Method (weights fit on TRAIN) | TRAIN Sharpe | VAL Sharpe | OOS Sharpe (t) | Walk-forward OOS Sharpe (yearly refit) |
|---|---|---|---|---|
| Equal | 0.40 | −0.50 | 0.20 (0.38) | 0.20 |
| Inverse vol | 0.58 | −0.41 | 0.26 (0.48) | 0.25 |
| Risk parity | 0.55 | −0.40 | 0.19 (0.35) | 0.22 |
| Max diversification | 0.52 | −0.39 | 0.13 (0.24) | 0.22 |
| Min variance | 0.70 | −0.28 | 0.20 (0.35) | 0.28 |
| Constrained mean-variance | 0.86 | −0.05 | 0.50 (0.93) | 0.42 |
| Capped Kelly | 0.99 | −0.22 | 0.63 (1.07) | 0.50 |

Every method is negative in 2020-21 and insignificant OOS (best t 1.07; best alpha t 1.9, capped
Kelly walk-forward). Diversification cuts drawdowns to 15-23% but does not create a significant
return. Class E.

### Survivorship bias (H02)

Long-only books, 2016-2024, next-open execution, CostModel costs. "Full" is the survivorship-free
liquid universe (1,907 names/day); "survivors" is the same universe restricted to names still listed
at the end of 2024 (1,572/day). There are 1,871 delistings in the panel, 645 of them distressed.

| Book | Full CAGR | Survivors CAGR | Bias (pp/yr) | Full Sharpe | Full max DD |
|---|---|---|---|---|---|
| Equal-weight universe (monthly) | 8.5% | 9.7% | +1.2 | 0.50 | −43% |
| 12-1 momentum, top decile | 11.3% | 12.8% | +1.5 | 0.51 | −46% |
| 5-day losers, bottom decile | 4.0% | 6.4% | +2.4 | 0.28 | −65% |
| 60-day volatility, top decile | 1.1% | 6.7% | **+5.5** | 0.21 | −73% |

Delisting-return sensitivity (distressed −100% or 0% instead of −30%) moves full-universe CAGR by
< 0.2 pp. The bias is mainly *which names are in the universe*, not the size of the last return. This
is a lower bound: delisted companies whose ticker was later reused are still missing (see Data
limitations).

## Statistical significance and multiple testing

- **Families:**
  - Hansen SPA p 0.40-1.00 and White RC p 0.51-1.00.
  - Deflated Sharpe probabilities ≤ 0.50.
  - PBO 0.00-0.61; residual momentum is the only family above 0.5, and its development result was not significant anyway.
  - Benjamini-Hochberg rejections: 0 in every family.
- **Effective trials** are far fewer than nominal: the 160 reversal variants behave like 2.0
  independent bets, the 40 momentum variants like 4.1. Correlated variants inflate search less than
  feared, but they also add no information.
- **Event families:** none survives removal of the top 5% of events.
- **Ledger:** 818 runs, 266 distinct configurations. OOS-period looks are listed per family in the
  audit section.

## Out-of-sample results

- **Not one family has an OOS t ≥ 2 net of costs**, on the first-run data or the corrected data.
- **Positive but insignificant OOS Sharpe:** H03 0.59, H04 0.32, H17 0.27, H23 0.21, H16 0.13. All
  were flat or negative in the 2020-21 validation window.
- **Options:** every realistic long/short book is negative OOS in every subset.
- **Part 49:** both trades are negative OOS.

## Walk-forward results

Rolling yearly re-selection (plan §10.9). At the start of each year, take the variant with the best
net Sharpe over all earlier years (at least 2 years of history), trade it for that year, and chain the
years. Selection never sees the year it trades. The first traded year is 2019 or 2020, depending on
each family's history needs. 2022-24 is reported separately.

| Family | Variants | Distinct picks | Walk-forward Sharpe (t) | CAGR | Max DD | Alpha t vs SPY | 2022-24 Sharpe (t) | Years positive |
|---|---|---|---|---|---|---|---|---|
| H01 reversal | 160 | 3 | −0.54 (−1.32) | −15.6% | −66% | −1.81 | −0.63 (−1.25) | 0 of 6 |
| H03 momentum matrix | 40 | 3 | 0.34 (0.90) | +4.8% | −30% | 0.94 | 0.56 (1.18) | 3 of 6 |
| H04 residual momentum | 16 | 3 | 0.00 (0.01) | −2.5% | −32% | 0.24 | 0.18 (0.36) | 2 of 5 |
| H16 MAX | 4 | 1 | −0.10 (−0.28) | −7.9% | −65% | 1.41 | 0.13 (0.25) | 2 of 6 |
| H17 IVOL | 8 | 2 | −0.12 (−0.33) | −8.4% | −71% | 1.09 | 0.22 (0.43) | 3 of 6 |
| H23 revisions | 2 | 1 | 0.08 (0.18) | −0.3% | −38% | 0.58 | 0.21 (0.35) | 2 of 5 |
| H07 gap reversal | 4 | 1 | −5.87 (−15.0) | −78% | −100% | −15.0 | −5.55 (−10.1) | 0 of 6 |

- Re-selecting every year does not rescue anything. The best walk-forward stream, momentum with
  Sharpe 0.34 (t 0.9), kept switching variants: 63-day consistency, then 12-1 in strong breadth, then
  12-1 in a calm market. That is the instability selection on noise produces.
- Portfolio walk-forward (H36, yearly re-fit weights across the five streams): OOS Sharpe 0.20-0.50,
  t ≤ 0.96.

## Options results (summary)

- **Information, yes. Money, no.** Implied volatility is a strong but upward-biased forecast: realised
  volatility is below implied 63-70% of the time. QuantLab's forecast:
  - beats raw IV in illiquid names;
  - is beaten by IV in liquid names;
  - adds a little information beyond IV (OOS t 3.9) but not enough to pay a ~10% spread per side.
- **No structure survived:**
  - long or short straddles, capped iron flies and both Part-49 directions;
  - ten cross-sectional sorts;
  - pre-earnings straddles and index straddle selling.
- **What would have to change for options to work:** fills much closer to mid than the quoted spread,
  or structures where spread per unit of risk is smaller (longer-dated, index, OTM put-writing). That
  is an execution problem, and it needs intraday quote data to test.

## Best signals

Best in the sense of reliable information, not tradeable return:
1. **Volatility and magnitude prediction** (H12; OOS rank IC 0.80 for 21-day volatility and
   0.36 for the 5-day absolute move, both well above trailing-volatility baselines).
2. **Implied volatility itself:** in liquid options it is the best available forecast of realised
   volatility. QuantLab should use it as an input, not try to beat it.
3. **12-1 momentum in strong breadth (H03):** positive in TRAIN (t 1.86) and OOS (Sharpe 0.59, t 1.0)
   but negative in 2020-21. Not significant after the family's 40 trials.
4. **Momentum's return is earned overnight (H06):** +4 to +6 bp a night, t 3.4-5.8 in TRAIN and OOS.
   This is a robust description, not a tradeable stand-alone edge.

## Failed signals

Everything in the tables above, all kept in the ledger:
- reversal (160 variants);
- momentum (40) and residual momentum (16);
- MAX and IVOL;
- gap reversal (signal exists, untradeable);
- intraday momentum and seasonality;
- the earnings premium, post-earnings drift and revisions;
- pairs;
- ten option sorts;
- pre-earnings straddles and index variance selling;
- ML direction and timing models;
- every portfolio combination.

## Regime dependence

Every selected variant has results split by SPY trend, SPY volatility tercile, credit (HYG/LQD),
rates (TLT) and dollar (UUP); see the `by_*` keys in `results/*.json`.
- The clearest pattern: reversal and long-volatility books did well only in 2020's turmoil and badly
  otherwise.
- No family qualified as class C, because no regime split was pre-registered as its own hypothesis.
  Adding one now, after seeing OOS, would be data snooping.
- Candidates for a future pre-registered test: reversal in high-volatility regimes; momentum when
  breadth is strong.

## Data limitations

- **Missing delisted companies.** Delisted companies whose ticker was later reused by a still-listed
  company (old Caesars, old SunPower, old Barnes, old CoreSite and others) are missing from the store.
  Their option and earnings data are now dropped rather than mis-assigned. H02 is therefore a lower
  bound. Delisting coverage is also weaker for 2016-17.
- **Splices.** Splices of two securities under one key without a zero-volume gap cannot be detected.
  Splices with a gap (268 found) are now excluded.
- **Security type** is name-based. Market cap is unknown; dollar volume stands in for size.
- **Options:**
  - 3 expiries and ~20 strikes per underlying; no volume or open interest;
  - archive-imported before 2021-05; Mon/Wed/Fri snapshots;
  - wings for capped structures are priced by a model (Black-Scholes at the 25-delta IV).
- **The 2020-24 earnings calendar is realised dates** (PIT_ASSUMED; backfilled). Calendar
  de-duplication used post-event volume for 3.4% of rows (disclosed, not a signal).
- **No bot database in this cloud.** The missed-opportunity engine (H13) and execution alpha (H39)
  are DATA-LIMITED. The export was blocked by the permission system and needs Harry.
- **Quality process.** The first data-quality report did count 862k zero-volume bars and 2,380 one-day
  moves above +100%, but nothing excluded them until the second review. Data-quality flags must gate
  the research panel, not just be reported.

## Overfitting risks

- **The largest false discovery here was in options.** Decile sorts showed t 5-10 in the first run;
  in the corrected run they show t 3-9 in ask-to-ask terms but lose money in every realistic
  implementation. This is exactly what Part 52 warns about: "a 3,000% options trade proves nothing".
- **Tree models overfit:** train AUC 0.70-0.82 against ~0.51 OOS for direction.
- **OOS is no longer virgin.** It was looked at up to 4 times per equity family (first run, panel v2,
  panel v3, walk-forward), and every look is disclosed. The selected variants never changed, so no choice was made
  on OOS, but the 2025+ holdout is now the only clean test.
- Earlier QuantLab work used 2021-24 for price/volume ideas, so equity OOS was not virgin for those
  families to begin with.

## Transaction-cost impact

Costs are the binding constraint almost everywhere:

| Book | Gross | Costs |
|---|---|---|
| Reversal | 0-5 bp/day | 5-30 bp/day (daily turnover 0.2-1.5x) |
| Gap reversal | ~10 bp/day | ~70 bp/day (in and out every day) |
| Option straddles | variance premium: a short straddle at mid earns −2% to +6% of premium per month | ~10% of premium per side (median spread), ~20% round trip |
| Momentum and low-vol books (low turnover) | | 2x costs moves t by < 0.5 |

## Execution impact

- Next-open versus next-close (MOC) execution, recorded per family: no class changes.
- Paper fills on Alpaca are optimistic for options. Real option execution would be worse than the
  bid/ask modelled here, not better.
- Gap reversal: entering at 09:35 instead of the opening print keeps ~62% of the (small) gross return. Costs, not fill timing, are what kill it.

## Portfolio impact

There is nothing to add to the bot's book. The equity streams are weak, mostly short the market, and
not independent: 3.1 effective bets out of 5. Their best combination (capped Kelly) has OOS Sharpe
0.63, t 1.07, and is negative in 2020-21.

## The six questions of Part 51

| Question | Answer |
|---|---|
| What does NOT work? | Daily price/volume/calendar/earnings rules on US stocks, after costs; option sorts and straddle trades after spreads; both Part-49 trades; ML for direction or timing |
| What might work? | Using movement/volatility forecasts to size risk (not to trade); option strategies only if fills near mid are achievable (untested); index put-writing (untested) |
| What data do we need? | The bot's database (free, needs Harry); the reused-ticker delisted companies (free); intraday option quotes only for an execution test |
| What evidence supports it? | H12 volatility/magnitude ICs, stable across TRAIN/VAL/OOS; H33 information beyond IV in OOS (small) |
| What evidence contradicts it? | Every realistic P&L in the options lab; IV beats the model in liquid names; H33 not significant in 2022 |
| What remains uncertain? | Execution quality in options; 2025+ behaviour (holdout untouched); regime-conditional variants (not pre-registered) |

## Recommended next steps (ranked by expected information value)

1. **Do not mine more daily US-equity patterns.** About 300 configurations have failed. The next
   "winner" from this space is most likely a false positive.
2. **Use the volatility/magnitude model as a risk tool** in the existing paper bot: size by predicted
   volatility and widen or tighten stops. This changes risk, not expected return; measure it forward.
3. **Recover the reused-ticker delisted companies** (free; plan above) and re-measure survivorship
   bias.
4. **Get the bot's database exported** (needs Harry) and run the missed-opportunity and execution
   studies when the 570 shadow outcomes have matured (Oct-Dec 2026).
5. **Spend the 2025+ holdout once, on a pre-registered shortlist**, and only if something earns a
   shortlist. Today nothing does, so keep it sealed.
6. **Options: buy ThetaData (~$80/mo) only for a specific, pre-registered execution test**, such as
   "can limit orders at mid ± x fill?". Do not buy it for more signal research.

## Per-batch summaries (Part 53 format)

**Batch 1: equity cross-section** (H01 reversal, H03 momentum matrix, H04 residual momentum, H16 MAX,
H17 IVOL, H23 revisions, H07 gap reversal; plus H02 survivorship and H06 overnight/intraday)
- **HYPOTHESIS:** daily cross-sectional rules earn a net excess return in liquid US stocks.
- **DATA:** survivorship-free daily bars 2016-24 (panel v3: 6,932 entities, 1,871 delistings); consensus
  estimates for H23.
- **METHOD:** 234 pre-registered variants; TRAIN selection; Newey-West t, DSR, SPA/RC and PBO;
  top-trade removal; 2x costs; delisting-return sensitivity; regime splits.
- **RESULT:** no family has a significant net edge in development. The best TRAIN t is 1.86 (12-1
  momentum when breadth is strong), and VALIDATION is negative.
- **OUT-OF-SAMPLE RESULT:** no family reaches t ≥ 2 (−1.12 to +1.01; the gap book −10.1).
- **ROBUSTNESS:**
  - identical selections and classes on three data versions (first run, panel v2, panel v3);
  - SPA p ≥ 0.40;
  - momentum's gross return is entirely overnight (t 3.4-5.8);
  - survivorship bias is 1.2-5.5 pp/yr for long-only books.
- **CONFIDENCE:** high.
- **NEXT ACTION:** retire these rules as stand-alone strategies. Keep the overnight/intraday finding for
  execution timing.

**Batch 2: options volatility lab** (H24 IV-RV, H33 forecast-IV and Part 49, H25 expected move, H27
option momentum, H28 term structure, H29 skew, H30 IV rank, H26 pre-earnings straddle, H35 index
variance premium)
- **HYPOTHESIS:** implied-vs-realised mispricing, option momentum, term/skew/IV-rank or earnings timing
  gives a tradeable option edge, and QuantLab predicts realised movement better than the options market.
- **DATA:** DoltHub end-of-day chains 2019-24; 390,637 one-month ATM straddle observations (1,911
  companies, 9% later delisted); earnings calendar 2020-24.
- **METHOD:**
  - decile sorts traded at ask in, bid out (or intrinsic);
  - capped iron flies;
  - all names, liquid and strict-ATM subsets;
  - Driscoll-Kraay encompassing regressions;
  - the pre-registered Part-49 two-way test;
  - the corrected earnings placebo.
- **RESULT:**
  - Information, yes: IV is above later realised volatility 63-70% of the time, and QuantLab's forecast
    beats the IV of illiquid options.
  - Money, no: every sort loses after spreads (all names −18% to −32% per trade; liquid −1% to −6%).
- **OUT-OF-SAMPLE RESULT:**
  - In liquid options, IV forecasts better than QuantLab (log-variance MSE 0.085 vs 0.120). The model
    adds little beyond IV (coefficient 0.13, t 3.9).
  - Both Part-49 trades are negative.
  - Pre-earnings straddles lose ~30% per trade after the spread.
- **ROBUSTNESS:** the first run's large option "edges" were partly wrong-company joins (corrected). The
  liquid and strict-ATM subsets agree.
- **CONFIDENCE:** high that the edges are not tradeable at quoted spreads. Medium that there is no edge
  at all, because execution near mid is untested.
- **NEXT ACTION:** only a pre-registered execution test (fills near mid) could change this. Do not buy
  data for more signal mining.

**Batch 3: events, calendar and intraday** (H20 earnings premium, H21 post-earnings drift, H09
seasonality, H08 intraday momentum, H15 pairs)
- **HYPOTHESIS:** earnings timing, earnings surprises, calendar effects, intraday momentum or pairs
  convergence earn net returns.
- **DATA:** calendar and EPS 2020-24 (75,508 clean events, PIT_ASSUMED); SPY/QQQ minute bars; daily
  bars.
- **METHOD:** event studies with beta hedge and two-sided costs; Benjamini-Hochberg across 13 calendar
  effects; GGR distance pairs.
- **RESULT:**
  - earnings premium ≤ 0 in every variant;
  - post-earnings drift TRAIN t 0.3-0.8, carried by the top 5% of events;
  - no calendar effect survives BH (min q 0.82);
  - intraday momentum and pairs ≈ 0.
- **OUT-OF-SAMPLE RESULT:** earnings premium −12 bp per event (t −0.2); drift t −0.6 (−542 bp without the
  top 5%); pairs Sharpe −0.18 / −0.41.
- **ROBUSTNESS:** including companies that later delisted removed the drift's apparent TRAIN
  significance.
- **CONFIDENCE:** medium-high. The calendar is backfilled, so it is not truly point-in-time.
- **NEXT ACTION:** retire. Re-test the earnings ideas only on a true point-in-time calendar (2025+
  holdout or a paid source).

**Batch 4: machine learning, portfolio construction and walk-forward** (H12, H36, walk-forward selection)
- **HYPOTHESIS:** separate models can predict direction, magnitude, volatility, tail and timing; weak
  independent streams combine into an edge; yearly re-selection finds what fixed selection missed.
- **DATA:** 899k stock-weeks (2016-24); the five selected equity streams; every family's variant
  streams.
- **METHOD:**
  - linear, random-forest and gradient-boosting models per target, fit on 2016-19;
  - seven portfolio methods, fit on TRAIN and re-fit yearly;
  - expanding-window yearly selection.
- **RESULT:**
  - magnitude (IC 0.36), volatility (IC 0.80) and tails (AUC 0.70) are predictable;
  - direction (0.51-0.52) and timing (0.50-0.52) are not;
  - combinations are negative in 2020-21.
- **OUT-OF-SAMPLE RESULT:** the same ICs hold OOS. Portfolio OOS Sharpe is ≤ 0.63 (t ≤ 1.07).
  Yearly walk-forward re-selection: every family's chained stream is insignificant (best: momentum,
  Sharpe 0.34, t 0.9).
- **ROBUSTNESS:** tree models overfit badly; the five streams are 3.1 independent bets.
- **CONFIDENCE:** high.
- **NEXT ACTION:** use the volatility/magnitude model for risk sizing in the paper bot, and measure that
  forward.
