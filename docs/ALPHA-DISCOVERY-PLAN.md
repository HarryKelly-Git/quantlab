# Alpha discovery plan (2026-10-06)

The goal of this phase is to find out whether the market gives QuantLab anything worth exploiting,
and to prove the answer either way. It is not to make QuantLab profitable. PAPER ONLY. Branch
`alpha-discovery`. Code: `src/quantlab/alpha/`. Records: `research/alpha/` (queue + append-only
ledger). Final report: `docs/ALPHA-DISCOVERY-REPORT-2026-10.md`.

---------------------------------------------------------------------------------------------------

## 1. What QuantLab already knows (from the repo audit)

**Infrastructure that works and is reused here, not rebuilt**
- **Point-in-time discipline:** truncation-invariance tests (`testing/pit.py`); raw prices for
  level rules and adjusted prices for ratios.
- **Statistics:** stationary bootstrap, PSR, Deflated Sharpe, Holm / Benjamini-Hochberg /
  Benjamini-Yekutieli and Newey-West in `validation/stats.py`; locked-holdout tokens in
  `validation/holdout.py`.
- **Cost model:** half-spread tier by 20-day median dollar volume (2/5/10/25 bps) plus 5 bps
  slippage (`core/costs.py`). The new engine uses the same tiers.
- **Trading machinery:** trade-plan simulator, walk-forward runner, promotion gates, research
  ledger, shadow book, exploration with a hypothesis ladder.
- **Execution:** Alpaca paper execution with opening-auction entries, limit fallback, partial-fill
  top-ups and reconciliation.
- **Options layer** (`options/`): live comparison of stock against defined-risk structures, an OPT
  paper book that is off by default, and marking from real option bars.

**Measured results (docs/SYSTEM-INVENTORY.md, SELECTION-EVIDENCE, UPSIDE-EVIDENCE,
REAL-MONEY-READINESS)**
- About 30 daily price/volume rules are flat or negative after costs: 8 strategies, 16 discovery
  cohorts, 4 catalyst hypotheses, 15 new-data tests.
- The best ranking (12-1 momentum, low volatility, high liquidity) earns +10/+23 bps per 5-day
  trade, not significant and still below SPY. Its robust feature is half the tail risk.
- **Characteristics that predict big moves predict size, not direction:**
  - Range contraction, time since earnings, volume spikes and distance below the 52-week high all
    raise the odds of a big move *down* about as much as *up*.
  - The most volatile quintile makes the most +10% winners and loses money on average.
- Post-earnings strength (EAR z >= 1.5) was negative out of sample (t −3.4). Price-reaction-based
  post-earnings drift doesn't survive costs.
- The SPY 200-day regime throttle is supported in one half-sample only and is unvalidated.
- Insider buying: flat at realistic timing; the one positive variant failed the holdout. House
  member purchases passed once, then failed on 2025-26.

## 2. What is disproven, what is uncertain

| Status | Item |
|---|---|
| Disproven (as tested) | Daily price/volume pattern rules on liquid US stocks as stand-alone long trades; price-reaction PEAD; insider and House disclosures as signals; the discovery score |
| Uncertain, never tested properly | Short-term reversal as a *market-neutral cross-section*; residual momentum; the overnight/intraday split; anything using implied volatility; earnings-calendar effects with a real calendar; analyst revisions; whether past tests were biased by survivorship (they were long-only on current listings) |
| Untested and blocked earlier by data | IV vs realised vol, expected move vs realised move, earnings straddles, option momentum, term structure, skew (no historical option data until now); delisted stocks (no list until now) |

## 3. Datasets: available now (probed 2026-10-06)

| Dataset | Coverage | Point-in-time status | Notes |
|---|---|---|---|
| **Alpaca SIP daily bars, active + inactive symbols** (`alpha/store.py`) | 2016-01 .. 2024-12; 32.8k tickers requested | PIT (raw for levels, adjusted for returns) | 19,169 inactive symbols: delisted, acquired, bankrupt. Entity-mapped `asof` keeps one company per series (META includes FB; BBBYQ shows Bed Bath falling to $0.075; WFM, TWTR, SIVB, FRC present). Security type is name-based (ASSUMED). |
| **DoltHub `post-no-preference/options` chain** (`alpha/options_store.py`) | 2019-02 .. 2024-12 exported (to 2026-10 in source) | Snapshot-dated EOD | Bid, ask, IV and Greeks for about 1,500-2,300 underlyings. 3 expiries (~2w, ~1m, ~2m) and ~10-25 strikes around the money. Weekly in 2019, **Mon/Wed/Fri 2020-24**, daily from 2025. **No volume, no open interest, no underlying price.** Anonymous publisher, so quality checks come first. |
| DoltHub volatility_history | same | snapshot-dated | Vendor IV/HV per underlying (definition undocumented; used as a cross-check only) |
| **DoltHub earnings calendar** | 2020-01 .. | **PIT_ASSUMED 2020-24** (backfilled: rows first appear *after* the events); true PIT from late 2024 (~38-41 days ahead) | Announcement date plus before/after-market flag. Duplicate projected dates exist and need cleaning. |
| DoltHub eps/sales estimates, eps history | dated consensus snapshots | snapshot-dated | Enables analyst-revision and earnings-surprise tests |
| Ken French data library | full | academic factors (CRSP-based, survivorship-free) | Factor-neutral residuals; replication benchmark (published reversal and momentum factor returns) |
| Nasdaq earnings API | reachable | historical pages are realised dates | Backup for the calendar |
| ETFs in the Alpaca store | 2016-24 | PIT | Regime proxies: SPY trend, TLT/IEF (rates), HYG vs LQD (credit), UUP (dollar), DBC/USO/GLD (commodities), VIXY |
| Alpaca minute bars | 2016+ | PIT | Cheap for SPY/QQQ intraday studies; too heavy for the whole cross-section |

## 4. Missing data

| Missing | Blocks | How to get it | Cost |
|---|---|---|---|
| Option volume / open interest, full chains, intraday option quotes, pre-2019 options | Liquidity filters on option trades; exact earnings-straddle timing; long option history | ThetaData. Free tier: EOD quotes + OI from 2023-06 only. Value ($40/mo): 1-min data, IV and Greeks from 2020. Standard ($80/mo): tick data from 2016. | $40-80 for one month of downloads |
| SEC fundamentals / 8-K events in this cloud | Quality/value, event database | `QUANTLAB_SEC_USER_AGENT` is not set here (the PC has it) | free; needs the env var |
| The bot's 570 shadow opportunities, decisions and fills | Missed-opportunity engine (Part 32); execution alpha (Part 39) | Export from the PC database. **The cross-session request was blocked by the permission system**; needs Harry | free |
| FRED (rates, credit spreads) | Macro regime labels | Blocked from this network; ETF proxies used instead | free |
| Short interest, borrow fees | Realistic short-side costs | paid vendors | flat 50 bps/yr assumption with x4 sensitivity |
| Market cap history | Size buckets | shares outstanding needs SEC | dollar volume used as the size proxy (labelled) |

**ThetaData decision (Part 41):** don't subscribe yet. The free DoltHub chains cover the core
questions (IV vs realised vol, expected vs realised move, straddle returns, term structure and
skew) at end-of-day resolution for 2019-2024. Subscribe for **one month of Standard** only if a
DoltHub-based options result reaches class B or better and needs confirmation with full chains,
open interest and intraday timing.

## 5. Research philosophy for this phase

**Alpha taxonomy.** Every hypothesis is tagged with one of these:

| | | | |
|---|---|---|---|
| A directional | B cross-sectional | C relative value | D volatility |
| E event-driven | F mean reversion | G momentum | H market-neutral |
| I options surface | J dispersion/correlation | K seasonality | L liquidity/microstructure |
| M information timing | N alternative data | O portfolio construction | P execution |

**What is being predicted is stated explicitly every time:**

| Target | Measured by | Traded through |
|---|---|---|
| DIRECTION | sign of the forward return | stock or delta |
| MAGNITUDE | absolute forward return | straddles, strangles, condors |
| VOLATILITY | forward realised volatility | vega and gamma structures |
| RELATIVE PERFORMANCE | return vs the cross-section, sector or market | market-neutral books |
| TIMING | when, not whether | entry and exit rules |

A signal that predicts MAGNITUDE but not DIRECTION is not a failure. It is a candidate for options,
and Part 49's central question is whether **QuantLab can forecast realised movement better than
the options market prices it**.

**Replication first** (Part 38): published effects are reproduced as published, with costs and
point-in-time data, before any change is tried.

## 6. Method

**Research splits** (`alpha/splits.py`, pinned by tests):

| Split | Equity | Options |
|---|---|---|
| TRAIN | 2016-2019 | 2019-02 .. 2021 |
| VALIDATION | 2020-2021 | 2022 |
| OOS | 2022-2024, **one** evaluation per locked spec (the registry refuses a second, different spec unless it is recorded as a new variant) | 2023-2024 |
| HOLDOUT | 2025+ is **not downloaded** into the research store. It needs a separate download and a logged unlock. | 2025+ |

Caveat: earlier QuantLab work used 2021-03..2024-11 for price/volume research, so OOS is not
virgin for those ideas. Forward paper trading remains the final test.

**Pre-registration.** Each family's variant grid (lookbacks, rankings, weightings, neutralisations)
is written into its experiment module *before* TRAIN is run. Every variant counts as a trial.

**Gate to OOS.** One variant per family, chosen on TRAIN+VALIDATION. The family's OOS slot is
spent exactly once, on that variant.

**Statistics on every family:**
- Newey-West t
- Deflated Sharpe using the family's trial count and trial variance
- Hansen SPA and White's Reality Check across the family's variants
- Probability of Backtest Overfitting (CSCV) across variants
- Effective number of independent trials
- Benjamini-Hochberg across families
- Permutation tests where cheap

**Costs.** The CostModel tiers, also shown at x2. Shorts pay 50 bps/yr borrow (x4 sensitivity).
Capacity is estimated with square-root impact at $1M and $10M.

**Robustness.**
- By year, by regime, by liquidity bucket, by volatility bucket.
- With the top 1/5/10% of trades removed.
- Delisting-return sensitivity: 0%, −30% (default for distressed names), −100%.
- With and without delisted names, to measure survivorship bias.

**Classification (Part 39)** in `alpha/registry.py`, with codes from the brief:

| Code | Class | Rule |
|---|---|---|
| A | STRONG | OOS t >= 2, DSR >= 0.95, SPA p < 0.05, survives 2x costs and removal of the top 5% of trades, >= 2/3 of OOS years positive |
| B | PROMISING | positive but at least one bar missed |
| C | REGIME-DEPENDENT | |
| D | IMPLEMENTATION PROBLEM | gross significant, net not |
| E | NO EDGE | |
| F | DATA-LIMITED | |
| G | OVERFIT | passed development, failed OOS, or PBO >= 0.5 |

Class A is a reason to start forward paper testing, not to trade.

**Research roles (Part 36), as pipeline stages rather than chat agents.** No single stage can
produce class A; the `Evidence` record needs fields filled by different stages.

| Role | Implemented by |
|---|---|
| Researcher | the literature register (section 8) |
| Data scientist | `alpha/quality.py` and the dataset table above |
| Quant | experiment modules |
| Skeptic | the robustness battery |
| Statistician | `alpha/mht.py` plus DSR/BH |
| Portfolio manager | correlation to existing books and SPY |
| Options specialist | structure pricing from the chain, with spreads |
| Execution specialist | cost and capacity sensitivity |
| Research director | `classify()` |

An independent adversarial review (a separate reviewer reading the code and the results looking
for look-ahead and errors) runs before the report is final.

## 7. Hypothesis queue (ranked by expected information value)

Expected information value = P(real and tradeable) x value if true + uncertainty removed, scaled
down by data and compute cost. The full queue is in `research/alpha/queue.json`.

| # | ID | Hypothesis | Category | Data | Predicts | Status now |
|---|---|---|---|---|---|---|
| 1 | H02 | Survivorship bias: how much did current-listings-only data flatter results? | infrastructure | store | — | testable now |
| 2 | H33 | **QuantLab forecasts realised vol better than IV prices it** (Part 49) | D/I | store + chain | magnitude/vol | testable now |
| 3 | H24 | IV vs realised vol and the variance risk premium per stock; Goyal-Saretto replication (straddles sorted on HV−IV) | D/I | chain | vol | testable now |
| 4 | H01 | Short-term reversal 1/2/3/5/10-day; raw/market/sector/residual rankings; equal/inverse-vol; dollar-/beta-neutral | F/B/H | store | relative | testable now |
| 5 | H26 | Pre-earnings straddle (Gao-Xing-Zhang), exit before the announcement | E/D | chain + calendar | vol timing | testable now (M/W/F cadence, PIT_ASSUMED calendar) |
| 6 | H25 | Expected move vs realised move buckets, with capped-risk structures | D/I | chain + store | magnitude | testable now |
| 7 | H27 | Option (straddle) momentum, controlling for underlying momentum | G/I | chain | option return | testable now |
| 8 | H28 | IV term-structure slope predicts straddle returns and realised vol | I | chain | vol | testable now |
| 9 | H29 | Put skew predicts stock returns (Xing-Zhang-Zhao) and option returns | I/A | chain + store | direction | testable now |
| 10 | H04 | Residual momentum (market, sector, factor neutral) | G/H | store + FF | relative | testable now |
| 11 | H03 | Momentum matrix: 1w-12m, absolute/relative/residual/sector/market-relative, acceleration, consistency, vol-adjusted, with conditioning (vol, volume, breadth, sector leadership) | G | store | relative | testable now |
| 12 | H06 | Overnight vs intraday decomposition of returns and anomalies | L/M | store | timing | testable now |
| 13 | H07 | Overnight strategies (close->open, open->close) by gap, momentum, volatility, earnings | L | store + calendar | timing | testable now |
| 14 | H16 | MAX/lottery effect (Bali-Cakici-Whitelaw) | B | store | relative | testable now |
| 15 | H17 | Idiosyncratic-volatility / low-vol anomaly (Ang et al.) | B | store | relative | testable now |
| 16 | H30 | IV rank / percentile and IV changes vs straddle and condor returns | D | chain | vol | testable now |
| 17 | H20 | Earnings announcement premium (pre-announcement drift) | E | store + calendar | direction | testable now |
| 18 | H21 | Post-earnings drift on EPS surprise (SUE), not on price reaction | E | eps_history + calendar | direction | testable now |
| 19 | H23 | Analyst EPS revision momentum (consensus snapshot changes) | E/N | eps_estimate | relative | testable now |
| 20 | H09 | Seasonality: day-of-week, turn-of-month, pre-holiday, OPEX, FOMC/CPI/NFP days | K | store + public calendars | timing | testable now (macro dates compiled) |
| 21 | H10 | Regime dependence of every surviving signal (SPY trend, VIXY, credit, rates, dollar, commodities) | C | store ETFs | — | after signals |
| 22 | H11/H12 | ML with separate targets (direction / magnitude / volatility / tail / timing); simple models first | ML | store + chain | all | testable now |
| 23 | H31 | Delta-hedged option returns vs idiosyncratic vol (Cao-Han) | I | chain + store | option return | testable now |
| 24 | H32 | Dispersion: SPY/QQQ IV vs constituent IVs, implied vs realised correlation | J | chain (if SPY covered) | correlation | check coverage |
| 25 | H08 | SPY intraday momentum (first 30 minutes predicts the last 30) | L | SPY minute bars | timing | testable now (small download) |
| 26 | H15 | Pairs / cointegration within statistical sectors | C | store | relative | testable now |
| 27 | H36 | Ensemble of independent validated signals; portfolio construction (EW, inverse vol, risk parity, max diversification, capped Kelly) | O | outputs | — | after signals |
| 28 | H37 | Meta-model: when *not* to trade a signal | O | outputs | — | after signals |
| 29 | H38 | Strategy-decay monitor (rolling Sharpe, hit rate, slippage, exposure drift) | infra | outputs | — | build with results |
| 30 | H13 | Missed-opportunity engine on the bot's 570 shadow opportunities | M | PC database | — | **DATA-LIMITED (export blocked)** |
| 31 | H39 | Execution alpha (auction vs limit vs close), measured from paper fills | P | PC fills | — | **DATA-LIMITED** |
| 32 | H35 | Index variance risk premium; capped-risk index structures | D | VIX / Cboe indices, SPY options | vol | partial (Cboe CSV access to check) |

What needs new data: H13, H39 (the PC export). Full-chain and open-interest versions of H24-H30
(ThetaData). Intraday microstructure beyond SPY/QQQ (minute or quote data at scale). Fundamentals
and 8-K event studies in this environment (the SEC user agent).

## 8. Literature register (Part 37)

Each external result is recorded with its source, sample, method, costs, survivorship treatment,
out-of-sample evidence and criticism. Entries are added to `research/alpha/literature.json` as
families are tested. The first entries:
- Goyal-Saretto (2009) IV-RV straddles
- Gao-Xing-Zhang (2018) pre-earnings straddles
- Heston et al. (2023) option momentum
- Cao-Han (2013) idiosyncratic vol
- Vasquez (2017) term structure
- Xing-Zhang-Zhao (2010) skew
- Lehmann (1990) / Lo-MacKinlay (1990) / de Groot et al. (2012) short-term reversal
- Blitz-Huij-Martens (2011) residual momentum
- Lou-Polk-Skouras (2019) overnight vs intraday
- Bali-Cakici-Whitelaw (2011) MAX
- Ang et al. (2006) idiosyncratic volatility
- Frazzini-Lamont (2007) earnings announcement premium
- Chan-Jegadeesh-Lakonishok (1996) revisions
- Gao-Han-Li-Zhou (2018) intraday momentum
- Lucca-Moench (2015) pre-FOMC drift
- Zarattini-Barbon-Aziz (2024) opening-range breakout (no costs, no OOS; replication found net ≈ 0)
- de Silva-Smith-So (retail losses in earnings options)

## 9. Absolute rules (Part 52, restated for this code)

Never delete a failed run (the ledger is append-only). Never re-tune after seeing OOS. Never use a
universe of today's survivors. Never treat paper profit as alpha. Never assume an academic result
still works, or that a stock signal carries over to options, or that cheap options are good or
expensive ones should be sold. Every short-volatility structure has a known maximum loss. If data
is missing or stale, do nothing.
