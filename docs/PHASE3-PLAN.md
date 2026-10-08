# Phase 3: convert the information QuantLab has into a tradeable advantage

> **Status (2026-10-07):** items 3-7 were run as the one-week alpha sprint
> ([pre-registration](ALPHA-SPRINT-PREREG.md), [final report](ALPHA-SPRINT-FINAL.md)). No edge survived. Item 2
> (bot-database analysis) is still waiting for Harry's export.

Baseline truth: [ALPHA-DISCOVERY-REPORT-2026-10.md](ALPHA-DISCOVERY-REPORT-2026-10.md), corrected after two audits.
Old hypotheses are not reopened. Rules change only with a documented reason. The 2025+ holdout stays
sealed. PAPER ONLY.

## Knowledge map

| | Content | Evidence |
|---|---|---|
| **Know** | **Direction:** 5-day AUC 0.51-0.52, a coin flip. **Timing:** 0.50-0.52, a coin flip. **Volatility:** 21-day rank IC 0.80, strong. **Magnitude:** IC 0.36, real. **Tails:** P(\|21d\| > 10%) AUC 0.70. | H12 (899k stock-weeks) |
| **Know** | In liquid options, IV forecasts realised vol better than QuantLab (OOS MSE 0.085 vs 0.120); QuantLab adds little beyond IV (coefficient 0.13, t 3.9 OOS; t 1.4 in 2022) | H33 corrected |
| **Know** | No option sort or straddle trade survives the quoted spread (~10% of premium per side); no equity rule survives costs; weak streams don't combine (3.1 effective bets) | report |
| **Think we know** | The forecast's extra information beyond IV is concentrated somewhere (liquidity, earnings, IV regime). It is untested by segment. | H33 segments |
| **Think we know** | Mid-price fills would change some option results. Untested: no intraday quotes. | H26/H33 economics at mid vs ask/bid |
| **Disproven** | Every daily equity rule tested; option decile sorts at quoted spreads; pre-earnings straddles at quoted spreads; both Part-49 trades; ML for direction/timing | report |
| **Unknown** | (1) Is the bot too conservative? (2) Does vol-forecast sizing improve the existing paper strategies? (3) Is the forecast well calibrated? (4) Do QuantLab's P(large move) beat option-implied probabilities anywhere? (5) Which structure expresses a vol view most cheaply? (6) How much of the spread can patient orders capture? | |
| **Data needed** | (1) Bot DB export (free; Harry). (2)-(3) Existing research store. (4)-(5) Existing DoltHub chains (end of day). (6) Intraday quotes; the options-data business case decides. | |

## Four prediction problems, kept separate

A **DIRECTION** (weak), B **MAGNITUDE** (meaningful), C **VOLATILITY** (strong), D **TIMING** (weak).
Every Phase 3 result states which problem it uses. Magnitude and volatility are never discarded
because direction failed. The question is how to earn from "a large move is more likely" without
knowing its sign, given what options already price.

Every economic result is reported three ways: at mid (optimistic), at a conservative fraction of the
spread captured (realistic), and near the worst executable side (pessimistic). A result that only
works at mid is **EXECUTION-SENSITIVE**, not alpha. A statistical improvement without after-cost value
is **STATISTICALLY USEFUL, NOT ECONOMICALLY TRADEABLE**.

## Queue (ranked by expected value per unit of effort)

| # | Item | Directive part | Status |
|---|---|---|---|
| 1 | HOLDOUT_LOCK: every access refused and logged; unlock only with a committed pre-registration; 1-evaluation budget | 3 | **DONE** (`alpha/holdout.py`, `research/alpha/HOLDOUT_LOCK.json`, tests) |
| 2 | Bot-DB importer: schema validation, quality checks, missed-opportunity analysis with date-clustered CIs and no verdict under 20 dates | 4 | **DONE** (`alpha/botdb.py`, `scripts/research/alpha/run_botdb.py`, [BOT-DB-IMPORT.md](BOT-DB-IMPORT.md)); waiting for Harry's export |
| 3 | Volatility forecast engine: 1/3/5/10/20-day vol, expected absolute move, P(large move), expected tail move; stored separately from direction; calibration curves (bucket → realised), rank IC, MSE/MAE, monotonicity | 5-6 | next |
| 4 | Vol-forecast sizing vs fixed sizing on the existing paper strategies: Sharpe, Sortino, drawdown, tails, turnover. Risk improvement without alpha. | 7 | next (the most likely near-term gain) |
| 5 | Large-move probabilities P(\|r\| > 3/5/8/10%) vs option-implied probabilities (from the straddle/strangle prices in the chains); conformal intervals | 19-22 | queued |
| 6 | Structure comparison (straddle, strangle, butterflies, condors, calendars, verticals) under 3 execution scenarios, liquid subset, pre-registered liquidity thresholds | 9-10, 13-14, 17-18 | queued |
| 7 | RV − IV residual model and where-it-works niches (IV rank, liquidity, earnings, maturity); ensemble of IV + HV + QuantLab, weights fit on TRAIN only | 11-12, 23-24 | queued |
| 8 | Options-data business case (current DoltHub vs ThetaData vs others, pricing verified from provider docs) | 15-16 | queued; no purchase before it |
| 9 | H50-H56 (magnitude-based option hypotheses), pre-registered in this file before any run | 26 | queued, after 3-7 |

Validation history: 2016-2024 (2022-24 has been examined repeatedly: research history, not a pristine
test). Final test: the sealed 2025+ holdout, once, on a pre-registered shortlist. Forward evidence: the
bot's paper record.
