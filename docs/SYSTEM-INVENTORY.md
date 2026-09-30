# QuantLab — full system inventory

As of 2026-09-30. A complete accounting of what has been built, what it has actually proven, what
has never run, and where the real improvements are. Written to be read top-to-bottom once.

---

## 1. What it is

A paper-only US-equity research lab and an Alpaca **PAPER** trader. There is no live mode: the
broker client refuses a non-paper host, `TRADING_MODE=PAPER` must be declared explicitly to start,
and no code path can move real money.

The design principle throughout is **point-in-time (PIT) honesty**: every decision for session D+1
uses only information available by 16:00 ET on session D, and executes at the D+1 open. This is
enforced by truncation-invariance tests, not just by convention — a feature must return the same
value when the dataset is cut off at the decision time as it does with the full history, or the test
fails.

Two operating modes:

- **STRICT** — trade only strategies with statistically significant, walk-forward-validated,
  positive expectancy after costs. Currently that is **zero strategies**, so STRICT places no orders.
- **EXPLORATION** (currently active, set in the git-ignored `config/local.yaml`) — also paper-trade
  the top-2 ranked discovery setups per session *without* proven edge, at 2% of equity each, max 5
  open, to accumulate forward evidence. All safety controls stay on.

## 2. Build history

24 commits, roughly 8 phases:

| Commits | Phase |
|---|---|
| `a7f1ad4` `788d978` | Foundation: config, DB schema with audit triggers, PIT data contracts, trade simulator, feature/strategy base classes |
| `cf93a6f` `9af0dfb` `6186ffa` `224ddc7` | Features, universe, 8 strategies, regime engine, arena, validation, experiment runner, reports, CLI |
| `b08f739` | Real data providers: Alpaca bars/actions/news, SEC EDGAR events + fundamentals, Nasdaq Trader reference |
| `e60627a` `ab497b0` | Paper execution layer (SimBroker, Alpaca paper broker, ledger, exits, journal, reconciliation); decision stack (stats → EV → no-trade → ranking → portfolio → risk → final decision); daily pipeline + replay |
| `02ea7ca` `e3923a0` | Walk-forward OOS runner, real-data audit, dashboard |
| `a50dd90` `c898f17` | Persistent Alpaca paper runner: preflight, trade_updates stream, idempotent orders, heartbeat, round-trip order test |
| `e736f69` `448cca3` | Market discovery layer (separate from validation), unified candidate pool, next-session/overnight mode, forward-outcome research |
| `731da6c` `ed7745f` `dc28c44` | Catalyst data (PIT earnings, full-universe news, PIT industry/sector, PIT fundamentals, other 8-K events), catalyst discovery + research, STRICT/EXPLORATION split |
| `abea0c2` … `eaf0d6d` | **Fill fix**: opening orders that expire unfilled now take the position; supervisor script; fallback gated by broker/state checks; expired exits get the same fallback |

## 3. Code inventory

**137 Python modules, 29,655 lines** in `src/quantlab`, plus **75 test files, 9,245 lines** (734
tests passing).

| Package | Modules | What it does |
|---|---|---|
| `strategies/` | 13 | 8 registered strategies: momentum_trend, mean_reversion, breakout, pead_ear, relative_strength, quality_momentum, extreme_reversal, sector_rotation |
| `execution/` | 11 | SimBroker, Alpaca paper broker (live-host refusal), ledger, order service, exits, journal, reconciliation |
| `features/` | 10 | Price, volume, event (EAR, days-since-earnings), as-of fundamentals, industry/sector, news counts |
| `discovery/` | 10 | 5 scored price/volume families + context families, catalyst discovery, catalyst research replay, source coverage |
| `data/` + `data/providers/` | 18 | Alpaca, SEC EDGAR (8-K items, companyfacts, SGML headers for historical SIC), Nasdaq Trader, news classifier, catalyst refresh |
| `ml/` | 7 | Model scaffolding |
| `decision/` | 6 | Stats, EV gate, no-trade rules, ranking, portfolio construction, risk chain |
| `pipeline/` | 5 | The daily runner, session jobs, steps |
| `core/` | 5 | Types, trade simulator, cost model, calendar |
| `validation/` | 4 | Walk-forward, holdout tokens, significance |
| `exploration/` | 4 | Exploration engine, hypothesis ladder, review queue |
| `experiments/` | 4 | Experiment registry and runner |
| `ai/` + `ai/providers/` | 7 | LLM assessment scaffolding |
| others | ~33 | testing (PIT assertions), shadow, monitoring, dashboard, universe, risk, research, regime, portfolio, db, backtest |

**Database**: 79 tables across 16 migrations.

**CLI**: 25 command groups — `init ingest validate universe backtest experiments holdout-unlock
status pause resume run-daily replay strategy data-audit walkforward paper discover
discovery-research next-session catalysts catalyst-research explore hypothesis dashboard demo`.

## 4. Data assets

452 MB on disk, 29 MB database.

| Kind | Sets | Range | Rows |
|---|---|---|---|
| bars (daily) | 33 | 2020-01-02 → 2026-09-29 | 6,845,288 |
| news | 59 | 2012-04-17 → 2026-09-29 | 2,077,983 |
| fundamentals | 26 | 1992-02-29 → 2034-03-05 | 3,582,409 |
| events (8-K etc.) | 29 | 2020-01-02 → 2026-09-29 | 545,449 |
| corporate actions | 7 | 2020-01-02 → 2026-09-29 | 52,436 |
| reference | 7 | — | 92,978 |

Also 15,731 universe snapshots and 798 quarantined symbols (unexplained split jumps etc.).

Notable data engineering: SEC `acceptanceDateTime` is used as true availability, never `filingDate`;
the companyfacts `frame` field is never used because it is look-ahead; fiscal Q4 is derived as
FY − YTD9, paired with the YTD9 that was known at filing time; historical SIC codes come from SGML
headers so industry classification is itself point-in-time.

## 5. What has actually been proven

**Nothing yet. Not one strategy has positive validated edge.**

Walk-forward out-of-sample, real data:

| Strategy | OOS trades | Verdict | Sharpe | Win rate |
|---|---|---|---|---|
| momentum_trend | 155 | NOT_SIGNIFICANT | 0.29 | 34.2% |
| mean_reversion | 952 | NOT_SIGNIFICANT | 0.02 | 51.3% |
| breakout | 309 | NOT_SIGNIFICANT | 0.23 | 46.3% |
| extreme_reversal | 840 | NOT_SIGNIFICANT | −1.15 | 47.1% |
| relative_strength | **0** | INSUFFICIENT_SAMPLE | — | — |
| sector_rotation | **0** | INSUFFICIENT_SAMPLE | — | — |
| quality_momentum | not run | — | — | — |
| pead_ear | not run | — | — | — |

Catalyst research (real data, 2021-03 → 2024-11), four hypotheses plus two controls:

- A: earnings + positive reaction → **flat**
- B: A + volume confirmation → **flat**
- C: material catalyst + reaction + volume → **flat**
- D: catalyst + strong industry relative strength → **negative** (t = −4.02)
- both controls → flat

**Conclusion: no post-earnings drift survives costs.**

Execution reality: **0 trades, 0 fills, 0 positions, ever.** Two orders have been placed in the
system's life (2026-09-29, UGP and RNG), and Alpaca **expired both unfilled** in the opening auction
— `opg` orders are eligible only for the cross and are not rested if they miss it. That bug is now
fixed with a bounded market-day fallback, but the fix has never filled an order in production.

## 6. Built but never used

These tables have zero rows. Each represents real code with no evidence behind it.

| Area | Tables at 0 | Read |
|---|---|---|
| **Trading outcomes** | `trades`, `fills`, `positions`, `position_log`, `trade_events`, `trade_plans` | Never traded. The whole point. |
| **Shadow outcomes** | `shadow_outcomes`, `shadow_outcome_details` — while `shadow_opportunities` has 570 | **Not yet, not broken** (corrected 2026-10-01). The `outcomes` step runs every evening; the 570 were recorded 2026-09-24/28/29, and an outcome is written only once its 5-60 session horizon has passed. First maturities ~2026-10-02, the longest in December. |
| **AI/LLM layer** | `ai_assessments`, `ai_calls`, `ai_objections`, `llm_cache` | Scaffolding only. Never called. |
| **ML layer** | `models`, `ml_predictions`, `model_status_log` | Scaffolding only. |
| **Hypothesis ladder** | `hypotheses`, `hypothesis_events`, `research_hypotheses`, `research_ledger` | The promotion workflow exists but has never been walked. |
| **Human lab** | `human_decisions`, `human_decision_context`, `human_notes`, `lab_reports`, `evidence_packets` | The review queue for real-money ideas has never produced one. |
| **Monitoring** | `health_checks` | No health history. |
| **Ranking** | `opportunity_rankings` | Superseded by `discovery_candidates` (1,314 rows). |
| **Holdout** | `holdout_tokens`, `holdout_token_uses`, `holdout_access_log` | Correct — the 2025-01-01 holdout is still locked and untouched. |

Roughly a third of the schema and several whole packages (`ai/`, `ml/`, parts of `research/`) are
unexercised. That is not automatically waste — but it is complexity you are carrying with no return.

---

## 7. Where the improvements actually are

Ranked by expected value, most valuable first.

### 1. Confirm a fill happens, then read the outcome loop as it matures
The system has never held a position. Until it does, every other improvement is speculative.
(Done 2026-09-30: HALO and TBBB filled.) *Correction 2026-10-01:* an earlier version of this item
called the 570 `shadow_opportunities` dormant and proposed a backfill. That was wrong. The
evening pipeline's `outcomes` step already scores them; none had matured when this was written.
The work is to **read** them as they mature from ~2026-10-02, not to build anything. The question
"does the discovery score predict anything?" was meanwhile answered historically on 334,904
observations: it does not (docs/SELECTION-EVIDENCE.md).

### 2. The stop is not a real stop
`OrderRequest` has no `stop_loss`, `order_class`, or bracket field. Nothing submits a stop order to
Alpaca. Stops are QuantLab-managed only: evaluated against **daily closes** in the evening pipeline
(~19:05 ET), with the exit submitted for the **next** open. A stop breached intraday Monday exits at
Tuesday's open — up to a full trading day of exposure beyond the stop, plus gap risk. With 10-session
holds on momentum names, gap risk is the dominant loss source and it is entirely unmanaged.

Options, cheapest first: (a) submit a native Alpaca stop order alongside each entry so the broker
holds protection; (b) bracket orders (entry + stop + target as one OTO/OCO); (c) an intraday check in
the runner's 5-minute tick against last trade price rather than waiting for the evening close.

(a) is the highest value per line of code.

### 3. There is no take-profit at all
No target price is set on exploratory trades. Exits come only from the stop, the 10-session time
limit, or thesis invalidation. `execution.exits.market_risk_max_drawdown` defaults to null — off.
There is no trailing stop, no partial exit, no scale-out. A momentum setup that runs 20% in four days
gives it all back waiting for session 10. Adding even a crude ATR-multiple target or a trailing stop
would change the return distribution more than any new strategy.

### 4. Two strategies emit zero out-of-sample trades
`relative_strength` and `sector_rotation` both returned **0** OOS trades. That is a bug or a gate so
strict it never fires — not a finding. Diagnosing them is cheap and gets two more real verdicts. Note
that `relative_strength` produces 0 trades as a *strategy* while "Momentum + Relative strength" is the
*discovery family label* on both of today's candidates; these are different code paths and the
overlap is confusing enough to be worth renaming.

### 5. pead_ear was never walk-forward tested
It is the strategy tied to the catalyst data — the phase that took the most effort. Running its
walk-forward closes the loop on that whole investment, and the catalyst research already predicts it
will fail. Run it anyway: a clean negative result lets you retire the catalyst code with confidence
instead of maintaining it on hope.

### 6. Daily bars are a structural ceiling
Everything is decided on daily closes and executed at the next open. There is no intraday data at
all. This means: the overnight gap is your largest single risk and you cannot see it coming; you
cannot manage a position within a day; and your search space is restricted to the single most
heavily mined dataset in finance. If you want a genuinely different result, this is the change that
makes different results *possible* — not another strategy on the same bars.

### 7. Forward evidence accumulates too slowly to learn from
2 trades/session ≈ 10/week. A 10-session hold means the first cohort of outcomes lands in ~2 weeks,
and statistical significance on a ~50-bps effect needs hundreds of trades. **Do not fix this by
raising the trade cap** — that inflates the sample without improving the information. Fix it by
making the *historical* replay the primary evidence engine and treating live exploration as a
correctness check on the machinery, not as the research.

### 8. The cost model deserves an audit
Reported walk-forward expectancy is around −100 bps/trade for momentum_trend, and costs turn several
flat results negative. That makes the cost assumption the single most decision-relevant number in
the system. It is worth proving it right. (Note: `core/costs.py` and `core/tradesim.py` currently have
uncommitted edits from a parallel session — resolve those before trusting any cost number.)

### 9. Machine reliability
The runner died silently once: no python process, 7h21m stale heartbeat, while the database still
said `RUNNING`. A whole session's planned trades were lost. `scripts/run_paper.cmd` now restarts it
on crash, but it cannot survive the PC sleeping or rebooting, and nothing alerts you. Worth having:
a Windows Task Scheduler entry with "run whether logged in or not" plus wake timers, and a heartbeat
staleness alert that reaches your phone.

### 10. Delete or finish the unused third
`ai/`, `ml/`, the hypothesis ladder, the human-lab tables and `opportunity_rankings` are carried but
unexercised. Every one is a thing you must reason about when changing anything near it. Pick: wire it
up this month, or delete it and recover the commit from git if you ever want it back.

---

## 8. The honest summary

The engineering is genuinely good. PIT discipline is real and enforced by tests. The safety model is
strong — paper-only is structural, not a flag. The data assets are substantial and correctly
point-in-time, which is rare and hard.

What is missing is a finding. Eight strategies, five discovery families, four catalyst hypotheses,
two controls: **all flat or negative after costs.** That is not a failure of the code; it is the
expected result of searching daily price/volume patterns on liquid US equities, which is the most
efficiently arbitraged search space that exists.

So the top of the improvement list is not "more strategies." It is: (1) make the machine actually
hold positions and score them, (2) give positions real risk management — a broker-held stop and some
form of profit-taking, (3) read the 570 recorded opportunities as their outcomes mature, and (4) decide
whether to keep searching the same space or change data. Items 1–3 are days of work with certain
value. Item 4 is the only one that could change the answer.
