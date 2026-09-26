# QuantLab architecture & contracts

QuantLab is a US-equity research lab, an autonomous **paper** trader and a human paper-trading lab.
It exists to find out whether any repeatable edge exists. It is allowed to conclude that none does.

**No live money.** There is no live-trading code path. The only broker endpoint the code will ever
contact is `https://paper-api.alpaca.markets`. That is enforced in config validation and again in
the broker client.

This document is the contract that all modules code against. When code and this document
disagree, one of them is a bug.

---------------------------------------------------------------------------------------------------

## 1. Layout

```
config/default.yaml        every tunable parameter (local.yaml overrides, git-ignored)
src/quantlab/
  config.py secrets.py logging_setup.py context.py cli.py
  core/        types.py calendar.py costs.py tradesim.py            <- shared vocabulary (FOUNDATION)
  db/          database.py migrations/NNN_*.sql                      <- SQLite audit trail (FOUNDATION)
  data/        schemas.py panel.py store.py providers/ ingest.py validation.py
  universe/    engine.py
  features/    base.py price.py volume.py relative.py event.py fundamental.py news.py market.py
  regime/      engine.py
  strategies/  base.py registry.py <family>.py arena.py
  backtest/    engine.py metrics.py
  validation/  splits.py holdout.py stats.py baselines.py walkforward.py
  experiments/ registry.py ab.py
  ml/          dataset.py models.py walkforward.py calibration.py registry.py meta.py
  ai/          providers/ evidence.py prompts.py schemas.py roles.py budget.py service.py
  decision/    ranking.py expected_value.py no_trade.py final.py
  portfolio/   construction.py
  risk/        engine.py
  execution/   broker.py sim_broker.py alpaca_paper.py ledger.py service.py reconcile.py exits.py journal.py
  shadow/      book.py outcomes.py
  counterfactual/ engine.py
  human/       service.py
  compare/     human_vs_bot.py
  research/    ledger.py promotion.py ideas.py
  monitoring/  health.py killswitch.py model_monitor.py llm_monitor.py
  pipeline/    daily.py report.py
  dashboard/   app.py templates/ static/
  testing/     pit.py fixtures.py                                    <- test harness (FOUNDATION)
tests/<subsystem>/test_*.py
```

## 2. Point-in-time (PIT) rules. These are the most important rules here.

1. **Information cutoff.** A decision "as of session D" may use only information with
   `available_at <= TradingCalendar.cutoff(D)`, where the cutoff is D at 16:00 America/New_York.
   Orders execute at the **next session's open**. Backtests and live paper trading use the same
   rule.
2. **Raw levels, adjusted ratios.** Bars are stored RAW (unadjusted). Level-based logic (price >=
   $5, dollar volume, share counts) uses raw fields only. Returns and ratios use `ret` (daily total
   return computed from raw data plus that day's actions) or the tri-scaled fields `aopen/ahigh/
   alow/aclose`. The scale of tri-scaled fields is arbitrary, so never compare their levels to
   anything raw.
3. **`DataBundle.truncate(D)`** returns exactly what was knowable at cutoff(D): panel rows up to D,
   actions with `ex_date <= D`, and events/fundamentals/news with `available_at <= cutoff(D)`.
4. **Truncation invariance** (`quantlab.testing.pit.assert_truncation_invariant`). Every feature,
   universe rule, regime metric, strategy score and ML feature matrix must give identical values at
   D whether it runs on the full bundle or on `bundle.truncate(D)`. Each such component needs a
   test that proves this.
5. **Never store full-history facts in metadata** that survive truncation. For example, a
   symbol's last bar is future delisting information.
6. **`pit_status`** (`core.types.PitStatus`: PIT > PIT_CONSERVATIVE > ASSUMED_STATIC > UNKNOWN) is
   carried by every dataset row, feature spec and candidate. UNKNOWN is not usable in historical
   research by default (`HISTORICAL_RESEARCH_PIT`). Reports print the weakest status involved.
7. **Fundamentals are as-of.** For each (symbol, concept, period), the value known at D is the
   one from the most recently filed row with `available_at <= cutoff(D)`. A restatement becomes
   visible only from its own filing. Never use XBRL `frame` or the frames API: they carry the
   latest restated value. Periods are keyed by (start, end) duration, never by the filing's
   fy/fp. Availability comes from the filing's acceptanceDateTime (join `accn` to submissions)
   when known (PIT). Otherwise it is the cutoff of the session after `filed` (PIT_CONSERVATIVE).
   XBRL numbers appear only when the 10-Q/10-K is filed, often days after the earnings release.
8. **Current metadata** (security type, sector/SIC) is ASSUMED_STATIC and must be labeled that way.
9. **LLM outputs are contaminated for history.** Models may know what happened after a
   historical date, so AI-filtered results are evaluated only with forward shadow/paper tests.
   Historical AI "backtests" must be labeled `CONTAMINATED_HISTORICAL` and never used for promotion.
10. **Survivorship.** The universe is built only from securities with data. Coverage of
    delisted names depends on the provider, so reports must state the survivorship status
    (`UNKNOWN` until verified). Positions in a symbol that stops trading are closed at the last
    close x (1 + `costs.delisting_return`).

## 3. Data contracts (`data/schemas.py`)

Kinds: `bars`, `corporate_actions`, `reference`, `events`, `fundamentals`, `news`. Providers return
frames that pass `schemas.conform(kind, df)`. UTC columns must be tz-aware and session-date
columns tz-naive. Every provider raises `ProviderError` on failure and never returns partial data
silently. `ProviderNotConfigured` means credentials are missing.

`MarketDataStore` (`data/store.py`) holds immutable, content-addressed Parquet datasets plus a
`datasets` registry. `store.snapshot()` returns the dataset_ids per kind; record it with every
experiment. `store.load_bundle(benchmarks, ...)` builds the `DataBundle`. Synthetic and real data
are never mixed.

`Panel` (`data/panel.py`): wide `sessions x symbols` frames. The raw fields are `open high low close
volume`. The derived fields are `ret tri aopen ahigh alow aclose dollar_volume split_ratio dividend`.
`panel.forward_returns(h)` is a **label** helper only; it uses future data by design.

## 4. Feature catalog (`features/`, names are a contract)

Register with `@FEATURES.feature(...)` (see `features/base.py`). Return a wide frame, or for
market-level features a single `__market__` column. The `FeatureSet(bundle)` computes lazily and
caches. `a*` means tri-scaled OHLC. `SMA(x,n)` is a trailing simple mean. `vol` means daily
`ret` std. `dv` means `dollar_volume` (raw close x raw volume).

| name | group | definition |
|---|---|---|
| ret_1d | price | `ret` |
| ret_5d, ret_20d, ret_60d, ret_120d, ret_252d | price | `aclose/aclose.shift(n)-1` |
| mom_12_1 | price | `aclose.shift(21)/aclose.shift(252)-1` |
| mom_6_1 | price | `aclose.shift(21)/aclose.shift(126)-1` |
| dist_ma20, dist_ma50, dist_ma200 | price | `aclose/SMA(aclose,n)-1` |
| ma50_over_ma200 | price | `SMA(aclose,50)/SMA(aclose,200)-1` |
| atr14_pct | price | `SMA(TR,14)/aclose` with TR from a-fields (TR uses prev aclose) |
| vol_20d, vol_60d | price | `std(ret, n)*sqrt(252)` |
| vol_ratio_20_60 | price | `vol_20d/vol_60d` |
| dist_high_52w | price | `aclose/max(ahigh,252)-1` (<= 0) |
| dist_low_52w | price | `aclose/min(alow,252)-1` (>= 0) |
| drawdown_252d | price | `aclose/max(aclose,252)-1` |
| breakout_55 | price | `aclose/max(ahigh.shift(1),55)-1` (> 0 = new 55-session high) |
| range_contraction_20_60 | price | `(max(ahigh,20)-min(alow,20))/(max(ahigh,60)-min(alow,60))` |
| ret_z_1d | price | `ret / (std(ret,20).shift(1))`: today's move in units of trailing daily vol |
| ret_z_3d | price | `ret_3d / (std(ret,20).shift(3)*sqrt(3))` |
| gap_1d | price | `aopen/aclose.shift(1)-1` |
| adv20, adv60 | volume | `median(dv, n)` (raw USD) |
| rel_volume_1d | volume | `dv / mean(dv.shift(1),20)` (dollar-based, split-invariant) |
| rel_volume_5d | volume | `mean(dv,5)/mean(dv.shift(5),60)` |
| volume_trend_20_60 | volume | `mean(dv,20)/mean(dv,60)` |
| rs_spy_20, rs_spy_63, rs_spy_126 | relative | `ret_n(stock)-ret_n(SPY)` |
| rs_sector_63 | relative | `ret_63(stock)-ret_63(sector ETF)` (sector map is ASSUMED_STATIC) |
| sector_rs_spy_63 | relative | `ret_63(sector ETF)-ret_63(SPY)` broadcast to member stocks (ASSUMED_STATIC) |
| xs_rank_ret_63, xs_rank_ret_126, xs_rank_mom_12_1 | relative | per-date percentile rank (0..1) among `fs.universe` members (all symbols with data if no universe) |
| beta_126 | relative | rolling cov(ret,spy_ret)/var(spy_ret) over 126 |
| corr_spy_60 | relative | rolling corr(ret, spy_ret) over 60 |
| days_since_earnings | event | sessions since the latest earnings `reaction_date` <= D among events available by cutoff(D) |
| ear_3d | event | abnormal return (`ret - spy ret`) summed over reaction_date-1..reaction_date+1. Valid from the close of reaction_date+1 until the next event; NaN before |
| ear_z | event | `ear_3d / (std(ret - spy_ret, 60) measured before the event * sqrt(3))` |
| event_rel_volume | event | dv on reaction_date / mean(dv, 20 sessions before reaction_date), carried forward like ear_3d |
| est_sessions_to_earnings | event | `63 - days_since_earnings` clipped at [0, 63] (MODEL estimate; no PIT schedule available) |
| sue | event | seasonal random-walk SUE from EPS: (EPS_q - EPS_{q-4}) / std of last 8 such diffs, valid from the filing's available_at |
| reaction_ret_1d | event | abnormal return (`ret - spy_ret`) on the latest earnings reaction session r (8-K/A refilings excluded). Known at the close of r, carried until the next event |
| reaction_z_1d | event | `reaction_ret_1d / std(ret - spy_ret, 60)` measured through r-1, carried like reaction_ret_1d |
| abn_ret_since_reaction | event | cumulative abnormal return from the close of r to D (0 at r). Continuation vs reversal of the reaction. Missing bars count as 0 |
| sec_material_1d | event | count of material 8-K filings (items other than 2.02/7.01/9.01) usable in (cutoff(D-1), cutoff(D)]. NaN when no SEC 8-K source |
| rev_growth_yoy | fundamental | Revenues_q / Revenues_{q-4} - 1 (earliest-filing rule) |
| ni_margin | fundamental | NetIncomeLoss_ttm / Revenues_ttm |
| roe | fundamental | NetIncomeLoss_ttm / StockholdersEquity (latest) |
| leverage | fundamental | LongTermDebt / Assets (latest) |
| eps_growth_yoy | fundamental | EPS_q / EPS_{q-4} - 1 (NaN if EPS_{q-4} <= 0) |
| ep_ttm | fundamental | EPS_ttm / raw close (earnings yield) |
| fundamental_age | fundamental | sessions since the latest fundamental fact became available |
| gross_margin | fundamental | GrossProfit_ttm / Revenues_ttm over the same 4 consecutive quarters (as-of) |
| op_margin | fundamental | OperatingIncomeLoss_ttm / Revenues_ttm over the same 4 consecutive quarters (as-of) |
| fcf_margin | fundamental | (OperatingCashFlow_FY - Capex_FY) / Revenues_FY for the latest fiscal year with all three (as-of; 10-Q cash flows are YTD only) |
| news_count_1d, news_count_5d | news | items with available_at in (cutoff(D-n), cutoff(D)] |
| news_count_z | news | `(news_count_1d - mean(news_count_1d.shift(1),60)) / std(...)` |
| news_company_1d | news | company-specific items (article tagged with <= 2 symbols, tags counted over all stored rows) usable at D |
| news_material_1d | news | company-specific items whose headline category (rules in `data/news_classify.py`) is a company event, usable at D |
| sic_code_asof | industry | SIC code from the header of the latest periodic report accepted by cutoff(D) (`events.sic_observation`). NaN before the first observation |
| industry_code | industry | 3-digit SIC group with >= 5 members at D, else 1000 + 2-digit group with >= 5 members, else NaN |
| industry_members | industry | members of the symbol's industry group at D (including itself) |
| industry_ret_20 | industry | equal-weighted leave-one-out 20-session return of the industry group |
| rs_industry_20 | industry | `ret_20(stock) - industry_ret_20` |
| industry_rank_63 | industry | percentile rank (0..1] of the group's mean 63-session return among groups at D |
| rs_sector_20_pit | industry | `ret_20(stock) - ret_20(sector ETF)`, sector ETF from the point-in-time SIC via `sectors.SIC_SECTOR_RANGES` |
| sector_rs_spy_63_pit | industry | `ret_63(sector ETF) - ret_63(SPY)` for the point-in-time sector |
| market_trend_200 | market | SPY `aclose/SMA(aclose,200)-1` |
| market_mom_60 | market | SPY 60-session return |
| market_vol_20 | market | SPY `std(ret,20)*sqrt(252)` |
| market_drawdown | market | SPY `aclose/cummax(aclose)-1` |
| breadth_50 | market | fraction of universe members with `dist_ma50 > 0` |
| breadth_200 | market | fraction of universe members with `dist_ma200 > 0` |
| sector_dispersion_63 | market | cross-sectional std of sector-ETF 63-session returns |

Sentiment features are **not** available. No PIT-safe sentiment source is configured, and an LLM
sentiment score is an AI_OPINION that would first have to be validated forward. Treat sentiment as
a hypothesis.

## 5. Strategies (`strategies/`)

Subclass `strategies.base.Strategy`. Implement `score(fs, universe) -> sessions x symbols` (NaN = no
signal, larger = stronger, NaN outside the universe). Optionally override `plan()`. Register in
`strategies/registry.py`. `build_strategies(config)` returns the enabled strategies with their
config params/versions. Families and ids:
`momentum_trend, mean_reversion, breakout, pead_ear, relative_strength, quality_momentum,
news_shock, extreme_reversal, sector_rotation`.
The `StrategyArena` runs all strategies. It keeps overlapping candidates (one symbol can have many
strategy candidates). It measures information content: forward-return IC per strategy, overlap and
correlation between strategies. It does **not** count votes.

## 6. Trade semantics (`core/tradesim.py`). One definition everywhere.

Signal at close D -> entry at open D+1. Stops/targets are evaluated on each held session's
**close**, and the exit fills at the next open (gaps are paid in full). A time exit comes after
`holding_sessions` sessions. Delisting applies `costs.delisting_return`. Costs come from
`core.costs.CostModel` (a half-spread tier by 20-session median dollar volume, plus slippage and
commission). Unknown liquidity is charged the worst tier. The backtester, shadow outcomes,
counterfactuals, human-decision evaluation and EV calibration all call `simulate_plan()` or a
vectorized equivalent that is tested against it.

## 7. Decision chain (bot, per candidate)

```
strategy candidates -> ranking (opportunity score; NOT a probability) -> ML predictions (calibrated)
 -> AI researcher -> AI adversary -> AI judge (ACCEPT/REJECT/WATCH/UNKNOWN)    [optional, budgeted]
 -> no-trade engine ("why should we NOT trade this?")
 -> expected value (empirical, shrunk to a no-edge prior; LLM confidence never becomes probability)
 -> portfolio construction (sizing, sector/correlation/exposure limits)
 -> risk chain: DATA -> SIGNAL -> STRATEGY -> EV -> PORTFOLIO -> RISK -> EXECUTION (any CRITICAL fail => NO TRADE)
 -> paper order (only if SYSTEM state is ACTIVE)
```
Every candidate, whether traded or not, is written to `candidates` and `decisions`, plus
`shadow_opportunities` with its `reject_stage`. Outcomes are tracked later in `shadow_outcomes`.
AI failures produce `UNKNOWN`, never an invented answer. With the default config, an UNKNOWN AI
review does not block a trade the quantitative layers accept (the AI layer is an optional filter),
but it is recorded. `ai.unknown_policy` may be set to `block`.

### 7a. Market discovery (`discovery/`, research only)

Discovery answers "what looks interesting today?" and is kept separate from validation ("may this
setup be paper-traded?"). It can never permit, size or place an order, and it changes no gate.

* **Funnel.** Full universe, then the basic data/liquidity filter (bar at D, price >= $1, median
  dollar volume >= $1M, >= 60 sessions), then discovery features, discovery setups (at least one
  family fired), top-N ranking, watchlist, validation, paper eligible, paper trade. The steps
  after the watchlist are read from the recorded decisions and orders of the session. They are
  never recomputed.
* **Scored families.** Five price/volume families, 0-20 points each, from cross-sectional
  percentile ranks of existing FeatureSet features: momentum, relative strength vs SPY,
  volume/activity, breakout/compression and mean reversion. The score is the sum of the known
  families' points, normalised to 0-100. It is a ranking, not a probability, and it has fixed
  equal weights.
* **Context families.** Earnings, news, fundamentals and sector are recorded only where the
  source covers the symbol, otherwise UNKNOWN (never 0). They are never scored.
* **Feature states.** VALID; UNKNOWN (not enough history, or the source does not cover the
  symbol); INVALID (a data-quality issue).
* **Candidate status.** DISCOVERED, WATCH, VALIDATION_PENDING, REJECTED, PAPER_ELIGIBLE, TRADED.
  This is separate from the derived strategy research status (SHADOW, WALK_FORWARD, VALIDATED,
  PAPER_ELIGIBLE, DISABLED). A SHADOW strategy's setups stay visible, but only a TRADE decision
  from the unchanged chain makes one PAPER_ELIGIBLE.
* **Near-misses and diagnostics.** Near-misses record what each setup passed and failed, and the
  exact stage and reason. Diagnostics flag zero or low discovery, unavailable or invalid
  features, and one rule blocking everything.
* **Pipeline step.** `discover` runs after `risk` and before `report`. It is durable, and a
  failure is recorded without blocking risk or reporting.
* **Forward outcomes.** Recorded at 1/3/5/10/20 sessions (`discovery_outcomes`) and never fed
  back into rules.
* **PIT.** `scan` sees only `bundle.truncate(D)` (truncation-invariance and corrupt-the-future
  tests).
* **One candidate pool.** The pool is the discovery setups plus every symbol with a recorded
  strategy decision that session. Origin is DISCOVERY, STRATEGY or BOTH. A strategy signal is
  never dropped because its discovery score is low or because the symbol was outside the scan
  (it then carries score UNKNOWN and a `discovery_scan` check). STRATEGY-origin rows are never
  "high ranked" or put on the watchlist by discovery. `DiscoveryEngine.core()` is the single
  scoring path shared by the scan and the research replay; `families.fired_masks()` is the single
  definition of a fired family (NaN never fires).
* **Next-session mode (`discovery/nextsession.py`, migration 102).** The end-of-day scan of D
  has information cutoff 16:00 ET of D and relevance NEXT_SESSION (the next NYSE session from
  the rule calendar in `data/audit.py`; early closes are not modelled). Information phases:
  REGULAR_SESSION (<= 16:00 D), POST_CLOSE (<= 20:00 D), OVERNIGHT (<= 04:00 next day),
  PRE_MARKET (< 09:30 of the next session), NEXT_SESSION (after the open: never an input).
  `overnight_refresh` records news/events with cutoff < available_at <= now into
  `overnight_updates`; it may promote DISCOVERED to WATCH, never to PAPER_ELIGIBLE.
  `preopen_recheck` writes `preopen_checks` and moves a candidate to UNKNOWN (stale or missing
  inputs), INVALIDATED (corporate action at the next open, known before now), REJECTED (data
  quality, kill switch) or keeps PAPER_ELIGIBLE only if the chain's TRADE decision and the
  strategy's eligibility still hold (`decide_transition`, a pure function). Both refuse to run
  once the next open has occurred, so the actual open is never used. Setups are conditional
  records (why / confirm / invalidate / missing), never predicted prices, and never orders.
* **Outcome timestamps.** `discovery_outcomes` measures from the next-session open (entry at the
  next adjusted open, cost-adjusted with `CostModel`) and records `known_before_open`
  (`discovered_at < next_open_at`), so a candidate produced after the fact is labelled as such.
* **Forward-outcome research (`discovery/research.py`).** Replays `core()` on past sessions with
  block FeatureSets (rows <= D only), attaches 1/3/5/10/20-session outcomes afterwards (never as
  inputs), stops before the locked holdout, and compares each family/combination with the
  same-date baseline (date-clustered t, Bonferroni, period halves, minimum observations and
  dates, else INSUFFICIENT_SAMPLE). Findings never change rules automatically.

* **Catalyst data (`data/sec_catalysts.py`, `data/news_classify.py`).** Everything SEC-derived is
  stored in the `events` kind so `DataBundle.truncate` applies the same `available_at` rule:
  `earnings_release` (8-K item 2.02; available_at = SEC acceptanceDateTime, timing PRE_MARKET /
  INTRADAY / POST_CLOSE / NON_SESSION; 8-K/A is flagged and never a new reaction), `periodic_report`
  (10-Q/10-K), `sec_8k` (items other than 2.02/7.01/9.01, with categories), `foreign_report`
  (20-F/40-F/6-K: earnings timing UNKNOWN), `ownership_13d`, `registration`, `sic_observation`
  (SIC from each periodic report's own SGML header, as of its acceptance; the submissions `sic` is a
  current snapshot and is only stored in `sec_registrant` rows whose available_at is the retrieval
  time). News is ingested market-wide (every tag kept; `n_tags` computed over all stored rows of an
  article before any symbol filter); headlines are classified by fixed keyword rules (never an LLM),
  analyst notes and market commentary are not company events. Fundamentals: companyfacts joined to
  acceptance times; fiscal Q4 is derived as FY - YTD9 from the versions known AS OF each update.
* **Catalyst discovery (`discovery/catalysts.py`).** Two families, thresholds fixed in
  `discovery.catalyst`: `post_earnings` (an 8-K 2.02 whose reaction session is within 3 sessions and
  a measurable response: |abnormal reaction z| >= 1.5 or dollar volume >= 2x) and `material_event`
  (company-specific material news or a material 8-K usable at D plus |move z| >= 1.5 or volume >= 2x).
  Direction is recorded (POSITIVE / NEGATIVE / MUTED), never assumed; consensus surprise is always
  UNKNOWN (no PIT consensus source). Catalysts never change the 0-100 technical score and never make
  anything paper eligible. Setup class: TECHNICAL + CATALYST, CATALYST-DRIVEN, TECHNICAL-ONLY (catalyst
  sources cover the symbol and found nothing) or UNKNOWN (they do not cover it). Positive,
  market-confirmed catalyst setups get a separate capped watchlist allowance (10).
* **Evidence chain.** Every candidate stores EVENT -> WHEN KNOWN -> PRICE RESPONSE -> VOLUME RESPONSE
  -> SECTOR/INDUSTRY -> FUNDAMENTALS -> VALIDATION -> RISK -> EV -> PAPER ELIGIBILITY with provenance
  (source, id, availability time). Stages 7-10 are read from the unchanged decision chain. The run
  keeps its dataset ids and information cutoff, so the decision state can be reconstructed.
* **Catalysts overnight.** `overnight_refresh` reads only rows with cutoff < available_at <= now. A
  post-close / overnight / pre-market 8-K 2.02 for a basic-universe symbol that is not yet a
  candidate CREATES a WATCH candidate (CATALYST-DRIVEN, price/volume PENDING, `created_by =
  OVERNIGHT_REFRESH`, info_cutoff_at = the refresh time). Material company news or a material 8-K
  promotes a DISCOVERED candidate to WATCH; analyst notes / commentary are CONTEXT_ONLY. Neither can
  create eligibility.
* **Catalyst research (`discovery/catalyst_research.py`).** Pre-registered groups A-D plus two
  controls, primary horizons 5 and 20 sessions, Bonferroni over groups x 2, date-clustered t,
  period halves and per-year results vs the same-date baseline; verdicts PROMISING / NEGATIVE / FLAT /
  INCONCLUSIVE. Holdout-safe. Findings never change rules.

## 8. Books & execution

Books: `BOT`, `HUMAN` (both paper), `SHADOW` (hypothetical), `BACKTEST`. The internal ledger
(`execution/ledger.py`) is the source of truth for each book. Brokers: `SimBroker` (fills at the
next session open with CostModel costs, the same model as the backtest) and `AlpacaPaperBroker`
(paper endpoint only). Broker state is reconciled against the ledger. A mismatch or unknown
broker state trips `SYSTEM_PAUSED`. The human book is always simulated internally with the same
fill model as the bot, so the human-vs-bot comparison is apples-to-apples. Human decisions are
append-only. A correction is a new row with `supersedes_id`.

A paper book is bound to ONE broker for the life of a database (`paper_book_bindings`, immutable):
a BOT book seeded from the Alpaca paper account can never be advanced by the SimBroker, or vice
versa. Order submission is idempotent: deterministic `client_order_id` (entry: candidate; exit:
trade + decision session), the local row is written as `pending_submit` before the broker call,
and a client id the broker already knows is adopted, never resubmitted. Fills are applied from the
broker's cumulative `filled_qty` (positive deltas only), so replayed events and polls are harmless.
Partial exits keep the trade open until the position is flat.

### 8a. Alpaca PAPER runner (`pipeline/runner.py`, `quantlab paper start`)

One long-lived process. Startup: preflight (`execution/preflight.py`: env `TRADING_MODE=PAPER` and
`LIVE_TRADING=false` exactly, paper endpoint, credentials present, account ACTIVE/unblocked/USD;
every attempt recorded in `paper_preflights`) -> single-instance session (`paper_runner_sessions`,
heartbeat) -> book binding + cash seeding (refused unless the BOT book and the paper account are
clean) -> market calendar (Alpaca `/v2/calendar`, cross-checked against the NYSE rules) ->
reconciliation -> `trade_updates` websocket (`execution/trade_stream.py`).

Schedule for decision session D (ET): from `paper.runner.process_after_et` (19:05; Alpaca rejects
`opg` orders 09:28-19:00) the runner ingests D, reconciles, and runs the unchanged
`DailyPipeline(D)` with an extra per-order guard: orders are `opg` market orders for D+1's opening
auction and may only be submitted until `paper.runner.order_cutoff_et` (09:25) on D+1. A session
processed later records its decisions but every order is refused ("execution window missed").
One `paper_session_jobs` row per (book, D) holds the pipeline run_id; a restart resumes that run,
so candidates and client ids are identical and nothing is resubmitted. Fills arrive on the stream
(REST polling is the fallback) and are applied to the ledger, then reconciled.

The runner pauses the system (and keeps monitoring) on: stale/invalid data, an unverifiable broker,
a reconciliation mismatch, a broker order unknown to QuantLab, a pipeline failure, or any
unexpected error. Strategy eligibility, EV, no-trade, portfolio and risk stay per-candidate gates
in the decision chain. With no ACTIVE strategy at stage PAPER/PROMOTED the runner reports
`NO PAPER-ELIGIBLE STRATEGY` and places no orders.

### 8b. Paper modes: STRICT and EXPLORATION (`exploration/`, `paper.mode`)

* **STRICT** (default): only TRADE decisions of the unchanged chain are traded (validated strategy,
  EV after costs, risk). Exploration selections are still recorded as SHADOW and tracked.
* **EXPLORATION**: additionally, a small fixed daily budget (`exploration.max_new_per_session`, 2) of
  the best-ranked discovery / strategy candidates is PAPER traded WITHOUT requiring validation,
  significant OOS results or positive historical EV. Always required, in both modes: valid bar and
  features at D, the research universe plus exploration's own price/ADV floor, no quarantine, kill
  switch ACTIVE, long-only direction, no existing position, position / session / total exposure caps,
  and at submission every execution-service gate (the same service strict orders use).
* **Timing.** The EOD pipeline step `explore` plans (`exploration_decisions`, PLANNED). The paper
  runner revalidates and submits from `exploration.submit_after_et` until the order cutoff; the
  actual open is never an input (after it, planned entries are CANCELLED_PREOPEN). Sim replays plan
  and submit at simulated close + 5 min.
* **Records.** `exploration_decisions` (complete pre-trade state), `exploration_events`
  (PLANNED / REVALIDATED / SUBMITTED / CANCELLED_PREOPEN / REFUSED), `exploration_outcomes`
  (1/3/5/10/20 sessions from the next open, cost-adjusted, MFE/MAE, stop breached, catalyst
  persisted) are append-only. Exploratory trades carry `strategy_id = EXPLORATION`.
* **Learning loop.** Exploratory outcomes vs watched-but-not-traded vs rejected vs strict vs SPY
  (`experiment_results`). Patterns become `research_hypotheses` that move one stage at a time
  (EXPLORATION_OBSERVED -> HISTORICAL_PIT_TEST -> WALK_FORWARD -> LOCKED_HOLDOUT -> PROSPECTIVE_PAPER ->
  STRICT_ELIGIBLE), by a named human, with evidence, never automatically; STRICT_ELIGIBLE changes no
  strategy status.

## 9. Research validity

* Walk-forward (`validation/walkforward.py`): for each window the out-of-sample backtest runs on
  `bundle.truncate(test_end)`. Nothing after the window exists for the engine or the features.
  Positions close at test_end, capital chains across windows, and SPY is chained over the same
  sessions. OOS trades are stored as `oos:<k>` segments, which the EV statistics prefer. Train
  windows only produce an in-sample reference (nothing is fitted; parameters are fixed).
* Data quarantines are dated. A symbol with a data problem found at session X is excluded from X
  onward, never retroactively; retroactive exclusion would be look-ahead.
* Data splits: `validation.in_sample`, then walk-forward windows with an embargo, then a
  **locked holdout** (`validation.holdout.start`). Any backtest touching the holdout raises
  `HoldoutLockedError` unless it is unlocked with a reason. Every unlock is written to
  `holdout_access_log` forever and shown in reports.
* Every experiment is registered before it runs (`experiments` row: git commit + dirty flag,
  config hash + full config, dataset snapshot, strategy/model/AI versions, cost assumptions, seeds,
  and `n_variants_tested`). Failed experiments are kept. The tables are append-only.
* Statistics: bootstrap CIs (stationary block bootstrap), the Probabilistic Sharpe Ratio, the
  Deflated Sharpe Ratio using `n_variants_tested`, and Benjamini-Hochberg across strategies.
  Report sample sizes. Do not conclude anything below `validation.min_trades_for_conclusion`.
* Baselines are always reported: SPY buy-and-hold, a simple 12-1 momentum baseline, the raw
  strategy, the strategy with ML filtering, and the strategy with AI filtering (forward-only).
* Method validation on synthetic data. In the null world (no planted edge) the research stack must
  NOT find a significant edge. In a planted-edge world it SHOULD find it.

## 10. Safety & failure handling

* `SYSTEM_PAUSED` (`monitoring/killswitch.py`) blocks all new paper orders. Positions stay visible,
  the reason is logged, and a human must resume it. Triggers: critical data/health failure, broker
  failure/unknown state, reconciliation mismatch, bot drawdown > `risk.max_drawdown_pause`,
  model failure, corrupt state, and (paper runner) stale data past the order deadline, broker
  unverifiable, unknown broker order, pipeline failure, unexpected runner error.
* Failures lead to UNKNOWN / NO TRADE / SYSTEM_PAUSED. The system never assumes success.
* Secrets come only from env vars (`.env` is git-ignored) and are redacted from logs.

## 11. Database

SQLite (WAL) at `project.db_path`. Migrations live in `src/quantlab/db/migrations/NNN_name.sql`.
Core schema is `001_core.sql`. Shared cross-subsystem tables are in `002_shared_research.sql`:
`backtest_trades` and `backtest_equity` (written by backtest/, read by decision/ and research/)
and `symbol_quarantine` (written by data validation, honored by the universe). Reserved ranges
for subsystem migrations: data 010-019,
features/universe 020-029, strategies 030-039, backtest/validation/experiments 040-049,
execution 050-059, ML 060-069, AI 070-079, decision/portfolio/risk 080-089,
human/shadow/research/monitoring 090-099, pipeline/dashboard 100-109. Audit tables are append-only
(enforced by triggers). Helpers: `Database.insert/insert_many/upsert/fetchone/fetchall/query_df/
transaction`, `to_json/from_json`.

## 12. Conventions

* Python 3.14, pandas 3 (copy-on-write, no chained assignment, no `inplace`; use
  `stack(future_stack=True)`), numpy 2, scikit-learn 1.9, pydantic 2.
* Modules stay small and explicit. Tunables live in config. Log with
  `quantlab.logging_setup.get_logger(__name__)` / `log_event(logger, msg, **data)`.
* Tests use SYNTHETIC data (`tests/conftest.py` fixtures `config`, `db`, `ctx`,
  `synthetic_market`, `bundle`). Network tests are marked `@pytest.mark.network` and are skipped by
  default. Never weaken a test to make code pass.
* Transparency labels (`core.types.InfoKind`): FACT / MODEL_OUTPUT / AI_OPINION / HYPOTHESIS /
  UNCERTAINTY. Everything shown to a human carries one.
