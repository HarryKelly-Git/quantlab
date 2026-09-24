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
7. **Fundamentals** use the earliest filing: for each (concept, period), take the value from the
   earliest filing with `available_at <= cutoff`. Later restatements are invisible until they
   were filed. Availability comes from the filing's acceptance time when known (PIT). Otherwise
   it is assumed to be the next session's cutoff (PIT_CONSERVATIVE).
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
| rev_growth_yoy | fundamental | Revenues_q / Revenues_{q-4} - 1 (earliest-filing rule) |
| ni_margin | fundamental | NetIncomeLoss_ttm / Revenues_ttm |
| roe | fundamental | NetIncomeLoss_ttm / StockholdersEquity (latest) |
| leverage | fundamental | LongTermDebt / Assets (latest) |
| eps_growth_yoy | fundamental | EPS_q / EPS_{q-4} - 1 (NaN if EPS_{q-4} <= 0) |
| ep_ttm | fundamental | EPS_ttm / raw close (earnings yield) |
| fundamental_age | fundamental | sessions since the latest fundamental fact became available |
| news_count_1d, news_count_5d | news | items with available_at in (cutoff(D-n), cutoff(D)] |
| news_count_z | news | `(news_count_1d - mean(news_count_1d.shift(1),60)) / std(...)` |
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

## 8. Books & execution

Books: `BOT`, `HUMAN` (both paper), `SHADOW` (hypothetical), `BACKTEST`. The internal ledger
(`execution/ledger.py`) is the source of truth for each book. Brokers: `SimBroker` (fills at the
next session open with CostModel costs, the same model as the backtest) and `AlpacaPaperBroker`
(paper endpoint only). Broker state is reconciled against the ledger. A mismatch or unknown
broker state trips `SYSTEM_PAUSED`. The human book is always simulated internally with the same
fill model as the bot, so the human-vs-bot comparison is apples-to-apples. Human decisions are
append-only. A correction is a new row with `supersedes_id`.

## 9. Research validity

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
  model failure, corrupt state.
* Failures lead to UNKNOWN / NO TRADE / SYSTEM_PAUSED. The system never assumes success.
* Secrets come only from env vars (`.env` is git-ignored) and are redacted from logs.

## 11. Database

SQLite (WAL) at `project.db_path`. Migrations live in `src/quantlab/db/migrations/NNN_name.sql`.
Core schema is `001_core.sql`. Reserved ranges for subsystem migrations: data 010-019,
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
