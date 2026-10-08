# Proposal: merger-aware delistings and neutral spin-off steps

Status: **PROPOSAL** on branch `proposal/delisting-and-spinoffs`. Not merged. PAPER research only; no
live path, broker client, runner/pipeline orchestration, kill switch or risk limit is touched.

## 1. Every delisting is booked at −30%, including takeovers

### Problem
Master closes a held or tracked symbol as DELISTED after `execution.delisting_missing_sessions` (5)
sessions without a bar and books the exit at last close x (1 + `costs.delisting_return`), i.e. −30%.
Among liquid US stocks most delistings are acquisitions, where holders receive about the last close
(the deal price). The −30% then turns takeover wins into losses. It is applied in:

- `backtest/engine.py`, the backtester;
- `core/tradesim.py::simulate_plan`. This also feeds `shadow/outcomes.py` (status `delisted`), which
  feeds `decision/stats_provider.py` and `monitoring/llm_monitor.py`. So it biases the bot's own outcome
  statistics downward.
- `ml/dataset.py::build_labels` (not changed here, see below).

`discovery/outcomes.py` does not apply a haircut: it records `DELISTED_OR_MISSING` with NULL numbers.
The −30% outcome rows come from `shadow/outcomes.py`.

Evidence (alpha-discovery branch, `docs/ALPHA-SPRINT-FINAL.md`, survivorship-free panel, OOS 2022-24):
all 13 delisted trades in the bot's book (S1) were cash acquisitions at or above the last price.

| Delisting rule | CAGR | Sharpe | Max drawdown | Net per trade |
|---|---|---|---|---|
| Master's −30% on every delisting | −8.0% | −0.77 | −29% | −0.9% |
| 0% for acquisitions, −30% distressed | −0.3% | +0.01 | −11% | −0.2% |

That research classified acquisitions from its own data, not from Alpaca's merger records.

**Coverage, checked 2026-10-08** against the research store's full Alpaca corporate-action download
(2016-2024, `var/alpha/external/corporate_actions.parquet` on the alpha-discovery side):

- **9 of the 13 deals have a merger record:** CSPR, TPTX, BHVN, TWTR, CCXI, ALBO, PRVB, BLU, NWLI. All
  are 2022-24.
- **4 have none:** CORI, ARII, DOVA (2018-19) and AKUS (2022).
- **Alpaca's merger records barely exist before 2020:** 2 records in 2017 and 36 in 2019, against
  668-972 a year in 2020-24.

So this change fixes most takeovers going forward, which is what the live bot sees. A backtest
reaching back before 2020 still books most early takeovers at −30%. The full effect in the table
above needs a complete merger history.

### Fix
1. **Provider** (`data/providers/alpaca_data.py`). Master already requested `cash_merger`,
   `stock_merger` and `stock_and_cash_merger`, then rejected them as `unmapped_type`. They are now
   mapped under their own `action_type`:
   - `symbol` = `acquiree_symbol`;
   - `ex_date` = `effective_date`;
   - `amount` = cash rate and `ratio` = acquirer/acquiree rate, both for audit only.

   The mapping is gated by `providers.alpaca.map_spin_offs_and_mergers`. It is `true` in
   `config/default.yaml`, and the code default `false` keeps the old behaviour. The gate exists
   because the existing test `test_unmapped_types_are_counted_and_logged_not_dropped` pins the old
   behaviour for a config without the flag, and tests are never weakened. All other types (stock
   dividends, unit splits, name changes...) are still counted and logged as unmapped.
2. **Panel** (`data/panel.py::build_panel`). A new bool field `merger` is True on the first
   CALENDAR session >= the effective date. A calendar session is used because the acquiree has no
   later bar. A record dated after the panel's last session is not placed.
3. **Rule** (`core/costs.py::CostModel.delisting_exit_return`).
   - If a merger takes effect on a session in [last bar − `costs.delisting_merger_lookback_sessions`
     (5), delisting session], the exit is booked at last close x (1 + `costs.delisting_return_merger`),
     default 0.0.
   - Otherwise the exit stays at −30%, the conservative default.
   - Only rows <= the delisting session are read (point-in-time). A merger dated after the booking
     session is never used.
4. Used by both the engine and `simulate_plan`, so the backtester and shadow outcomes agree.
   `diagnostics["delistings_merger"]` counts the merger exits.
5. The shadow tracker gets the merger information without any pipeline change. `pipeline/daily.py`
   hands it a bare `Panel`, so the information has to live on the panel: carrying it as a
   PIT-truncating panel field is what makes that possible.

## 2. Spin-off days are booked as raw price drops

### Problem
`build_panel` builds `ret` and the tri-scaled prices from RAW bars plus splits and cash dividends.
On a spin-off ex-date the parent's raw drop is booked as a real return. Example: NVS −12% on
2019-04-09, when Alcon was spun off; holders lost nothing. In the bot this causes:

- fake losses, so stops can fire. `execution/exits.py` compares stops with `aclose`.
- fake mean-reversion and extreme-reversal signals;
- distorted momentum and volatility features;
- distorted shadow outcomes.

### Fix
1. **Provider.** `spin_offs` records are mapped (same flag) to `action_type="spin_off"` on the parent
   (`source_symbol`, `ex_date`). `ratio` = new shares per parent share, for audit only.
2. **Panel.** The record is placed with the existing `_effective_action_dates`, on the first traded
   session >= ex_date. On that session a **negative** `ret` is set to 0. This is a NEUTRAL step, as if
   a distribution of equal value had been paid. A positive `ret` is left unchanged.
   - `tri` and `aopen/ahigh/alow/aclose` follow from `ret` automatically.
   - Raw OHLCV, `dividend` and `split_ratio` are untouched. Level logic and the paper ledger
     (`execution/ledger.py` reads `dividend`/`split_ratio`) see no change.
   - A bool field `spin_off` marks the sessions where the step was applied.
   - This is an approximation, and the `build_panel` docstring says so. The exact value of the
     distributed shares is unknown, and the parent's own move that session is lost with the drop.
3. **Backtester** (`backtest/engine.py`). The engine's net return comes from RAW cash flows, so it
   needed one more change. On a flagged session it credits the distributed shares like a cash
   dividend: per pre-split share, previous close − (close x split + dividend), counted in `dividends`.
   - The credit uses that session's close, so it is booked at the close step.
   - A lot sold at that session's open keeps its entitlement. The credit is booked to its trade, and
     the cash arrives at the close, so entries at that open never see cash derived from that close.
   - Without this credit the engine would still book the −12% while `simulate_plan` would not.
   - With it, the two differ only by the same second-order terms as for a cash dividend. The engine
     holds the credit as cash. The tri path reinvests it in the stock and charges the sell cost on it.
     The tests pin both exact formulas and this bound.

## What is NOT covered (deliberately or by limitation)

**Code left unchanged**
- **ML labels** (`ml/dataset.py::build_labels`, `ml/monitor.py`) still use −30% for every delisting.
  The merger information is now on the panel (`panel.merger`), so this is a small follow-up. But it
  changes the training labels of an enabled model (`ml.enabled: true`), so it needs its own
  before/after comparison.
- **Simulated broker / exit engine settlement** (`execution/sim_broker.py`, `execution/exits.py`)
  still settles a delisting at −30%. This affects the human lab's SimBroker book. These are
  execution-layer modules and were left alone on purpose. `quantlab paper start` trades on Alpaca
  paper, where Alpaca itself settles real corporate actions.
- **The bot's real paper ledger P&L** on a spin-off is unchanged: what Alpaca paper does with
  spin-offs is UNKNOWN (docs/EXTERNAL-SERVICES.md).

**Records the vendor does not have**
- **Spin-offs missing from Alpaca's list** are still booked raw. The list has about 120 spin-off
  records for 2016-24, and the sprint found it misses most spin-offs.
- Nothing in master reports unexplained moves of spin-off size. `data/validation.py` (`extreme_returns`)
  warns only at |ret| > `validation.data.extreme_return` (0.8) without an earnings event.
  `data/audit.py` reconciles splits and dividends only. Thresholds were not changed.
- The alpha-discovery branch flags such days with a price rule (`alpha/ca_fixes.py`). That rule was
  not ported.
- **Missing or late merger records.** Alpaca does not guarantee when records appear. A record that
  arrives after the delisting rule fires does not help: shadow outcome rows are write-once (DB
  triggers), so rows already written at −30% stay. The runner's incremental ingest fetches only the
  last `paper.runner.ingest_lookback_days` (10) days, so a record published later is never fetched
  by it.

**Not modelled**
- **Distressed delistings with a merger record**, for example an asset sale of a failing company,
  would get 0%. The window (last bar − 5 to the delisting session) limits but does not remove this.
- **Ticker reuse after a merger** (BHVN) and **deal terms** (the cash rate is stored but not used)
  are not modelled.

**Re-ingest needed for history.** The corporate-action datasets stored on the bot's PC were fetched
without this mapping, so the merger and spin-off records were dropped. For history, re-ingest:
`.venv\Scripts\python -m quantlab.cli ingest --kinds corporate_actions --start 2016-01-01 --end <today>`.
Run it from the repo root. A large symbol list is fetched market-wide. Loading de-duplicates
corporate actions on (symbol, ex_date, action_type, source_id), latest retrieval wins
(`data/store.py`), so overlapping windows never double-count a dividend.

## Test evidence
Offline synthetic suite, `python -m pytest`:

| Run | Result |
|---|---|
| Baseline, pristine `origin/master` (2788eb7) | 963 passed, 0 failed |
| This branch | 985 passed, 0 failed (963 unchanged + 22 new) |

No existing test was modified.

New tests:
- `tests/backtest/test_engine_delisting_merger.py`:
  - a merger within the window exits at the merger return (config value honoured; default 0.0 means
    the last close);
  - no record keeps −30%;
  - a merger dated **after** the delisting session is not used (PIT);
  - a merger before the lookback is not used;
  - engine and `simulate_plan` agree on every case;
  - weekend effective dates go to the next calendar session;
  - the `merger` field is truncation-invariant (`assert_truncation_invariant` on a rebuilt panel).
- `tests/data/test_panel_spin_offs.py`:
  - a raw −12% with a record gives `ret == 0` and continuous `aclose`, with raw fields, dividend and
    split untouched;
  - without a record it stays −12%;
  - a positive move on the date is unchanged;
  - the record only touches its own symbol;
  - truncation invariance of `build_panel`;
  - with no record, engine and `simulate_plan` are identical (fake stop-out);
  - with a record, both are neutral (no fake stop), with exact formulas and the second-order bound.
- `tests/data/test_alpaca_spin_offs_mergers.py`:
  - flag on: mapped under their own types (never as split/dividend), acquiree and effective_date
    used, malformed record rejected, other types still counted unmapped;
  - flag off or absent: everything stays unmapped;
  - provider output feeds `build_panel`.

## Expected effect on the live bot
- **Shadow outcomes** (EV statistics, LLM monitor): new outcome rows for symbols that delist with a
  merger record exit at the last close instead of −30%. Existing rows are immutable. How often this
  happens in the bot's shadow book is **UNKNOWN**. The research above suggests most delistings of the
  liquid names it tracks are acquisitions.
- **Spin-off sessions** (only those Alpaca lists, after re-ingest):
  - features and signals no longer see the fake drop;
  - the exit engine's stop no longer fires on it;
  - shadow outcomes are neutral on that day.

  The number of such sessions in the bot's universe is **UNKNOWN**. The sprint's different,
  price-based rule found about 25 fake in-universe days in nine years. Signals on those names can
  change for the bundle window (900 days).
- **Backtests** change only where these records exist. The config hash changes because of the new
  keys.
- **Memory:** two bool frames (sessions x symbols, 1 byte per cell). For the runner's 900-day bundle,
  that is about 0.6 MB per 1,000 symbols per frame.
- **Rollback:**
  - For mergers, set `costs.delisting_return_merger: -0.30`.
  - For new ingests, set `providers.alpaca.map_spin_offs_and_mergers: false`. Records already stored
    stay.
  - The spin-off step has no config switch. Disabling it means reverting the commit or using a
    corporate-action snapshot without those records.
