# QuantLab

A research lab for US equities, an autonomous **paper** trader, and (planned) a human paper-trading
lab. **No live money.** The only broker endpoint in the code is `https://paper-api.alpaca.markets`.
Config validation enforces this, the broker client enforces it, and a repository-scan test checks it.

The system is built to find out whether any edge exists. It is allowed to conclude that none does.
Contracts are in [ARCHITECTURE.md](ARCHITECTURE.md). Verified external-API facts are in
[docs/EXTERNAL-SERVICES.md](docs/EXTERNAL-SERVICES.md).

## Status (2026-09-25, updated after the walk-forward work)

| area | state |
|---|---|
| Data: store, PIT panel, validation, ingest | working + tested |
| Providers: Alpaca (bars / corporate actions / news), SEC EDGAR (8-K timing, fundamentals), Nasdaq Trader (reference) | working against the REAL APIs (2026-09-25). SIP feed confirmed. Spin-offs are NOT applied (Alpaca spin_off records unmapped): parent returns show a false drop on spin dates |
| Features: price, volume, relative, market, event (EAR), fundamental (as-of), news counts | working. Every feature passes a truncation (look-ahead) test |
| Universe, regime | working + tested |
| Strategies (9): momentum_trend, mean_reversion, breakout, relative_strength, extreme_reversal, sector_rotation, pead_ear, quality_momentum, news_shock (disabled) | working + look-ahead tested |
| Backtester, metrics, bootstrap / PSR / deflated Sharpe, locked holdout, experiment registry, reports | working + tested. A planted edge is found; a null world is not significant |
| Decision stack: EV (empirical, shrunk to a no-edge prior), no-trade rules, portfolio construction, risk chain, final decision | working + tested |
| Paper execution: SimBroker, Alpaca **paper** broker, ledger, exits, journal, reconciliation | working + tested (Alpaca paper not yet exercised live) |
| Daily paper pipeline, replay, shadow book + outcomes, daily report, kill switch | working end to end on synthetic data |
| Dashboard (read-only, localhost) | working + tested |
| ML (`quantlab/ml/`), AI layer (`quantlab/ai/`), research ledger, health / LLM monitors | **partial, untested, NOT wired in**. Do not rely on them |
| Human paper lab, counterfactual engine, human-vs-bot comparison | **not implemented** |
| Walk-forward OOS runner | working + tested, including a corrupt-the-future no-look-ahead test. First real run: `momentum_trend` on ~5,240 currently listed stocks, 4 OOS windows 2023-2024: **NOT_SIGNIFICANT** (see below) |
| Real-data audit (`data-audit`) | working. SPY/XLK/AAPL/MSFT 2020-01-02..2026-09-24: **SUITABLE_SMALL_SAMPLE_ONLY**. SIP feed (median SPY volume 70.9M shares/day), 0 missing NYSE sessions, splits and dividends verified, SEC 8-K timing PIT |
| ML/AI-filtered baselines | **not implemented** |

## Latest research result (real data, 2026-09-25)

`momentum_trend` v1.0.0 (fixed parameters, no search), walk-forward OOS 2023-01-18..2024-12-31, 4 windows:
155 trades, net +4.9% (gross +5.8%), expectancy -1.0% per trade (95% CI -5.9%..+3.5%), Sharpe 0.29,
max drawdown -11.0%, average exposure 27%. SPY over the same windows returned +48.1%; the strategy beat it
in 1 of 4 windows. The top 5 trades account for 274% of P&L. **Verdict: NOT_SIGNIFICANT.** No evidence of edge.
Caveats: the universe comes from a current listings snapshot (survivorship-biased, which flatters long-only results),
spin-offs are not applied, and window-end forced exits account for 26% of trades.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env      # then fill in the values yourself; never commit .env
```

`.env` needs `QUANTLAB_SEC_USER_AGENT` (a descriptive UA with your email) and Alpaca **paper** keys
(`ALPACA_PAPER_KEY_ID`, `ALPACA_PAPER_SECRET_KEY`). Alpaca's docs say "Paper Only" accounts (non-US)
get **IEX data only**. IEX is about 2.5% of volume, with history from 2020. The feed actually used is
recorded on every dataset. If only IEX is available, the dollar-volume liquidity filters are not
reliable.

## Commands

```
.venv\Scripts\python -m quantlab.cli demo                    # offline SYNTHETIC end-to-end backtest + report
.venv\Scripts\python -m quantlab.cli ingest --synthetic      # or: ingest --start 2020-01-01 (real providers)
.venv\Scripts\python -m quantlab.cli validate
.venv\Scripts\python -m quantlab.cli backtest --strategy momentum_trend
.venv\Scripts\python -m quantlab.cli data-audit                # REAL data check: SPY,XLK,AAPL,MSFT from 2020
.venv\Scripts\python -m quantlab.cli walkforward --strategy momentum_trend --data real
.venv\Scripts\python -m quantlab.cli run-daily               # one paper-pipeline session (latest)
.venv\Scripts\python -m quantlab.cli replay --start 2019-10-01 --end 2019-12-20
.venv\Scripts\python -m quantlab.cli strategy list|evaluate|promote|status ...
.venv\Scripts\python -m quantlab.cli status | pause --reason ... | resume --reason ... --actor human:you
.venv\Scripts\python -m quantlab.cli dashboard               # http://127.0.0.1:8765
.venv\Scripts\python -m pytest                               # test suite (offline, synthetic data)
```

## Alpaca PAPER runner (real-time paper trading)

```
.venv\Scripts\python -m quantlab.cli paper preflight   # check PAPER mode + account, no orders
.venv\Scripts\python -m quantlab.cli paper start       # the persistent runner (foreground; Ctrl+C stops it cleanly)
.venv\Scripts\python -m quantlab.cli paper status      # RUNNING / STALE / STOPPED, heartbeat, phase, last job
.venv\Scripts\python -m quantlab.cli paper stop        # clean stop from another terminal
.venv\Scripts\python -m quantlab.cli dashboard         # then open http://127.0.0.1:8765/live
.venv\Scripts\python -m quantlab.cli paper order-test --confirm   # optional plumbing check, see below
```

Requirements: `.env` must contain `TRADING_MODE=PAPER` and `LIVE_TRADING=false` exactly (a missing or
different value refuses to start; there is no live mode), plus the Alpaca **paper** keys. The paper
account must be dedicated to QuantLab: no positions and no open orders that QuantLab did not
create. On first start the BOT ledger is bound to the Alpaca paper account and seeded with its
cash, so the BOT book in this database can no longer be driven by the simulated broker
(`run-daily`/`replay`). Use a separate `project.db_path` for simulated replays.

What it does:

* **Startup.** Preflight, recorded in `paper_preflights`: environment, paper endpoint, account ACTIVE
  and unblocked. Then a single-instance session with a heartbeat, the market calendar from Alpaca
  (cross-checked with the NYSE rules), reconciliation, and the `trade_updates` websocket.
* **After each close (D).** From 19:05 ET it ingests D's bars and corporate actions, reconciles, and
  runs the existing daily pipeline: validation, fills and marks, exit rules, features, strategies,
  candidates, EV/no-trade/portfolio/risk gates, then orders. Orders are `opg` market-on-open orders
  for the next session's opening auction. Alpaca rejects `opg` orders between 09:28 and 19:00 ET.
* **Missed deadline.** If D is processed after 09:25 ET on the next session (for example, the runner
  was down), it still records candidates and decisions, but refuses every order ("execution window
  missed").
* **During the session.** Fills arrive on the stream and are applied to the ledger. REST polling is
  the fallback. Everything is reconciled against the broker.
* **Restarts.** A restart resumes the same pipeline run for D: same candidates, same deterministic
  `client_order_id`s. Before submitting, the runner asks the broker whether it already has that id.
  A restart or reconnect therefore cannot duplicate an order.
* **Kill switch.** `SYSTEM_PAUSED` stays authoritative. The runner pauses trading on stale or invalid
  data, an unverifiable broker, a reconciliation mismatch, an unknown broker order, a pipeline
  failure or an unexpected error. It keeps running so orders and positions stay monitored. Resume
  requires a human (`resume --actor human:you` or the dashboard's system page).
* **No eligible strategy.** If no strategy is ACTIVE at stage PAPER/PROMOTED, the runner stays in
  SHADOW, shows **NO PAPER-ELIGIBLE STRATEGY**, and places no orders.

`/live` on the dashboard re-renders every 15 s. It shows:

* **System:** runner state, heartbeat, session, stream, data freshness, last successful pipeline,
  strategy status.
* **Signals:** candidates with TRADE/REJECT, rejection stage and reason.
* **Orders:** recent paper orders with broker ids and fill times/prices.
* **Portfolio:** broker vs ledger cash/equity, positions, P/L.
* **Performance:** paper equity curve vs SPY, expectancy, drawdown.
* **Audit:** errors, kill-switch events, data-quality issues, a decision trace.

It is read-only. The only writes are pause/resume on `/system`.

`paper order-test --confirm` places **one** non-marketable paper order: 1 share BUY limit at half
the last close, cancelled immediately. It checks the order and websocket path against the real
paper account. It is not a strategy order, never touches the ledger, and is refused while
`SYSTEM_PAUSED`.

Strategies start as SHADOW. They record every decision but place no orders. A strategy places
paper orders only when it is ACTIVE at stage PAPER or PROMOTED **and** its validated history
passes the EV gate. Promotion needs evidence; a human override is possible but is logged
permanently. Synthetic backtests never feed the EV gate unless `decision.stats.allow_synthetic`
is set explicitly, which is for tests and demos only.
