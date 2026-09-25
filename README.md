# QuantLab

A research lab for US equities, an autonomous **paper** trader, and (planned) a human paper-trading
lab. **No live money.** The only broker endpoint in the code is `https://paper-api.alpaca.markets`.
Config validation enforces this, the broker client enforces it, and a repository-scan test checks it.

The system is built to find out whether any edge exists. It is allowed to conclude that none does.
Contracts are in [ARCHITECTURE.md](ARCHITECTURE.md). Verified external-API facts are in
[docs/EXTERNAL-SERVICES.md](docs/EXTERNAL-SERVICES.md).

## Status (2026-09-25)

| area | state |
|---|---|
| Data: store, PIT panel, validation, ingest | working + tested |
| Providers: Alpaca (bars / corporate actions / news), SEC EDGAR (8-K timing, fundamentals), Nasdaq Trader (reference) | implemented + tested with fake HTTP. **Not yet run against real APIs** (needs keys, see below) |
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
| Walk-forward runner, ML/AI-filtered baselines | **not implemented** |

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
.venv\Scripts\python -m quantlab.cli run-daily               # one paper-pipeline session (latest)
.venv\Scripts\python -m quantlab.cli replay --start 2019-10-01 --end 2019-12-20
.venv\Scripts\python -m quantlab.cli strategy list|evaluate|promote|status ...
.venv\Scripts\python -m quantlab.cli status | pause --reason ... | resume --reason ... --actor human:you
.venv\Scripts\python -m quantlab.cli dashboard               # http://127.0.0.1:8765
.venv\Scripts\python -m pytest                               # test suite (offline, synthetic data)
```

Strategies start as SHADOW. They record every decision but place no orders. A strategy places
paper orders only when it is ACTIVE at stage PAPER or PROMOTED **and** its validated history
passes the EV gate. Promotion needs evidence; a human override is possible but is logged
permanently. Synthetic backtests never feed the EV gate unless `decision.stats.allow_synthetic`
is set explicitly, which is for tests and demos only.
