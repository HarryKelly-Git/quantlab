# QuantLab: project instructions

US-equity research lab + autonomous PAPER trader + human paper-trading lab. Contracts are in
[ARCHITECTURE.md](ARCHITECTURE.md). Verified external API facts are in
[docs/EXTERNAL-SERVICES.md](docs/EXTERNAL-SERVICES.md). Read both before changing a subsystem.

## Absolute rules
- **No live money.** Never add a live-trading code path, live endpoint, or "live mode" flag. The only
  broker URL is `https://paper-api.alpaca.markets` (enforced in `config.py` and in the broker client).
- **Never fabricate** data, backtest results, fills or LLM outputs. Missing means `UNKNOWN`. Failure
  means `NO TRADE` / `SYSTEM_PAUSED`.
- **Point-in-time.** Obey the rules in ARCHITECTURE.md section 2. Any new feature, signal, universe rule or ML
  input needs a truncation-invariance test (`quantlab.testing.pit.assert_truncation_invariant`).
- **Synthetic data is labeled** (`is_synthetic`) and never presented as market evidence.
- **Never weaken or delete a test** to make code pass. Never delete failed experiments, rejected
  opportunities, human decisions or ledger entries (DB triggers enforce this).
- **Secrets** come only from env vars / the git-ignored `.env`. Never log, print or commit them.
- **Complexity must be earned.** A new component is a hypothesis. Keep it only if it beats the
  simpler baseline out-of-sample or forward. Record the comparison in the research ledger.

## Relationship to the parent doctrine
`../CLAUDE.md` (Upside Engine v2) governs **real-money** stock verdicts for Harry. QuantLab outputs
are paper-trading research: hypotheses, not BUY verdicts. If QuantLab output is used to answer "should
I buy X with real money", the v2 doctrine applies in full.

## Commands (Windows, from repo root)
```
.venv\Scripts\python -m pytest                   # full test-suite (synthetic data, offline)
.venv\Scripts\python -m quantlab.cli --help      # CLI
```

## Conventions
Python 3.14, pandas 3 (copy-on-write, `stack(future_stack=True)`, no `inplace`), numpy 2, sklearn 1.9,
pydantic 2. Tunables go in `config/default.yaml`. Migrations are numbered SQL files in their
reserved ranges (ARCHITECTURE.md section 11). Log through `quantlab.logging_setup`.
