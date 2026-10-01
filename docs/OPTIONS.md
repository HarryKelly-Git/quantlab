# Options layer (PAPER only)

`src/quantlab/options/` compares, for one thesis, buying the stock (with its stop) against a small
fixed grid of defined-risk option structures, records every evaluation, and can run a separate
PAPER options book `OPT`. Paper order submission is **OFF by default** (`options.paper_trading:
false`). Nothing here is a real-money recommendation; real-money questions go through the Upside
Engine v2 doctrine.

## Architecture

| module | role |
|---|---|
| `data.py` | `OptionsDataClient`: contracts (`paper-api /v2/options/contracts`), INDICATIVE snapshots (`/v1beta1/options/snapshots/{underlying}?feed=indicative`), historical daily option bars (`/v1beta1/options/bars`), underlying spot (IEX latest trade) and daily bars. Injectable HTTP, `get_secret` credentials, `next_page_token` pagination, shared rate limiters. GET only. |
| `liquidity.py` | Rejects: no quote, one-sided, crossed/locked, spread > `max_spread_pct` of mid, bid < `min_bid`, quote older than `max_quote_age_minutes` (or no timestamp), OI < `min_open_interest` when known, not tradable. Null OI is recorded `UNKNOWN` and fails only if `unknown_open_interest_fails`. Every failed rule is stored. |
| `structures.py` | Long call, long put, call debit spread, put debit spread. The constructor refuses any short leg not covered by a long leg that bounds its loss (`NakedShortError`), credit/ratio structures, mixed expiries, debit >= width. Entry at the executable side (buy at ask, sell at bid). Exact breakeven, max loss (= debit x multiplier), max gain, payoff at expiry; exit before expiry by Black-Scholes at an explicit IV (MODEL). |
| `pricing.py` | Black-Scholes price/greeks/IV. MODEL only: American exercise and dividends ignored, flat vol. Never used for a fill. |
| `distribution.py` | `MoveDistribution` (end returns + weights, optional path: max favourable / max adverse). `as_distribution(np.ndarray)` accepts a plain array, e.g. the 21 quantiles of `exploration.upside.move_distribution` (caveat: those are the stock trade's net returns after stop and costs, so using them as underlying moves double-charges the stock and hides moves beyond the stop from the options). `empirical_distribution` = overlapping windows of the underlying's own history (UNCONDITIONAL: assumes no edge). |
| `compare.py` | Expiry: first listed expiry >= `horizon + expiry_buffer_sessions` NYSE sessions. Strikes: nearest |delta| 0.70/0.50/0.30, ATM, +1 expected move (ATM IV x sqrt(h/252)), one spread long ATM / short +1 EM. Per expression: expected P&L per $ at risk, P(profit), P(reaching breakeven) (path-based when given), contractual max loss, expected loss given loss. Choice: STOCK if its E[P&L]/risk > `min_expected_pnl_per_risk`; a structure only if it beats max(stock, floor) by `min_option_advantage_per_risk`; else NO_TRADE. Fixed rules, not fitted. |
| `book.py` | Append-only recording; the OPT ledger (premium journal + broker-reported fills, multiplier on every fill); `cotenant_cash_offset`; OPT reconciliation (`us_option` positions, unknown `qlopt-` orders). |
| `execution.py` | Gated OPT paper executor: limit + day only (DB CHECK constraints too), buy at ask / sell at bid or `*_mid_fraction` toward mid, never better than mid; long leg first, short leg only for the filled long quantity; closes buy the short back first; caps (positions, contracts, premium per trade and total, allocation); kill switch blocks every OPT order; entries refused on/after the close-by date (`close_before_expiry_sessions` before expiry). |
| `marking.py` | Marks evaluations and OPT structures from historical option daily bars (real traded closes). A leg without a bar that session makes the mark `UNKNOWN`; never interpolated. Stock leg marked from the underlying's raw close, so stock and options outcomes are measured separately. |

Migration `055_options.sql` (execution range 050-059): `options_eval_runs`, `options_evaluations`
(one row per thesis x expression), `options_liquidity_checks`, `options_structures`(+events),
`options_orders`(+events), `options_fills`, `options_cash_events`, `options_refusals`,
`options_marks`. Journal/evaluation tables are append-only by trigger; state rows are never deleted.

CLI: `quantlab options evaluate --symbol X --horizon N [--direction LONG|SHORT] [--stop P]
[--returns-file F] [--paper --qty N]`, `quantlab options status`, `quantlab options mark`.

## Shared Alpaca paper account (integration with the stock runner)

The OPT book trades in the same paper account as the BOT book. Changes outside `options/`:
* `execution/reconcile.py`: `Reconciler(asset_classes=("us_equity",), cash_offset=...)`. Positions
  are filtered by Alpaca's `asset_class` (OCC-symbol fallback when absent); the other books' net cash
  is added to the ledger cash before the cash check. Defaults keep the old behaviour for stock-only
  accounts.
* `pipeline/runner.py` (4 lines): imports `OPT_ORDER_PREFIX, cotenant_cash_offset`; a `qlopt-` order
  on the trade stream or in the open-order list is not "unknown to QuantLab"; the reconciliation
  passes `cash_offset=cotenant_cash_offset(db, broker)`, which also counts broker fills on open OPT
  orders not yet applied by the OPT ledger (no race).
* Still a mismatch, by design: an exercise/assignment that creates a stock position, or any option
  fee the paper broker might charge (UNVERIFIED: the paper docs say regulatory fees are not
  simulated). Both pause the stock runner (fail safe). OPT never holds into expiry.

## What is measured

Per evaluation (forward, from today's indicative quotes): every expression's expected P&L per $ at
risk, P(profit), P(breakeven), max loss, E[loss | loss], the chosen expression and the reason, all
inputs (thesis, full distribution, settings, quote timestamps, feed) and every liquidity rejection.
Later, `marking.py` attaches what actually traded: daily marks of each evaluated structure and of the
stock, so "stock vs options" is measured on real traded prices going forward, separately per
expression. Marks use the day's last trade (close), which is not an executable bid/ask.

## Data gaps (verified 2026-10-01 on this account) and what would close them

| gap | consequence | closes it |
|---|---|---|
| Historical option QUOTES `/v1beta1/options/quotes` -> HTTP 404: no historical bid/ask or NBBO | No realistic historical options backtest. Nothing here backtests options; history is used only to mark forward evaluations. | A historical OPRA NBBO vendor: Databento (OPRA.PILLAR), ThetaData, Polygon.io/Massive options quotes, ORATS (historical EOD/intraday with quotes), Cboe DataShop. |
| Only `feed=indicative`; `feed=opra` -> HTTP 403 "OPRA agreement is not signed" | Quotes are Alpaca indicative values, not the OPRA NBBO; spreads/ages are approximate and labelled `feed='indicative'` everywhere. | Sign the OPRA agreement and take a paid Alpaca market-data plan with real-time OPRA (Algo Trader Plus). |
| Historical option bars only from Feb 2024 | Marking/learning only for evaluations after Feb 2024 (in practice: forward from now). | Same historical vendors as above. |
| Open interest is current-only and often null | No point-in-time OI; null OI = `UNKNOWN` (configurable to fail). | OI history from ORATS / Cboe DataShop / ThetaData. |
| Greeks/IV: present in the indicative snapshot for MOST contracts (TGT 2026-09-30: 916 of 1232); absent mainly for one-sided or deep ITM/OTM quotes | When absent, IV is computed from the quote MID by Black-Scholes for DESCRIPTION only (`iv_source=MODEL_FROM_MID`); if no IV exists the strike cannot enter the delta grid and a pre-expiry exit is `UNKNOWN` (structure ineligible). | OPRA data plus a vendor IV surface (ORATS) would replace the model. |
| Paper option fills | Alpaca paper fills marketable limits against its quote, ignoring size; option fees UNVERIFIED in paper; tick-size rules for limits UNVERIFIED (limits are rounded to $0.01; a sub-tick price may be rejected, which is recorded, never assumed filled). Whether option DAY orders are accepted outside market hours is UNVERIFIED. | Verify with a deliberately tiny paper order once `paper_trading` is enabled by the operator. |

Outside market hours every indicative quote is older than `max_quote_age_minutes` (30), so an
evening `options evaluate` rejects every contract as `QUOTE_STALE` and can only choose STOCK or
NO_TRADE. That is intended: stale quotes are never treated as executable.
