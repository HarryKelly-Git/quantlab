# Momentum breakout: pre-registered research plan

Status: **PRE-REGISTERED. Written 2026-10-06, before any outcome was computed.** No price data had
been downloaded when this was written. Any change after the first outcome must be added as a dated
amendment at the bottom, saying why, and it counts as a new variant in the multiple-testing count.
Nothing above the amendments section may be edited once Stage 2 starts.

Branch: `momentum-breakout` (from `master`). Master's paper bot is Strategy 0 and is not touched.
PAPER ONLY. This is research into whether an edge exists. It is allowed to conclude that none does.

## 0. Prior: what we already know

Everything QuantLab has tested so far lost to holding SPY after costs (docs/REAL-MONEY-READINESS.md).
That includes a daily "close above the 55-day high after contraction" breakout family, which was
flat. So the prior for this idea is **low**. The new parts are the market filter, the strength
filters and an intraday entry (5-minute opening-range breakout) with a tight stop. Those parts
change the payoff shape, so the idea is worth one clean test. One test, not a search.

## 1. The question

On days when the market is in an uptrend, does buying strong US stocks with a tight setup, on a
5-minute opening-range breakout, make money after costs, and more than (a) random entries with the
same mechanics, (b) the same entry on momentum stocks without the setup filters, and (c) holding
SPY or QQQ?

Stock level only. Options are a later stage and happen only if the stock level passes (section 9).

## 2. Data and periods

| Item | Rule |
|---|---|
| Daily bars | Alpaca `adjustment=raw`, `feed=sip` (fallback `iex` is recorded per row, as on master). Returns from QuantLab's own PIT panel (`data/panel.py`). |
| Minute bars | Alpaca 1-minute SIP bars, 09:30-16:00 ET, **only for (symbol, session) pairs where the daily setup fired the evening before** (core set, section 4) plus the random-entry sample (section 6). Nothing else is downloaded. |
| Development (all choices) | 2016-01-04 .. 2021-12-31 |
| Out-of-sample (touched once, frozen config only) | 2022-01-03 .. 2024-12-31 |
| 2025-01-01 onward | **Not used.** That holdout has been read twice already, so it is no longer a fresh test. |
| Final test | Forward paper trading with the frozen rule (section 8). |

Known data weaknesses, stated up front:
- **Survivorship bias.** Alpaca has no historical asset list. The symbol list is today's listings
  plus whatever delisted names we can still query. Dead momentum stocks will be under-represented,
  which flatters every long-only test. The random-entry baseline is drawn from the same list, so the
  comparison *between* them is less biased than either absolute number.
- **Sector map is today's** (`sectors.py`, ASSUMED_STATIC).
- **Feed terms.** Alpaca says Paper-Only accounts are entitled to IEX data only; master already uses
  SIP history. If SIP is refused, the run stops and reports it. It does not silently switch.
- Minute bars before 2020 come from the same SIP history. They are real, not approximated.

## 3. Universe (as of the close of session D)

Master's universe rules (`config/default.yaml: universe`): common stock on a main exchange,
raw close >= $5, history >= 260 sessions. One stricter rule for an intraday entry: **20-session
median raw dollar volume >= $20M** (below that, 5-minute bars are thin and the cost model's
cheapest tier does not apply).

## 4. Stage 1: the daily setup scanner (all values known at the close of D)

All return/ratio maths uses the tri-scaled fields (`aclose`, `ahigh`, `alow`), all level rules use
raw fields, per ARCHITECTURE.md section 2. Each metric is output as a number so it can be audited.
Each boolean below is a separate column so the ladder can switch it on and off.

**Core (always on):**
- `regime_on`: SPY close > SMA10(SPY) **and** SMA10(SPY) > SMA20(SPY). *(Interpretation of Harry's
  "SPY 10/20 SMA rule". If Harry meant a different version, he must say so before Stage 2 starts.)*
- `momentum`: 63-session return (`aclose_D / aclose_{D-63} - 1`) is in the **top 5%** of the
  universe on D (cross-sectional percentile >= 0.95).

**Six add-on features (tested one at a time, in this fixed order):**
1. `sector_rs`: the stock's sector ETF has a higher 63-session return than SPY.
2. `stock_rs`: the stock's 63-session return beats its sector ETF's by at least 10 percentage points.
3. `impulse`: within the last 63 sessions, a leg up of **>= 30%** from a low to a later high
   (`max ahigh / min alow before it - 1`).
4. `consolidation`: the 63-session high was set **3 to 40 sessions ago**, and close >= 85% of it,
   and close >= SMA20.
5. `contraction`: ATR5% / ATR20% <= 0.75 (recent daily ranges are tighter than usual).
6. `volume_dryup`: 5-session average volume / 50-session average volume <= 0.80.

A stock "fires" for session D+1 when it is in the universe, `regime_on` is true, and every
switched-on condition is true at the close of D.

**Point-in-time proof:** every column must pass `assert_truncation_invariant` (computing on all
data and reading D gives the same value as computing on data that stops at D). That test is part of
Stage 1 and must pass before any outcome is computed.

## 5. Trade rules (Stage 2, fixed now)

**Entry (session D+1):** opening range = high and low of the 09:30-09:34 one-minute bars (first 5
minutes). Buy-stop at the opening-range high. Triggered if any minute bar from 09:35 to 15:30 trades
above it. Fill = max(that bar's open, OR high), plus costs. One entry per stock per setup. No entry
if the opening range has fewer than 3 bars with volume (thin open), or if OR high/OR low - 1 > 8%
(stop too wide). Missing minute data = NO TRADE, counted and reported.

**Initial stop:** the opening-range low.

**Three exits (all tested, none chosen afterwards by picking the best one on OOS):**
- X1 Intraday: stop at OR low during the entry session; otherwise sell at the entry-session close.
- X2 Trend trail: stop at OR low on the entry session. After that, stop stays at OR low; exit at the
  next open after the first daily close below SMA10. Max 60 sessions.
- X3 Partial: as X2, but sell half at the close of the 3rd session after entry, then move the stop
  on the rest to the entry price and trail with a close below SMA20. Max 60 sessions.

On later sessions, a stop hit is detected with the daily low; the fill is min(open, stop), so gaps
are paid in full. If the stop and the exit rule trigger on the same bar, the worse price is used.

**Costs:** master's `CostModel` (half-spread tier by 20-day median dollar volume + 5 bps slippage
per side), plus an extra **5 bps on the entry** because buy-stops fill worse than the touch. Every
result is also shown at 2x costs.

**Measurement unit:** net return per trade in %, and in R (R = entry minus initial stop).

## 6. The test ladder (development data 2016-2021 only)

1. **Step 0, core:** regime + momentum + ORB. This is also "momentum without consolidation".
2. **Steps 1-6:** add features 1-6 one at a time, in the order above. A feature is **kept** only if,
   with X2 as the reference exit, it (a) raises mean net R per trade versus the current ladder,
   (b) does so in at least 2 of the 3 two-year blocks (2016-17, 2018-19, 2020-21), and (c) leaves at
   least 150 development trades. Otherwise it is dropped and the next one is tested against the
   unchanged ladder. Exits X1 and X3 are reported for every step but do not drive the keep/drop.
3. The result is **one frozen rule** (the kept features). It is written to the amendments section
   with its git commit before the out-of-sample run.

Variants counted for multiple testing: 7 ladder steps x 3 exits = **21**, used in the Deflated
Sharpe Ratio. Any extra variant tried for any reason is added to this count.

**Baselines, all on the same sessions and with the same cost model:**
- SPY buy-and-hold and QQQ buy-and-hold.
- Random entries: for each signal session, the same number of stocks drawn at random from that
  day's universe (fixed seed 20261006), same ORB entry and exits. This tests whether the stock
  selection adds anything beyond the entry/exit mechanics.
- Momentum without consolidation = Step 0.

**Robustness, reported for every step:** results with the top 1%, 5% and 10% of winning trades
removed; results at 2x costs; each two-year block separately; walk-forward 6-month windows (rules
are fixed, so nothing is refitted; the windows show stability over time).

**Portfolio view** (for the vs-SPY comparison): start $100k, risk 0.5% of equity per trade on the
OR-low stop, position capped at 10% of equity, at most 10 open positions; when more trades trigger
than there are slots, take them in order of trigger time. This is the only sizing rule tested.

## 7. Pass / fail on out-of-sample 2022-2024 (frozen rule, one run)

The stock-level test **PASSES** only if all of these hold, with the pre-registered reference exit X2
(X1 and X3 are reported; if X2 fails and another exit passes, that is reported as a lead for forward
testing, not a pass):
1. At least 100 OOS trades (fewer = INCONCLUSIVE, not pass).
2. Mean net return per trade > 0 with a block-bootstrap (by date) t-stat >= 2.
3. Beats the random-entry baseline: the difference in mean net R per trade has a 95% bootstrap CI
   above zero.
4. Still positive after removing the top 5% of winners.
5. Portfolio view: total return beats SPY over 2022-2024, max drawdown no worse than SPY's, and
   positive in both 2022 and 2023-24.
6. Still positive at 2x costs.

Anything less is reported as FAIL or INCONCLUSIVE in plain language, with the numbers.

## 8. Forward paper test (the final test)

Only if section 7 passes. The frozen rule runs on Alpaca PAPER for at least 3 months and ~50 closed
trades, separately from master's bot. It passes if results fall inside the backtest's expected range
and beat SPY over the same window. Same bar as docs/REAL-MONEY-READINESS.md.

## 9. Options (later stage, only if the stock level passes)

Not designed yet. Any options result for dates before 2024 is labelled **APPROXIMATION** (no real
historical option quotes on this account; prices would be modelled).

## 10. How work is done

Plain rule modules in `src/quantlab/momentum_breakout/`, no LLM agents in the decision path. Each
step is committed and pushed. Results go in the research ledger and in this branch's docs, including
failures.

## Amendments

_(none yet)_
