# Intraday news reaction: pre-registration (2026-10-08)

Committed BEFORE any intraday data was downloaded. PAPER research, 2017-2024 only; the 2025+ holdout
stays sealed.

## Question

Vistra's +10.8% on 2026-10-06 happened DURING the session. The Google-Constellation headline came at
9:32am ET, and VST closed far above its open.

The daily event study showed that buying at the NEXT open after such days loses money. This study tests
the remaining window. When a substantive single-stock headline arrives during regular hours and the
stock has already reacted strongly within 15 minutes, does the move continue to the close or the next
close, after realistic intraday costs?

## Data (fixed)

- **Days:** a seeded random 25% of the 2017-01-01 .. 2024-12-31 trading days (numpy seed 7), stratified
  by year.
- **News:** every Alpaca/Benzinga article created on those days.
- **Qualifying headline:**
  - created 09:30-15:00 ET;
  - substantive (not a price recap, using the event study's RECAP regex);
  - not an M&A-target headline (the event study's ma_target regex);
  - tagged with 1 or 2 symbols (roundups excluded);
  - the stock was in the research liquid universe at the prior close;
  - only the first qualifying headline per stock per day counts.
- **Prices:** SIP 1-minute bars (raw prices; nothing splits intraday) for the headline stocks and SPY,
  09:20-16:01 ET. SIP daily closes are used for the next-day exit.

## Measurement

**Prices:**
- τ = `created_at` rounded up to the minute.
- P0 = close of the last 1-minute bar before τ, within 10 minutes; otherwise skip (stale).
- P15 = close of the last bar in [τ+10, τ+15] minutes; otherwise skip.
- Reaction r0 = P15 / P0 − 1.

**Signal:** |r0| >= 2%, direction s = sign(r0).

**Trade:** enter at P15.

| Exit | Price |
|---|---|
| I1 | the last regular-session 1-minute bar's close |
| I2 | the next session's close |

**Return:** s x (P_exit/P15 − 1) − s x (SPY over the same window) − cost.

**Cost:** round trip = 2 x (2 x master's half-spread tier by MDV20 + 5 bps). Half-spreads are doubled
because spreads widen in the minutes after news.

## Hypotheses and decision

| ID | Test | Role |
|---|---|---|
| IN1 | LONG after r0 >= +2%, exit at the close: mean net excess > 0 | primary; the bot is long-only |
| IN2 | LONG after r0 >= +2%, exit at the next close: mean net excess > 0 | primary |
| IN3 | SHORT after r0 <= −2% (I1 and I2) | descriptive |

**Splits:** TRAIN 2017-19, VAL 2020-21, OOS 2022-24.

**Statistics:** date-clustered t. With 2 primary tests, significance means t >= 2.5.

| Verdict | Condition |
|---|---|
| SURVIVES | TRAIN and VAL > 0 with t >= 2.5, AND OOS > 0 with t >= 2.5 |
| PROMISING (forward paper-shadow only) | > 0 in all three splits, but weaker |
| FAILS | Otherwise |

**Descriptive only:** reaction deciles, time-of-day, category (event study's keyword classes), year.

Thresholds are not tuned. A change is a new labelled variant reported next to the original.

## Caveat

**Benzinga timestamps often trail the original wire.** Bloomberg broke the Vistra loan story before
Benzinga relayed it. The measured reaction r0 may therefore already contain the first move; that is the
realistic position for a bot reading this feed.
