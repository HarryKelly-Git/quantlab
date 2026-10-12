# Dip buying: pre-registration (2026-10-08)

Committed BEFORE any result was computed. PAPER research. Nothing from 2025 on is used.

**Why.** Harry has done well buying dips and asked for it to be explored properly. The bot already has
two dip-type strategies:

| Strategy | 2022-24 Sharpe (survivorship-free replay) |
|---|---|
| mean_reversion | −0.51 |
| extreme_reversal | −0.75 |

Short-term reversal also failed (H01). This study tests the dip-buying forms with the best published
prior, judged by Harry's own yardstick: money compounding per unit of time.

**The yardstick: return per unit of time, done correctly.**
- **Primary:** each rule is run as a portfolio with realistic capital limits, and judged by CAGR, i.e.
  how fast the money compounds.
- **Per-trade "return per day held"** is also reported. It cannot be the verdict on its own: a +2% day
  "annualises" to +14,000%, and capital cannot be re-deployed instantly. Portfolio CAGR is the honest
  version of "10% in a week beats 20% in a month".

**Splits:**

| Data | TRAIN | VAL | OOS |
|---|---|---|---|
| Ken French daily market | 1963-07..1999 | 2000-12 | 2013-24 |
| Survivorship-free stock store | 2016-19 | 2020-21 | 2022-24 |

## Part I. Index dips (Ken French daily US market, total return)

Cash earns the 1-month T-bill rate. Costs are 5 bps per unit of weight traded. Borrowing costs RF + 1%/yr.

| ID | Rule |
|---|---|
| I1 | **RSI(2) dip in an uptrend (Connors).** At a close where the market is above its 200-day average and RSI(2) < 10, go 100% market at that close (market-on-close). Exit at the first close above the 5-day average, or after 10 days. Otherwise T-bills. |
| I1-lag | I1 entered one day later. A robustness check, reported, not a separate verdict. |
| I2 | **Buy-and-hold plus extra on dips.** Always 100% market. While an I1 signal is active, hold 200% (the extra 100% is borrowed). |
| I3 | **Buy-and-hold plus extra after a correction.** Always 100% market. For 252 trading days after the market first closes 10% or more below its prior peak, hold 150%. The window resets once a new peak is made. |

**Verdicts, against buy-and-hold:**
- **I1** passes on return per unit of RISK if its Sharpe beats BH in TRAIN, VAL and OOS with the OOS 90%
  12-month-block CI above 0. Its CAGR is also reported. I1 holds the market only 10-20% of the time, so its
  CAGR will be lower; its value would be as a component.
- **I2 and I3** pass as **ABSOLUTE IMPROVEMENT** if CAGR beats BH in all three splits and the OOS Sharpe is
  not lower. Otherwise **HIGHER RETURN, HIGHER RISK** if the OOS CAGR is higher, else **NO IMPROVEMENT**.

## Part II. Single-stock dips (survivorship-free store, liquid universe, corrected delisting)

**Signal, at a close t, for a stock in the liquid universe, all known at t:**
- above its 200-day average;
- its 52-week high, as of 10 sessions earlier, within 10% of that day's close: a strong stock;
- down 8% or more over the last 5 sessions: the dip.

**Trade:**
- Buy at the next open.
- Exit at the first close at or above the close 5 sessions before the signal (the dip recovered), or
  after 20 sessions, whichever comes first. No stop.
- Costs: master's tiers, the round trip by liquidity (as in the event study).

**Variants:**

| ID | Universe |
|---|---|
| S1 | All liquid stocks |
| S2 | The top 200 by 60-day median dollar volume (large caps, like VST or UBER) |

**Portfolio:**
- Up to 10 positions, each 10% of equity at entry.
- If more signals arrive than free slots, the deepest dips are taken first.
- Unused cash earns T-bills.

**Verdicts, against SPY buy-and-hold:**

| Verdict | Condition |
|---|---|
| **BEATS SPY** | Portfolio CAGR above SPY in TRAIN, VAL and OOS, and Sharpe above SPY with the OOS 90% CI above 0 |
| **BETTER RISK-ADJUSTED ONLY** | The Sharpe conditions hold without the CAGR condition |
| **FAILS** | Otherwise |

**Also reported:**
- per-trade mean excess return vs SPY, with date-clustered t;
- hit rate;
- average days held;
- return per day held;
- worst trade;
- results by year.

## Multiple testing

Part I has three rules and Part II has two. Any pass is a candidate for forward paper shadowing, never
proof.

---

## Results (added 2026-10-08, after the runs; the rules above were not changed)

Files: `research/alpha/results/dips/{I_index,S_stocks}.json`; ledger `DIP_I1`, `DIP_I2`, `DIP_I3`, `DIP_S1`, `DIP_S2`.

**Part I: index dips** (Ken French daily market; buy-and-hold is 12.6% / 2.2% / 14.5% a year in the three
splits):

| Rule | 1963-99 return/yr | 2000-12 | 2013-24 | 2013-24 worst fall | Time in market | Verdict |
|---|---|---|---|---|---|---|
| I1 RSI(2) dip, cash otherwise | 3.9% | 3.1% | 4.3% | −9% | 10-14% | FAILS |
| I1, entered a day late | 6.8% | 4.8% | 3.7% | −10% | | (robustness) |
| I2 hold + 2x during dips | 9.6% | 2.6% | **17.2%** | −38% | | HIGHER RETURN, HIGHER RISK |
| I3 hold + 1.5x for a year after a 10% correction | 12.7% | −1.0% | 16.5% | −45% | | HIGHER RETURN, HIGHER RISK |

Per-trade I1 results:

| Period | Trades | Mean per trade | Winners | Average hold | Per day held |
|---|---|---|---|---|---|
| 2013-24 | 100 | +0.37% | 73% | 3.4 days | +0.11% (about twice buy-and-hold's daily rate) |
| 1963-99 | 324 | −0.13% | | | |

**Index dip-buying is an era effect.** It paid well in the recent bull market and lost in 1963-99. Adding
leverage on dips raised 2013-24 returns but lost to plain holding in 1963-99 (I2) or 2000-12 (I3).

**Part II: single-stock dips** (stocks in an uptrend that fell 8% or more in 5 days; buy next open, exit
on recovery or after 20 days):

| Variant | 2016-19 return/yr | 2020-21 | 2022-24 | Per-trade excess vs SPY, 2022-24 | Recovered within 20 days | Verdict |
|---|---|---|---|---|---|---|
| S1 all liquid | 2.6% | 16.4% | **−10.4%** | −1.1% (t −1.2) | ~20% | FAILS |
| S2 large caps | 4.1% | 8.6% | **−6.5%** | −1.4% (t −2.1) | ~25-35% | FAILS |
| SPY | 14.5% | 23.4% | 8.9% | | | |

**Sharp dips in individual strong stocks did not recover often enough to pay.** This matches the bot's
own dip strategies (mean_reversion −0.51 and extreme_reversal −0.75 Sharpe in 2022-24). A dip buyer's
success in recent years is better explained by buying while the whole market rose. The index-level
results show that effect, and its dependence on the era.
