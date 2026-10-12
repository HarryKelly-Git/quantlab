# Should the bot take more risk? (2026-10-08)

Harry asked for more risk. Here is the arithmetic of compounding, from data, not opinion.

**The rule (Kelly).** Long-run growth ≈ exposure × expected excess return − ½ × exposure² × variance.

- Growth peaks at the "Kelly" exposure = expected excess return / variance.
- Beyond it, more risk LOWERS growth.
- With zero expected edge, the best exposure is zero: more risk only adds swings.

## 1. The bot's own picks
On survivorship-free 2016-24 data, the bot's book made the following:

| Period | Return/yr | Sharpe |
|---|---|---|
| 2016-19 | +1.6% | 0.25 |
| 2020-21 | +0.1% | 0.06 |
| 2022-24 | −0.3% | 0.01 |

Expected edge ≈ 0, so its Kelly exposure ≈ 0. Bigger positions would multiply its swings without
raising its expected return. In 2022-24, bot plus SPY made 4.8% a year against SPY's 8.8%
(`research/alpha/results/improvement/B_bot.json`).

## 2. The one asset with a proven premium: the US market (Ken French daily, 1963-2024)

| Exposure (rebalanced daily, borrowing at T-bills + 1%) | Return/yr | Worst fall | 1987 crash day | 2007-09 |
|---|---|---|---|---|
| 1.0x (just hold) | 10.7% | −55% | −17% | −55% |
| 1.5x | 12.3% | −73% | −26% | −72% |
| 2.0x | 13.1% | −87% | −35% | −84% |
| 3.0x | 12.4% | −98% | −52% | −95% |

**The formula's answer depends heavily on the period:**

| Period | Kelly exposure |
|---|---|
| 1963-2024 | 2.7x |
| 1963-99 | 3.8x |
| 2013-24 | 4.6x |
| 2000-12 | **0.5x** (any leverage lost money) |

**In practice:**
- Beyond about 1.5x, extra risk adds little return and brings near-wipeout drawdowns.
- Estimation error and fat tails push the safe level lower still.
- Half-Kelly or less, about 1.2-1.5x, is where professionals stop.

## 3. What "risk more" should mean, ranked by evidence

1. **Not** bigger bets on the bot's current strategies (no edge to scale).
2. **More time in the market for money that is otherwise idle.**
   - The bot holds about 66% cash.
   - In paper this is a learning choice.
   - In a real account, idle cash should at least earn T-bills: +2.5 points a year in 2022-24 (B5a).
3. **Moderate market leverage switched on only in uptrends.** M6 (2x when the market is above its
   10-month average) returned 18.5% a year in 2013-24 against 14.5%, with a −36% worst fall against −25%.
   Higher return, higher risk, no better per unit of risk.
4. **Any real-money leverage decision** follows the Upside Engine v2 doctrine; this note is paper
   research.

Changing the bot's risk limits (`risk.*`, `paper.*` in the config) is Harry's decision on the PC. These
numbers are the input to that decision, not a change.
