# Improvement program: brainstorm and Batch 1 pre-registration (2026-10-08)

PAPER research. Part B is committed BEFORE any Batch 1 result was computed. Nothing from 2025 on is used.

**Why a new program.** About 50 hypotheses failed. They were mostly about finding a NEW edge. This program
also asks a different question: what would make what Harry already has (the bot, and the ETF core)
measurably better? The single biggest drag found so far is in the bot's own book. In the replay, 69 of
1,325 trades lost more than 25% each; together they lost $46k, while all trades together made +$5.6k
(`research/alpha/results/new_areas/bot_tail_losses.json`).

## Part A. Brainstorm (ranked by expected value per unit of cost)

Scale used below:
- **Prior:** the chance it passes its bar, given the literature and this lab's 50 failures.
- **Value if it passes:** how much it changes Harry's outcomes.

### Batch 1 (pre-registered in Part B, runs now)

| # | Idea | Mechanism | Prior | Value if it passes | Cost |
|---|---|---|---|---|---|
| B1 | Bot skips stocks with < 1 year of trading history | IPOs and de-SPACs have the worst tail losses (SKYH, CXAI, AISP, CIIG) | Low-medium | Medium | 1 replay |
| B2 | Bot skips trades whose stop is > 20% away | A 3xATR stop on a wild stock is no stop (11 were at or below $0). Gaps go through it anyway | Low-medium | Medium | 1 replay |
| B3 | B1 + B2 together | Both target the blow-ups | Low-medium | Medium | 1 replay |
| B4 | Prune the bot to strategies that worked in 2016-21 | Drop strategies with negative early Sharpe | Low (selection is noisy) | Medium | 7 replays |
| B5 | Idle cash earns T-bill interest, or sits in SPY | The bot is 65% cash on average | Certain for T-bills (it is arithmetic); low for "beats SPY" | Real for a real account | Arithmetic |
| M1 | ETF core: profitability (quality) tilt | Novy-Marx 2013; one of the most robust factors | Low-medium after costs | High (real money, decades) | Ken French data |
| M2 | ETF core: value tilt | Fama-French 1992 | Low (value lost badly 2013-24) | High | same |
| M3 | ETF core: momentum tilt | Jegadeesh-Titman 1993 | Low-medium | High | same |
| M4 | ETF core: equal blend of M1-M3 | Diversified factors | Low-medium | High | same |
| M5 | ETF core: industry momentum (top 10 of 49 industries) | Moskowitz-Grinblatt 1999 | Low | Medium | same |
| M6 | ETF core: 2x market only when above its 10-month average | Leverage plus trend (Gayed-Bilello) | Low-medium (higher return, higher risk) | High, with real risk | same |

### Batch 2 (next; needs a design decision or data)

| # | Idea | Blocker | Prior |
|---|---|---|---|
| D1 | Insider buying clusters (SEC Form 4), the most robust public-information signal in the literature (Cohen-Malloy-Pomorski) | The SEC requires a contact email in the User-Agent. **Needs Harry's OK to use an email.** Master already has a Form 4 importer | Medium |
| D2 | Single-stock quality/value from SEC XBRL fundamentals, with filing dates (point-in-time) | Same SEC email | Low-medium |
| D3 | Post-earnings drift with a real point-in-time calendar | Calendar data (paid, or forward capture) | Low |
| X1 | Meta-labeling: a model that predicts which of the bot's own candidates blow up | Design; run after B1-B3 show whether simple rules capture it | Low-medium |
| X2 | Exit rules: one pre-registered variant (trailing stop at 2xATR after +1 ATR gain) | Design | Low |
| X3 | Bot as a "core plus satellite": 70% SPY plus the bot on the rest | Arithmetic, after B5 | Certain arithmetic; tells whether the bot subtracts value |
| O1 | Forward shadow of the straddle rule | Needs live option quotes daily; record-only | Low-medium |

### Considered and rejected (and why)

| Idea | Why not |
|---|---|
| More news, faster news, LLM news reading | Tested: prices finish moving within ~15 minutes of the headline (IN1/IN2); an LLM would also leak look-ahead in research |
| More option-selling | Fails after spreads (6.2% loss per trade at the bid) |
| Re-tuning strategy parameters (lookbacks, thresholds) | Classic overfitting; 160-variant families already showed it |
| Crypto, FX, futures | Outside this lab's data and scope; the futures-lab branch exists separately |
| Shorting | The bot is long-only by design; borrow costs and squeeze risk |
| Buying more data before an edge exists | Spend only when a rule passes at conservative fills |

### The honest frame

The highest-value levers for Harry's wealth are outside QuantLab: savings rate, low-cost ETFs and time.
QuantLab can still add value in three ways:
- stop the bot from losing money (B1-B3);
- earn interest on its idle cash (B5);
- find out whether any tilt of the ETF core has beaten the plain market after costs in the decade after
  it was published (M1-M6).

## Part B. Batch 1 pre-registration

**Common rules**
- Splits as before:

  | Data | TRAIN | VAL | OOS |
  |---|---|---|---|
  | Bot replay | 2016-19 | 2020-21 | 2022-24 |
  | Ken French | 1963-07..1999 | 2000-12 | 2013-24 |

- One variant per idea, exactly as written; no thresholds are tuned.
- Results go to `research/alpha/results/improvement/`, with ledger rows `IP_<id>`.
- Ten tests run at once, so one could pass by luck. A pass therefore also needs the same sign in all
  three splits. Any pass is reported as a candidate for forward paper shadowing, never as proven.

### Bot tests (B1-B5): master's replay, corrected delisting convention, current sizing (S1)

**B1. Young stocks:** drop a candidate whose entity has fewer than 252 sessions with a bar in the
research store up to and including the signal day. Candidates before 2017-01-03 are kept, because the
store starts in 2016 and age cannot be measured.

**B2. Wide stops:** drop a candidate whose plan stop is missing, non-positive, or more than 20% below the
plan's entry reference price (`stop_price / entry_ref_price < 0.80`).

**B3.** Drop if B1 OR B2 would drop.

**Verdict for B1-B3:** the E4 IMPROVEMENT rule.
- OOS Sharpe +0.10 or more, with the 90% monthly-block CI above 0;
- OOS CAGR not lower;
- TRAIN and VAL Sharpe differences both above 0.

**B4. Pruning.**
- Each of the 6 replayed strategies runs alone. A strategy is kept if its Sharpe is >= 0 in BOTH TRAIN
  and VAL.
- The pruned book (kept strategies only, same engine and sizing) is compared with the full book.
- Verdict: OOS only, because TRAIN and VAL were used to select. Pass = OOS Sharpe +0.10 or more with
  the CI above 0, and OOS CAGR not lower.
- If no strategy or all strategies are kept, the verdict is NO CHANGE.

**B5. Cash.** Daily, from the base book's equity, cash and positions:
- **B5a:** uninvested cash earns the 1-month T-bill rate (Ken French RF).
- **B5b:** uninvested cash is held in SPY, with 1 bp per unit traded on daily changes of the cash
  weight.
- B5a is reported as arithmetic, with no verdict.
- **B5b verdict:** "the bot adds value over SPY" if B5b's Sharpe is above SPY buy-and-hold in all three
  splits, with the OOS CI above 0. Otherwise, holding SPY alone is better.

### ETF-core tests (M1-M6): Ken French data

**Data and timing**
- Monthly value-weighted portfolio returns.
- Market = Mkt-RF + RF.
- Rebalance at month end.

**Costs**
- Each tilt pays a 0.25%/yr fund-cost drag, typical of factor ETFs.
- The portfolios' own internal turnover is not in the data, so this is optimistic for momentum.

**Rules**
- **M1:** "Hi 30" of operating profitability.
- **M2:** "Hi 30" of book-to-market.
- **M3:** the mean of the top 3 deciles of prior 12-2 return.
- **M4:** equal weights of M1, M2 and M3, rebalanced monthly.
- **M5:** each month, the 10 of 49 industries with the highest prior 12-2 month return, equal weight,
  5 bps per unit traded.
- **M6:** while the market's total-return index is above its 10-month average at month end, hold 2x the
  market's daily excess return plus RF, rebalanced daily. Financing costs RF + 1%/yr on the borrowed
  half, plus a 0.95%/yr fund fee. Otherwise hold T-bills. 5 bps per switch.

**Verdicts:** as in Area A, versus buy-and-hold:

| Verdict | Condition |
|---|---|
| RISK-ADJUSTED IMPROVEMENT | Sharpe above BH in TRAIN, VAL and OOS, and the OOS 90% 12-month-block CI of the difference above 0 |
| ABSOLUTE IMPROVEMENT | The above, plus OOS CAGR >= BH |
| RISK REDUCTION ONLY | Lower max drawdown without either improvement |
| HIGHER RETURN, HIGHER RISK | M6 only: OOS CAGR above BH but Sharpe not improved |
| NO IMPROVEMENT | Otherwise |

**Also reported:**
- worst calendar year;
- the 2008 and 2022 drawdowns;
- the share of OOS months the tilt beat the market.
