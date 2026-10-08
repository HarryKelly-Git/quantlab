# What the bot needs, Vistra decoded, and whether big moves repeat (2026-10-07)

PAPER research. Nothing here is a verdict on Harry's real-money Vistra position; real-money decisions
follow the Upside Engine v2 doctrine.

- **Pre-registration:** [EVENT-STUDY-PREREG.md](EVENT-STUDY-PREREG.md), with amendments 1-2.
- **Results:** `research/alpha/results/events/`.
- **Code:** `src/quantlab/alpha/events_study.py`, `scripts/research/alpha/events_*.py`.

## 1. Vistra, decoded

Sources: Alpaca/Benzinga news timestamps and SIP daily bars.

| Day | What happened | VST | Others |
|---|---|---|---|
| Fri 2026-10-02, 3:27pm ET | Bloomberg: US to offer Vistra a ~$4B loan to boost nuclear output | 140.02 (+0.2%; volume 2.3x, the news hit near the close) | |
| Mon 10-05 | $4.2B DOE loan confirmed, for nuclear uprates whose power Meta already bought | 144.89 (**+3.5%**) | CEG +3.9%, TLN +3.5% |
| Tue 10-06 | **Google signs a 20-year deal for Constellation's reactors**: the nuclear/IPP group re-rates | 160.50 (**+10.8%**) | CEG +12.2%, TLN +12.4%, NRG +7.0%, OKLO +7.2%, XLU +3.0%, SPY +0.5% |
| Wed 10-07 (intraday) | Follow-through | ~166.7 (+3.8%) | NRG +4.9%, CEG ~flat |

**About two-thirds of the gain was a COMPETITOR's deal re-rating the whole sector, not Vistra's own news.**

Neither event existed in the price history beforehand:
- a government loan decision;
- a Google-Constellation contract.

Price patterns could not have timed them. What QuantLab can say in advance is that Vistra is a
high-volatility stock in a news-dense theme, so big moves are likely. It cannot say which way they go.

## 2. Does it repeat? 18,266 jumps like this, 2016-2024

**Setup:**
- **Jump:** +8% or more on at least 2x normal dollar volume, liquid US stocks, survivorship-free data,
  2016-2024.
- **Holdout:** the 2025+ holdout was not touched. Vistra's 2026 move motivated the test but is not data
  in it.
- **News classification:** 112,182 Benzinga headlines (new data for QuantLab), classified by fixed
  rules. An LLM was not used for classification, because it would already know what happened next.
- **Measurement:** buy at the next open, the bot's convention. Return vs SPY, after costs.

| What the bot buys the next morning | n | 20-day net excess: 2016-19 | 2020-21 | 2022-24 |
|---|---|---|---|---|
| Jump WITH substantive news (excluding takeover targets) | 10,573 | −0.7% (t −2.7) | −1.3% (t −2.0) | −0.7% (t −2.0) |
| Jump with NO news | 6,515 | −2.2% | −1.2% | −1.8% (t −3.1) |
| Jump on a sector "theme day" (sector ETF up >= 2%, like 10-06) | 3,586 | −5.0% | +0.5% | −1.0% |
| Theme-day jump WITH news | 1,442 | −1.9% | +1.7% | +0.1% |
| Laggard peers on theme days (the catch-up idea) | 39,973 | −0.2% (5-day) | 0.0% | −0.5% (t −2.2) |
| Jump to a new 52-week high | 5,134 | 0.0% | −1.8% | −1.0% |
| ATM call bought after a news jump, held to expiry (2019+) | 4,468 | −2.2% at the ask | −23.2% | −8.8% (+2.6% at mid) |

**Answer:**
- Yes, it repeats, but what repeats is **underperformance**. After a big jump the average stock lags
  the market over the next 1, 5, 20 and 60 days, news or not.
- News-driven jumps lag less than no-news jumps: by +1.5 points in 2016-19 and +1.2 in 2022-24, but
  not in 2020-21 (−0.1). They still lag, and the difference is not significant at the pre-registered
  bar.
- Laggards do not catch up.
- Calls after news jumps are expensive: the median spread is 16% of the premium, and implied vol is high
  after the jump. They lose at the ask in every period.

**Verdicts:**

| Test | Verdict |
|---|---|
| E1 news drift | FAILS |
| E2a sector-wide / E2a-v2 theme | FAILS / FAILS |
| E2b laggards / E2b-v2 | FAILS / FAILS |
| E3 calls | FAILS (EXECUTION-SENSITIVE: positive only at mid) |

**The opposite trade does not work either.** Fading jumps is the "lottery stock" anomaly, which QuantLab
already tested as H16. It failed after costs, and the bot is long-only.

**E4: should the bot stop buying right after jumps?** 22% of its entries follow a jump day (67% of
breakout entries). The filter was a pre-registered one-shot test:

| Period | Sharpe change |
|---|---|
| 2016-19 | +0.20 |
| 2020-21 | +0.25 |
| 2022-24 | −0.51 |

All CIs include 0, so the verdict is **NO IMPROVEMENT**. Inside the bot's book, its after-jump trades
(mostly breakouts) did slightly better than its other trades: +0.8% against −0.6% per trade. Nothing in
the bot changes.

## 2b. Reacting the same day: intraday news study (2026-10-08)

Pre-registration: [INTRADAY-NEWS-PREREG.md](INTRADAY-NEWS-PREREG.md). Results:
`research/alpha/results/intraday/intraday_news.json`.

**Sample:**
- **Days:** 503 random trading days, 2017-2024.
- **News:** 371,792 Benzinga articles, giving 48,773 qualifying single-stock headlines in regular hours.
- **Prices:** SIP 1-minute bars.

**Trade:** buy 15 minutes after a headline when the stock has already moved at least 2% (787 cases). Hold
to the close (IN1) or the next close (IN2). Results are net of doubled post-news spreads (median 0.18%
round trip), vs SPY.

| Test | 2017-19 | 2020-21 | 2022-24 |
|---|---|---|---|
| IN1: long to the close | +0.58% (t 1.0) | −0.66% (t −1.4) | −0.03% (t −0.1) |
| IN2: long to the next close | +1.39% (t 1.9) | −0.68% | +0.36% (t 0.8) |

**Verdict: both FAIL.**

- **The move is done within ~15 minutes.** After that, the remaining drift to the close, before costs,
  is a few hundredths of a percent. The strongest up-reactions (top decile, +1.5% reaction) add +0.08%;
  the strongest down-reactions add a 0.04% further drop. Both are smaller than the cost of trading.
- **Benzinga often trails the original wire.** Only 1.6% of headlines still showed a 2% move after their
  timestamp, because the price had usually reacted before the headline arrived.

A retail-speed bot reading this feed is too late to capture moves like Vistra's.

## 3. What the bot does not have, ranked by expected value

| # | Missing | Evidence it matters | Cost | Verdict |
|---|---|---|---|---|
| 1 | **A validated edge.** About 50 hypotheses have failed (30 earlier, about 20 in October, the sprint, E1-E4). The bot's six replayable strategies earn about 0 on survivorship-free data. | Every report | Research time | The binding constraint. More data or machinery without an edge just adds costs. |
| 2 | **Same-day reaction to news.** Vistra's +10.8% happened DURING Oct 6 (it opened +4.5%). A close-to-next-open bot only sees day 2, and on average the day-2+ drift is negative. | Sections 2 and 2b | Tested 2026-10-08: 48,773 headlines with minute bars | **FAILS.** Prices finish reacting within ~15 minutes of the Benzinga headline, and the leftover drift is below costs. Do not build. |
| 3 | **News awareness as a RISK input.** The bot has no news feed at all (news_shock is disabled). | Takeover targets pin, binary events (FDA, earnings) gap | Free (Alpaca/Benzinga) | Useful for avoiding traps, such as buying a pinned takeover target or holding through a known binary event. Not proven as alpha. |
| 4 | **A point-in-time earnings calendar.** 7,105 of 18,266 jumps are earnings-driven, and earnings drive option-period volatility. | Sprint P4/P6: the earnings-aware model forecasts option vol better | A paid calendar, or forward capture from Nasdaq | Needed before any serious earnings-volatility options test |
| 5 | **Real option quotes for live and research.** The paper bot sees only Alpaca "indicative" quotes (OPRA needs a paid plan). There is no historical option volume, open interest or intraday quotes; research uses DoltHub end-of-day chains. | docs/OPTIONS.md (master); sprint P5 | Alpaca paid plan with OPRA; ThetaData $40-80/month | Buy only when an options rule passes validation at conservative fills. None does yet. |
| 6 | **Correct corporate actions.** Alpaca misses most spin-offs; master books raw-price drops on spin-off days. | Sprint-c: 94 fake days | Free fix (the price rule) or a paid actions feed | Fix in research done; master's live data has a smaller version of the issue |
| 7 | **The bot's own forward record.** | P1 is blocked | 5 minutes of Harry's time | Do it: it is free evidence |

## 4. Taking options seriously

**What exists:**
- **On master:** a complete options machine:
  - stock-vs-structure comparison (long calls/puts, debit spreads);
  - liquidity rejects;
  - a gated PAPER options book that is OFF by default;
  - marking from real option bars.
- **In research:** 2019-24 end-of-day chains, the four-level fill engine, and IV and vol-forecast
  models.

**What the evidence says, after spreads:**

| Use | Result |
|---|---|
| Implied vol as a forecast | Efficient; QuantLab adds about 0.2-0.5% of explained variance |
| Buying straddles when the forecast beats implied | Paid only in 2023 |
| Strangles | Same pattern: 2023 only |
| Calls after news jumps | Lose at the ask |
| Selling volatility | Loses after spreads |
| Pre-earnings straddles | About −30% |

Median spreads: about 6% (ATM straddles), 14% (25-delta strangles), 16% (ATM calls after jumps).

**What options are actually good for:** defined-risk, convex exposure to a thesis that is ALREADY right.
They multiply an edge and charge for it; they do not create one.

**A serious options plan, in order:**
1. **Forward paper-shadow O1** (straddles when QuantLab's forecast move is at least 1.2x the implied
   move). Record only, no orders. Score it on excess over buying every straddle on the same dates. It is
   the only options rule with positive out-of-sample point estimates.
2. **Turn on master's OPT paper book at minimum size** (Harry's decision), only to verify fills, fees
   and assignment handling against the four-level model. Do not expect profit from it.
3. **Get a point-in-time earnings calendar, then pre-register one earnings-volatility test**, priced at
   the ask: the event class that drives most big moves.
4. **Buy ThetaData only if step 1 or 3 passes validation** at conservative fills.

## 5. Bottom line

Big moves like Vistra's are driven by news nobody could see in the price history. Across 18,266 of them:
- buying the next morning lost money against the market;
- sector peers did not catch up;
- calls were too expensive.

The bot is not missing a pattern-matcher; it is missing an edge.

Same-day reaction to Benzinga headlines was tested too (section 2b), and prices finish moving within ~15
minutes. The cheapest remaining evidence is the bot's own forward record (P1). That needs at least 20
trading days of bot history, about two more weeks from 2026-10-08.
