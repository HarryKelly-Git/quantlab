# How to make the bot better: options, swing trades, anything under a month (2026-10-06)

A review of everything QuantLab has measured, set against published research. PAPER ONLY.
Nothing here is a real-money recommendation. Ranked by expected value per hour and per dollar.

## 1. Where QuantLab really stands

- About 30 stock rules tested, point-in-time and after costs: 8 strategies, 16 discovery groups,
  4 earnings/news ideas, 15 new-data tests (insiders, Congress, news, overnight). **None beats holding
  SPY after costs.** The two holdout "passes" failed when re-checked.
- The best ranking found (selection score = 12-1 momentum, low volatility, high liquidity) earns
  +10 / +23 bps per 5-day trade, not significant, and still a little below SPY. Its real value is
  half the tail risk.
- **The most important finding is in docs/UPSIDE-EVIDENCE.md:** every characteristic that predicts
  big moves (compressed range, time since earnings, volume spikes, distance below the high)
  predicts the *size* of the next move, not its *direction*. Down moves get more likely about as
  much as up moves. Ranking stocks for "highest upside" really ranks them for volatility, which adds
  big losers as fast as big winners. The most volatile fifth of stocks makes the most +10% winners
  and **loses money on average**.
- The machine works: paper fills, partial-fill top-ups, reconciliation, a kill switch and
  point-in-time tests. Exploration currently holds 10 positions (its cap), about 18% invested.

## 2. The key insight: predicting move SIZE is exactly what options pay for

A stock trade only pays if you get the direction right. An option position can pay when you get
the **size** of the move right relative to what the option price already assumes (implied
volatility, IV). QuantLab can already forecast size and cannot forecast direction. The published
evidence says volatility-based option trades are some of the strongest effects ever documented,
much stronger than directional stock effects:

| Effect | What it says | Source |
|---|---|---|
| IV vs realised vol | Sort stocks by (historical vol - ATM IV). Buy straddles where options look cheap, sell where they look dear: up to ~8.8%/month after costs (1996-2006). | [Goyal & Saretto 2009](https://www.cis.upenn.edu/~mkearns/finread/CrossOptions.pdf) |
| Pre-earnings straddle | ATM straddle bought 3 days before earnings and closed at the announcement: +3.34% on average. Straddles normally lose money. | [Gao, Xing & Zhang, JFQA 2018](https://resolve.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/anticipating-uncertainty-straddles-around-earnings-announcements/7B34877AD5E06304BA3C55FBA3219FDD) |
| Option momentum | Straddles with high past returns keep outperforming over 6-36 months. Pre-cost Sharpe ~3x stock momentum; survives reasonable costs. | [Heston et al., JF 2023](https://alphaarchitect.com/option-momentum/) |
| Idiosyncratic vol | Delta-hedged option returns fall as the stock's idiosyncratic vol rises: options on wild stocks are overpriced. | [Cao & Han 2013](https://optionmetrics.com/research/j-cao-and-b-han-cross-section-of-option-returns-and-idiosyncratic-stock-volatility/) |
| Index put-writing | Cboe PUT index (selling cash-secured S&P puts) since 1986: ~S&P-like return, ~10% vs ~15% volatility, max drawdown −33% vs −51%. This is the variance risk premium. | [Cboe](https://www.cboe.com/insights/posts/generating-income-and-managing-risk-cash-secured-put-writing-in-a-low-equity-return-environment) |
| **What loses** | Retail option buyers lose 5-9% around earnings (10-14% on the most-hyped names). They overpay for volatility, pay wide spreads and react slowly. | [de Silva, Smith & So](https://mitsloan.mit.edu/ideas-made-to-matter/retail-investors-lose-big-options-markets-research-shows) |

**So the "perfect" options layer is not "buy calls on the best stock picks".** That is the pattern
the research shows losing money. It is: trade volatility when it is mispriced, with defined risk,
using the size forecasts QuantLab already has. Directional calls on stock picks (the
momentum-breakout options stage) stay as a measured comparison, not the main bet.

Caveats, stated plainly:
- These results are mostly from 1996-2019 and some may have shrunk since publication. Each one
  needs its own pre-registered test.
- Goyal-Saretto's long-short version sells straddles, which has unlimited risk. QuantLab forbids
  naked shorts, so the short side must be iron condors or iron butterflies (defined risk). That
  changes the payoff and must be tested as such.
- **None of it can be tested on today's data.** Alpaca gives no historical option quotes, only
  indicative quotes now and daily bars from Feb 2024 (docs/OPTIONS.md). Testing it properly needs
  historical OPRA bid/ask quotes.

## 3. Ranked plan

### Options (the user's top priority)
1. **Buy historical option quotes.** [ThetaData](https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html):
   Standard ~US$80/month gives tick NBBO quotes and ~8 years of history (Value ~$40 has 4 years and
   1-minute snapshots; check the current tiers before buying). One month is enough to download a
   research set. Without it every options backtest is a model (APPROXIMATION). **Highest ROI action
   in this document.**
2. **Test 1: pre-earnings straddle (Gao-Xing-Zhang).** Buy an ATM straddle 3 sessions before a
   *scheduled* earnings date and sell it at the last close before the announcement, so it never
   holds through the announcement and avoids the IV crush. Simple, defined risk, about 4-day holds.
   Needs a **point-in-time earnings calendar** (dates known in advance). QuantLab's current dates
   come from 8-K filings, which arrive after the event. The decision trace shows
   `next earnings date UNKNOWN`, so this gap must be closed first.
3. **Test 2: IV vs forecast-vol sort (Goyal-Saretto, defined-risk version).** Monthly: compare
   each liquid stock's ATM IV with its forecast volatility, using QuantLab's own size model
   (upside table, ATR, range contraction). Buy straddles or strangles where forecast > IV by a fixed
   margin. Where IV > forecast, sell defined-risk iron condors. One pre-registered rule, fixed margins.
4. **Test 3: option momentum (Heston et al.).** Rank by past straddle returns. Needs the same quote
   history, so it is cheap to add once item 1 exists.
5. Keep OPT paper trading off until one of these passes on real quotes. Then run it forward on
   Alpaca paper (Alpaca paper supports multi-leg `mleg` orders, so straddles and condors fill as
   one order with no partial-leg risk;
   [docs](https://docs.alpaca.markets/docs/options-level-3-trading)).

### Swing trades and anything under a month (stocks)
1. **Weekly short-term reversal in large caps** (already on QuantLab's untested list). Buy last
   week's losers relative to their industry, sell or avoid the winners, liquid names only, with
   low-turnover construction. Research reports ~30-50 bps/week net in large caps
   ([summary](https://quantpedia.com/strategies/short-term-reversal-in-stocks)). It fits QuantLab's
   own hints: 1-month return leans negative (SELECTION-EVIDENCE) and mean reversion at 20 sessions
   was the best hold arm. Daily data is enough; testable this week.
2. **Momentum breakout with an opening-range entry** (in progress, PR #1). Prior is **low**. The
   headline ORB paper (1,637% 2016-23) charged no spread or slippage and had no out-of-sample
   period, and an independent replication on indices found net ≈ 0
   ([CXO](https://www.cxoadvisory.com/technical-trading/day-trading-with-an-opening-range-breakout-strategy),
   [replication](https://www.mql5.com/en/blogs/post/776235)). Finish it because it is cheap and
   pre-registered, but don't expect it.
3. **Earnings-announcement premium:** stocks tend to drift up into scheduled announcements. Needs
   the same earnings calendar as options test 1, so do both together.
4. **Revenue-surprise drift and analyst-revision momentum** (on the untested list). Needs estimate
   data QuantLab doesn't have. Defer.

### Machine and data (cheap and certain)
1. **Add delisted stocks** (survivorship-free history). Every long-only result so far is flattered
   until this is done.
2. **Load sector data:** relative_strength and sector_rotation produce 0 trades because no sector
   map is loaded. That's a data gap, not a verdict.
3. **Score the shadow outcomes weekly** (570+ opportunities maturing since 2 Oct): free,
   out-of-sample evidence.
4. **A heartbeat alert to the phone** when the runner goes stale. Missed opens are the main
   reliability risk.
5. **Measure the bot against the capital it actually uses.** It is ~18% invested, so "account vs
   SPY" understates any edge and overstates safety. Report excess return on deployed capital, plus
   a fully invested version.

### What not to do
- Rank for "highest upside" on stocks. QuantLab's own data shows this is ranking for volatility,
  and it lowers net return.
- Buy short-dated calls on picks or before earnings (the documented retail-loss pattern).
- Loosen the test bar or raise trade caps to get faster "results".

## 4. What would change the answer

Real historical option quotes (item 1) and a point-in-time earnings calendar. Those two data
sources unlock the only ideas left with strong published evidence that match QuantLab's own
findings.
