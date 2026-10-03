# Real-money readiness and improvement plan (2026-10-03)

QuantLab is PAPER ONLY, and that is a hard rule in the code and in CLAUDE.md. This doc lists what would
have to be true before that rule could even be reconsidered, and where better results could plausibly
come from. It is research guidance, not investment advice.

## Where the evidence stands

| What was tested | Result |
|---|---|
| 8 strategies, 16 discovery cohorts, 4 catalyst hypotheses (price/volume/earnings) | flat or negative after costs |
| 15 new-data tests (insider, 13D, news, overnight) | 1 tiny pass (too small to trade), 14 flat |
| Live exploration rule, full backtest 2021-24 | **FAIL**: +2.9% vs SPY +59% |
| Quality/value monthly book 2021-24 | **FAIL**: +22.9% vs SPY +61%, excess t -0.05 |
| Insider "fast" buys, holdout #1 | **FAIL** (t 1.79 vs 1.96) |
| House member purchases, 2021-24 then holdout #2 | marginal pass (t 2.44), then **FAIL** on 2025-26 (t 0.34) |

**Nothing tested so far beats simply holding SPY after costs.** That is the bar any real-money use has
to clear, because holding an index fund costs almost nothing and takes no time.

## Gate list: ALL must be true before any real money

### 1. Evidence of an edge (none exists today)
- [ ] One strategy passes a **pre-registered backtest**: beats SPY after costs in both halves, excess
      t >= 2, drawdown no worse than SPY's.
- [ ] It passes a **fresh-sample check** it was never tuned on. The 2025+ holdout has now been used
      twice, so the next check has to be **forward paper data**.
- [ ] **Forward paper results**: at least 3 months and ~50 closed trades with the rule frozen. Returns
      must fall inside the backtest's expected range and beat SPY over the same window.

### 2. The machine is proven
- [ ] 20 consecutive trading days with no missed open, no runner crash, and clean reconciliation every
      day. Partial fills are fixed as of 2026-10-03; the watchdog and Task Scheduler are in place.
- [ ] **Measured fill quality**: realised slippage vs the cost model on 50+ fills. Paper fills are
      optimistic, so real costs should be assumed worse.
- [ ] Kill switch, daily-loss stop and max-position caps tested end to end, including a forced trigger.
- [ ] A decision on broker-held stops, backed by data (currently OFF: the re-baseline showed no gain).

### 3. Safety build (does not exist, by design)
- [ ] A separate, reviewed live-execution path. It would need its own keys, hard notional caps, and a
      manual confirm step at first. Today the code refuses any non-paper URL.
- [ ] Paper keeps running in parallel, so live-vs-paper drift is visible every day.
- [ ] Written stop rules for the live sleeve. For example: stop if live trails paper by more than X%,
      or if drawdown exceeds Y%.

### 4. Harry's own constraints (from the v2 doctrine)
- [ ] Only risk capital, as a small sleeve. The core stays in whatever Harry already uses.
- [ ] NZ tax: FIF rules apply above NZD 50k of foreign shares, and frequent trading may be taxed as
      income. Check with an accountant before a bot trades for you.
- [ ] Time cost: does monitoring this take hours away from the software business?

**Realistic timeline:** at least 4-6 months away, and only if a strategy passes gate 1. If none does,
the correct outcome is that the bot stays a paper research tool.

## How to improve success rate and returns (ranked by expected value)

1. **Stop testing more price/volume rules.** About 30 have failed. More of the same is how false
   positives get found.
2. **Fix the data before testing more ideas.** Each of these has quietly limited past tests:
   - **Survivorship bias.** The store holds only current listings, so every long backtest is flattered.
     Adding delisted stocks makes all results honest.
   - **Fundamentals coverage** is only ~300-400 names per month. Widening it is the reason to re-test
     quality/value at all.
   - **23-25% of House filings are scanned PDFs.** OCR is a low priority now that the family is retired.
3. **Use the forward evidence the bot already produces.** Shadow outcomes (570+ recorded
   opportunities) started maturing on Oct 2. They are free, out-of-sample and need no new trades. Score
   them weekly; this is how any lead gets confirmed now that the holdout is spent.
4. **Cut costs, which eat ~40% of the small gross edge.** Fewer, longer holds and higher-liquidity
   names lower the cost per trade. The 20-day hold arm does not beat the 5/10-day arms, so test longer
   holds only as a pre-registered change.
5. **Exposure, not just picks.** The live rule averages ~16% invested, so even a real edge earns
   little. Any future strategy should be tested as a fully invested book against SPY, as the
   quality/value test was.
6. **Ideas still untested, with published support:** short-term reversal in liquid large caps
   (weekly), post-earnings drift measured on revenue surprises, and analyst *revision* momentum (not
   target levels). Each gets one pre-registered test with the same bar, run on survivorship-free data.
7. **Keep the bar fixed.** Pre-register, no tuning after results, Bonferroni when testing several at
   once. That is why the results above can be trusted. Loosening it would produce a "winner" that
   loses real money.

## Next concrete steps
1. Keep the paper runner going and treat its P&L as a machinery test.
2. Score the shadow outcomes weekly (forward evidence).
3. Add delisted symbols to the price store, then re-run the two FAILED backtests to measure how much
   survivorship bias flattered them.
4. Pre-register and run idea 6 (reversal, revenue drift, revision momentum) on the improved data.
