# AI trading bots: what the evidence says, and what QuantLab should take from it (2026-10-08)

PAPER research. Nothing here is a real-money recommendation. Real-money questions follow the
Upside Engine v2 doctrine (`../CLAUDE.md`).

**Method.** About 35 web searches and fetches on 2026-10-08, preferring papers, fund documents, filings
and competition pages over blogs. The bracketed numbers are entries in the source list (section 6).

**Labels.**
- **Claimed** means the vendor or authors say it. It is not verified.
- **UNVERIFIED** means I found the figure only in a secondary or paywalled source, or sources disagree.
- **UNKNOWN** means I found no evidence either way.
- Computations of my own are marked *(computed here)*.

---

## 1. Summary

1. No AI trading system found has an independently verified multi-year record of beating the index after costs; the longest public one, AIEQ (2017-), trails the S&P 500 badly [19][20][21].
2. Live LLM contests are short and mostly lose: Alpha Arena season 1 (crypto, ~2 weeks) had 4 of 6 models down 30-63% [3][4][5]; in season 1.5 (US stocks) most systems lost and "trade too much" [7].
3. LLM-trading papers win on narrow, short tests and fade on broad, long ones (FINSABER, 20 years, 100+ symbols [11]; StockBench [8]); some headline baselines are simply wrong [10].
4. Commercial "AI signal" products publish backtests or simulations, not audited live records [28][29][30]; 7 of the 12 AI ETFs Morningstar tracked have closed [25].
5. The academic failure list is settled: overfitting and multiple testing [32][33][37][38], post-publication decay [36], costs on high-turnover signals [39], look-ahead incl. LLM training-data contamination [13][14].
6. What survives out-of-sample is mostly risk technology, not prediction: higher significance hurdles [37][38], low turnover with a buy/hold band [39], vol targeting for tail risk [44], fractional Kelly [46], diversified trend-following (modest returns, deep drawdowns) [42][43].
7. QuantLab already does most of what failed systems skipped (pre-registration, PIT and survivorship-free data, costs, sealed holdout, DSR/PBO, no LLM price prediction).
8. Gaps to close (section 5): a lab-wide trial counter, turnover budgets, implementation-shortfall reports, decay kill-rules, and pre-registered vol-target and trend-overlay tests.

---

## 2. Systems reviewed

"Verified" means checked against a primary or official source, or recomputed. Everything else is labelled.

| # | System | Type | Claimed performance | Verified live / out-of-sample performance | Main failure or weakness |
|---|---|---|---|---|---|
| 1 | **Alpha Arena season 1** (Nof1) | 6 frontier LLMs, $10k each, crypto perpetuals on Hyperliquid, about Oct 18 to Nov 3, 2025 [3] | "The first benchmark designed to measure AI's investing abilities" (wording on a nof1-branded page, see note A) | Qwen3 Max about **+22%**, DeepSeek about **+5%**. GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5 and Grok 4 all lost, between about 31% and 63% depending on the report [3][4][5]. Final balances differ between reports (UNVERIFIED in detail). | One 2-week run, leveraged crypto. Secondary reports [6] (UNVERIFIED): GPT-5 used up to 17.2x leverage; Gemini made 238 trades and paid about $1,331 in fees (about 13% of capital); Claude was 100% long with no stops. The public dashboard omitted drawdown, Sharpe and funding costs [2]. |
| 2 | **Alpha Arena season 1.5** | 8 LLMs, US stocks, 4 prompt "modes" (New Baseline, Monk Mode, Situational Awareness, Max Leverage), to Dec 3, 2025 [1][4] | Same as above | Winner: the "Mystery Model" (Grok 4.20) at **+12.11%** [1], profitable in all 4 modes [4]. As of Dec 8, "only Grok was still profitable" [4]. Bloomberg: "Most of the systems lose money. They trade too much. They make wildly different decisions when given identical instructions" [7]. "Only 6 of 32 result sets were profitable" and "the field lost about a third of its capital" are UNVERIFIED (paywalled or secondary). | Overtrading. Instability under identical prompts. About 2 weeks of data, so there is no statistical power. |
| 3 | **TradingAgents** (arXiv 2412.20138) | Multi-agent LLM "trading firm" (analysts, bull/bear researchers, risk, trader) | "Superiority over baseline models" in cumulative return, Sharpe and max drawdown [9] | Paper: about 3 months (Jan to Mar 2024) on a handful of large tech stocks [9]. A widely quoted AAPL result (Jun 19 to Nov 19, 2024: agent +26.62%, Sharpe 8.21, **buy-and-hold -5.23%**) has the wrong-signed baseline: AAPL rose **+9.12%** dividend-adjusted [10]. I confirmed this with QuantLab's own SIP bars (2024-06-20 to 2024-11-19: +9.12% adjusted, +8.87% raw) *(computed here)*. Which paper version or README carries those numbers: UNVERIFIED. | Tiny sample, short window, wrong baseline, no transaction costs modelled [10]. A Sharpe of 8 on one stock over 105 days is an artefact signature. Independent live record: UNKNOWN. |
| 4 | **ai-hedge-fund** (virattt, GitHub) | Open-source LLM "investor persona" agents (Buffett, Munger, ...) | None. README: "for educational purposes only and is not intended for real trading" [18] | No live record: UNKNOWN. Current docs say the LLMs only form theses and **deterministic code does sizing, orders and risk** (UNVERIFIED for the current version) [18]. | Persona prompts are untested as alpha. The project is popular on GitHub, which is not evidence that it makes money. |
| 5 | **StockBench** (arXiv 2510.02209, v2 Mar 2026) | Benchmark: LLM agents trade a stock basket daily on news, prices and fundamentals, Mar to Jul 2025 | n/a | "Most models struggle to outperform the simple buy-and-hold baseline". Also: "strong performance on static financial question-answering [does] not necessarily translate into effective trading behavior" [8]. Agents had shallower drawdowns than buy-and-hold (secondary summary). | 5-month window. Buy-and-hold made only about 0.4%, so the rankings are noisy. |
| 6 | **FINSABER** (arXiv 2505.07078, KDD 2026 D&B track) | Re-test of published LLM timing strategies over about 20 years and 100+ symbols | The original papers claimed LLM advantages | "Previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation." LLM strategies are "overly conservative in bull markets ... overly aggressive in bear markets" [11]. | Survivorship and data-snooping in the original evaluations. Regime blindness. The authors' conclusion is that regime awareness and risk control matter more than architectural complexity [11]. |
| 7 | **Lopez-Lira and Tang: ChatGPT news scores** (arXiv 2304.07619) | LLM scores headlines, then a long-short daily portfolio | Sep-2024 version: 38 bp/day pre-cost, ">650%" cumulative Oct 2021 to Dec 2023; ">300% (150%) at 5 (10) bp per round trip" [12] | Out-of-sample relative to the model's training cutoff (backtest, not live). The latest version reports "strategy returns decline as LLM adoption rises" [12]. | Daily turnover, so very cost-sensitive (650% falls to 150% at 10 bp). Strongest in small stocks and negative news, which are hard to short. Decays as the signal spreads. |
| 8 | **AIEQ**: Amplify AI Powered Equity ETF (EquBot / IBM Watson), launched Oct 2017 | AI-selected US equity ETF, about 160 holdings | "Leveraging the power of artificial intelligence" across news, filings and social media [19] | **2019-2025 cumulative: AIEQ +118% vs S&P 500 TR +205%**, from YCharts calendar-year returns [20] *(computed here)*. Beat the index in 2 of 7 years. 2022: **-31.9% vs -18.1%**. 5-year annualised to 2026-07-31: **NAV +4.5% vs S&P 500 TR +12.9%**. 1 year: +15.8% vs +19.6% [21]. Since inception: +10.03%/yr NAV [19]. | Underperformance with higher drawdown. Reported **turnover 804%** and a 0.75% fee [20]. Morningstar/MarketWatch (Aug 2026): "recent performance has been worse, relative to the S&P 500" [22]. |
| 9 | **Other AI-managed ETFs** | Various (EquBot AIIQ, Qraft QRFT/AMOM/NVQ, Alpha Intelligent AILG/AILV, BUZZ) | Marketing varies | AIIQ liquidated July 2022 [23]. QRFT liquidated July 2026 [24]. Morningstar's Bryan Armour: **7 of 12 AI-powered ETFs tracked from about 2020 have closed**, mostly on low assets [25]. 2019: 3 of 4 AI ETFs trailed the S&P 500 [26]. Short-window counterpoint: in 2026 YTD to Aug 4, 5 of 8 large-cap AI funds beat the S&P 500, while only 2 of 9 small/mid-cap funds beat the Russell 2000 [27]. | Fees, turnover, closures. **Survivorship:** the surviving AI funds overstate how the category did. |
| 10 | **Danelfin** | AI score (1-10) for the probability of beating the market over 3 months | Backtested outperformance of top-score stocks. Reviews quote inconsistent annualised alpha figures (about +15% vs +21%) [28] | No independent audit of live results found: **UNKNOWN**. The headline figures are labelled backtests [28]. Anecdote: MarketWatch's check of one (unnamed) AI rating firm found its top-10 stocks averaged -4% over 3 months vs S&P +2.7% [22]. | Backtest-only marketing. Inconsistent numbers. |
| 11 | **Tickeron** | AI pattern signals and "AI trading bots" | Of 34 bots one reviewer looked at, most claimed **40-169% annualised** [29] | Independent verification: none found, **UNKNOWN**. "Audited" means self-audited per the reviewers [29]. | Self-published, implausibly high returns. Affiliate-driven review ecosystem. |
| 12 | **Trade Ideas "Holly"** | Overnight strategy search, next-day intraday alerts | Simulated about 25%/yr, about 62% win rate (2025, company figures per a review) [30] | Simulation assumes fills at the signal price [30]. One reviewer's 6-month live account has no trade log: **UNVERIFIED**. | Strategies are re-selected nightly on win rate, which is data snooping by construction. Fill optimism. |
| 13 | **Quantopian** crowd-sourced fund (2011-2020) | Thousands of user algorithms; Point72 committed up to $250M (2016) | Crowd of quants, up to $50M per strategy | Its own study of **888 algorithms**: backtest Sharpe predicts out-of-sample Sharpe with **R² < 0.025**. The more a strategy was backtested, the bigger the gap [32]. Investors got capital back in Feb 2020 after the market-neutral strategy lagged. Shut down Nov 2020 [31]. | Overfitting at scale. |
| 14 | **FinRL / deep RL** | Open-source RL trading library and contests | Papers report backtest gains | FinRL's own authors list "backtesting overfitting" and survivorship bias as core obstacles [48], and say earlier DRL results "may suffer from the false positive issue due to overfitting" [48]. Independent live record: **UNKNOWN**. | Low signal-to-noise, short test windows (e.g. 2 months in 2022) [48]. |
| 15 | **Freqtrade** (crypto bot framework) | Open-source strategies, backtest, dry-run, live | Community strategies post backtests | The framework ships a `lookahead-analysis` tool because computing all indicators on the full dataframe produces look-ahead that "would not be possible in dry or live modes" [50]. Aggregate live-vs-backtest outcome for users: **UNKNOWN** (anecdotes only). | Look-ahead and warm-up bugs. Backtest-to-live gap. |
| 16 | **QuantConnect LEAN** | Cloud and open-source engine | n/a | Docs: algorithms "usually perform differently between backtesting and live trading". It runs an out-of-sample backtest alongside each live deployment to measure the gap [49]. | Data and fill-model gaps between backtest and live. |
| 17 | **Human baseline: retail day traders** | Not AI. The benchmark these bots replace. | Course sellers' claims | Brazil: **97%** of those who day traded futures for more than 300 days lost money; 0.5% earned more than a bank teller's salary [40]. Taiwan 1992-2006: **less than 1%** were predictably profitable net of fees [41]. | Costs, overtrading, no skill persistence. |
| - | Jesse, Hummingbot, Zipline/backtrader communities, Kavout | Frameworks or products | - | Not researched within budget: **UNKNOWN** | - |

**Note A.** A separate domain, `nof1.info`, shows "Deepseek +117.4%, Qwen +56.3%, GPT-5 +39.4%" next to "Earn consistent
returns with our proven strategies / START TRADING NOW". Those numbers contradict every report of the actual
season-1 results [3][4][5]. Treat it as unverified marketing on a look-alike site. It is a live example of
why claimed AI-bot returns need a primary source.

---

## 3. Failure modes, ranked by how often they appear

The counts are my tally across the 17 systems above. They involve judgement calls, so read the ranking as rough.

| Rank | Failure mode | Seen in | Sourced examples |
|---|---|---|---|
| 1 | **No credible live or out-of-sample record.** Backtest, simulation or a few weeks only. | about 11 of 17 | Tickeron's 40-169% bots are self-published [29]. Danelfin's figures are labelled backtests [28]. TradingAgents was tested over 3 months [9]. Alpha Arena seasons were about 2 weeks each [3][4]. |
| 2 | **Does not beat buy-and-hold or the index** (or mis-states the benchmark) | about 8 | AIEQ: +118% vs +205% (2019-2025) [20]. StockBench: "most models struggle to outperform ... buy-and-hold" [8]. TradingAgents' AAPL baseline had the wrong sign [10]. |
| 3 | **Costs, turnover, overtrading** | about 8 | Bloomberg on Alpha Arena: "They trade too much" [7]. Gemini's fees were about 13% of capital (UNVERIFIED) [6]. AIEQ turnover is 804% [20]. Lopez-Lira's 650% falls to 150% at 10 bp [12]. Novy-Marx and Velikov: few anomalies with more than 50% monthly turnover survive costs [39]. |
| 4 | **Overfitting and multiple testing** | about 6 | Quantopian: backtest Sharpe R² < 0.025 vs out-of-sample [32]. More than 45 variants on 5 years of data almost guarantees an in-sample Sharpe near 1 with an expected out-of-sample Sharpe near 0 [35]. Hou-Xue-Zhang: 64% of 447 anomalies fail at t ≥ 1.96, and 85% at t ≥ 3 [37]. Holly re-selects strategies nightly by win rate [30]. |
| 5 | **Look-ahead, including LLM training-data contamination** | about 4, plus the literature | GPT sentiment backtests inside the training window are distorted ("look-ahead" plus a "distraction effect" from company knowledge). Anonymised headlines did *better* in-sample [13]. A Lookahead Propensity measure amplifies the LLM signal by about 32% in-sample, and the effect is insignificant after the cutoff [14]. Freqtrade needed a dedicated detector [50]. |
| 6 | **Regime dependence** | about 4 | FINSABER: too conservative in bull markets, too aggressive in bear markets [11]. AIEQ 2022: -31.9% vs -18.1% [20]. Trend-followers: -20.4% from May 2024 to May 2025, the second-worst drawdown since 2000 [43]. |
| 7 | **Leverage, no stops, concentration** | about 3 | Alpha Arena season 1: GPT-5 at up to 17.2x leverage, Claude 100% long with no stops (UNVERIFIED) [6]. Season 1.5 had a "Max Leverage" mode [4]. |
| 8 | **LLM inconsistency and hallucination** | about 3, plus the literature | "Wildly different decisions when given identical instructions" [7]. FinanceBench (2023 models): GPT-4-Turbo with retrieval was wrong or refused on 81% of filing questions [17]. StockBench: QA skill does not translate into trading [8]. |
| 9 | **Optimistic simulated fills** | about 3 | Holly assumes fills at the signal price [30]. QuantConnect reconciles the backtest-live gap [49]. Alpaca paper ignores order size and market impact (QuantLab `docs/EXTERNAL-SERVICES.md`). |
| 10 | **Decay after publication or adoption** | about 2, plus the literature | McLean-Pontiff, 97 predictors: returns are 26% lower out-of-sample and 58% lower after publication [36]. LLM news-signal returns fall as adoption rises [12]. |
| 11 | **Survivorship (of products and of universes)** | about 2 | 7 of 12 AI ETFs closed [25]. FINSABER: narrow ticker sets overstated LLM results [11]. |

---

## 4. What has real evidence of working

Most of what survives out-of-sample controls risk, cost or false discovery rather than predicting returns.

| Technique | Evidence | Effect size (as reported) | Caveats |
|---|---|---|---|
| **Higher significance hurdles for new signals** | Harvey-Liu-Zhu reviewed 316 factors and propose t ≥ 3.0 for new ones [38]. Hou-Xue-Zhang [37]. Deflated Sharpe and PBO [33][34]. | Moving from t ≥ 1.96 to t ≥ 3 raises anomaly failure from 64% to 85% [37]. | A filter against false positives, not a source of returns. |
| **Choose strategies by risk stability, not backtest Sharpe** | Wiecki et al., 888 Quantopian algorithms [32] | Volatility and max drawdown in the backtest *do* predict out-of-sample values. An ML model on backtest features reached R² 0.17. A portfolio chosen by prediction beat the top-backtest-Sharpe portfolio out-of-sample [32]. | The out-of-sample window was short (Jun 2015 to Feb 2016) [32]. |
| **Low turnover and a buy/hold band (hysteresis)** | Novy-Marx and Velikov, 23 anomalies [39] | Most anomalies with less than 50% monthly turnover keep significant net spreads when designed to save costs. A buy/hold spread is "the most effective cost mitigation technique" [39]. | Costs are estimated from effective spreads (small-trader view), not market impact [39]. |
| **Diversified trend-following (time-series momentum)** | Hurst-Ooi-Pedersen: 67 futures markets, 1880-2016 [42] | Positive net returns **in every decade since 1880**. The weakest decade was +4.1%/yr. Combined net Sharpe about 0.76-0.77 after 2/20 fees in the summaries. The per-asset average is about 0.4 (versions differ) [42]. | Recent live index: SG Trend lost **20.4%** from May 2024 to May 2025 [43]. It has had 16 drawdowns greater than 10% since 2000 [43]. The evidence is multi-asset *futures*: a US-equity-ETF-only version is much less diversified (UNKNOWN effect). |
| **Volatility targeting (as a risk tool)** | Harvey et al. 2018, 60+ assets, from 1926 [44] | Sharpe rises only for equities and credit (about 0.40 to 0.48-0.51 for equities in a secondary summary). **Extreme returns become less likely in all asset classes** [44]. | Cederburg et al. 2020: regression-based "volatility-managed" portfolios **lower** the Sharpe ratio out-of-sample for 72 of 103 strategies [45]. Use simple vol scaling for tail risk, not as alpha. |
| **Fractional Kelly sizing** | MacLean, Thorp and Ziemba [46] | Full Kelly maximises long-run growth but "can be very risky in the short term". Fractional Kelly trades growth for safety [46]. In the idealised lognormal model, half-Kelly keeps about 75% of the growth rate, and the chance of ever halving falls from 50% (full) to 12.5% (half) and 0.8% (quarter) *(standard result, computed here)*. | Real edges are estimated with error, which makes full Kelly worse still. Kelly on a zero edge is zero. |
| **Short-term mean reversion (Connors RSI(2) and similar)** | Connors' own tests [47] | 1995-2007, **no transaction costs, no data-mining correction** [47] | I found no independent post-2008 test with costs: **UNVERIFIED**. Claims that it "still works" are paywalled or promotional. Treat it as a hypothesis, not established. |
| **LLMs on text, out of sample** | Lopez-Lira and Tang [12]. Chronologically consistent models (He et al.) suggest look-ahead in some settings is modest [16]. | Positive but decaying, and very cost-sensitive [12] | No independent live record. Contamination tests are now available [14][15]. |

**Not supported:** using LLM personas as stock pickers, live LLM discretionary trading, "AI score" subscriptions, or
AI-labelled ETFs as alpha sources. The only multi-year public record (AIEQ) lags the index [19]-[22].

---

## 5. Concrete lessons for QuantLab's bot

**What QuantLab already does**, verified in the repo on 2026-10-08:
- Research and validation:
  - pre-registered tests;
  - survivorship-free data;
  - truncation-invariance (PIT) tests;
  - a sealed 2025+ holdout;
  - DSR and PBO-CSCV, plus Holm, BH and BY corrections and a stationary bootstrap (`alpha/mht.py`);
  - a DSR p ≤ 0.10 promotion gate (`research/promotion.py`).
- Costs: half-spread tiers plus 5 bp slippage (`config/default.yaml`).
- Sizing and risk limits:
  - quarter-Kelly as a portfolio option (`alpha/portfolio.py`);
  - 0.5% risk per trade;
  - a 10% position cap;
  - gross exposure ≤ 1.0;
  - a 20% drawdown pause;
  - a kill switch.
- Other safeguards:
  - an SPY 200-day regime throttle;
  - LLMs are never used for price prediction or news classification (ARCHITECTURE.md section 2 rule 9, section 7a);
  - documented awareness that Alpaca paper fills are optimistic.

Most failures in section 3 are therefore already guarded against. The lessons below are the gaps.

**L1. Verify every benchmark from raw data before any number is published.**
- *Failure seen:* TradingAgents' wrong-signed buy-and-hold [10]. Benchmark-free vendor claims [28][29].
- *Already:* the readiness bar is "beat SPY after costs" (`docs/REAL-MONEY-READINESS.md`).
- *Adopt:*
  - Every report recomputes SPY buy-and-hold over the *exact* window from stored bars, on the same price or
    total-return basis. Alpaca paper does not simulate dividends.
  - Prints the SPY figure next to every return.
  - Fails if the benchmark is missing (`UNKNOWN`, never assumed).
- *Impact:* removes a whole class of reporting error. Return impact: none.

**L2. Count every trial globally and deflate against the true count.**
- *Failure seen:* Quantopian R² < 0.025 [32]. 45 variants on 5 years of data "guarantees" an in-sample Sharpe near 1 [35].
- *Already:* DSR and PBO per experiment; Holm/BH within families.
- *Adopt:*
  - A **lab-wide trial ledger**. About 30+ stock rules and dozens of variants have been tested so far
    (`docs/IMPROVEMENT-RESEARCH-2026-10.md`).
  - DSR's N comes from that ledger, not from one sprint.
  - New signals need t ≥ 3 [38], not 2.
- *Impact:* fewer false promotions. Nothing has passed yet, so the current book is unaffected. Long-run value: UNKNOWN but positive in expectation.

**L3. Give each strategy a turnover budget and a buy/hold band.**
- *Failure seen:* Alpha Arena overtrading [7]. AIEQ's 804% turnover [20]. The cost collapse of Lopez-Lira-style daily signals [12].
- *Already:* realistic per-fill costs.
- *Adopt:*
  - Report turnover and cost drag per strategy as standard metrics.
  - Use hysteresis: enter in the top decile, exit only below, say, the top 30%. Per [39], this is the most effective cost reducer.
  - Cap new positions per week. Pre-register the band and compare it with no band.
- *Impact:* cost saved scales with the turnover cut. The size on QuantLab's own trade log is UNKNOWN until measured.

**L4. Measure implementation shortfall continuously.**
- *Failure seen:* Holly's fills at the signal price [30]. The backtest-live gaps QuantConnect reconciles [49].
- *Already:* the ledger books actual fill prices. `entry_ref_price` is stored. Paper fills are known to ignore size.
- *Adopt:*
  - A standing per-trade report: decision reference price vs fill vs `CostModel` estimate.
  - A forward-vs-backtest tracking chart for the exploration book: re-run the backtest over the paper window and compare.
- *Impact:* catches cost-model drift and code divergence between backtest and paper. It cannot reveal real market impact, because paper has none: UNKNOWN.

**L5. Keep LLMs out of prediction. If an LLM is ever used on text, apply contamination controls.**
- *Failure seen:* in-sample look-ahead and "distraction" [13]; a memorisation-amplified signal [14]; unstable decisions [7]; hallucinated numbers [17].
- *Already:* no LLM price prediction. Headlines are classified by fixed rules because the LLM "would already know what happened next" (`docs/WHAT-THE-BOT-NEEDS.md`).
- *Adopt for any future LLM text feature:*
  - evaluation only on data after the model's training cutoff;
  - entity masking (anonymised headlines) [13];
  - fixed prompts and settings, with an agreement check across repeated calls;
  - every number an LLM emits must match a sourced field, or it becomes `UNKNOWN`.
  - Optionally, run a Look-Ahead-Bench-style in-sample vs out-of-sample comparison [15].
- *Impact:* protects validity. Return impact: UNKNOWN.

**L6. Keep risk limits hard-coded and outside any model.**
- *Failure seen:* Alpha Arena leverage and no stops [6]. Persona bots that hand sizing to the LLM.
- *Already:* no leverage (gross exposure ≤ 1.0), stops, position caps, drawdown pause, kill switch. The ai-hedge-fund project reached the same design: deterministic code owns sizing and risk [18].
- *Adopt:* no change. Add a test that no LLM output can reach sizing or order code paths.
- *Impact:* prevents the worst failure seen in live contests. Return impact: none.

**L7. Pre-register a portfolio-level volatility-targeting overlay as a risk test, not an alpha test.**
- *Evidence:*
  - fewer extreme returns in every asset class [44];
  - a Sharpe gain only for equities and credit [44];
  - regression-tuned versions fail out-of-sample [45].
- *Already:* per-trade ATR stops and risk per trade. No book-level vol target (I found no `vol_target` in config).
- *Adopt:* a simple rule (e.g. scale exposure to a target realised vol, 20-day half-life, no fitted parameters), tested against the unscaled book on the pre-registered metrics (max drawdown, 1% tail, Sharpe).
- *Impact:* tail-risk reduction is likely given [44]. Sharpe impact for a long-only US-equity book: small or UNKNOWN.

**L8. Treat Kelly as a ceiling and size at zero until an edge passes the holdout.**
- *Evidence:* [46]. Edge estimates are noisy.
- *Already:* quarter-Kelly capped.
- *Adopt:* derive the Kelly input from the *shrunk, post-holdout* edge, not the backtest edge. Log the implied full-Kelly fraction next to the size actually used.
- *Impact:* prevents over-sizing on overfit edges. Return impact: none until an edge exists.

**L9. Make regime splits a required part of every result.**
- *Failure seen:* FINSABER [11]; AIEQ in 2022 [20].
- *Already:* period splits in event studies (`docs/WHAT-THE-BOT-NEEDS.md`); the SPY 200-day throttle.
- *Adopt:*
  - Require performance by pre-defined regime (SPY above or below its 200-day average, high or low VIX, the 2020-21 vs 2022-24 style splits) in every promotion packet.
  - Pre-register a test of the throttle itself against no throttle.
- *Impact:* exposes bull-only strategies before paper. Return impact: UNKNOWN.

**L10. Test a simple diversified trend baseline (paper only, ETFs).**
- *Evidence:* [42]. Honest caveats: [43] and the narrower universe.
- *Already:* `momentum_trend` and `sector_rotation` strategies exist. No multi-asset time-series-momentum book.
- *Adopt:* a pre-registered TSMOM on liquid ETFs (equity, bond, gold, commodity and currency ETFs; Alpaca has no futures). Compare it with SPY and a 60/40 portfolio after costs. Judge it as a *diversifier* (correlation, drawdown overlap), not as a SPY-beater.
- *Impact:* historically modest Sharpe with long flat spells. For an ETF-only version: UNKNOWN.

**L11. Add decay monitoring and kill rules for anything promoted.**
- *Evidence:* returns 58% lower after publication [36]; LLM signals decay with adoption [12].
- *Already:* holdout and pre-registration.
- *Adopt:*
  - At promotion, pre-register the expected forward range. When setting expectations from a published effect, start from about half its paper size [36].
  - Pre-register a demotion rule, e.g. forward excess return below the backtest's 5th percentile over N trades.
- *Impact:* limits time spent on dead signals. Size: UNKNOWN.

**L12. Do not import "popular" bot strategies on reputation.**
- *Failure seen:* GitHub popularity (TradingAgents, ai-hedge-fund) and vendor claims are not evidence [10][28][29].
- *Already:* CLAUDE.md requires a component to beat its simpler baseline.
- *Adopt:* the RSI(2) / Connors family [47] and LLM-persona selection enter only as pre-registered hypotheses with costs, a post-2008 split and the sealed holdout.
- *Impact:* stops time and paper capital going to the commonest failure in section 3.

---

## 6. Sources

1. Nof1, Alpha Arena home page, as indexed by search on 2026-10-08 ("official competition has ended as of December 3rd, 2025 ... Mystery Model was the winner with a 12.11%"). The live page fetched later showed no season text. https://nof1.ai/
2. Tech in Asia, "DeepSeek leads AI trading contest as GPT-5 posts big loss" (Oct 2025). https://www.techinasia.com/news/deepseek-leads-ai-trading-contest-as-gpt-5-posts-big-loss
3. GN Crypto, "Qwen wins Alpha Arena Season 1 with 22% returns". https://www.gncrypto.news/news/qwen-wins-alpha-arena-season-1-with-22-percent-returns
4. Forklog, "AI model Grok 4.2 triumphs in trading tournament" (season 1 balances, season 1.5 modes and winner). https://forklog.com/en/ai-model-grok-4-2-triumphs-in-trading-tournament/
5. Jiemian, "Alibaba's Qwen wins global AI investing contest as U.S. rivals post losses". https://en.jiemian.com/article/13592098.html
6. AlgoAlpha blog, "ChatGPT vs. a Trading Algorithm in 2026" (secondary: leverage, fees and trade counts are UNVERIFIED). https://algoalpha.co/blog/chatgpt-vs-trading-algorithm
7. Bloomberg (Justina Lee, 2026-05-06), "AI Bots Auditioning for Wall Street Trading Are Mostly Losing" (excerpt only, paywalled). https://news.bloomberglaw.com/banking-law/ai-bots-auditioning-for-wall-street-trading-are-mostly-losing
8. Chen et al., "StockBench: Can LLM Agents Trade Stocks Profitably in Real-world Markets?", arXiv 2510.02209 (v2, Mar 2026). https://arxiv.org/abs/2510.02209
9. Xiao et al., "TradingAgents: Multi-Agents LLM Financial Trading Framework", arXiv 2412.20138 (v7, Jun 2025). https://arxiv.org/abs/2412.20138
10. T-row, "The most-starred LLM trading paper claims buy-and-hold lost 5.23%; it actually gained 9.12%" (DEV Community). https://dev.to/trow126/the-most-starred-llm-trading-paper-claims-buy-and-hold-lost-523-it-actually-gained-912-1jj6
11. Li, Kim, Cucuringu and Ma, "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" (FINSABER), arXiv 2505.07078. https://arxiv.org/abs/2505.07078
12. Lopez-Lira and Tang, "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models", arXiv 2304.07619. https://arxiv.org/abs/2304.07619
13. Glasserman and Lin, "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis", arXiv 2309.17322. https://arxiv.org/abs/2309.17322
14. Gao, Jiang and Yan, "A Test of Lookahead Bias in LLM Forecasts", arXiv 2512.23847. https://arxiv.org/abs/2512.23847
15. "Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance", arXiv 2601.13770. https://arxiv.org/abs/2601.13770
16. He, Lv, Manela and Wu, "Chronologically Consistent Large Language Models", arXiv 2502.21206. https://arxiv.org/abs/2502.21206
17. Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering", arXiv 2311.11944. https://arxiv.org/abs/2311.11944
18. virattt/ai-hedge-fund (README disclaimer and design notes). https://github.com/virattt/ai-hedge-fund
19. Amplify ETFs, AIEQ fact sheet (since-inception NAV +129.80% cumulative, +10.03% annualised). https://amplifyetfs.com/wp-content/uploads/files/Amplify_AIEQ_FactSheet.pdf
20. YCharts, AIEQ (calendar-year returns 2019-2025 vs S&P 500 TR; turnover 804%). https://ycharts.com/companies/AIEQ
21. Charles Schwab, AIEQ performance report (data as of 2026-07-31). https://www.schwab.wallst.com/cgi-bin/upload.dll/file.pdf?z0b8f7d0az45db7658469b416288adb7e8180fb383
22. Morningstar / MarketWatch, "Has AI gotten any better at stock picking?" (2026-08-26). https://www.morningstar.com/news/marketwatch/2026082598/has-ai-gotten-any-better-at-stock-picking
23. SEC filing, ETF Series Solutions 497 liquidation supplement for AIIQ (2022). https://www.sec.gov/Archives/edgar/data/1540305/000089418922004560/equbotaiiqliquidationstick.htm
24. Exchange Traded Concepts, "to close and liquidate QRAFT AI-Enhanced U.S. Large Cap ETF (QRFT)" (Jun 2026). https://finviz.com/news/364968/exchange-traded-concepts-to-close-and-liquidate-qraft-ai-enhanced-us-large-cap-etf-nyse-qrft
25. The Daily Upside, "AI-powered ETFs aren't living up to investor expectations" (quotes Morningstar's Bryan Armour). https://www.thedailyupside.com/etf/thematics-sectors/ai-powered-etfs-arent-living-up-to-investor-expectations/
26. CFA Institute Enterprising Investor, "AI: What have you done for us lately?" (2019). https://rpc.cfainstitute.org/blogs/enterprising-investor/2019/ai-what-have-you-done-for-us-lately
27. Oninvest, "AI-managed investing: which smid-cap funds have outperformed the Russell 2000 YTD?" (2026-08-11). https://en.oninvest.com/article/ai-managed-investing-which-smid-cap-funds-have-outperformed-the-russell-2000-ytd
28. Danelfin reviews quoting company figures (I could not trace each figure to one page; UNVERIFIED). https://diyai.io/ai-tools/finance/reviews/danelfin-review/ and https://tooliverse.ai/tools/danelfin
29. Liberated Stock Trader, Tickeron review (34 bots, 40-169% claimed annualised). https://www.liberatedstocktrader.com/tickeron-review/
30. Trade Ideas Holly review (simulated figures, fills at the signal price). https://beginnersinai.org/trade-ideas-review/ ; company page: https://trade-ideas.com/compare
31. Quantopian, Wikipedia (2016 Point72 commitment, Feb 2020 capital return, Nov 2020 shutdown). https://en.wikipedia.org/wiki/Quantopian ; eFinancialCareers: https://www.efinancialcareers.com/news/2020/11/quantopian-shutdown
32. Wiecki, Campbell, Lent and Stauth, "All that Glitters Is Not Gold: Comparing Backtest and Out-of-Sample Performance on a Large Cohort of Trading Algorithms" (2016). https://papers.ssrn.com/abstract=2745220 ; summary: https://quantpedia.com/quantopians-academic-paper-about-in-vs-out-of-sample-performance-of-trading-alg/
33. Bailey, Borwein, López de Prado and Zhu, "The Probability of Backtest Overfitting". https://papers.ssrn.com/abstract=2326253
34. Bailey and López de Prado, "The Deflated Sharpe Ratio" (2014). URL from memory, not re-checked in this review: https://papers.ssrn.com/abstract=2460551
35. CXO Advisory, "Measuring Investment Strategy Snooping Bias" (summary of the Bailey et al. overfitting work, including the 45-variant illustration). https://www.cxoadvisory.com/big-ideas/measuring-investment-strategy-snooping-bias/
36. McLean and Pontiff, "Does Academic Research Destroy Stock Return Predictability?", Journal of Finance 2016. https://www.gwern.net/doc/economics/2016-mclean.pdf ; CFA digest: https://rpc.cfainstitute.org/research/cfa-digest/2016/06/does-academic-research-destroy-stock-return-predictability-digest-summary
37. Hou, Xue and Zhang, "Replicating Anomalies", NBER w23394. https://www.nber.org/papers/w23394
38. Harvey, Liu and Zhu, "...and the Cross-Section of Expected Returns" (t ≥ 3 hurdle), summary. https://www.advisorperspectives.com/articles/2015/08/18/why-you-shouldn-t-trust-most-financial-research
39. Novy-Marx and Velikov, "A Taxonomy of Anomalies and Their Trading Costs", RFS 2016 / NBER w20721. https://www.nber.org/papers/w20721
40. Chague, De-Losso and Giovannetti, "Day Trading for a Living?" (FGV working paper). https://repositorio.fgv.br/items/e87d04fc-4b4c-4a56-ab5a-1c84c1afde1b/full
41. Barber, Lee, Liu and Odean, "The Cross-Section of Speculator Skill: Evidence from Day Trading". https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf
42. Hurst, Ooi and Pedersen, "A Century of Evidence on Trend-Following Investing", JPM 2017 (AQR). https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing ; Swedroe summary: https://www.etf.com/sections/index-investor-corner/swedroe-why-financial-trends-persist
43. Cambridge Associates, "Does Trend-Following's Recent Struggle Signal That the Strategy Is Structurally Broken?" https://www.cambridgeassociates.com/insight/does-trend-followings-recent-struggle-signal-that-the-strategy-is-structurally-broken/
44. Harvey, Hoyle, Korgaonkar, Rattray, Sargaison and Van Hemert, "The Impact of Volatility Targeting", JPM 2018, summary. https://quantpedia.com/the-impact-of-volatility-targeting-on-equities-bonds-commodities-and-currencies
45. Cederburg, O'Doherty, Wang and Yan, "On the Performance of Volatility-Managed Portfolios", JFE 2020. https://www.lehigh.edu/~xuy219/research/COWY.pdf ; summary: https://alphaarchitect.com/does-portfolio-timing-based-on-volatility-signals-outperform-buy-and-hold/
46. MacLean, Thorp and Ziemba, "Good and bad properties of the Kelly criterion". https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
47. CXO Advisory, "A Few Notes on Short-term Trading Strategies That Work" (Connors, 1995-2007, no costs). https://www.cxoadvisory.com/technical-trading/a-few-notes-on-short-term-trading-strategies-that-work/
48. Gort, Liu et al., "Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting", arXiv 2209.05559. https://arxiv.org/abs/2209.05559 ; FinRL-Meta, arXiv 2211.03107: https://arxiv.org/abs/2211.03107
49. QuantConnect docs, Live Trading: Reconciliation. https://quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation
50. Freqtrade docs, Lookahead analysis. https://www.freqtrade.io/en/stable/lookahead-analysis/
