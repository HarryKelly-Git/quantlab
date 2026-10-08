# Insider buying: pre-registration (2026-10-08)

Committed BEFORE any result was computed. PAPER research. Only filings dated 2016-01-01..2024-12-31 are used.

**Why.** Officer and director open-market purchases are among the most robust public-information
signals in the literature:
- Lakonishok-Lee 2001;
- Cohen-Malloy-Pomorski 2012, about 0.8%/month for "opportunistic" buys;
- the effect is strongest in small caps.

Harry's Uber idea (CEO bought $10M in September 2026) is this signal. QuantLab has never tested it.

**Data.** OpenInsider screener pages, which republish SEC Form 4 filings, downloaded into an isolated folder
and treated as untrusted data. One query per calendar quarter, with these filters:
- open-market purchases (P);
- buyers that are officers or directors;
- transaction value $50k or more;
- price $5 or more.

The SEC's own datasets need a contact email (pending), so this is the stand-in. Rows the parser cannot
read are counted and dropped; none are guessed.

**Point in time.**
- The signal day t is the last trading session on or before the filing date (ET calendar date).
- Entry is at the next open (t+1). A filing during market hours on day D therefore enters at D+1: a
  conservative lag.
- The ticker is resolved to the company that held it at t (research store, renames and ticker reuse
  handled).
- The stock must be in the liquid universe at t.

**Signals.** Each is counted once per company per 30 days.

| ID | Rule |
|---|---|
| A | Any qualifying purchase |
| B | The CEO or CFO bought |
| C | A cluster: 2 or more distinct insiders bought within 10 calendar days. The event is the filing that completes the cluster |
| D | B or C, AND the stock was 20% or more below its 52-week high at t (the "falling knife" screen used for the 2026-10-08 list) |

**Measurement.**
- Excess return vs SPY from the t+1 open to the t+h close, for h = 5, 20 and 60 sessions.
- Net of master's round-trip cost tiers, under the corrected delisting convention.
- Date-clustered t-statistics.
- **Primary horizon: 60 sessions**, since the insider effect builds over months. 20 is secondary.

**Splits:** TRAIN 2016-19, VAL 2020-21, OOS 2022-24.

**Verdict per signal, at the primary horizon:**

| Verdict | Condition |
|---|---|
| **SURVIVES** | Mean net excess > 0 with t >= 2.5 in all three splits |
| **PROMISING** (forward paper-shadow only) | Mean > 0 in all three splits, without the t-statistic bar |
| **FAILS** | Otherwise |

Four signals are tested, so a single SURVIVES is still a candidate for forward shadowing, not a trading
rule.

---

## Results (added 2026-10-08, after the run; the rules above were not changed)

File: `research/alpha/results/insider/insider_study.json`; ledger `INS_A`..`INS_D`; code
`scripts/research/alpha/insider_study.py`.

**Data.**
- 32,629 qualifying purchases (36 quarterly queries, none hit the 5,000-row page cap).
- 2,316 could not be matched to a company in the store.
- 15,881 were in stocks outside the liquid universe on the signal day.
- **14,432 usable.**

**Net excess return vs SPY over 60 sessions** (date-clustered t in brackets):

| Signal | Events | 2016-19 | 2020-21 | 2022-24 | Verdict |
|---|---|---|---|---|---|
| A any officer/director buy | 7,950 | −0.80% (−2.0) | +1.22% (1.3) | −0.46% (−1.0) | FAILS |
| B CEO or CFO buy | 2,568 | −0.84% (−1.0) | +1.57% (1.2) | −0.57% (−0.8) | FAILS |
| C cluster (2+ insiders in 10 days) | 1,943 | +0.06% (0.1) | +0.81% (0.5) | +0.64% (0.7) | PROMISING (forward shadow only) |
| D CEO/CFO or cluster after a 20%+ fall | 2,343 | −0.27% (−0.3) | +3.40% (2.1) | +0.23% (0.3) | FAILS |

**Reading.**
- In the stocks the bot can actually trade, insider purchases did not beat SPY after costs over the
  following three months, apart from 2020-21.
- Clusters were positive in all three periods, but too small to separate from noise.
- Half of all purchases are in small, illiquid stocks. The published effect is concentrated there, and
  it was not tested here because the bot cannot trade them at its liquidity standard.
- For Uber specifically: a CEO purchase in a liquid large cap has, on average, not been a reliable
  60-day edge in 2016-24. It remains one data point in a real-money decision, which follows the Upside
  Engine v2 doctrine.
