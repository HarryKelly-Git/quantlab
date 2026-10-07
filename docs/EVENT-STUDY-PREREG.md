# Event study: do big news-driven moves repeat in a tradeable way? (pre-registration, 2026-10-07)

Committed BEFORE any event outcome was computed. PAPER research.

## Motivation

Harry's Vistra trade went from 145 to about 166 in two days:

| Day | News | VST | Peers |
|---|---|---|---|
| Fri 2026-10-02, 3:27pm ET | Bloomberg: a ~$4B DOE nuclear loan for Vistra | | |
| Mon 2026-10-05 | Loan confirmed | +3.5% | |
| Tue 2026-10-06 | Google signs a 20-year deal for Constellation's reactors; the whole nuclear/IPP group re-rates | +10.8% | CEG +12.2%, TLN +12.4%, NRG +7.0%; SPY +0.5% |

The question is whether moves LIKE this repeat in a way a close-of-day bot can exploit, measured across
thousands of past examples.

## Scope and data

- **Not used:** the 2026 example itself is not data here. The 2025+ holdout stays sealed: no 2025+ prices
  or news are used.
- **Prices:** survivorship-free research store, 2016-2024, sprint-c corrected panel.
- **News (new data):** Alpaca/Benzinga articles, symbol-tagged.
  - Timestamps are `created_at`, the earliest availability.
  - Only articles for event dates are downloaded, 2016-2024.
- **Options:** DoltHub end-of-day chains, 2019-2024.

## Events (fixed)

**Universe:** the research liquid universe at t: price >= $5, MDV20 >= $5M, >= 252 sessions, common
stock.

**UP event at the close of day t:**
- close-to-close total return >= +8%;
- that day's dollar volume >= 2x its trailing 20-session median;
- at most one event per stock per 20 sessions (the first).

**DOWN events** (<= −8%, same rules) are recorded as the control group.

**News window:** (16:00 ET on t−1, 16:00 ET on t]. Everything is known at t's close, so it is
point-in-time for a bot that enters at the t+1 open.

**Classes** (all known at t's close):

- **NEWS:** at least 1 article tagged with the stock in the window whose headline is NOT a price recap.
  - A recap headline matches, case-insensitive:
    `shares (are )?(trading|moving)|stock is (trading|moving|soaring|falling)|why is .* (stock|shares)|what'?s going on with|movers|mid-?day|pre-?market|after-?hours|52-week high|52-week low|gap(ping)? (up|down)|session|stocks? (moving|to watch)|top (gainers|losers)|unusual options`
  - Otherwise the event is **NO-NEWS**: no article, or only recaps.
- **CATEGORY:** keyword match on the substantive headlines, first match wins.

  | Category | Keywords |
  |---|---|
  | M&A target | acquire, to buy, buyout, takeover, merger agreement, to be acquired, go private |
  | earnings | earnings, EPS, revenue, guidance, quarter, Q1-Q4, results, outlook |
  | analyst | upgrade, downgrade, price target, initiates, reiterates |
  | clinical | FDA, approval, trial, phase, data readout |
  | deal | contract, agreement, partnership, deal, award, loan, order, selects |
  | other | anything else |

- **SECTOR-WIDE:** at least 3 OTHER stocks in the same statistical sector are up >= +5% on t. Otherwise
  **IDIOSYNCRATIC**.
- **LAGGARD PEERS** (for E2b) of a sector-wide day: same-sector liquid stocks with a day-t return of
  +1% or less.

**Outcomes:**
- Entry at the t+1 OPEN, the bot's convention. Exit at the close of t+h, for h = 1, 5, 20, 60.
- Excess = stock return minus SPY over the same window.
- Net = excess minus the round-trip cost: 2 x (master's half-spread tier by MDV20 + 5 bps slippage).
- A delisting inside the window books the panel's delisting return (0% for acquisitions, −30% if
  distressed).

## Hypotheses (four primary tests)

| ID | Claim | Primary metric |
|---|---|---|
| E1 | NEWS UP events (excluding M&A targets, which pin to the deal price) CONTINUE. NO-NEWS UP events do not (Chan 2003: drift after news, reversal after no-news moves). | mean net 20-day excess of NEWS-non-M&A events > 0, and the NEWS minus NO-NEWS difference > 0 |
| E2a | SECTOR-WIDE UP events continue more than IDIOSYNCRATIC ones | difference in mean net 20-day excess |
| E2b | LAGGARD PEERS on sector-wide days catch up | mean net 5-day excess of laggards > 0 |
| E3 | The options expression of E1: buy the ~30-DTE ATM CALL at the first chain snapshot after a NEWS-non-M&A UP event, held to expiry | mean net return per premium at the four fill levels of the sprint engine (MID / CONSERVATIVE / PESSIMISTIC = ask / WORST); classified ROBUST / EXECUTION-SENSITIVE / NONVIABLE |

## Splits and decision rule

**Splits:**

| Data | TRAIN | VAL | OOS |
|---|---|---|---|
| Equity | 2016-19 | 2020-21 | 2022-24 |
| Options (E3) | 2019-21 | 2022 | 2023-24 |

**Statistics:** date-clustered t (events entering on the same day share one market).

**Bonferroni over the 4 primary tests:** |t| >= 2.5 counts as significant.

**Verdicts** (for E1, E2a and E2b on the primary metric):

| Verdict | Condition |
|---|---|
| SURVIVES | TRAIN and VAL both > 0 with t >= 2.5, AND OOS > 0 with t >= 2.5 (or > 0 in at least 2 of the 3 OOS years with t >= 2) |
| PROMISING | TRAIN and VAL > 0, OOS > 0 but weaker. Forward paper-SHADOW only. |
| FAILS | Otherwise |

**E3:** ROBUST at PESSIMISTIC fills in OOS is required.

**Descriptive only, never a basis for selection:**
- category tables;
- horizons other than the primary;
- DOWN events;
- year-by-year results.

Thresholds are not tuned. Changing one creates a new labelled variant whose result is reported next to
the original.
