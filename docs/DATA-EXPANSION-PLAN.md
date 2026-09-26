# Data expansion plan (discovery context families)

## Status (updated 2026-09-26, after the catalyst phase)

Items 1-5 below are now IMPLEMENTED and ingested on a scratch copy of the database (never the live
paper DB); item 6 remains UNKNOWN by design. What was built and verified:

| # | Source | Implementation | Real ingestion (scratch copy) | PIT validation |
|---|---|---|---|---|
| 1 | SEC 8-K 2.02 earnings | `data/sec_catalysts.py`, `catalysts ingest-sec` | 5,229 symbols, 21 chunks, 88,736 releases 2020-01..2026-09 | AAPL matches the verified acceptance times to the second; mean abs abnormal move peaks on the computed reaction day (5.7% pre-market / 6.7% post-close vs 2.0-2.6% the day before); 0 of 51,770 events available after the reaction they are measured against |
| 2 | Market-wide news | `get_news_market`, `catalysts ingest-news` | 2021-01..2024-12 and 2026-06..2026-09 (2025-01..2026-05 not ingested: holdout / not needed yet) | available_at = created_at on every row; revisions > 60 s flagged PIT_CONSERVATIVE (5%); no backward leak at a test cutoff |
| 3 | Point-in-time SIC | `sic_observation` events from filing headers | 4,994 symbols, 11,898 observations, 489 SIC changes (243 de-SPACs from 6770) | the current snapshot is only in `sec_registrant` rows (available_at = retrieval time) |
| 4 | Fundamentals | `catalysts ingest-facts` (companyfacts + acceptance join) | 3,572,544 facts, 4,954 symbols | as-of values change only from the filing's first usable session; truncation-invariant on real data; fiscal Q4 derived as-of (fixes a pre-existing Q4 drop) |
| 5 | Other SEC events | same pass as 1 | 178,892 material 8-Ks, 96,425 periodic reports, 107,208 foreign reports, 15,254 SC 13D, 7,739 S-1/S-3 | acceptance time, like 1 |
| 6 | Estimate revisions | not implemented | - | UNKNOWN: no point-in-time source |

The plan as written before implementation follows.

_Written 2026-09-26. A plan only: nothing here is ingested yet. Facts about providers come from
[EXTERNAL-SERVICES.md](EXTERNAL-SERVICES.md) (`[An]` = item n of its Alpaca Market Data section, `[Sn]` = item n of its SEC EDGAR section). Anything
not verified there is marked UNVERIFIED or UNKNOWN._

## Why this matters now

The forward-outcome research (`quantlab discovery-research`, report `research_5aeec91f9b4a4209`,
2021-03-12..2024-11-27, holdout-safe) found every price/volume family and combination **FLAT**
against the same-date baseline at 5 sessions, net of costs. Momentum and relative strength vs SPY
are close to the same signal (per-date Spearman 0.94 of family points). More price/volume
arithmetic is unlikely to help. The context families (earnings, news, fundamentals, sector) are
the untested inputs, and they are UNKNOWN for almost the whole universe today.

## Current coverage (real, non-synthetic datasets)

| Dataset | Provider | Symbols | Span | Used by discovery |
|---|---|---|---|---|
| bars | Alpaca | ~5,200 | 2020-01-02..2026-09-24 | yes (all scored families) |
| corporate_actions | Alpaca | 2,900 | 2020-01-02..2026-09-24 | pre-open recheck only |
| news | Alpaca (Benzinga) | 4 | 2017-05-27..2026-09-24 | context, 4 symbols only |
| events (8-K) | SEC EDGAR | 2 | 2020-01-29..2026-07-31 | context, 2 symbols only |
| fundamentals | none | 0 | none | UNKNOWN everywhere |
| sector / industry | none (sector ETF bars only) | 0 | none | UNKNOWN per symbol |
| estimate revisions | none | 0 | none | UNKNOWN everywhere |
| pre-market prices | none | 0 | none | pre-open recheck marks UNKNOWN |

## Priority order

### 1. Historical earnings events for the full universe (SEC 8-K item 2.02)

* **Provider.** SEC EDGAR submissions JSON per CIK, or the nightly bulk `submissions.zip` [S27].
  Free. 10 requests/s fair-access limit [S2]; a declared User-Agent is required [S3].
* **Historical depth.** `filings.recent` plus the older pages give the full filing history [S6][S7].
  Bar history starts 2020, so 2019 onward is enough.
* **PIT integrity.** High. `acceptanceDateTime` is true UTC [S8] and is the real acceptance
  instant even when `filingDate` rolls forward [S9]. Earnings 8-Ks are often accepted after the
  16:00 ET close (e.g. 16:30 ET) [S11], so an announcement belongs to the next session's
  information set. That is exactly the overnight-catalyst case the next-session mode handles.
* **Coverage.** US domestic filers. Foreign private issuers file 6-K/20-F, which have no item 2.02
  and stay UNKNOWN. ETFs have no earnings (NOT APPLICABLE, not zero).
* **Timestamps.** Per filing, to the second.
* **Survivorship risk.** Medium. `company_tickers_exchange.json` and the submissions `tickers`
  field are current only [S28][S30]. Delisted or renamed tickers need a CIK mapping from
  historical sources (formerNames has name history, not ticker history). Unmapped symbols must
  stay UNKNOWN, never "no earnings".
* **Look-ahead risk.** Low if availability = `acceptanceDateTime`. Never use `filingDate` or
  `reportDate` as the availability time.
* **Storage.** Small: only 8-K rows are kept (the existing `events` schema). The bulk ZIP is about
  1.6 GB to download once [S27]; the per-CIK route is about 5k requests (roughly 10 minutes at
  10 req/s).
* **Safe for discovery?** Yes, as context ("earnings announced after the last close", "sessions
  since last earnings"). Forward earnings **dates** (the calendar) are not in EDGAR, so "earnings
  expected tomorrow" stays UNKNOWN. No verified PIT source for the calendar exists.

### 2. Full-universe news (Alpaca news, Benzinga)

* **Provider.** Alpaca `/v1beta1/news` [A31]. Already integrated for 4 symbols.
* **Historical depth.** Back to 2015 [A34].
* **PIT integrity.** Medium. `available_at = created_at` is safe. Sorting and paging use
  `updated_at` [A33], and REST appears to return only the latest revision of headline, summary and
  symbols (EXTERNAL-SERVICES caveats). So the headline text and the symbol tags may be revised
  after the fact.
* **Coverage.** Single source (Benzinga) [A34]. Small caps are thinly covered. Absence of news is
  not evidence of no news: a symbol with no article stays "no article found", never zero sentiment.
* **Timestamps.** `created_at` per article, to the second.
* **Survivorship risk.** Low for market-wide pages (query with `symbols` empty). The tags use the
  ticker at the time of the article, so a symbol join needs the same ticker history as item 1.
* **Look-ahead risk.** Medium for text features (revised headlines), low for event presence and
  counts keyed on `created_at`.
* **Storage.** UNKNOWN until measured. The docs quote 130+ articles/day [A34]; the 4 ingested
  symbols alone hold 105,346 rows. Measure one month of market-wide pages before a full backfill.
  Rate limit shared with bars on the Basic plan (200/min, UNVERIFIED for news) [A14].
* **Safe for discovery?** Yes for event presence ("new article after the cutoff", article count),
  which the overnight refresh already uses. Not for headline sentiment until revision behaviour
  is verified.

### 3. Point-in-time sector / industry (SIC to Fama-French industries)

* **Provider.** SEC per-filing SGML headers (`STANDARD INDUSTRIAL CLASSIFICATION`) [S29][S30], mapped
  with Kenneth French's industry definitions (`Siccodes12.zip`, `Siccodes49.zip`) [S32].
* **Historical depth.** Every filing header, so back before 2020.
* **PIT integrity.** High only from the headers. The submissions `sic` field is a **current-only
  snapshot** [S30]; using it historically is look-ahead and must not be done.
* **Coverage.** SEC filers with an SIC code. ETFs and some ADRs have no useful SIC (UNKNOWN).
* **Timestamps.** SIC as of each filing's acceptance time; carry it forward until the next filing.
* **Survivorship risk.** Medium (same ticker-to-CIK problem as item 1).
* **Look-ahead risk.** Low from headers; high from the snapshot field.
* **Storage.** Tiny (one row per symbol per filing year). Cost is requests: one header per 10-K per
  year is about 30k requests for 5k symbols over 6 years (about 50 minutes at 10 req/s).
* **Safe for discovery?** Yes. It would let relative strength be measured against the industry
  instead of SPY, which is the only way to make "relative strength" differ from momentum (the
  research shows RS vs SPY is momentum minus a per-date constant).

### 4. Point-in-time fundamentals (SEC XBRL companyfacts)

* **Provider.** SEC companyfacts per CIK or bulk `companyfacts.zip` (about 1.4 GB) [S27].
* **Historical depth.** XBRL starts 2009 [S26].
* **PIT integrity.** High if each fact's availability is its filing's `acceptanceDateTime`
  (join `accn` to the submissions history [S14]). The `frame` field sits on the **last-filed**
  fact and is look-ahead; never use it as a selector [S17]. The 8-K 2.02 does not feed
  companyfacts; the quarter's numbers arrive with the later 10-Q/10-K [S12].
* **Coverage.** US GAAP filers. Company-extension and dimensional facts are excluded [S22] (e.g.
  multi-class share counts are missing). Revenue concept names change over time and need a
  fallback chain [S23].
* **Timestamps.** Per filing (via the accession join).
* **Survivorship risk.** Medium (ticker-to-CIK, as item 1).
* **Look-ahead risk.** High if `frame` or `filed` date is used; low with the acceptance join.
* **Storage.** Large raw (1.4 GB compressed); kept facts are modest if limited to a concept list.
  The `fundamentals` schema already exists.
* **Safe for discovery?** Yes as slow context (valuation, growth), but low value at 1-20 session
  horizons. Do after 1-3.

### 5. Other SEC events (8-K items, offerings, ownership filings)

* **Provider.** Same submissions data as item 1 (`items` column [S11], `form`).
* **PIT integrity / timestamps.** Same as item 1 (`acceptanceDateTime`).
* **Coverage.** US filers. Item meaning must be mapped per item code.
* **Survivorship / look-ahead.** As item 1. 8-K/A amendments must be dated by their own
  acceptance time, not the original's.
* **Storage.** Small (a superset of item 1's rows).
* **Safe for discovery?** Yes as event flags. Almost free once item 1 exists.

### 6. Estimate revisions (analyst consensus)

* **Provider.** None available. No free point-in-time source is verified.
* **PIT integrity.** Current consensus snapshots are look-ahead when used historically and must not
  be substituted for history.
* **Decision.** Not planned. Stays UNKNOWN. Revisit only with a vendor that provides as-of
  snapshots with timestamps (UNVERIFIED which vendors do and at what cost).

## Also missing for full overnight catalyst coverage

* **Pre-market prices.** The pre-open recheck marks the pre-market price UNKNOWN. On the Basic plan,
  SIP history must be at least 15 minutes old [A10] and paper-only accounts may get IEX only [A11];
  whether pre-market bars are usable before the open is UNVERIFIED.
* **Forward earnings calendar.** See item 1: not available, stays UNKNOWN.
* **Corporate actions stream.** Pre-open uses the REST corporate-actions dataset (2,900 symbols).
  Whether the SSE stream is available on Basic is UNVERIFIED (EXTERNAL-SERVICES open questions).

## Rules for any new source

* Every row carries `available_at` (UTC) and `pit_status`. Discovery reads rows with
  `available_at <= cutoff` only.
* A symbol the source does not cover is UNKNOWN, never zero and never "no event".
* Current snapshots are never back-filled into history.
* A new context family stays unscored until the forward-outcome research shows it beats the
  same-date baseline out of sample, and the comparison is recorded in the research ledger.
