# Congress & insider trades (free sources): CONTEXT ONLY

`alt_trades` holds insider Form 4 filings (SEC EDGAR) and US House periodic transaction reports
(House Clerk). It is **visible context and recorded data, never a trading signal**: no feature from it
is in `SCORED`, `SCORED_POINTS`, the selection score, sizing or any order path. PAPER ONLY.

## Evidence status

A pre-registered study (`../research/2026-10-01-new-data-tests/`) found insider buying **FLAT** at
realistic timing (usable from the filing, not the trade), and its one positive variant **failed the
locked 2025+ holdout**. So this layer is context for the learning loop (outcomes with vs without
insider/congress activity, recorded on every exploration decision), not a signal, until forward
evidence says otherwise. Promotion to anything scored needs a new pre-registered test.

## Sources

| source | what | discovery | availability (PIT) |
|---|---|---|---|
| insider | SEC Form 4 / 4/A, all issuers | daily form index `daily-index/<Y>/QTR<q>/form.<YYYYMMDD>.idx` + `getcurrent` Atom feed (filings not yet in an index); each filing's `<accession>.txt` | `available_at` = the `<ACCEPTANCE-DATETIME>` in the submission header (EDGAR clock, read as US Eastern; if it were UTC this would only make it later). `PIT` |
| congress | House PTRs only | annual index `financial-pdfs/<YEAR>FD.zip` (XML: name, FilingType `P`, StateDst, FilingDate, DocID); PDF `ptr-pdfs/<YEAR>/<DocID>.pdf` | FilingDate is date-only: `available_at` = cutoff of the session **after** it (the repo's date-only rule, `conservative_available_at`). `PIT_CONSERVATIVE` |

The transaction date is never availability (PTRs arrive up to 45 days after the trade).

Mapping (`data/schemas.py` documents every column):
* Insider side: BUY = code P with acquired (A) in the non-derivative table; SELL = code S with
  disposed (D); everything else (grants, exercises, tax withholding, gifts, every derivative row) OTHER.
  Value = shares x price for BUY/SELL, else NaN (UNKNOWN, never 0). Joint filings: one row per line,
  owners joined in `actor`.
* House side: P = BUY, S / S (partial) = SELL, E = OTHER; options `[OP]` and non-stock assets are OTHER
  (a put purchase is bearish: direction ambiguous); the original type stays in `raw_json`. Amount =
  the disclosed range; "Over $X" has an UNKNOWN upper bound.
* `record_status`: PARSED, NO_SYMBOL (no ticker named), NO_TRANSACTIONS (Form 4 without lines),
  UNPARSEABLE (scanned paper PTR, unreadable PDF: recorded once per DocID, **never guessed**).
* Dedupe key `(source, record_id)`: `form4:<accession>:<hash(table,code,date,shares,price,A/D)>[#n]`,
  `house:<DocID>:<line>` / `house:<DocID>:unparseable`. Amendments (4/A, amended PTRs) have their own
  accession / DocID and are separate disclosures.

## Operation

* **Evening refresh** (`data/catalyst_refresh.py`, part `alt`; only the market-wide daily refresh,
  never the pre-open candidate refresh): incremental (stored accessions / DocIDs are never fetched
  again), oldest first, bounded by `alt_data.<source>.max_filings_per_run` and
  `max_seconds_per_run`, short HTTP timeouts with one retry. Anything left is reported as
  `remaining` (status PARTIAL) and fetched by the next run. An outage (403, rate limit, network, five
  failed filings in a row, missing PDF library) reports FAILED; nothing is ever raised into the
  pipeline.
* **CLI**: `quantlab alt ingest [--days N] [--source insider|congress]`, `quantlab alt recent
  [--symbol X]`, `quantlab alt status`, `quantlab alt import-insider --path <parquet>`.
* **Politeness**: SEC requests carry `QUANTLAB_SEC_USER_AGENT` and share the process-wide `sec-edgar`
  limiter (8 req/s < SEC's 10/s). House Clerk: 1 req/s (`providers.house_clerk`). Without the SEC
  user agent the insider part is SKIPPED (UNKNOWN, not zero).
* **PDF text** needs `pypdf` (added to `pyproject.toml`; install with `.venv\Scripts\pip install pypdf`).
  Without it the congress part reports FAILED and no PTR is marked UNPARSEABLE, so it is retried later.

## Features and context (features/alt.py, discovery family `smart_money`)

`congress_buys_30d`, `congress_sells_30d`, `congress_net_30d`, `insider_buys_30d`,
`insider_sells_30d`, `insider_net_value_30d`: disclosures **usable** in the last 30 calendar days.
Insider counts use directors/officers only (the import's and the study's population). Each is
UNKNOWN until its source has delivered a whole 30-day window (from its earliest stored record), then
a quiet symbol is a real 0 (both sources cover every issuer / House member). Truncation invariance is
tested (`tests/features/test_alt_features.py`). The family is recorded on every discovery candidate
and every exploration decision's `pre_trade_json["smart_money"]`; it never fires anything scored.

## Historical backfill

* Insiders: `quantlab alt import-insider --path ../research/2026-10-01-new-data-tests/data/insider_tx_2017-2026.parquet`
  maps the research parquet (built from the SEC **Insider Transactions Data Sets**, quarterly
  `<YYYY>q<q>_form345.zip`) into `alt_trades`: directors/officers, open-market P/S, price > 0, one
  owner per accession. Its FILING_DATE is date-only, so availability is the session-after cutoff
  (PIT_CONSERVATIVE); record ids equal the live parser's, so an overlap never double-counts.
  Coverage note: the import has no NO_SYMBOL / OTHER rows and no owner names (actor = owner CIK).
* Congress: `quantlab alt ingest --source congress --days N` walks the annual indexes (2008+).

## Coverage gaps

* **Senate: not covered.** The official eFD site refuses automated access (403) and must not be
  bypassed; the free community dataset stopped in 2020. Senate trades are UNKNOWN, not zero.
* Scanned paper House PTRs are UNPARSEABLE (counted in `quantlab alt status`).
* The House posting lag after FilingDate is not published; the session-after rule assumes it is
  at most one session.
* A refresh outage longer than `refresh_days` (insider 5, congress 60) leaves a gap: back-fill it with
  `quantlab alt ingest --days N`.
