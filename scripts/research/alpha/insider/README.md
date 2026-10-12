# Insider-purchase screen (2026-10-08). A screen, NOT a validated signal

Data: OpenInsider pages, which republish SEC Form 4 filings, downloaded into an isolated scratch folder and
treated as untrusted data. Price, volume and news context come from Alpaca. No order path.

1. Download the pages (polite, ~1.5 s apart). Pages used:
   - `/latest-cluster-buys`;
   - `/top-insider-purchases-of-the-week` and `-of-the-month`;
   - a 90-day screener of officer and director purchases at $5 or more and $100k or more;
   - one page per ticker for the history check.
2. Parse each page: `python -I parse_oi.py PAGE.html > page.json`.
3. Score: `.venv/bin/python score_insiders.py <folder with oi_*.json>`.

Scoring rules were fixed before any forward return was looked at:

| Points | Condition |
|---|---|
| +2 | CEO or CFO bought (+1 for another top officer) |
| +2 / +3 | 2 / 3 or more distinct insiders |
| +1 / +2 | $1M / $5M or more in total |
| +1 | Some buyer's stake rose 10% or more |
| +1 | Bought 20% or more below the 52-week high |
| −3 | Only 10% holders bought |
| −2 | Illiquid (median daily value under $5M) |
| flag | Under $5, recent listing, or a round-price offering |

Results: `research/alpha/results/insider/`.

The historical test of this rule (Batch 2, D1 in docs/IMPROVEMENT-PROGRAM.md) needs the SEC's own Form 4
datasets. The SEC requires a contact email in the request header, and that is pending Harry's OK.
