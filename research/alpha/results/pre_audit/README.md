# First-run results, before the 2026-10-06 adversarial audit

These are copies of `research/alpha/results/` and `queue.json` as they stood at commit 45bc577, before
the audit fixes (C1, C2, M1-M6, see docs/ALPHA-DISCOVERY-REPORT-2026-10.md). Do not cite them as
current results. Several are known to be wrong:
- the options decile families for the full universe were contaminated by wrong-company joins (C2);
- the H26 placebo landed on the previous earnings event (C1);
- the calendar, options and estimate joins dropped companies that later delisted (M1);
- the H33 t-statistics were overstated (M4).

They are kept, not deleted (Part 52), so every first-run number in the ledger can be compared with
its corrected re-run.
