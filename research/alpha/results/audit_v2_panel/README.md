# Equity families re-run on panel v2 (2026-10-06/07)

Panel v2 contains the first audit's data fixes: twins de-duplicated (M2), the distress flag on
adjusted prices, and fixed spy_vol bins. TRAIN statistics start at each book's first active day.

These are the six equity families re-run under override tag `audit-2026-10`. Every selected variant
and class matched the first run (`../pre_audit/`).

These results were then superseded by panel v3, which drops zero-volume filler bars and uses the
contiguous-run twin rule; see ALPHA-DISCOVERY-PLAN.md section 10.8. They are kept so each step can be
compared, and their OOS looks are counted in the ledger.
