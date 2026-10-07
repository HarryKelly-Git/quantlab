# Importing the paper bot's database (Phase 3, missed-opportunity engine)

The bot's own records are the freshest evidence QuantLab has. These are the 570+ shadow opportunities,
with the reason each was rejected and the bot's own forward outcomes: returns, MFE/MAE and stop/target
hits. Only the `shadow_*` tables are read, and the analysis is descriptive. No filter is changed on a
single sample.

## 1. On the PC (Windows, repo root, any time; ~1 minute)

Make a consistent copy while the bot keeps running (SQLite online backup; the live file is untouched):

```
.venv\Scripts\python -c "import sqlite3; sqlite3.connect('var/quantlab.db').backup(sqlite3.connect('var/quantlab_export.db'))"
```

## 2. Get the copy to the cloud session (pick one)

- **Git (simplest).** `var/` is git-ignored, so force-add it on a separate branch:
  ```
  git checkout -b bot-db-export
  git add -f var/quantlab_export.db
  git commit -m "bot db export"
  git push -u origin bot-db-export
  git checkout master
  ```
  The file holds paper-account records only. Secrets live in `.env` and are never in the database.
- **Ask the PC Claude chat to do step 1 and the git push.** Approve its permission prompt yourself. The
  earlier cross-session attempt was blocked because the approval has to come from you.

## 3. In the cloud session

```
git fetch origin bot-db-export
git show origin/bot-db-export:var/quantlab_export.db > var/quantlab_export.db
.venv/bin/python scripts/research/alpha/run_botdb.py var/quantlab_export.db
```

The run writes `research/alpha/results/botdb_missed_opportunities.json`. It covers schema validation,
quality checks, and traded-vs-rejected results by horizon, decision, reject stage, top reject reasons,
strategy and score rank. It also asks whether QuantLab is too conservative. That verdict needs at least
20 distinct as-of dates, and the confidence intervals resample whole dates, because same-day candidates
share one market.
