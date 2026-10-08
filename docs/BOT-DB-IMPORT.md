# Importing the paper bot's database (Phase 3, missed-opportunity engine)

The bot's own records are the freshest evidence QuantLab has. These are the 570+ shadow opportunities,
with the reason each was rejected and the bot's own forward outcomes: returns, MFE/MAE and stop/target
hits. Only the `shadow_*` tables are read, and the analysis is descriptive. No filter is changed on a
single sample.

## 1. On the PC (Windows, QuantLab folder, any time; ~1 minute)

Open PowerShell IN the QuantLab folder: the one that contains `.venv` and `scripts`. Either type `powershell`
in File Explorer's address bar there, or run `cd "<path to quantlab>"` first. In PowerShell the command must
start with `.\`, or PowerShell reports "The module '.venv' could not be loaded".

Make a consistent copy while the bot keeps running (SQLite online backup; the live file is untouched):

```
.\.venv\Scripts\python.exe -c "import sqlite3; sqlite3.connect('var/quantlab.db').backup(sqlite3.connect('var/quantlab_export.db'))"
```

## 2. Get the copy to the cloud session

> **The GitHub repository `HarryKelly-Git/quantlab` is PUBLIC (checked 2026-10-07).** Do NOT push the
> database to any branch while it is public. A pushed file is world-readable and may be cached even
> after deletion.

Pick one:

- **Upload it into the Claude chat (preferred).** Attach `var/quantlab_export.db` to a message in the
  cloud session. It lands in `/mnt/user-data/uploads/`.

  If the file is too large to attach, export only the five tables the analysis reads:
  `shadow_opportunities`, `shadow_outcomes`, `shadow_outcome_details`, `risk_checks`, `decisions`.
  Ready-made request for the PC Claude chat, which you approve yourself:

  > "Create var/quantlab_export.db containing only the tables shadow_opportunities, shadow_outcomes,
  > shadow_outcome_details, risk_checks and decisions copied from var/quantlab.db (read-only; do not
  > modify var/quantlab.db), then tell me the file size."
- **Git, ONLY after making the repository private** (GitHub: Settings -> General -> Danger Zone ->
  Change visibility). `var/` is git-ignored, so force-add it on a separate branch:
  ```
  git checkout -b bot-db-export
  git add -f var/quantlab_export.db
  git commit -m "bot db export"
  git push -u origin bot-db-export
  git checkout master
  ```
  The file holds paper-account records only. Secrets live in `.env` and are never in the database.

## 3. In the cloud session

Uploaded file:
```
.venv/bin/python scripts/research/alpha/run_botdb.py /mnt/user-data/uploads/quantlab_export.db
```

Private-repo git route:
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
