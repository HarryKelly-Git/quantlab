-- 092_lab_reports.sql — persisted counterfactual and human-vs-bot reports.
--
-- Reports are research evidence: once saved they are never edited (a newer report is a new row),
-- so a conclusion can always be traced to the exact numbers and parameters it was drawn from.

CREATE TABLE lab_reports (
    report_id     TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,                     -- counterfactual | human_vs_bot
    created_at    TEXT NOT NULL,
    period_start  TEXT,
    period_end    TEXT,
    params_json   TEXT NOT NULL,
    config_hash   TEXT,
    report_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_lab_reports_kind ON lab_reports(kind, created_at);

CREATE TRIGGER trg_lab_reports_no_update BEFORE UPDATE ON lab_reports
BEGIN SELECT RAISE(ABORT, 'lab_reports are append-only'); END;
CREATE TRIGGER trg_lab_reports_no_delete BEFORE DELETE ON lab_reports
BEGIN SELECT RAISE(ABORT, 'lab_reports are append-only'); END;
