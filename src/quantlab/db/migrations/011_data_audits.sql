-- 011_data_audits.sql: real-data suitability audits (data/audit.py). Append-only.
CREATE TABLE data_audits (
    audit_id          TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL,     -- NOT_RUN | FAILED | INCONCLUSIVE | DATA_LIMITATION | SUITABLE_SMALL_SAMPLE_ONLY | SYNTHETIC_ONLY
    reason            TEXT,
    symbols_json      TEXT NOT NULL,
    start_date        TEXT,
    end_date          TEXT,
    feed              TEXT,
    dataset_ids_json  TEXT,
    checks_json       TEXT NOT NULL
);

CREATE TRIGGER trg_data_audits_no_update BEFORE UPDATE ON data_audits
BEGIN SELECT RAISE(ABORT, 'data_audits are append-only'); END;
CREATE TRIGGER trg_data_audits_no_delete BEFORE DELETE ON data_audits
BEGIN SELECT RAISE(ABORT, 'data_audits are append-only'); END;
