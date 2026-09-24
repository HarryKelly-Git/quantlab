-- 080_decision_stack.sql - decision/portfolio/risk subsystem (reserved range 080-089).
--
-- `candidates` rows are immutable once written (001 trigger), so the ranking produced later in the
-- decision chain cannot be written back into candidates.rank / candidates.opportunity_score.
-- It is stored here instead, append-only, one row per ranked opportunity (symbol x direction x day)
-- per run. The dashboard and the report read the full evidence record from record_json.
-- opportunity_score is a RANKING in [0, 1], NOT a probability (see decision/ranking.py).

CREATE TABLE opportunity_rankings (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT,
    as_of_date            TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    direction             TEXT NOT NULL,
    rank                  INTEGER NOT NULL,
    opportunity_score     REAL NOT NULL,
    primary_candidate_id  TEXT,
    candidate_ids_json    TEXT NOT NULL,
    strategies_json       TEXT NOT NULL,
    components_json       TEXT NOT NULL,
    record_json           TEXT NOT NULL,
    is_synthetic          INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL
);
CREATE INDEX ix_opp_rankings_date ON opportunity_rankings(as_of_date, rank);
CREATE INDEX ix_opp_rankings_run ON opportunity_rankings(run_id);

CREATE TRIGGER trg_opp_rankings_no_update BEFORE UPDATE ON opportunity_rankings
BEGIN SELECT RAISE(ABORT, 'opportunity_rankings are append-only'); END;
CREATE TRIGGER trg_opp_rankings_no_delete BEFORE DELETE ON opportunity_rankings
BEGIN SELECT RAISE(ABORT, 'opportunity_rankings are append-only'); END;
