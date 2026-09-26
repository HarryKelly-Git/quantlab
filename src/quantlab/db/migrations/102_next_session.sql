-- 102_next_session.sql: unified candidate pool (origin), next-session setups, overnight refresh,
-- pre-open rechecks and discovery research summaries. Research/monitoring records only: nothing
-- here can place an order.

-- Candidate pool: where the candidate came from and what it means for the NEXT session.
-- DISCOVERY | STRATEGY | BOTH
ALTER TABLE discovery_candidates ADD COLUMN origin TEXT;
-- NEXT_SESSION
ALTER TABLE discovery_candidates ADD COLUMN relevance TEXT;
-- exchange-calendar date the setup is for
ALTER TABLE discovery_candidates ADD COLUMN next_session TEXT;
-- latest information timestamp used (UTC)
ALTER TABLE discovery_candidates ADD COLUMN info_cutoff_at TEXT;
-- when the candidate was produced (UTC)
ALTER TABLE discovery_candidates ADD COLUMN discovered_at TEXT;
-- conditional setup: why / confirm / invalidate / missing / paper
ALTER TABLE discovery_candidates ADD COLUMN setup_json TEXT;

ALTER TABLE discovery_runs ADD COLUMN next_session TEXT;
ALTER TABLE discovery_runs ADD COLUMN info_cutoff_at TEXT;
ALTER TABLE discovery_runs ADD COLUMN calendar_source TEXT;

-- Forward outcomes from the NEXT-SESSION OPEN (the first executable price) with modeled costs,
-- plus whether the candidate was known before that open (research must be able to tell).
-- raw open of the next session
ALTER TABLE discovery_outcomes ADD COLUMN next_open REAL;
-- D close -> next open (adjusted)
ALTER TABLE discovery_outcomes ADD COLUMN gap_ret REAL;
-- next open -> close of D+h (adjusted)
ALTER TABLE discovery_outcomes ADD COLUMN open_ret REAL;
-- modeled round-trip cost (fraction)
ALTER TABLE discovery_outcomes ADD COLUMN cost_ret REAL;
-- open_ret - cost_ret
ALTER TABLE discovery_outcomes ADD COLUMN net_ret REAL;
ALTER TABLE discovery_outcomes ADD COLUMN discovered_at TEXT;
-- exchange open of the next session (UTC)
ALTER TABLE discovery_outcomes ADD COLUMN next_open_at TEXT;
ALTER TABLE discovery_outcomes ADD COLUMN known_before_open INTEGER;

-- Overnight refresh: information that became available after the close of D (post-close,
-- overnight, pre-market) and what it did to a candidate. Append-only.
CREATE TABLE overnight_updates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_run_id   TEXT NOT NULL,
    discovery_id       TEXT,
    symbol             TEXT NOT NULL,
    refreshed_at       TEXT NOT NULL,        -- wall-clock time of the refresh (UTC)
    info_cutoff_at     TEXT NOT NULL,        -- only items available_at <= this were used
    phase              TEXT NOT NULL,        -- POST_CLOSE | OVERNIGHT | PRE_MARKET
    kind               TEXT NOT NULL,        -- news | earnings_event | corporate_action
    available_at       TEXT NOT NULL,
    source_id          TEXT,
    detail_json        TEXT,
    effect             TEXT NOT NULL,        -- CATALYST_ADDED | PROMOTED_TO_WATCH | NONE
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_overnight_updates_run ON overnight_updates(discovery_run_id);

-- Pre-open recheck of next-session candidates, run BEFORE the next session opens. The actual open
-- is never an input here. Append-only.
CREATE TABLE preopen_checks (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_run_id   TEXT NOT NULL,
    discovery_id       TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    checked_at         TEXT NOT NULL,        -- must be < next_open_at
    next_session       TEXT NOT NULL,
    next_open_at       TEXT NOT NULL,
    status_before      TEXT NOT NULL,
    status_after       TEXT NOT NULL,        -- PAPER_ELIGIBLE | REJECTED | UNKNOWN | INVALIDATED | WATCH | VALIDATION_PENDING | DISCOVERED
    reason             TEXT NOT NULL,
    checks_json        TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_preopen_checks_run ON preopen_checks(discovery_run_id);

-- Discovery research (forward outcomes of point-in-time discovery replays). Summaries only; the
-- per-observation table is written as a parquet file next to the report.
CREATE TABLE discovery_research_runs (
    research_id        TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    is_synthetic       INTEGER NOT NULL,
    period_start       TEXT NOT NULL,
    period_end         TEXT NOT NULL,
    n_dates            INTEGER NOT NULL,
    n_observations     INTEGER NOT NULL,
    params_json        TEXT NOT NULL,
    summary_json       TEXT NOT NULL,        -- per family/combination x horizon statistics + verdicts
    redundancy_json    TEXT NOT NULL,
    report_path        TEXT,
    data_path          TEXT
);

CREATE TRIGGER trg_overnight_updates_no_update BEFORE UPDATE ON overnight_updates
BEGIN SELECT RAISE(ABORT, 'overnight_updates are append-only'); END;
CREATE TRIGGER trg_overnight_updates_no_delete BEFORE DELETE ON overnight_updates
BEGIN SELECT RAISE(ABORT, 'overnight_updates are append-only'); END;
CREATE TRIGGER trg_preopen_checks_no_update BEFORE UPDATE ON preopen_checks
BEGIN SELECT RAISE(ABORT, 'preopen_checks are append-only'); END;
CREATE TRIGGER trg_preopen_checks_no_delete BEFORE DELETE ON preopen_checks
BEGIN SELECT RAISE(ABORT, 'preopen_checks are append-only'); END;
CREATE TRIGGER trg_discovery_research_runs_no_update BEFORE UPDATE ON discovery_research_runs
BEGIN SELECT RAISE(ABORT, 'discovery_research_runs are append-only'); END;
CREATE TRIGGER trg_discovery_research_runs_no_delete BEFORE DELETE ON discovery_research_runs
BEGIN SELECT RAISE(ABORT, 'discovery_research_runs are append-only'); END;
