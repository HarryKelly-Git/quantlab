-- 101_discovery.sql: market discovery layer (quantlab/discovery). Research records only: nothing here
-- can place an order. Discovery, candidate status and near-miss rows are append-only per run; forward
-- outcomes are append-only per (candidate, horizon).

-- One row per discovery run (a pipeline session or a standalone `quantlab discover`).
CREATE TABLE discovery_runs (
    discovery_run_id   TEXT PRIMARY KEY,
    run_id             TEXT,                 -- pipeline run (NULL for a standalone scan)
    as_of_date         TEXT NOT NULL,        -- decision session D (information cutoff = D close)
    created_at         TEXT NOT NULL,
    is_synthetic       INTEGER NOT NULL,
    funnel_json        TEXT NOT NULL,        -- real stage counts (scanned -> ... -> trades)
    blockers_json      TEXT NOT NULL,        -- why-no-paper-trades breakdown
    families_json      TEXT NOT NULL,        -- per-family data availability / trigger counts
    config_json        TEXT NOT NULL,        -- thresholds and weights used (fixed, from config)
    dataset_ids_json   TEXT
);
CREATE INDEX ix_discovery_runs_asof ON discovery_runs(as_of_date);

-- Every discovered setup (a symbol that fired >= 1 family), with its score, factors and status.
CREATE TABLE discovery_candidates (
    discovery_id       TEXT PRIMARY KEY,
    discovery_run_id   TEXT NOT NULL,
    as_of_date         TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    discovery_score    REAL,                 -- 0..100 ranking score (NULL = UNKNOWN). NOT a probability of profit
    score_coverage     REAL NOT NULL,        -- share of score weight whose inputs were available
    rank               INTEGER NOT NULL,
    families_json      TEXT NOT NULL,        -- fired families with strength and direction bias
    dimensions_json    TEXT NOT NULL,        -- per-dimension 0..100 score, or null = UNKNOWN
    factors_json       TEXT NOT NULL,        -- every input: value, source, as_of, pit_status
    catalyst_json      TEXT,                 -- news/earnings items used (headline, source, available_at)
    direction_bias     TEXT NOT NULL,        -- BULLISH | BEARISH | NEUTRAL (description, not a trade)
    status             TEXT NOT NULL,        -- DISCOVERED | WATCH | VALIDATION_PENDING | REJECTED | PAPER_ELIGIBLE | TRADED
    high_quality       INTEGER NOT NULL,
    on_watchlist       INTEGER NOT NULL,
    block_stage        TEXT,                 -- stage that stops a paper trade (NULL when traded)
    block_reason       TEXT,
    checks_json        TEXT NOT NULL,        -- passed/failed checks with stage + reason (near-miss source)
    strategy_links_json TEXT NOT NULL,       -- strategy candidates for the same symbol/session + decisions
    is_synthetic       INTEGER NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE (discovery_run_id, symbol)
);
CREATE INDEX ix_discovery_candidates_asof ON discovery_candidates(as_of_date, discovery_score);
CREATE INDEX ix_discovery_candidates_symbol ON discovery_candidates(symbol, as_of_date);

-- Diagnostics: suspiciously empty or one-sided discovery, missing data sources.
CREATE TABLE discovery_diagnostics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_run_id   TEXT NOT NULL,
    as_of_date         TEXT NOT NULL,
    level              TEXT NOT NULL,        -- INFO | WARN | CRITICAL
    code               TEXT NOT NULL,        -- ZERO_DISCOVERIES | LOW_DISCOVERY_RATE | DATA_MISSING | DATA_STALE | ONE_RULE_REJECTS_ALL | ...
    message            TEXT NOT NULL,
    details_json       TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_discovery_diagnostics_run ON discovery_diagnostics(discovery_run_id);

-- Forward outcomes of discovered candidates (traded or not). Research only: never fed back into rules.
CREATE TABLE discovery_outcomes (
    discovery_id       TEXT NOT NULL,
    horizon_sessions   INTEGER NOT NULL,     -- 1 | 3 | 5 | 10 | 20
    as_of_date         TEXT NOT NULL,
    end_date           TEXT NOT NULL,
    ret                REAL,                 -- split/dividend-adjusted close-to-close return
    spy_ret            REAL,
    excess_ret         REAL,
    mfe                REAL,                 -- max favourable excursion (adjusted high vs D close)
    mae                REAL,                 -- max adverse excursion (adjusted low vs D close)
    price_confirmed    INTEGER,              -- close stayed above D's close at the horizon
    volume_confirmed   INTEGER,              -- mean relative dollar volume over the window >= 1
    catalyst_persisted INTEGER,              -- NULL when no news feed covers the window
    status             TEXT NOT NULL,        -- MATURED | DELISTED_OR_MISSING
    is_synthetic       INTEGER NOT NULL,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (discovery_id, horizon_sessions)
);

CREATE TRIGGER trg_discovery_runs_no_update BEFORE UPDATE ON discovery_runs
BEGIN SELECT RAISE(ABORT, 'discovery_runs are append-only'); END;
CREATE TRIGGER trg_discovery_runs_no_delete BEFORE DELETE ON discovery_runs
BEGIN SELECT RAISE(ABORT, 'discovery_runs are append-only'); END;
CREATE TRIGGER trg_discovery_candidates_no_update BEFORE UPDATE ON discovery_candidates
BEGIN SELECT RAISE(ABORT, 'discovery_candidates are append-only'); END;
CREATE TRIGGER trg_discovery_candidates_no_delete BEFORE DELETE ON discovery_candidates
BEGIN SELECT RAISE(ABORT, 'discovery_candidates are append-only'); END;
CREATE TRIGGER trg_discovery_diagnostics_no_update BEFORE UPDATE ON discovery_diagnostics
BEGIN SELECT RAISE(ABORT, 'discovery_diagnostics are append-only'); END;
CREATE TRIGGER trg_discovery_diagnostics_no_delete BEFORE DELETE ON discovery_diagnostics
BEGIN SELECT RAISE(ABORT, 'discovery_diagnostics are append-only'); END;
CREATE TRIGGER trg_discovery_outcomes_no_update BEFORE UPDATE ON discovery_outcomes
BEGIN SELECT RAISE(ABORT, 'discovery_outcomes are append-only'); END;
CREATE TRIGGER trg_discovery_outcomes_no_delete BEFORE DELETE ON discovery_outcomes
BEGIN SELECT RAISE(ABORT, 'discovery_outcomes are append-only'); END;
