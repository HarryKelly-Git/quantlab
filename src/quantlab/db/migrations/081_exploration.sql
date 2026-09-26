-- PAPER_EXPLORATION mode (paper only). Exploratory decisions are separate from the strict decision
-- chain (decisions table): they never make a strategy paper eligible and never feed EV estimates.
-- Every table is append-only: a decision is never rewritten after the fact; status changes are
-- new rows in exploration_events, outcomes are new rows in exploration_outcomes.

-- One row per exploratory candidate considered for a session: SELECTED (planned trade), SHADOW
-- (selected while paper.mode = STRICT: tracked, never traded), WATCHED_NOT_TRADED (next-ranked,
-- kept for comparison) or SKIPPED (failed an exploration safety check). pre_trade_json holds the
-- complete pre-trade state exactly as known at the decision cutoff.
CREATE TABLE exploration_decisions (
    decision_id        TEXT PRIMARY KEY,
    session_date       TEXT NOT NULL,
    next_session       TEXT,
    symbol             TEXT NOT NULL,
    mode               TEXT NOT NULL,
    selection          TEXT NOT NULL,
    rank               INTEGER,
    discovery_run_id   TEXT,
    discovery_id       TEXT,
    origin             TEXT,
    setup_type         TEXT,
    setup_class        TEXT,
    families           TEXT,
    catalyst_families  TEXT,
    discovery_score    REAL,
    ref_price          REAL,
    qty                REAL,
    stop_price         REAL,
    holding_sessions   INTEGER,
    strict_blocker     TEXT,
    reason             TEXT NOT NULL,
    info_cutoff_at     TEXT NOT NULL,
    pre_trade_json     TEXT NOT NULL,
    is_synthetic       INTEGER NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE (session_date, symbol, mode)
);
CREATE INDEX ix_exploration_decisions_session ON exploration_decisions(session_date, selection);

-- Lifecycle of a decision (PLANNED -> REVALIDATED -> SUBMITTED | CANCELLED_PREOPEN | REFUSED).
CREATE TABLE exploration_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id        TEXT NOT NULL,
    event              TEXT NOT NULL,
    at                 TEXT NOT NULL,
    order_id           TEXT,
    details_json       TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_exploration_events_decision ON exploration_events(decision_id);

-- Forward outcomes from the next-session open (entry) to the close h sessions later. Measured for
-- every decision (traded or not) so traded vs watched vs rejected can be compared.
CREATE TABLE exploration_outcomes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id        TEXT NOT NULL,
    horizon_sessions   INTEGER NOT NULL,
    entry_date         TEXT,
    end_date           TEXT,
    entry_price        REAL,
    gross_ret          REAL,
    cost_ret           REAL,
    net_ret            REAL,
    spy_ret            REAL,
    mfe                REAL,
    mae                REAL,
    stop_breached      INTEGER,
    thesis_valid       INTEGER,
    catalyst_persisted INTEGER,
    computed_at        TEXT NOT NULL,
    UNIQUE (decision_id, horizon_sessions)
);

-- Exploration -> validation research workflow. A hypothesis only moves one stage at a time, with
-- a named human actor and an evidence reference. Nothing advances it automatically, and reaching
-- STRICT_ELIGIBLE changes no strategy status (strategy promotion is the separate, unchanged process).
CREATE TABLE research_hypotheses (
    hypothesis_id      TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    definition_json    TEXT NOT NULL,
    created_by         TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE TABLE hypothesis_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id      TEXT NOT NULL,
    from_stage         TEXT,
    to_stage           TEXT NOT NULL,
    actor              TEXT NOT NULL,
    evidence           TEXT NOT NULL,
    created_at         TEXT NOT NULL
);

CREATE TRIGGER trg_exploration_decisions_no_update BEFORE UPDATE ON exploration_decisions
BEGIN SELECT RAISE(ABORT, 'exploration_decisions are append-only'); END;
CREATE TRIGGER trg_exploration_decisions_no_delete BEFORE DELETE ON exploration_decisions
BEGIN SELECT RAISE(ABORT, 'exploration_decisions are append-only'); END;
CREATE TRIGGER trg_exploration_events_no_update BEFORE UPDATE ON exploration_events
BEGIN SELECT RAISE(ABORT, 'exploration_events are append-only'); END;
CREATE TRIGGER trg_exploration_events_no_delete BEFORE DELETE ON exploration_events
BEGIN SELECT RAISE(ABORT, 'exploration_events are append-only'); END;
CREATE TRIGGER trg_exploration_outcomes_no_update BEFORE UPDATE ON exploration_outcomes
BEGIN SELECT RAISE(ABORT, 'exploration_outcomes are append-only'); END;
CREATE TRIGGER trg_exploration_outcomes_no_delete BEFORE DELETE ON exploration_outcomes
BEGIN SELECT RAISE(ABORT, 'exploration_outcomes are append-only'); END;
CREATE TRIGGER trg_research_hypotheses_no_update BEFORE UPDATE ON research_hypotheses
BEGIN SELECT RAISE(ABORT, 'research_hypotheses are append-only'); END;
CREATE TRIGGER trg_research_hypotheses_no_delete BEFORE DELETE ON research_hypotheses
BEGIN SELECT RAISE(ABORT, 'research_hypotheses are append-only'); END;
CREATE TRIGGER trg_hypothesis_events_no_update BEFORE UPDATE ON hypothesis_events
BEGIN SELECT RAISE(ABORT, 'hypothesis_events are append-only'); END;
CREATE TRIGGER trg_hypothesis_events_no_delete BEFORE DELETE ON hypothesis_events
BEGIN SELECT RAISE(ABORT, 'hypothesis_events are append-only'); END;
