-- 001_core.sql — QuantLab core schema.
-- Conventions:
--   * Timestamps are ISO-8601 UTC strings ("...+00:00"); session dates are "YYYY-MM-DD" (US/Eastern trading date).
--   * JSON payloads are TEXT columns ending in _json.
--   * AUDIT tables are append-only: triggers abort any UPDATE or DELETE. Corrections are NEW rows
--     that reference what they supersede. This protects human decisions, candidates, shadow
--     opportunities, experiments and the research ledger from after-the-fact editing.
--   * STATE tables (positions, system_state, orders.status) may be updated; every change is also
--     written to an append-only *_log / *_events table.

-- ---------------------------------------------------------------------------------------------
-- Runs, pipeline checkpoints, system state
-- ---------------------------------------------------------------------------------------------
CREATE TABLE runs (
    run_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,                 -- pipeline | backtest | walk_forward | ingest | ml_train | report | ...
    mode          TEXT,                          -- RESEARCH | BOT_PAPER | HUMAN_PAPER
    as_of_date    TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',  -- running | succeeded | failed | aborted
    git_commit    TEXT,
    git_dirty     INTEGER,
    config_hash   TEXT,
    config_json   TEXT,
    error         TEXT,
    notes         TEXT
);

CREATE TABLE pipeline_steps (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    step         TEXT NOT NULL,
    step_order   INTEGER NOT NULL,
    status       TEXT NOT NULL,                  -- pending | running | succeeded | failed | skipped
    started_at   TEXT,
    finished_at  TEXT,
    error        TEXT,
    output_json  TEXT,
    PRIMARY KEY (run_id, step)
);

CREATE TABLE system_state (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    state       TEXT NOT NULL,                   -- ACTIVE | SYSTEM_PAUSED
    reason      TEXT,
    changed_at  TEXT NOT NULL,
    changed_by  TEXT NOT NULL
);

CREATE TABLE system_state_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    state       TEXT NOT NULL,
    reason      TEXT,
    changed_at  TEXT NOT NULL,
    changed_by  TEXT NOT NULL,
    trigger     TEXT,                            -- which check/trigger caused it
    details_json TEXT
);

CREATE TABLE health_checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    check_name  TEXT NOT NULL,
    component   TEXT NOT NULL,                   -- data | provider | broker | model | llm | reconciliation | system
    passed      INTEGER NOT NULL,
    severity    TEXT NOT NULL,                   -- CRITICAL | WARNING | INFO
    reason      TEXT,
    details_json TEXT,
    created_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------------------------
-- Data provenance & quality
-- ---------------------------------------------------------------------------------------------
CREATE TABLE datasets (
    dataset_id     TEXT PRIMARY KEY,             -- content-addressed: <kind>_<hash>
    kind           TEXT NOT NULL,                -- bars | corporate_actions | reference | fundamentals | events | news
    provider       TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    retrieved_at   TEXT NOT NULL,
    path           TEXT NOT NULL,                -- relative to data_dir
    content_hash   TEXT NOT NULL,
    row_count      INTEGER NOT NULL,
    symbol_count   INTEGER,
    start_date     TEXT,
    end_date       TEXT,
    params_json    TEXT,
    pit_notes      TEXT,
    is_synthetic   INTEGER NOT NULL DEFAULT 0    -- 1 => SYNTHETIC test data, never market evidence
);

CREATE TABLE data_quality_issues (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT,
    dataset_id   TEXT,
    check_name   TEXT NOT NULL,
    severity     TEXT NOT NULL,                  -- CRITICAL | WARNING | INFO
    symbol       TEXT,
    session_date TEXT,
    detail       TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE security_master (
    symbol         TEXT NOT NULL,
    source         TEXT NOT NULL,                -- nasdaq_trader | alpaca | sec | synthetic
    name           TEXT,
    exchange       TEXT,
    security_type  TEXT,                         -- COMMON | ETF | PREFERRED | WARRANT | RIGHT | UNIT | FUND | OTHER | UNKNOWN
    is_test_issue  INTEGER,
    cik            TEXT,
    sic            TEXT,
    sector         TEXT,
    industry       TEXT,
    status         TEXT,                         -- active | inactive | unknown
    pit_status     TEXT NOT NULL DEFAULT 'ASSUMED_STATIC',
    retrieved_at   TEXT NOT NULL,
    raw_json       TEXT,
    PRIMARY KEY (symbol, source)
);

CREATE TABLE universe_snapshots (
    as_of_date  TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    included    INTEGER NOT NULL,
    reason      TEXT,
    run_id      TEXT,
    PRIMARY KEY (as_of_date, symbol)
);

CREATE TABLE regime_snapshots (
    as_of_date  TEXT PRIMARY KEY,
    label       TEXT,
    metrics_json TEXT NOT NULL,
    run_id      TEXT,
    created_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------------------------
-- Strategies & models (registry + append-only status history)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE strategies (
    strategy_id   TEXT NOT NULL,
    version       TEXT NOT NULL,
    family        TEXT NOT NULL,
    description   TEXT,
    params_json   TEXT NOT NULL,
    feature_deps_json TEXT,
    status        TEXT NOT NULL DEFAULT 'SHADOW',   -- ACTIVE | SHADOW | PAUSED | RETIRED
    stage         TEXT NOT NULL DEFAULT 'RESEARCH', -- IDEA ... PROMOTED
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (strategy_id, version)
);

CREATE TABLE strategy_status_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    version      TEXT NOT NULL,
    from_status  TEXT, to_status TEXT,
    from_stage   TEXT, to_stage  TEXT,
    reason       TEXT NOT NULL,
    evidence_json TEXT,
    changed_at   TEXT NOT NULL,
    changed_by   TEXT NOT NULL
);

CREATE TABLE models (
    model_id      TEXT NOT NULL,
    version       TEXT NOT NULL,
    kind          TEXT NOT NULL,                 -- logistic | random_forest | hist_gradient_boosting | meta | ...
    target        TEXT NOT NULL,
    features_json TEXT NOT NULL,
    params_json   TEXT,
    train_start   TEXT, train_end TEXT,
    trained_at    TEXT NOT NULL,
    experiment_id TEXT,
    status        TEXT NOT NULL DEFAULT 'MONITORED',
    artifact_path TEXT,
    metrics_json  TEXT,
    PRIMARY KEY (model_id, version)
);

CREATE TABLE model_status_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    version     TEXT NOT NULL,
    from_status TEXT, to_status TEXT NOT NULL,
    reason      TEXT NOT NULL,
    evidence_json TEXT,
    changed_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------------------------
-- Candidates and everything decided about them
-- ---------------------------------------------------------------------------------------------
CREATE TABLE candidates (
    candidate_id      TEXT PRIMARY KEY,
    run_id            TEXT,
    as_of_date        TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    strategy_id       TEXT NOT NULL,
    strategy_version  TEXT NOT NULL,
    direction         TEXT NOT NULL,
    score             REAL NOT NULL,
    rank              INTEGER,
    opportunity_score REAL,
    features_json     TEXT,
    reasons_json      TEXT,
    entry_convention  TEXT NOT NULL DEFAULT 'next_open',
    entry_ref_price   REAL,
    stop_price        REAL,
    target_price      REAL,
    holding_sessions  INTEGER,
    invalidation      TEXT,
    risk_json         TEXT,
    pit_status        TEXT NOT NULL,
    is_synthetic      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_candidates_date ON candidates(as_of_date);
CREATE INDEX ix_candidates_symbol ON candidates(symbol, as_of_date);

CREATE TABLE ml_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  TEXT,
    model_id      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    as_of_date    TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    target        TEXT NOT NULL,
    probability   REAL,
    prediction    REAL,
    created_at    TEXT NOT NULL
);

CREATE TABLE evidence_packets (
    packet_id     TEXT PRIMARY KEY,
    candidate_id  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    packet_hash   TEXT NOT NULL,
    packet_json   TEXT NOT NULL
);

CREATE TABLE ai_calls (
    call_id        TEXT PRIMARY KEY,
    run_id         TEXT,
    candidate_id   TEXT,
    role           TEXT NOT NULL,                -- researcher | adversary | judge | idea_generator
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    request_hash   TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    cached_tokens  INTEGER,
    cost_usd       REAL,
    status         TEXT NOT NULL,                -- ok | cache_hit | error | invalid_schema | budget_exceeded | disabled
    error          TEXT,
    response_json  TEXT
);

CREATE TABLE ai_assessments (
    assessment_id  TEXT PRIMARY KEY,
    candidate_id   TEXT NOT NULL,
    call_id        TEXT,
    role           TEXT NOT NULL,                -- researcher | adversary | judge
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    decision       TEXT NOT NULL,                -- ACCEPT | REJECT | WATCH | UNKNOWN
    summary        TEXT,
    assessment_json TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE ai_objections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   TEXT NOT NULL,
    assessment_id  TEXT,
    category       TEXT NOT NULL,
    severity       TEXT NOT NULL,                -- HARD_FAIL | MATERIAL_CONCERN | MINOR_CONCERN | UNKNOWN
    text           TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'ai',
    created_at     TEXT NOT NULL
);

CREATE TABLE decisions (
    decision_id    TEXT PRIMARY KEY,
    candidate_id   TEXT NOT NULL,
    run_id         TEXT,
    book           TEXT NOT NULL DEFAULT 'BOT',
    decision       TEXT NOT NULL,                -- TRADE | NO_TRADE | WATCH | UNKNOWN
    reject_stage   TEXT NOT NULL DEFAULT 'NONE',
    reasons_json   TEXT,
    ai_decision    TEXT,
    ev_json        TEXT,
    no_trade_json  TEXT,
    sizing_json    TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX ix_decisions_candidate ON decisions(candidate_id);

CREATE TABLE risk_checks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  TEXT NOT NULL,
    decision_id   TEXT,
    check_name    TEXT NOT NULL,
    passed        INTEGER NOT NULL,
    severity      TEXT NOT NULL,
    reason        TEXT,
    details_json  TEXT,
    created_at    TEXT NOT NULL
);

-- ---------------------------------------------------------------------------------------------
-- Shadow book: every serious opportunity, traded or not, and what happened afterwards
-- ---------------------------------------------------------------------------------------------
CREATE TABLE shadow_opportunities (
    opportunity_id    TEXT PRIMARY KEY,
    candidate_id      TEXT NOT NULL,
    as_of_date        TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    strategy_id       TEXT NOT NULL,
    strategy_version  TEXT NOT NULL,
    score             REAL,
    quant_reasoning   TEXT,
    ml_json           TEXT,
    ai_decision       TEXT,
    objections_json   TEXT,
    bot_decision      TEXT NOT NULL,             -- TRADE | NO_TRADE | WATCH | UNKNOWN
    reject_stage      TEXT NOT NULL,             -- RejectStage
    reject_reason     TEXT,
    entry_convention  TEXT NOT NULL DEFAULT 'next_open',
    entry_ref_price   REAL,
    stop_price        REAL,
    target_price      REAL,
    holding_sessions  INTEGER,
    created_at        TEXT NOT NULL,
    is_synthetic      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_shadow_date ON shadow_opportunities(as_of_date);

CREATE TABLE shadow_outcomes (
    opportunity_id  TEXT NOT NULL REFERENCES shadow_opportunities(opportunity_id),
    horizon_sessions INTEGER NOT NULL,
    measured_at     TEXT NOT NULL,
    entry_date      TEXT,
    entry_price     REAL,
    exit_date       TEXT,
    exit_price      REAL,
    ret             REAL,                        -- follows the TradePlan (stop/target/time), next-open entry
    ret_hold        REAL,                        -- plain buy-and-hold over the horizon
    mfe             REAL,
    mae             REAL,
    hit_stop        INTEGER,
    hit_target      INTEGER,
    benchmark_ret   REAL,
    excess_ret      REAL,
    status          TEXT NOT NULL,               -- complete | delisted | no_data
    PRIMARY KEY (opportunity_id, horizon_sessions)
);

-- ---------------------------------------------------------------------------------------------
-- Human paper-trading lab
-- ---------------------------------------------------------------------------------------------
CREATE TABLE human_decisions (
    decision_id       TEXT PRIMARY KEY,
    candidate_id      TEXT,                      -- NULL for self-sourced ideas
    symbol            TEXT NOT NULL,
    action            TEXT NOT NULL,             -- BUY | PASS | WATCH | SELL | WAIT
    conviction        INTEGER,                   -- 1..5 optional
    notes             TEXT,
    decided_at        TEXT NOT NULL,             -- server clock, UTC
    ref_session_date  TEXT NOT NULL,             -- last completed session known when deciding
    ref_price         REAL,                      -- last raw close known when deciding
    fill_convention   TEXT NOT NULL DEFAULT 'next_open_after_decision',
    supersedes_id     TEXT,                      -- corrections are new rows
    created_at        TEXT NOT NULL
);
CREATE INDEX ix_human_decisions_symbol ON human_decisions(symbol, decided_at);

CREATE TABLE human_notes (
    note_id       TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,                 -- strategy_idea | observation | hypothesis | stock_idea | market_observation | note
    symbol        TEXT,
    text          TEXT NOT NULL,
    hypothesis_id TEXT,
    created_at    TEXT NOT NULL
);

-- ---------------------------------------------------------------------------------------------
-- Paper execution, positions, trade journal (BOT and HUMAN books kept separate by `book`)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE orders (
    order_id         TEXT PRIMARY KEY,
    client_order_id  TEXT NOT NULL UNIQUE,
    book             TEXT NOT NULL,
    broker           TEXT NOT NULL,              -- sim | alpaca_paper
    candidate_id     TEXT,
    decision_id      TEXT,
    human_decision_id TEXT,
    trade_id         TEXT,
    purpose          TEXT NOT NULL,              -- entry | exit
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,              -- buy | sell
    qty              REAL NOT NULL,
    order_type       TEXT NOT NULL,              -- market | limit
    time_in_force    TEXT NOT NULL,              -- opg | day | gtc
    limit_price      REAL,
    created_at       TEXT NOT NULL,
    submitted_at     TEXT,
    status           TEXT NOT NULL,
    broker_order_id  TEXT,
    filled_qty       REAL NOT NULL DEFAULT 0,
    filled_avg_price REAL,
    last_update_at   TEXT NOT NULL,
    raw_json         TEXT
);
CREATE INDEX ix_orders_book_status ON orders(book, status);

CREATE TABLE order_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT NOT NULL,
    event       TEXT NOT NULL,                   -- created | submitted | status_change | fill | cancel | error | reconcile
    status      TEXT,
    at          TEXT NOT NULL,
    raw_json    TEXT
);

CREATE TABLE fills (
    fill_id        TEXT PRIMARY KEY,
    order_id       TEXT NOT NULL,
    book           TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    qty            REAL NOT NULL,
    price          REAL NOT NULL,
    commission     REAL NOT NULL DEFAULT 0,
    modeled_cost   REAL NOT NULL DEFAULT 0,      -- spread/slippage charged by the fill model (sim)
    filled_at      TEXT NOT NULL,
    session_date   TEXT,
    broker_fill_id TEXT,
    source         TEXT NOT NULL                 -- sim | alpaca_paper
);

CREATE TABLE positions (
    book           TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    qty            REAL NOT NULL,
    avg_cost       REAL NOT NULL,
    trade_id       TEXT,
    opened_at      TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (book, symbol)
);

CREATE TABLE trades (
    trade_id          TEXT PRIMARY KEY,
    book              TEXT NOT NULL,
    candidate_id      TEXT,
    decision_id       TEXT,
    human_decision_id TEXT,
    symbol            TEXT NOT NULL,
    direction         TEXT NOT NULL DEFAULT 'LONG',
    strategy_id       TEXT,
    strategy_version  TEXT,
    model_version     TEXT,
    ai_model          TEXT,
    status            TEXT NOT NULL,             -- OPEN | CLOSED
    qty               REAL NOT NULL,
    entry_date        TEXT,
    entry_price       REAL,
    stop_price        REAL,
    target_price      REAL,
    planned_exit_date TEXT,
    exit_date         TEXT,
    exit_price        REAL,
    exit_reason       TEXT,
    gross_pnl         REAL,
    costs             REAL,
    net_pnl           REAL,
    ret               REAL,
    holding_sessions  INTEGER,
    mae               REAL,
    mfe               REAL,
    regime            TEXT,
    sector            TEXT,
    journal_json      TEXT,                      -- full audit trail snapshot (evidence, checks, objections)
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX ix_trades_book ON trades(book, status);

CREATE TABLE trade_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id   TEXT NOT NULL,
    event      TEXT NOT NULL,                    -- opened | stop_updated | exit_signal | closed | note
    at         TEXT NOT NULL,
    details_json TEXT
);

CREATE TABLE portfolio_snapshots (
    book            TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,
    cash            REAL NOT NULL,
    equity          REAL NOT NULL,
    gross_exposure  REAL NOT NULL,
    positions_count INTEGER NOT NULL,
    peak_equity     REAL NOT NULL,
    drawdown        REAL NOT NULL,
    open_risk       REAL,
    details_json    TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (book, as_of_date)
);

CREATE TABLE reconciliations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    book          TEXT NOT NULL,
    at            TEXT NOT NULL,
    status        TEXT NOT NULL,                 -- ok | mismatch | broker_unavailable
    broker_json   TEXT,
    internal_json TEXT,
    diffs_json    TEXT
);

-- ---------------------------------------------------------------------------------------------
-- Experiments, holdout protection, research ledger
-- ---------------------------------------------------------------------------------------------
CREATE TABLE experiments (
    experiment_id      TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    kind               TEXT NOT NULL,            -- backtest | walk_forward | holdout | ab | ml | counterfactual | baseline
    hypothesis_id      TEXT,
    created_at         TEXT NOT NULL,
    git_commit         TEXT,
    git_dirty          INTEGER,
    config_hash        TEXT NOT NULL,
    config_json        TEXT NOT NULL,
    dataset_ids_json   TEXT,
    strategy_versions_json TEXT,
    model_versions_json TEXT,
    ai_models_json     TEXT,
    cost_assumptions_json TEXT,
    seeds_json         TEXT,
    period_start       TEXT,
    period_end         TEXT,
    n_variants_tested  INTEGER NOT NULL DEFAULT 1,  -- for multiple-testing corrections
    uses_synthetic_data INTEGER NOT NULL DEFAULT 0,
    touches_holdout    INTEGER NOT NULL DEFAULT 0,
    notes              TEXT
);

CREATE TABLE experiment_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  TEXT NOT NULL REFERENCES experiments(experiment_id),
    status         TEXT NOT NULL,                -- succeeded | failed
    metrics_json   TEXT,
    artifacts_path TEXT,
    conclusion     TEXT,
    completed_at   TEXT NOT NULL
);

CREATE TABLE holdout_access_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    at             TEXT NOT NULL,
    experiment_id  TEXT,
    period_start   TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    reason         TEXT NOT NULL,
    actor          TEXT NOT NULL
);

CREATE TABLE hypotheses (
    hypothesis_id   TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    source          TEXT NOT NULL,               -- human | ai | system
    title           TEXT NOT NULL,
    hypothesis      TEXT NOT NULL,
    rationale       TEXT,
    required_data   TEXT,
    proposed_test   TEXT,
    expected_failure_conditions TEXT,
    source_note_id  TEXT,                        -- human_notes.note_id when converted from a human note
    status          TEXT NOT NULL DEFAULT 'PROPOSED',
    updated_at      TEXT NOT NULL
);

CREATE TABLE research_ledger (
    entry_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id  TEXT,
    entry_date     TEXT NOT NULL,
    entry_type     TEXT NOT NULL,                -- idea | implementation | experiment | result | conclusion | next_action | status_change
    text           TEXT NOT NULL,
    experiment_id  TEXT,
    author         TEXT NOT NULL,                -- human | ai | system
    created_at     TEXT NOT NULL
);

CREATE TABLE reports (
    report_id    TEXT PRIMARY KEY,
    run_id       TEXT,
    as_of_date   TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'daily',
    path         TEXT,
    markdown     TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE llm_cache (
    request_hash  TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- ---------------------------------------------------------------------------------------------
-- Immutability triggers for audit tables
-- ---------------------------------------------------------------------------------------------
CREATE TRIGGER trg_human_decisions_no_update BEFORE UPDATE ON human_decisions
BEGIN SELECT RAISE(ABORT, 'human_decisions is append-only: record a new decision with supersedes_id'); END;
CREATE TRIGGER trg_human_decisions_no_delete BEFORE DELETE ON human_decisions
BEGIN SELECT RAISE(ABORT, 'human_decisions is append-only'); END;

CREATE TRIGGER trg_candidates_no_update BEFORE UPDATE ON candidates
BEGIN SELECT RAISE(ABORT, 'candidates are immutable once recorded'); END;
CREATE TRIGGER trg_candidates_no_delete BEFORE DELETE ON candidates
BEGIN SELECT RAISE(ABORT, 'candidates are immutable once recorded'); END;

CREATE TRIGGER trg_decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
CREATE TRIGGER trg_decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;

CREATE TRIGGER trg_risk_checks_no_update BEFORE UPDATE ON risk_checks
BEGIN SELECT RAISE(ABORT, 'risk_checks are append-only'); END;
CREATE TRIGGER trg_risk_checks_no_delete BEFORE DELETE ON risk_checks
BEGIN SELECT RAISE(ABORT, 'risk_checks are append-only'); END;

CREATE TRIGGER trg_shadow_opps_no_update BEFORE UPDATE ON shadow_opportunities
BEGIN SELECT RAISE(ABORT, 'shadow_opportunities are append-only'); END;
CREATE TRIGGER trg_shadow_opps_no_delete BEFORE DELETE ON shadow_opportunities
BEGIN SELECT RAISE(ABORT, 'shadow_opportunities are append-only: rejected opportunities are never deleted'); END;

CREATE TRIGGER trg_shadow_outcomes_no_update BEFORE UPDATE ON shadow_outcomes
BEGIN SELECT RAISE(ABORT, 'shadow_outcomes are append-only'); END;
CREATE TRIGGER trg_shadow_outcomes_no_delete BEFORE DELETE ON shadow_outcomes
BEGIN SELECT RAISE(ABORT, 'shadow_outcomes are append-only'); END;

CREATE TRIGGER trg_experiments_no_update BEFORE UPDATE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are immutable; results go to experiment_results'); END;
CREATE TRIGGER trg_experiments_no_delete BEFORE DELETE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are never deleted (failed experiments are research data)'); END;

CREATE TRIGGER trg_experiment_results_no_update BEFORE UPDATE ON experiment_results
BEGIN SELECT RAISE(ABORT, 'experiment_results are append-only'); END;
CREATE TRIGGER trg_experiment_results_no_delete BEFORE DELETE ON experiment_results
BEGIN SELECT RAISE(ABORT, 'experiment_results are append-only'); END;

CREATE TRIGGER trg_holdout_log_no_update BEFORE UPDATE ON holdout_access_log
BEGIN SELECT RAISE(ABORT, 'holdout_access_log is append-only'); END;
CREATE TRIGGER trg_holdout_log_no_delete BEFORE DELETE ON holdout_access_log
BEGIN SELECT RAISE(ABORT, 'holdout_access_log is append-only'); END;

CREATE TRIGGER trg_ledger_no_update BEFORE UPDATE ON research_ledger
BEGIN SELECT RAISE(ABORT, 'research_ledger is append-only'); END;
CREATE TRIGGER trg_ledger_no_delete BEFORE DELETE ON research_ledger
BEGIN SELECT RAISE(ABORT, 'research_ledger is append-only'); END;

CREATE TRIGGER trg_hypotheses_no_delete BEFORE DELETE ON hypotheses
BEGIN SELECT RAISE(ABORT, 'hypotheses are never deleted; set status REJECTED'); END;

CREATE TRIGGER trg_state_log_no_update BEFORE UPDATE ON system_state_log
BEGIN SELECT RAISE(ABORT, 'system_state_log is append-only'); END;
CREATE TRIGGER trg_state_log_no_delete BEFORE DELETE ON system_state_log
BEGIN SELECT RAISE(ABORT, 'system_state_log is append-only'); END;

CREATE TRIGGER trg_strategy_log_no_update BEFORE UPDATE ON strategy_status_log
BEGIN SELECT RAISE(ABORT, 'strategy_status_log is append-only'); END;
CREATE TRIGGER trg_strategy_log_no_delete BEFORE DELETE ON strategy_status_log
BEGIN SELECT RAISE(ABORT, 'strategy_status_log is append-only'); END;

CREATE TRIGGER trg_strategies_no_delete BEFORE DELETE ON strategies
BEGIN SELECT RAISE(ABORT, 'strategies are never deleted; set status RETIRED'); END;

CREATE TRIGGER trg_model_log_no_update BEFORE UPDATE ON model_status_log
BEGIN SELECT RAISE(ABORT, 'model_status_log is append-only'); END;
CREATE TRIGGER trg_model_log_no_delete BEFORE DELETE ON model_status_log
BEGIN SELECT RAISE(ABORT, 'model_status_log is append-only'); END;

CREATE TRIGGER trg_models_no_delete BEFORE DELETE ON models
BEGIN SELECT RAISE(ABORT, 'models are never deleted; set status RETIRED'); END;

CREATE TRIGGER trg_order_events_no_update BEFORE UPDATE ON order_events
BEGIN SELECT RAISE(ABORT, 'order_events are append-only'); END;
CREATE TRIGGER trg_order_events_no_delete BEFORE DELETE ON order_events
BEGIN SELECT RAISE(ABORT, 'order_events are append-only'); END;

CREATE TRIGGER trg_fills_no_update BEFORE UPDATE ON fills
BEGIN SELECT RAISE(ABORT, 'fills are append-only'); END;
CREATE TRIGGER trg_fills_no_delete BEFORE DELETE ON fills
BEGIN SELECT RAISE(ABORT, 'fills are append-only'); END;

CREATE TRIGGER trg_trade_events_no_update BEFORE UPDATE ON trade_events
BEGIN SELECT RAISE(ABORT, 'trade_events are append-only'); END;
CREATE TRIGGER trg_trade_events_no_delete BEFORE DELETE ON trade_events
BEGIN SELECT RAISE(ABORT, 'trade_events are append-only'); END;

CREATE TRIGGER trg_trades_no_delete BEFORE DELETE ON trades
BEGIN SELECT RAISE(ABORT, 'trades are never deleted'); END;

CREATE TRIGGER trg_ai_calls_no_update BEFORE UPDATE ON ai_calls
BEGIN SELECT RAISE(ABORT, 'ai_calls are append-only'); END;
CREATE TRIGGER trg_ai_calls_no_delete BEFORE DELETE ON ai_calls
BEGIN SELECT RAISE(ABORT, 'ai_calls are append-only'); END;

CREATE TRIGGER trg_ai_assessments_no_update BEFORE UPDATE ON ai_assessments
BEGIN SELECT RAISE(ABORT, 'ai_assessments are append-only'); END;
CREATE TRIGGER trg_ai_assessments_no_delete BEFORE DELETE ON ai_assessments
BEGIN SELECT RAISE(ABORT, 'ai_assessments are append-only'); END;

CREATE TRIGGER trg_ai_objections_no_update BEFORE UPDATE ON ai_objections
BEGIN SELECT RAISE(ABORT, 'ai_objections are append-only'); END;
CREATE TRIGGER trg_ai_objections_no_delete BEFORE DELETE ON ai_objections
BEGIN SELECT RAISE(ABORT, 'ai_objections are append-only'); END;

CREATE TRIGGER trg_datasets_no_update BEFORE UPDATE ON datasets
BEGIN SELECT RAISE(ABORT, 'datasets are immutable (content-addressed)'); END;
CREATE TRIGGER trg_datasets_no_delete BEFORE DELETE ON datasets
BEGIN SELECT RAISE(ABORT, 'datasets are immutable (content-addressed)'); END;

CREATE TRIGGER trg_evidence_no_update BEFORE UPDATE ON evidence_packets
BEGIN SELECT RAISE(ABORT, 'evidence_packets are immutable'); END;
CREATE TRIGGER trg_evidence_no_delete BEFORE DELETE ON evidence_packets
BEGIN SELECT RAISE(ABORT, 'evidence_packets are immutable'); END;

CREATE TRIGGER trg_human_notes_no_update BEFORE UPDATE ON human_notes
BEGIN SELECT RAISE(ABORT, 'human_notes are append-only'); END;
CREATE TRIGGER trg_human_notes_no_delete BEFORE DELETE ON human_notes
BEGIN SELECT RAISE(ABORT, 'human_notes are append-only'); END;

CREATE TRIGGER trg_ml_predictions_no_update BEFORE UPDATE ON ml_predictions
BEGIN SELECT RAISE(ABORT, 'ml_predictions are append-only'); END;
CREATE TRIGGER trg_ml_predictions_no_delete BEFORE DELETE ON ml_predictions
BEGIN SELECT RAISE(ABORT, 'ml_predictions are append-only'); END;
