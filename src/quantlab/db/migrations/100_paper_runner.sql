-- 100_paper_runner.sql: the persistent Alpaca PAPER runner (pipeline/runner.py).
-- The mode/endpoint CHECK constraints make a non-paper runner session impossible to record.

-- One row per runner process. Mutable status/heartbeat; everything that happened is in
-- paper_runner_events (append-only).
CREATE TABLE paper_runner_sessions (
    session_id         TEXT PRIMARY KEY,
    book               TEXT NOT NULL,
    mode               TEXT NOT NULL CHECK (mode = 'PAPER'),
    endpoint           TEXT NOT NULL CHECK (endpoint = 'https://paper-api.alpaca.markets'),
    broker             TEXT NOT NULL,
    pid                INTEGER,
    host               TEXT,
    started_at         TEXT NOT NULL,
    last_heartbeat_at  TEXT,
    status             TEXT NOT NULL,     -- RUNNING | STOPPED | CRASHED | REFUSED
    phase              TEXT,              -- human-readable current activity
    stream_status      TEXT,              -- connecting | connected | disconnected | unauthorized | stopped
    detail_json        TEXT,              -- latest broker snapshot, next action, eligible strategies
    stop_requested_at  TEXT,
    stop_reason        TEXT,
    stopped_at         TEXT,
    error              TEXT
);
CREATE INDEX ix_paper_runner_sessions_started ON paper_runner_sessions(started_at);

CREATE TABLE paper_runner_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT,
    at            TEXT NOT NULL,
    level         TEXT NOT NULL,          -- INFO | WARN | ERROR | CRITICAL
    kind          TEXT NOT NULL,          -- preflight | schedule | job | stream | reconcile | killswitch | error | ...
    message       TEXT NOT NULL,
    details_json  TEXT
);
CREATE INDEX ix_paper_runner_events_session ON paper_runner_events(session_id, id);

-- Startup preflight audit (never contains credentials; the account reference is masked).
CREATE TABLE paper_preflights (
    preflight_id    TEXT PRIMARY KEY,
    session_id      TEXT,
    at              TEXT NOT NULL,
    ok              INTEGER NOT NULL,
    trading_mode    TEXT,
    live_trading    TEXT,
    endpoint        TEXT,
    account_ref     TEXT,
    account_status  TEXT,
    currency        TEXT,
    equity          REAL,
    cash            REAL,
    buying_power    REAL,
    positions_json  TEXT,
    open_orders     INTEGER,
    checks_json     TEXT NOT NULL,
    reason          TEXT
);

-- Every order update seen from the broker (trade_updates stream or REST poll). Replayed stream
-- events are ignored by the unique index, so a reconnect can never double-count anything here.
CREATE TABLE broker_order_updates (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT,
    received_at       TEXT NOT NULL,
    source            TEXT NOT NULL,      -- stream | poll | test
    event             TEXT NOT NULL,      -- new | accepted | partial_fill | fill | canceled | rejected | ...
    order_id          TEXT,               -- QuantLab order id (NULL when the broker order is unknown locally)
    client_order_id   TEXT,
    broker_order_id   TEXT,
    execution_id      TEXT,
    status            TEXT,
    event_qty         REAL,
    event_price       REAL,
    filled_qty        REAL,
    filled_avg_price  REAL,
    event_at          TEXT,
    raw_json          TEXT
);
CREATE UNIQUE INDEX ux_broker_order_updates ON broker_order_updates(
    COALESCE(broker_order_id, ''), event, COALESCE(execution_id, ''), COALESCE(event_at, ''));
CREATE INDEX ix_broker_order_updates_client ON broker_order_updates(client_order_id);

-- One job per (book, decision session): which pipeline run processed it and whether orders were
-- allowed. A restart resumes the SAME run_id (same candidates => same client_order_ids), which is
-- what makes a restart unable to duplicate an order.
CREATE TABLE paper_session_jobs (
    book            TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,
    next_session    TEXT,
    run_id          TEXT,
    status          TEXT NOT NULL,        -- running | succeeded | failed | stale_data
    orders_allowed  INTEGER NOT NULL,
    reason          TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (book, as_of_date)
);

-- Which broker a paper book's ledger is bound to. Immutable: a book seeded from the Alpaca paper
-- account can never be advanced by the simulated broker (or vice versa) in the same database.
CREATE TABLE paper_book_bindings (
    book          TEXT PRIMARY KEY,
    broker        TEXT NOT NULL,
    bound_at      TEXT NOT NULL,
    details_json  TEXT
);

CREATE TRIGGER trg_paper_runner_events_no_update BEFORE UPDATE ON paper_runner_events
BEGIN SELECT RAISE(ABORT, 'paper_runner_events are append-only'); END;
CREATE TRIGGER trg_paper_runner_events_no_delete BEFORE DELETE ON paper_runner_events
BEGIN SELECT RAISE(ABORT, 'paper_runner_events are append-only'); END;
CREATE TRIGGER trg_paper_preflights_no_update BEFORE UPDATE ON paper_preflights
BEGIN SELECT RAISE(ABORT, 'paper_preflights are append-only'); END;
CREATE TRIGGER trg_paper_preflights_no_delete BEFORE DELETE ON paper_preflights
BEGIN SELECT RAISE(ABORT, 'paper_preflights are append-only'); END;
CREATE TRIGGER trg_broker_order_updates_no_update BEFORE UPDATE ON broker_order_updates
BEGIN SELECT RAISE(ABORT, 'broker_order_updates are append-only'); END;
CREATE TRIGGER trg_broker_order_updates_no_delete BEFORE DELETE ON broker_order_updates
BEGIN SELECT RAISE(ABORT, 'broker_order_updates are append-only'); END;
CREATE TRIGGER trg_paper_book_bindings_no_update BEFORE UPDATE ON paper_book_bindings
BEGIN SELECT RAISE(ABORT, 'paper_book_bindings are immutable'); END;
CREATE TRIGGER trg_paper_book_bindings_no_delete BEFORE DELETE ON paper_book_bindings
BEGIN SELECT RAISE(ABORT, 'paper_book_bindings are immutable'); END;
