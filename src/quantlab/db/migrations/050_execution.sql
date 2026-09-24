-- 050_execution.sql — paper execution: ledger cash journal, corporate-action applications, position
-- history, immutable trade plans, order intents, refused orders and simulated-broker state.
-- Books (BOT / HUMAN) are always kept apart by the `book` column; every query filters on it.

-- Cash is never stored as a mutable balance: it is the SUM of this append-only journal, so every
-- dollar in a book can be traced to a fill, commission, dividend or the starting deposit.
CREATE TABLE ledger_cash_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    book          TEXT NOT NULL,
    session_date  TEXT,
    kind          TEXT NOT NULL,                 -- starting_cash | buy | sell | commission | dividend
    amount        REAL NOT NULL,                 -- signed: + adds cash, - removes cash
    symbol        TEXT,
    trade_id      TEXT,
    ref_type      TEXT,                          -- fill | corporate_action | init
    ref_id        TEXT,
    details_json  TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX ix_ledger_cash_book ON ledger_cash_events(book);
CREATE INDEX ix_ledger_cash_trade ON ledger_cash_events(trade_id);

-- One row per (book, symbol, session, action type) actually applied to a holding. The primary key
-- makes re-running a session idempotent (a split can never be applied twice).
CREATE TABLE ledger_corporate_actions (
    book          TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    session_date  TEXT NOT NULL,
    action_type   TEXT NOT NULL,                 -- split | cash_dividend
    ratio         REAL,
    amount        REAL,                          -- USD per pre-split share (dividends)
    qty_before    REAL NOT NULL,
    qty_after     REAL NOT NULL,
    cash_amount   REAL NOT NULL DEFAULT 0,       -- dividend cash credited (0 when not credited)
    credited      INTEGER NOT NULL DEFAULT 1,    -- 0 => dividend known but not credited (broker does not simulate it)
    trade_id      TEXT,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (book, symbol, session_date, action_type)
);

-- Append-only history of the mutable `positions` state table.
CREATE TABLE position_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    book             TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    trade_id         TEXT,
    qty_before       REAL NOT NULL,
    qty_after        REAL NOT NULL,
    avg_cost_before  REAL,
    avg_cost_after   REAL,
    reason           TEXT NOT NULL,              -- fill | split
    ref_id           TEXT,
    session_date     TEXT,
    at               TEXT NOT NULL
);
CREATE INDEX ix_position_log_book ON position_log(book, symbol);

-- The plan a trade was opened with (raw prices as of the signal session). Immutable: the exit
-- engine converts the ORIGINAL stop/target via tri-scaled prices exactly like core.tradesim.
CREATE TABLE trade_plans (
    trade_id          TEXT PRIMARY KEY,
    book              TEXT NOT NULL,
    signal_date       TEXT,
    entry_convention  TEXT NOT NULL DEFAULT 'next_open',
    entry_ref_price   REAL,
    stop_price        REAL,
    target_price      REAL,
    holding_sessions  INTEGER,
    direction         TEXT NOT NULL DEFAULT 'LONG',
    invalidation      TEXT,
    created_at        TEXT NOT NULL
);

-- Why an order exists: decision session, plan, journal snapshot, exit reason. Immutable.
CREATE TABLE order_intents (
    order_id      TEXT PRIMARY KEY,
    book          TEXT NOT NULL,
    session_date  TEXT NOT NULL,                 -- decision session (fills happen strictly after it)
    purpose       TEXT NOT NULL,                 -- entry | exit
    intent_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX ix_order_intents_book_session ON order_intents(book, session_date);

-- Orders the execution service refused to create (SYSTEM_PAUSED, limits, missing data ...).
CREATE TABLE execution_refusals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    book               TEXT NOT NULL,
    purpose            TEXT NOT NULL,            -- entry | exit
    symbol             TEXT,
    qty                REAL,
    reason             TEXT NOT NULL,
    candidate_id       TEXT,
    decision_id        TEXT,
    human_decision_id  TEXT,
    trade_id           TEXT,
    session_date       TEXT,
    details_json       TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_execution_refusals_book ON execution_refusals(book, created_at);

-- Simulated broker state per book (a STATE table: one JSON document, updated in place) plus an
-- append-only event log of everything the simulated broker did.
CREATE TABLE sim_broker_state (
    book         TEXT PRIMARY KEY,
    state_json   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE sim_broker_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    book             TEXT NOT NULL,
    session_date     TEXT,
    event            TEXT NOT NULL,              -- submitted | filled | rejected | expired | canceled | split | dividend
    client_order_id  TEXT,
    symbol           TEXT,
    details_json     TEXT,
    at               TEXT NOT NULL
);
CREATE INDEX ix_sim_broker_events_book ON sim_broker_events(book, session_date);

CREATE TRIGGER trg_ledger_cash_no_update BEFORE UPDATE ON ledger_cash_events
BEGIN SELECT RAISE(ABORT, 'ledger_cash_events are append-only'); END;
CREATE TRIGGER trg_ledger_cash_no_delete BEFORE DELETE ON ledger_cash_events
BEGIN SELECT RAISE(ABORT, 'ledger_cash_events are append-only'); END;

CREATE TRIGGER trg_ledger_ca_no_update BEFORE UPDATE ON ledger_corporate_actions
BEGIN SELECT RAISE(ABORT, 'ledger_corporate_actions are append-only'); END;
CREATE TRIGGER trg_ledger_ca_no_delete BEFORE DELETE ON ledger_corporate_actions
BEGIN SELECT RAISE(ABORT, 'ledger_corporate_actions are append-only'); END;

CREATE TRIGGER trg_position_log_no_update BEFORE UPDATE ON position_log
BEGIN SELECT RAISE(ABORT, 'position_log is append-only'); END;
CREATE TRIGGER trg_position_log_no_delete BEFORE DELETE ON position_log
BEGIN SELECT RAISE(ABORT, 'position_log is append-only'); END;

CREATE TRIGGER trg_trade_plans_no_update BEFORE UPDATE ON trade_plans
BEGIN SELECT RAISE(ABORT, 'trade_plans are immutable'); END;
CREATE TRIGGER trg_trade_plans_no_delete BEFORE DELETE ON trade_plans
BEGIN SELECT RAISE(ABORT, 'trade_plans are immutable'); END;

CREATE TRIGGER trg_order_intents_no_update BEFORE UPDATE ON order_intents
BEGIN SELECT RAISE(ABORT, 'order_intents are immutable'); END;
CREATE TRIGGER trg_order_intents_no_delete BEFORE DELETE ON order_intents
BEGIN SELECT RAISE(ABORT, 'order_intents are immutable'); END;

CREATE TRIGGER trg_execution_refusals_no_update BEFORE UPDATE ON execution_refusals
BEGIN SELECT RAISE(ABORT, 'execution_refusals are append-only'); END;
CREATE TRIGGER trg_execution_refusals_no_delete BEFORE DELETE ON execution_refusals
BEGIN SELECT RAISE(ABORT, 'execution_refusals are append-only'); END;

CREATE TRIGGER trg_sim_broker_events_no_update BEFORE UPDATE ON sim_broker_events
BEGIN SELECT RAISE(ABORT, 'sim_broker_events are append-only'); END;
CREATE TRIGGER trg_sim_broker_events_no_delete BEFORE DELETE ON sim_broker_events
BEGIN SELECT RAISE(ABORT, 'sim_broker_events are append-only'); END;

CREATE TRIGGER trg_sim_broker_state_no_delete BEFORE DELETE ON sim_broker_state
BEGIN SELECT RAISE(ABORT, 'sim_broker_state rows are never deleted'); END;
