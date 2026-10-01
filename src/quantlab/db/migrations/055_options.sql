-- 055_options.sql: options layer (src/quantlab/options/). Execution range 050-059.
-- Evaluations (stock vs defined-risk option structures, indicative quotes) and the separate PAPER
-- options book `OPT` (structures, orders, fills, premium cash journal, marks). The OPT book never
-- shares tables with BOT/HUMAN. Every contract quantity is in CONTRACTS; every price is per share;
-- USD amounts are price x qty x multiplier with the multiplier stored on every fill.

-- One row per thesis evaluation: every input (thesis, distribution, settings) and the choice.
CREATE TABLE options_eval_runs (
    evaluation_id        TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    session_date         TEXT NOT NULL,          -- NY date the quotes belong to
    symbol               TEXT NOT NULL,
    direction            TEXT NOT NULL,          -- LONG | SHORT
    horizon_sessions     INTEGER NOT NULL,
    horizon_date         TEXT NOT NULL,
    spot                 REAL NOT NULL,
    spot_source          TEXT,
    stock_stop           REAL NOT NULL,
    feed                 TEXT NOT NULL,          -- always 'indicative' on this account
    expiration           TEXT,                   -- chosen expiry (NULL = none qualified)
    expiry_reason        TEXT,
    distribution_label   TEXT NOT NULL,
    distribution_json    TEXT NOT NULL,
    chosen_expression    TEXT NOT NULL,          -- STOCK | <structure_id> | NO_TRADE
    reason               TEXT NOT NULL,
    contracts_checked    INTEGER NOT NULL,
    contracts_passed     INTEGER NOT NULL,
    grid_notes_json      TEXT,
    data_notes_json      TEXT,
    settings_json        TEXT NOT NULL,
    is_synthetic         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_options_eval_runs_symbol ON options_eval_runs(symbol, session_date);

-- One row per thesis x expression (the stock and each candidate structure) per evaluation.
CREATE TABLE options_evaluations (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id             TEXT NOT NULL REFERENCES options_eval_runs(evaluation_id),
    session_date              TEXT NOT NULL,
    symbol                    TEXT NOT NULL,
    expression_id             TEXT NOT NULL,
    kind                      TEXT NOT NULL,     -- STOCK | SHORT_STOCK | LONG_CALL | LONG_PUT | *_DEBIT_SPREAD
    expiration                TEXT,
    legs_json                 TEXT,              -- per leg: OCC symbol, side, strike, bid, ask, quote_time, iv
    feed                      TEXT NOT NULL,
    quote_time_min            TEXT,
    quote_time_max            TEXT,
    entry_cost                REAL,              -- USD per unit at the executable side
    risk_usd                  REAL,
    max_loss_usd              REAL,
    max_gain_usd              REAL,              -- NULL = unbounded
    breakeven                 REAL,
    expected_pnl_usd          REAL,
    expected_pnl_per_risk     REAL,
    p_profit                  REAL,
    p_breakeven               REAL,
    max_loss_per_risk         REAL,
    expected_loss_given_loss  REAL,
    eligible                  INTEGER NOT NULL,
    ineligible_reason         TEXT,
    chosen                    INTEGER NOT NULL,
    label                     TEXT NOT NULL,     -- MODEL_OUTPUT
    notes_json                TEXT,
    created_at                TEXT NOT NULL,
    UNIQUE (evaluation_id, expression_id)
);
CREATE INDEX ix_options_evaluations_eval ON options_evaluations(evaluation_id);

-- Every contract the liquidity filter saw for an evaluation, with every failed rule.
CREATE TABLE options_liquidity_checks (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id      TEXT NOT NULL REFERENCES options_eval_runs(evaluation_id),
    contract_symbol    TEXT NOT NULL,
    passed             INTEGER NOT NULL,
    reasons_json       TEXT NOT NULL,
    bid                REAL,
    ask                REAL,
    spread_pct         REAL,
    quote_time         TEXT,
    quote_age_minutes  REAL,
    open_interest      REAL,
    oi_status          TEXT NOT NULL,            -- KNOWN | UNKNOWN
    feed               TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX ix_options_liq_eval ON options_liquidity_checks(evaluation_id);

-- OPT paper structures: a STATE row per structure (status changes in place) + append-only events.
CREATE TABLE options_structures (
    structure_id      TEXT PRIMARY KEY,          -- internal id (opt_...)
    book              TEXT NOT NULL DEFAULT 'OPT' CHECK (book = 'OPT'),
    evaluation_id     TEXT,
    kind              TEXT NOT NULL,
    underlying        TEXT NOT NULL,
    expiration        TEXT NOT NULL,
    multiplier        REAL NOT NULL,
    legs_json         TEXT NOT NULL,
    qty               INTEGER NOT NULL,          -- structures requested
    max_loss_usd      REAL NOT NULL,             -- worst case at submission (long legs at their limits)
    horizon_date      TEXT,
    close_by_date     TEXT NOT NULL,             -- never held into expiry
    status            TEXT NOT NULL,             -- OPENING | OPEN | CLOSING | CLOSED | ABANDONED
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE options_structure_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    structure_id  TEXT NOT NULL,
    event         TEXT NOT NULL,
    details_json  TEXT,
    at            TEXT NOT NULL
);
CREATE INDEX ix_options_structure_events ON options_structure_events(structure_id);

-- OPT orders: one STATE row per order (single leg, LIMIT, DAY only) + append-only events.
CREATE TABLE options_orders (
    order_id          TEXT PRIMARY KEY,
    client_order_id   TEXT NOT NULL UNIQUE,      -- always starts with qlopt-
    book              TEXT NOT NULL DEFAULT 'OPT' CHECK (book = 'OPT'),
    structure_id      TEXT NOT NULL REFERENCES options_structures(structure_id),
    leg_role          TEXT NOT NULL,             -- long | short
    purpose           TEXT NOT NULL,             -- open | close
    contract_symbol   TEXT NOT NULL,
    side              TEXT NOT NULL,             -- buy | sell
    qty               INTEGER NOT NULL,          -- contracts
    order_type        TEXT NOT NULL CHECK (order_type = 'limit'),
    time_in_force     TEXT NOT NULL CHECK (time_in_force = 'day'),
    limit_price       REAL NOT NULL,
    multiplier        REAL NOT NULL,
    quote_json        TEXT,                      -- the indicative quote the limit was derived from
    status            TEXT NOT NULL,
    broker            TEXT NOT NULL,
    broker_order_id   TEXT,
    filled_qty        REAL NOT NULL DEFAULT 0,
    filled_avg_price  REAL,
    created_at        TEXT NOT NULL,
    last_update_at    TEXT NOT NULL,
    raw_json          TEXT
);
CREATE INDEX ix_options_orders_status ON options_orders(status);

CREATE TABLE options_order_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      TEXT NOT NULL,
    event         TEXT NOT NULL,
    details_json  TEXT,
    at            TEXT NOT NULL
);

-- Fills exactly as the broker reported them (cumulative deltas, never a model or mid price).
CREATE TABLE options_fills (
    fill_id           TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL REFERENCES options_orders(order_id),
    structure_id      TEXT NOT NULL,
    contract_symbol   TEXT NOT NULL,
    side              TEXT NOT NULL,
    qty               REAL NOT NULL,             -- contracts in this delta
    price             REAL NOT NULL,             -- per share
    multiplier        REAL NOT NULL,
    cash_amount       REAL NOT NULL,             -- signed USD: -qty*price*multiplier for buys
    cum_filled_qty    REAL NOT NULL,
    session_date      TEXT,
    filled_at         TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE (order_id, cum_filled_qty)
);

-- Premium journal of the OPT book (allocation + every fill). The OPT book's net trading cash is
-- what moves the shared Alpaca paper account's cash outside the stock ledger.
CREATE TABLE options_cash_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    book          TEXT NOT NULL DEFAULT 'OPT' CHECK (book = 'OPT'),
    kind          TEXT NOT NULL,                 -- allocation | premium_paid | premium_received
    amount        REAL NOT NULL,
    fill_id       TEXT,
    structure_id  TEXT,
    details_json  TEXT,
    created_at    TEXT NOT NULL
);

-- Orders the OPT executor refused to create (gated off, kill switch, caps, naked short ...).
CREATE TABLE options_refusals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id  TEXT,
    structure_id   TEXT,
    reason         TEXT NOT NULL,
    details_json   TEXT,
    created_at     TEXT NOT NULL
);

-- Marks from historical option daily bars (real traded closes). A leg without a bar that session
-- makes the mark UNKNOWN; nothing is ever interpolated. A later KNOWN row may follow an UNKNOWN one.
CREATE TABLE options_marks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_type        TEXT NOT NULL,               -- evaluation | position
    ref_id          TEXT NOT NULL,               -- evaluation_id | structure_id (OPT)
    expression_id   TEXT NOT NULL,               -- STOCK | <structure expression id>
    session_date    TEXT NOT NULL,
    status          TEXT NOT NULL,               -- KNOWN | UNKNOWN
    value_per_unit  REAL,                        -- per share (stock: close; structure: sum of signed leg closes)
    pnl_usd         REAL,                        -- per unit vs the executable entry cost
    pnl_per_risk    REAL,
    legs_json       TEXT,
    reason          TEXT,
    source          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (ref_type, ref_id, expression_id, session_date, status)
);
CREATE INDEX ix_options_marks_ref ON options_marks(ref_type, ref_id);

CREATE TRIGGER trg_options_eval_runs_no_update BEFORE UPDATE ON options_eval_runs
BEGIN SELECT RAISE(ABORT, 'options_eval_runs are append-only'); END;
CREATE TRIGGER trg_options_eval_runs_no_delete BEFORE DELETE ON options_eval_runs
BEGIN SELECT RAISE(ABORT, 'options_eval_runs are append-only'); END;
CREATE TRIGGER trg_options_evaluations_no_update BEFORE UPDATE ON options_evaluations
BEGIN SELECT RAISE(ABORT, 'options_evaluations are append-only'); END;
CREATE TRIGGER trg_options_evaluations_no_delete BEFORE DELETE ON options_evaluations
BEGIN SELECT RAISE(ABORT, 'options_evaluations are append-only'); END;
CREATE TRIGGER trg_options_liq_no_update BEFORE UPDATE ON options_liquidity_checks
BEGIN SELECT RAISE(ABORT, 'options_liquidity_checks are append-only'); END;
CREATE TRIGGER trg_options_liq_no_delete BEFORE DELETE ON options_liquidity_checks
BEGIN SELECT RAISE(ABORT, 'options_liquidity_checks are append-only'); END;
CREATE TRIGGER trg_options_structure_events_no_update BEFORE UPDATE ON options_structure_events
BEGIN SELECT RAISE(ABORT, 'options_structure_events are append-only'); END;
CREATE TRIGGER trg_options_structure_events_no_delete BEFORE DELETE ON options_structure_events
BEGIN SELECT RAISE(ABORT, 'options_structure_events are append-only'); END;
CREATE TRIGGER trg_options_structures_no_delete BEFORE DELETE ON options_structures
BEGIN SELECT RAISE(ABORT, 'options_structures rows are never deleted'); END;
CREATE TRIGGER trg_options_orders_no_delete BEFORE DELETE ON options_orders
BEGIN SELECT RAISE(ABORT, 'options_orders rows are never deleted'); END;
CREATE TRIGGER trg_options_order_events_no_update BEFORE UPDATE ON options_order_events
BEGIN SELECT RAISE(ABORT, 'options_order_events are append-only'); END;
CREATE TRIGGER trg_options_order_events_no_delete BEFORE DELETE ON options_order_events
BEGIN SELECT RAISE(ABORT, 'options_order_events are append-only'); END;
CREATE TRIGGER trg_options_fills_no_update BEFORE UPDATE ON options_fills
BEGIN SELECT RAISE(ABORT, 'options_fills are append-only'); END;
CREATE TRIGGER trg_options_fills_no_delete BEFORE DELETE ON options_fills
BEGIN SELECT RAISE(ABORT, 'options_fills are append-only'); END;
CREATE TRIGGER trg_options_cash_no_update BEFORE UPDATE ON options_cash_events
BEGIN SELECT RAISE(ABORT, 'options_cash_events are append-only'); END;
CREATE TRIGGER trg_options_cash_no_delete BEFORE DELETE ON options_cash_events
BEGIN SELECT RAISE(ABORT, 'options_cash_events are append-only'); END;
CREATE TRIGGER trg_options_refusals_no_update BEFORE UPDATE ON options_refusals
BEGIN SELECT RAISE(ABORT, 'options_refusals are append-only'); END;
CREATE TRIGGER trg_options_refusals_no_delete BEFORE DELETE ON options_refusals
BEGIN SELECT RAISE(ABORT, 'options_refusals are append-only'); END;
CREATE TRIGGER trg_options_marks_no_update BEFORE UPDATE ON options_marks
BEGIN SELECT RAISE(ABORT, 'options_marks are append-only'); END;
CREATE TRIGGER trg_options_marks_no_delete BEFORE DELETE ON options_marks
BEGIN SELECT RAISE(ABORT, 'options_marks are append-only'); END;
