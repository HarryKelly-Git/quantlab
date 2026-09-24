-- 002_shared_research.sql — tables shared across subsystems (written by one, read by others).

-- Written by backtest/ (experiments), read by decision/ (EV + no-trade strategy statistics),
-- research/ (promotion evidence) and the dashboard.
CREATE TABLE backtest_trades (
    experiment_id    TEXT NOT NULL REFERENCES experiments(experiment_id),
    trade_id         TEXT NOT NULL,
    segment          TEXT NOT NULL DEFAULT 'full',   -- full | in_sample | oos:<window> | holdout
    strategy_id      TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    signal_date      TEXT NOT NULL,
    entry_date       TEXT,
    exit_date        TEXT,
    exit_reason      TEXT,
    qty              REAL,
    entry_price      REAL,                           -- raw fill price
    exit_price       REAL,
    gross_ret        REAL,
    cost_ret         REAL,
    net_ret          REAL,
    pnl              REAL,
    holding_sessions INTEGER,
    mfe              REAL,
    mae              REAL,
    score            REAL,
    regime           TEXT,
    sector           TEXT,
    PRIMARY KEY (experiment_id, trade_id)
);
CREATE INDEX ix_bt_trades_strategy ON backtest_trades(strategy_id, strategy_version);

CREATE TABLE backtest_equity (
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    segment         TEXT NOT NULL DEFAULT 'full',
    date            TEXT NOT NULL,
    equity          REAL NOT NULL,
    cash            REAL NOT NULL,
    gross_exposure  REAL NOT NULL,
    positions       INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, segment, date)
);

-- Written by data/validation.py, read by the universe/pipeline: symbols excluded because their
-- data failed integrity checks (e.g. unexplained split-like jump). Rows are append-only; a
-- quarantine ends by inserting a row with a to_date (or a newer superseding row).
CREATE TABLE symbol_quarantine (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    reason      TEXT NOT NULL,
    from_date   TEXT,
    to_date     TEXT,
    run_id      TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX ix_quarantine_symbol ON symbol_quarantine(symbol);

CREATE TRIGGER trg_bt_trades_no_update BEFORE UPDATE ON backtest_trades
BEGIN SELECT RAISE(ABORT, 'backtest_trades are append-only'); END;
CREATE TRIGGER trg_bt_trades_no_delete BEFORE DELETE ON backtest_trades
BEGIN SELECT RAISE(ABORT, 'backtest_trades are append-only'); END;
CREATE TRIGGER trg_bt_equity_no_update BEFORE UPDATE ON backtest_equity
BEGIN SELECT RAISE(ABORT, 'backtest_equity is append-only'); END;
CREATE TRIGGER trg_bt_equity_no_delete BEFORE DELETE ON backtest_equity
BEGIN SELECT RAISE(ABORT, 'backtest_equity is append-only'); END;
CREATE TRIGGER trg_quarantine_no_update BEFORE UPDATE ON symbol_quarantine
BEGIN SELECT RAISE(ABORT, 'symbol_quarantine is append-only'); END;
CREATE TRIGGER trg_quarantine_no_delete BEFORE DELETE ON symbol_quarantine
BEGIN SELECT RAISE(ABORT, 'symbol_quarantine is append-only'); END;
