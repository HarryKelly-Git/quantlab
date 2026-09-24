-- 040_holdout_tokens.sql — single-use holdout unlock tokens (validation/holdout.py).
--
-- Every unlock writes a holdout_access_log row (001) AND a holdout_tokens row pointing at it.
-- Only the SHA-256 of the token is stored, never the token itself. A token is consumed by
-- inserting into holdout_token_uses; its PRIMARY KEY makes a second use impossible, even from
-- another process. All three tables are append-only, so the audit trail can never be edited.

CREATE TABLE holdout_tokens (
    token_hash     TEXT PRIMARY KEY,
    access_log_id  INTEGER NOT NULL REFERENCES holdout_access_log(id),
    issued_at      TEXT NOT NULL,
    period_start   TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    experiment_id  TEXT,
    actor          TEXT NOT NULL
);

CREATE TABLE holdout_token_uses (
    token_hash     TEXT PRIMARY KEY REFERENCES holdout_tokens(token_hash),
    used_at        TEXT NOT NULL,
    period_start   TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    experiment_id  TEXT
);

CREATE TRIGGER trg_holdout_tokens_no_update BEFORE UPDATE ON holdout_tokens
BEGIN SELECT RAISE(ABORT, 'holdout_tokens are append-only'); END;
CREATE TRIGGER trg_holdout_tokens_no_delete BEFORE DELETE ON holdout_tokens
BEGIN SELECT RAISE(ABORT, 'holdout_tokens are append-only'); END;
CREATE TRIGGER trg_holdout_token_uses_no_update BEFORE UPDATE ON holdout_token_uses
BEGIN SELECT RAISE(ABORT, 'holdout_token_uses are append-only'); END;
CREATE TRIGGER trg_holdout_token_uses_no_delete BEFORE DELETE ON holdout_token_uses
BEGIN SELECT RAISE(ABORT, 'holdout_token_uses are append-only'); END;
