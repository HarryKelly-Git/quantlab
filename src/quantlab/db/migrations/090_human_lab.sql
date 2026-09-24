-- 090_human_lab.sql — human paper-trading lab: decision context + linear correction chains.
--
-- human_decisions (001) is append-only. Two additions here:
--   * A partial UNIQUE index on supersedes_id: a decision can be superseded at most once, so every
--     correction chain is linear (A <- B <- C) and "the effective decision" is unambiguous. A fork
--     (two corrections of the same row) is refused by the database itself, not only by the service.
--   * human_decision_context: what the human was shown when deciding (server-built snapshot of the
--     opportunity card, whether the bot's own verdict was visible, data freshness). Needed to audit
--     anchoring effects in the human-vs-bot comparison. Append-only like the decision it describes.

CREATE UNIQUE INDEX IF NOT EXISTS ux_human_decisions_supersedes
    ON human_decisions(supersedes_id) WHERE supersedes_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_human_decisions_candidate ON human_decisions(candidate_id);

CREATE TABLE human_decision_context (
    decision_id           TEXT PRIMARY KEY REFERENCES human_decisions(decision_id),
    bot_decision_visible  INTEGER NOT NULL,          -- 1 => the bot's verdict was on the card
    data_last_session     TEXT,                      -- latest session with stored bars at decision time
    data_stale            INTEGER NOT NULL DEFAULT 0,-- 1 => stored bars older than monitoring.max_data_staleness_sessions
    display_json          TEXT,                      -- opportunity card as built server-side at decision time
    created_at            TEXT NOT NULL
);

CREATE TRIGGER trg_human_decision_context_no_update BEFORE UPDATE ON human_decision_context
BEGIN SELECT RAISE(ABORT, 'human_decision_context is append-only'); END;
CREATE TRIGGER trg_human_decision_context_no_delete BEFORE DELETE ON human_decision_context
BEGIN SELECT RAISE(ABORT, 'human_decision_context is append-only'); END;
