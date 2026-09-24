-- 095_monitoring_guards.sql — database-level guards for the kill switch (monitoring/killswitch.py).
--
-- WHY: SYSTEM_PAUSED is the last line of defence against trading on bad data or unknown broker
-- state. The Python KillSwitch enforces the rules, but anything that can write SQL could bypass it.
-- These triggers make the rules hold for every writer:
--   * the single system_state row can never be deleted (deleting it would let AppContext
--     re-initialize the system as ACTIVE, silently undoing a pause);
--   * the state value must be ACTIVE or SYSTEM_PAUSED;
--   * every change must be written to system_state_log FIRST (same state/changed_at/changed_by as
--     the newest log row), so no state change is ever unlogged;
--   * leaving SYSTEM_PAUSED requires a human actor ('human' or 'human:<name>').

CREATE TRIGGER trg_system_state_no_delete BEFORE DELETE ON system_state
BEGIN SELECT RAISE(ABORT, 'system_state is never deleted (a pause must be resumed by a human via KillSwitch.resume)'); END;

CREATE TRIGGER trg_system_state_insert_valid BEFORE INSERT ON system_state
WHEN NEW.state NOT IN ('ACTIVE', 'SYSTEM_PAUSED')
BEGIN SELECT RAISE(ABORT, 'system_state.state must be ACTIVE or SYSTEM_PAUSED'); END;

CREATE TRIGGER trg_system_state_update_valid BEFORE UPDATE ON system_state
WHEN NEW.state NOT IN ('ACTIVE', 'SYSTEM_PAUSED')
BEGIN SELECT RAISE(ABORT, 'system_state.state must be ACTIVE or SYSTEM_PAUSED'); END;

CREATE TRIGGER trg_system_state_change_logged BEFORE UPDATE ON system_state
WHEN NOT EXISTS (
    SELECT 1 FROM system_state_log l
    WHERE l.id = (SELECT MAX(id) FROM system_state_log)
      AND l.state = NEW.state
      AND l.changed_at = NEW.changed_at
      AND l.changed_by = NEW.changed_by
)
BEGIN SELECT RAISE(ABORT, 'system_state changes must be logged to system_state_log first (use monitoring.killswitch.KillSwitch)'); END;

CREATE TRIGGER trg_system_state_human_resume BEFORE UPDATE ON system_state
WHEN OLD.state = 'SYSTEM_PAUSED' AND NEW.state = 'ACTIVE'
     AND NOT (lower(NEW.changed_by) = 'human' OR lower(NEW.changed_by) LIKE 'human:_%')
BEGIN SELECT RAISE(ABORT, 'resuming from SYSTEM_PAUSED requires a human actor (human review is required)'); END;

CREATE INDEX IF NOT EXISTS ix_health_checks_created ON health_checks(created_at);
CREATE INDEX IF NOT EXISTS ix_health_checks_run ON health_checks(run_id);
CREATE INDEX IF NOT EXISTS ix_ai_calls_started ON ai_calls(started_at);
CREATE INDEX IF NOT EXISTS ix_ai_assessments_candidate ON ai_assessments(candidate_id);
CREATE INDEX IF NOT EXISTS ix_reconciliations_book ON reconciliations(book, at);
