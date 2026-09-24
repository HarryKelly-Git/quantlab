-- 096_research_guards.sql — database-level guards for the research ledger and the strategy
-- promotion framework (research/ledger.py, research/promotion.py).
--
-- WHY: the research record is only trustworthy if nobody can quietly rewrite it.
--   hypotheses : the text of a hypothesis is fixed once proposed; only status/updated_at change,
--                and only right after a matching 'status_change' ledger entry
--                ("STATUS <old> -> <new>: <reason>") has been appended. A hypothesis always starts
--                PROPOSED, and an AI-authored status change can never set SUPPORTED
--                (AI may generate ideas but cannot declare success).
--   strategies : stage/status changes must be preceded by a strategy_status_log row whose
--                to_stage/to_status equal the new values, so every promotion, demotion, pause or
--                retirement is logged with its reason and evidence.

CREATE TRIGGER trg_hypotheses_insert_proposed BEFORE INSERT ON hypotheses
WHEN NEW.status IS NOT 'PROPOSED'
BEGIN SELECT RAISE(ABORT, 'new hypotheses must start as PROPOSED (status changes go through ResearchLedger.set_status)'); END;

CREATE TRIGGER trg_hypotheses_fixed_fields BEFORE UPDATE ON hypotheses
WHEN NEW.hypothesis_id IS NOT OLD.hypothesis_id
  OR NEW.created_at IS NOT OLD.created_at
  OR NEW.source IS NOT OLD.source
  OR NEW.title IS NOT OLD.title
  OR NEW.hypothesis IS NOT OLD.hypothesis
  OR NEW.rationale IS NOT OLD.rationale
  OR NEW.required_data IS NOT OLD.required_data
  OR NEW.proposed_test IS NOT OLD.proposed_test
  OR NEW.expected_failure_conditions IS NOT OLD.expected_failure_conditions
  OR NEW.source_note_id IS NOT OLD.source_note_id
BEGIN SELECT RAISE(ABORT, 'hypotheses are fixed once proposed: only status/updated_at may change (ResearchLedger.set_status)'); END;

CREATE TRIGGER trg_hypotheses_status_logged BEFORE UPDATE OF status ON hypotheses
WHEN NEW.status IS NOT OLD.status AND NOT EXISTS (
    SELECT 1 FROM research_ledger r
    WHERE r.entry_id = (SELECT MAX(entry_id) FROM research_ledger WHERE hypothesis_id = NEW.hypothesis_id)
      AND r.entry_type = 'status_change'
      AND substr(r.text, 1, length('STATUS ' || OLD.status || ' -> ' || NEW.status || ':'))
          = 'STATUS ' || OLD.status || ' -> ' || NEW.status || ':'
)
BEGIN SELECT RAISE(ABORT, 'hypothesis status changes must be logged first (use ResearchLedger.set_status)'); END;

CREATE TRIGGER trg_hypotheses_ai_cannot_support BEFORE UPDATE OF status ON hypotheses
WHEN NEW.status = 'SUPPORTED' AND EXISTS (
    SELECT 1 FROM research_ledger r
    WHERE r.entry_id = (SELECT MAX(entry_id) FROM research_ledger WHERE hypothesis_id = NEW.hypothesis_id)
      AND (lower(r.author) = 'ai' OR lower(r.author) LIKE 'ai:%')
)
BEGIN SELECT RAISE(ABORT, 'an AI author can never mark a hypothesis SUPPORTED'); END;

CREATE TRIGGER trg_strategies_change_logged BEFORE UPDATE ON strategies
WHEN (NEW.stage IS NOT OLD.stage OR NEW.status IS NOT OLD.status) AND NOT EXISTS (
    SELECT 1 FROM strategy_status_log l
    WHERE l.id = (SELECT MAX(id) FROM strategy_status_log
                  WHERE strategy_id = NEW.strategy_id AND version = NEW.version)
      AND l.to_stage IS NEW.stage
      AND l.to_status IS NEW.status
)
BEGIN SELECT RAISE(ABORT, 'strategies.stage/status changes must be logged to strategy_status_log first (use research.promotion.PromotionManager)'); END;

CREATE INDEX IF NOT EXISTS ix_research_ledger_hypothesis ON research_ledger(hypothesis_id, entry_id);
CREATE INDEX IF NOT EXISTS ix_strategy_status_log_key ON strategy_status_log(strategy_id, version, id);
CREATE INDEX IF NOT EXISTS ix_shadow_opps_strategy ON shadow_opportunities(strategy_id, strategy_version, as_of_date);
CREATE INDEX IF NOT EXISTS ix_shadow_opps_candidate ON shadow_opportunities(candidate_id);
CREATE INDEX IF NOT EXISTS ix_trades_strategy ON trades(strategy_id, strategy_version, book);
