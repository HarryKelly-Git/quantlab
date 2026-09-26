-- Catalyst discovery (research only). Candidate catalyst records, setup class and the evidence
-- chain; the run-level catalyst panel, source coverage and the basic-universe symbol list (so an
-- overnight catalyst can only create a candidate for a symbol that passed the scan's basic filter).
-- Comments stay on their own lines: the migration splitter needs statements that end a line with ";".
ALTER TABLE discovery_candidates ADD COLUMN setup_class TEXT;
ALTER TABLE discovery_candidates ADD COLUMN catalyst_families TEXT;
ALTER TABLE discovery_candidates ADD COLUMN catalyst_record_json TEXT;
ALTER TABLE discovery_candidates ADD COLUMN evidence_chain_json TEXT;
-- EOD_SCAN (end-of-day scan) or OVERNIGHT_REFRESH (created by a post-close / overnight catalyst)
ALTER TABLE discovery_candidates ADD COLUMN created_by TEXT;
ALTER TABLE discovery_runs ADD COLUMN catalysts_json TEXT;
ALTER TABLE discovery_runs ADD COLUMN source_coverage_json TEXT;
ALTER TABLE discovery_runs ADD COLUMN basic_symbols_json TEXT;
