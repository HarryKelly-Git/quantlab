-- 091_shadow_outcome_details.sql — audit detail for every shadow_outcomes row.
--
-- shadow_outcomes (001) holds the headline numbers. This sidecar records HOW each number was
-- produced so it can be reproduced and audited: exit reasons, gross vs cost split, the buy-and-hold
-- leg, the cost model used and — for point-in-time audits — the last session present in the panel
-- when the outcome was measured (proves no unseen session was used). Written once, never changed.

CREATE TABLE shadow_outcome_details (
    opportunity_id    TEXT NOT NULL,
    horizon_sessions  INTEGER NOT NULL,
    method            TEXT NOT NULL,                 -- e.g. tradesim.simulate_plan/v1
    direction         TEXT,                          -- LONG | SHORT
    exit_reason       TEXT,                          -- STOP | TARGET | TIME | DELISTED (plan leg)
    gross_ret         REAL,                          -- plan leg before costs
    cost_ret          REAL,                          -- modeled round-trip cost fraction
    holding_sessions  INTEGER,                       -- plan leg sessions held
    hold_exit_date    TEXT,                          -- buy-and-hold leg
    hold_exit_reason  TEXT,
    hold_gross_ret    REAL,
    benchmark         TEXT,
    panel_last_date   TEXT NOT NULL,                 -- last session in the panel at measurement time
    cost_model_json   TEXT,
    measured_at       TEXT NOT NULL,
    PRIMARY KEY (opportunity_id, horizon_sessions),
    FOREIGN KEY (opportunity_id, horizon_sessions) REFERENCES shadow_outcomes(opportunity_id, horizon_sessions)
);

CREATE TRIGGER trg_shadow_outcome_details_no_update BEFORE UPDATE ON shadow_outcome_details
BEGIN SELECT RAISE(ABORT, 'shadow_outcome_details are append-only'); END;
CREATE TRIGGER trg_shadow_outcome_details_no_delete BEFORE DELETE ON shadow_outcome_details
BEGIN SELECT RAISE(ABORT, 'shadow_outcome_details are append-only'); END;
