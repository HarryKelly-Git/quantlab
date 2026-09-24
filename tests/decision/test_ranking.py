"""OpportunityRanker: opportunity_score is bounded [0,1], a ranking (not a probability), and
comparable across strategies via percentile + shrunk quality."""
from __future__ import annotations

import pytest

from quantlab.core.types import Direction, MLPrediction
from quantlab.decision.ranking import OpportunityRanker
from quantlab.decision.stats_provider import StrategyStats

from ..shadow.support import make_candidate


class _StatsSource:
    def __init__(self, table):
        self.table = table

    def get(self, strategy_id, version, regime=None, as_of=None):
        return self.table.get((strategy_id, version), StrategyStats.empty(strategy_id, version))


def test_score_bounded_in_unit_interval():
    ranker = OpportunityRanker.__new__(OpportunityRanker)
    ranker.config = None
    ranker.weights = OpportunityRanker.DEFAULT_WEIGHTS
    ranker.quality_k = 50.0
    ranker.in_sample_weight = 0.25
    cands = [make_candidate(symbol="A", strategy_id=f"s{i}", score=float(i)) for i in range(5)]
    records = ranker.rank(cands)
    assert len(records) == 1
    assert 0.0 <= records[0].opportunity_score <= 1.0


def test_no_candidates_gives_no_records(config):
    ranker = OpportunityRanker(config)
    assert ranker.rank([]) == []


def test_lone_candidate_gets_neutral_percentile(config):
    ranker = OpportunityRanker(config)
    cand = make_candidate(symbol="A", strategy_id="s1", score=1.23)
    records = ranker.rank([cand])
    assert records[0].components["signal"] == pytest.approx(0.5)


def test_percentile_is_scale_invariant_monotone_rescaling(config):
    ranker = OpportunityRanker(config)
    raw = [make_candidate(symbol=f"S{i}", strategy_id="mom", score=float(i)) for i in range(1, 6)]
    rescaled = [make_candidate(symbol=f"S{i}", strategy_id="mom", score=float(i) ** 3 + 100) for i in range(1, 6)]
    r1 = {r.symbol: r.components["signal"] for r in ranker.rank(raw)}
    r2 = {r.symbol: r.components["signal"] for r in ranker.rank(rescaled)}
    assert r1 == pytest.approx(r2)


def test_higher_raw_score_ranks_higher_within_strategy(config):
    ranker = OpportunityRanker(config)
    cands = [make_candidate(symbol=f"S{i}", strategy_id="mom", score=float(i)) for i in range(5)]
    records = ranker.rank(cands)
    ordered = sorted(records, key=lambda r: r.rank)
    scores = [r.opportunity_score for r in ordered]
    symbols_desc_score = [r.symbol for r in sorted(records, key=lambda r: -r.candidates[0].score)]
    assert [r.symbol for r in ordered] == symbols_desc_score
    assert scores == sorted(scores, reverse=True)


def test_rank_is_per_day_not_global(config):
    ranker = OpportunityRanker(config)
    cands = [
        make_candidate(symbol="A", strategy_id="mom", score=1.0, as_of="2024-01-03"),
        make_candidate(symbol="B", strategy_id="mom", score=2.0, as_of="2024-01-03"),
        make_candidate(symbol="C", strategy_id="mom", score=1.0, as_of="2024-01-04"),
    ]
    records = ranker.rank(cands)
    day1 = [r for r in records if str(r.as_of_date) == "2024-01-03"]
    day2 = [r for r in records if str(r.as_of_date) == "2024-01-04"]
    assert {r.rank for r in day1} == {1, 2}
    assert [r.rank for r in day2] == [1]


def test_opposite_directions_stay_separate_records(config):
    ranker = OpportunityRanker(config)
    cands = [make_candidate(symbol="A", strategy_id="mom", direction=Direction.LONG),
            make_candidate(symbol="A", strategy_id="mom2", direction=Direction.SHORT)]
    records = ranker.rank(cands)
    assert len(records) == 2
    assert {r.direction for r in records} == {Direction.LONG, Direction.SHORT}


def test_quality_only_used_when_stats_provided(config):
    ranker = OpportunityRanker(config)
    cand = make_candidate(symbol="A", strategy_id="mom")
    no_stats = ranker.rank([cand])
    assert no_stats[0].components["quality"] is None
    assert "quality" not in no_stats[0].weights

    stats = {("mom", "1.0.0"): StrategyStats(strategy_id="mom", version="1.0.0", n=200, t_stat=4.0,
                                             source="walk_forward_oos")}
    with_stats = ranker.rank([cand], stats=_StatsSource(stats))
    assert with_stats[0].components["quality"] is not None
    assert with_stats[0].components["quality"] > 0.5   # positive t-stat -> above neutral


def test_quality_shrinks_toward_neutral_with_small_n(config):
    ranker = OpportunityRanker(config)
    cand = make_candidate(symbol="A", strategy_id="mom")
    strong_evidence = {("mom", "1.0.0"): StrategyStats(strategy_id="mom", version="1.0.0", n=10_000, t_stat=5.0,
                                                        source="walk_forward_oos")}
    weak_evidence = {("mom", "1.0.0"): StrategyStats(strategy_id="mom", version="1.0.0", n=1, t_stat=5.0,
                                                      source="walk_forward_oos")}
    q_strong = ranker.rank([cand], stats=_StatsSource(strong_evidence))[0].components["quality"]
    q_weak = ranker.rank([cand], stats=_StatsSource(weak_evidence))[0].components["quality"]
    assert q_strong > q_weak > 0.5


def test_ml_component_neutral_when_batch_has_no_usable_predictions(config):
    ranker = OpportunityRanker(config)
    cand = make_candidate(symbol="A", strategy_id="mom")
    uncalibrated = {cand.candidate_id: MLPrediction("m", "1", "up", 0.9, calibrated=False, status="ACTIVE")}
    records = ranker.rank([cand], ml=uncalibrated)
    assert records[0].components["ml"] is None
    assert "ml" not in records[0].weights


def test_ml_component_used_when_calibrated_and_active(config):
    ranker = OpportunityRanker(config)
    cand = make_candidate(symbol="A", strategy_id="mom")
    preds = {cand.candidate_id: MLPrediction("m", "1", "up", 0.9, calibrated=True, status="ACTIVE")}
    records = ranker.rank([cand], ml=preds)
    assert records[0].components["ml"] == pytest.approx(0.9)


def test_multi_strategy_confirmation_beats_single_strategy_but_not_unboundedly(config):
    ranker = OpportunityRanker(config)
    single = [make_candidate(symbol="A", strategy_id="mom", score=1.0)]
    confirmed = [make_candidate(symbol="A", strategy_id="mom", score=1.0),
                make_candidate(symbol="A", strategy_id="reversal", score=1.0)]
    rec_single = ranker.rank(single)[0]
    rec_confirmed = ranker.rank(confirmed)[0]
    assert rec_confirmed.components["confirmation"] > 0.0
    assert rec_single.components["confirmation"] == 0.0
    assert rec_confirmed.opportunity_score > rec_single.opportunity_score
    assert rec_confirmed.opportunity_score <= 1.0


def test_three_mediocre_signals_do_not_beat_one_strong_signal(config):
    """Confirmation is a small-weight input, not a vote count. Percentile is WITHIN a strategy's
    same-day candidates, so peers are needed to make "strong" (near the top of its pool) and
    "mediocre" (near the bottom of its pool) meaningful."""
    ranker = OpportunityRanker(config)
    cands = [
        # symbol A: one strategy, near the top of its own (4-candidate) pool -> high percentile,
        # no other strategy confirms it.
        make_candidate(symbol="A", strategy_id="mom", score=100.0),
        make_candidate(symbol="X1", strategy_id="mom", score=1.0),
        make_candidate(symbol="X2", strategy_id="mom", score=2.0),
        make_candidate(symbol="X3", strategy_id="mom", score=3.0),
    ]
    for i in range(3):
        # symbol B: three DIFFERENT strategies, each time near the BOTTOM of a 2-candidate pool
        # (a stronger peer in the same strategy) -> low percentile in every one, but 3 strategies
        # agree on B, so confirmation is high.
        cands.append(make_candidate(symbol="B", strategy_id=f"s{i}", score=0.01))
        cands.append(make_candidate(symbol=f"Y{i}", strategy_id=f"s{i}", score=9.0))
    records = {r.symbol: r for r in ranker.rank(cands)}
    rec_strong, rec_many = records["A"], records["B"]
    assert rec_strong.components["signal"] > 0.5
    assert rec_many.components["signal"] < 0.5
    assert rec_many.components["confirmation"] > rec_strong.components["confirmation"]
    assert rec_strong.opportunity_score > rec_many.opportunity_score


def test_weights_must_be_nonnegative_with_positive_signal_weight(config):
    with pytest.raises(ValueError):
        OpportunityRanker(config.with_overrides({"ranking": {"weights": {"signal": 0}}}))
    with pytest.raises(ValueError):
        OpportunityRanker(config.with_overrides({"ranking": {"weights": {"quality": -1}}}))


def test_persist_ranking_round_trips(db, config):
    from quantlab.decision.ranking import persist_ranking
    ranker = OpportunityRanker(config)
    cand = make_candidate(symbol="A", strategy_id="mom")
    records = ranker.rank([cand])
    n = persist_ranking(db, records, run_id="run1")
    assert n == 1
    row = db.fetchone("SELECT * FROM opportunity_rankings WHERE symbol='A'")
    assert row is not None
    assert row["opportunity_score"] == pytest.approx(records[0].opportunity_score)
    assert row["rank"] == 1
