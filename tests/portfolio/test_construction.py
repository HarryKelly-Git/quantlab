"""PortfolioConstructor: equal-risk sizing math, whole shares, every cap (position weight, cash
buffer, gross exposure, sector weight, open risk, positions, correlation), and explainability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.construction import BookState, HeldPosition, PortfolioConstructor

from ..shadow.support import make_candidate, make_panel


def _panel_with_correlation_structure(n=90):
    """HELD/CORR move together; ANTI is their mirror image; INDEP is unrelated noise."""
    rng = np.random.default_rng(7)
    r_held = rng.normal(0.0, 0.01, n)
    r_corr = r_held + rng.normal(0.0, 0.0005, n)
    r_anti = -r_held
    r_indep = rng.normal(0.0, 0.01, n)

    def to_closes(r):
        c = [100.0]
        for x in r[1:]:
            c.append(c[-1] * (1 + x))
        return c

    closes = {"HELD": to_closes(r_held), "CORR": to_closes(r_corr), "ANTI": to_closes(r_anti),
             "INDEP": to_closes(r_indep), "NEW": to_closes(r_indep * 0 + 0.0001)}
    panel, _ = make_panel(closes, start="2020-01-02")
    return panel


def _book(equity=100_000.0, cash=100_000.0, positions=None, open_risk=0.0):
    return BookState(equity=equity, cash=cash, positions=positions or {}, open_risk=open_risk)


@pytest.fixture
def corr_panel():
    return _panel_with_correlation_structure()


# -- equal-risk sizing math -------------------------------------------------------------------
def test_equal_risk_sizing_formula(config, corr_panel):
    pc = PortfolioConstructor(config)
    cand = make_candidate(symbol="INDEP", ref=100.0, stop=95.0)
    intents, rejections = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    expected_raw = pc.risk_per_trade * 100_000.0 / 5.0
    assert intents[0].sizing["raw_qty"] == pytest.approx(expected_raw)


def test_whole_shares_floored(config, corr_panel):
    pc = PortfolioConstructor(config)
    # per_share_risk chosen so the equal-risk qty is fractional AND stays under every cap
    # (weight cap = 0.10*equity/entry = 1000; raw = 0.005*equity/0.6 = 833.33)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.4)
    intents, rejections = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    raw = intents[0].sizing["raw_qty"]
    assert raw != int(raw)
    assert not intents[0].sizing["binding_caps"]
    assert intents[0].qty == int(raw // 1)


def test_no_stop_is_rejected(config, corr_panel):
    pc = PortfolioConstructor(config)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=None)
    intents, rejections = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    assert not intents
    assert rejections[0].reason == "no stop"


def test_zero_distance_stop_is_rejected(config, corr_panel):
    pc = PortfolioConstructor(config)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=10.0)
    _, rejections = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    assert rejections[0].reason == "no stop"


def test_qty_rounding_to_zero_is_rejected(config, corr_panel):
    cfg = config.with_overrides({"portfolio": {"risk_per_trade": 0.0000001}})
    pc = PortfolioConstructor(cfg)
    cand = make_candidate(symbol="INDEP", ref=100.0, stop=95.0)
    intents, rejections = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    assert not intents
    assert "zero shares" in rejections[0].reason


# -- continuous caps clip rather than reject ---------------------------------------------------
def test_max_position_weight_clips_qty(config, corr_panel):
    pc = PortfolioConstructor(config)
    cand = make_candidate(symbol="INDEP", ref=100.0, stop=99.0)   # raw qty = 500, weight cap = 100
    intents, rejections = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    intent = intents[0]
    assert intent.qty == 100
    assert intent.sizing["binding_caps"] == ["max_position_weight"]
    assert intent.sizing["position_weight"] == pytest.approx(0.10)


def test_cash_buffer_clips_qty(config, corr_panel):
    pc = PortfolioConstructor(config)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)   # raw qty = 500
    book = _book(equity=100_000.0, cash=6_000.0)                # available cash after buffer = 1000 -> cap 100
    intents, rejections = pc.build(book, [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    assert intents[0].qty == 100
    assert intents[0].sizing["binding_caps"] == ["cash_buffer"]


def test_max_gross_exposure_clips_qty(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"HELD": HeldPosition(qty=960, price=100.0)}          # gross = 96,000 of 100,000 equity
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)    # raw qty = 500
    book = _book(equity=100_000.0, cash=500_000.0, positions=held)
    intents, rejections = pc.build(book, [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    # remaining gross capacity = 100,000*1.00 - 96,000 = 4,000 -> qty cap = 400
    assert intents[0].qty == 400
    assert intents[0].sizing["binding_caps"] == ["max_gross_exposure"]


def test_max_sector_weight_clips_qty(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"HELD": HeldPosition(qty=290, price=100.0, sector="XLK")}   # 29,000 of 30% cap (30,000)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)            # raw qty = 500
    book = _book(equity=100_000.0, cash=500_000.0, positions=held)
    intents, rejections = pc.build(book, [cand], corr_panel, corr_panel.dates[-1],
                                   sector_map={"INDEP": "XLK"})
    assert not rejections
    assert intents[0].qty == 100   # remaining sector capacity 1,000 / entry 10
    assert intents[0].sizing["binding_caps"] == ["max_sector_weight"]
    assert intents[0].sector == "XLK"


def test_max_open_risk_clips_qty(config, corr_panel):
    cfg = config.with_overrides({"portfolio": {"risk_per_trade": 0.01}})   # raw qty = 1000
    pc = PortfolioConstructor(cfg)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)              # per-share risk = 1
    book = _book(equity=100_000.0, cash=500_000.0, open_risk=0.035)        # 3.5% of 4% cap used
    intents, rejections = pc.build(book, [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    assert intents[0].qty == 500     # remaining risk budget = 0.005*100,000 / 1
    assert intents[0].sizing["binding_caps"] == ["max_open_risk"]


# -- discrete constraints reject outright ------------------------------------------------------
def test_max_positions_reached_rejects(config, corr_panel):
    cfg = config.with_overrides({"portfolio": {"max_positions": 1}})
    pc = PortfolioConstructor(cfg)
    held = {"HELD": HeldPosition(qty=10, price=100.0)}
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)
    intents, rejections = pc.build(_book(positions=held), [cand], corr_panel, corr_panel.dates[-1])
    assert not intents
    assert "max_positions" in rejections[0].reason


def test_symbol_already_held_rejects(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"INDEP": HeldPosition(qty=10, price=100.0)}
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)
    intents, rejections = pc.build(_book(positions=held), [cand], corr_panel, corr_panel.dates[-1])
    assert not intents
    assert rejections[0].reason == "symbol already held in book"


def test_duplicate_symbol_in_batch_second_one_rejected(config, corr_panel):
    pc = PortfolioConstructor(config)
    c1 = make_candidate(symbol="INDEP", ref=10.0, stop=9.0, strategy_id="s1")
    c2 = make_candidate(symbol="INDEP", ref=10.0, stop=9.0, strategy_id="s2")
    intents, rejections = pc.build(_book(), [c1, c2], corr_panel, corr_panel.dates[-1])
    assert len(intents) == 1 and intents[0].candidate_id == c1.candidate_id
    assert len(rejections) == 1 and rejections[0].candidate_id == c2.candidate_id
    assert "duplicate" in rejections[0].reason


def test_non_positive_equity_rejects_all(config, corr_panel):
    pc = PortfolioConstructor(config)
    cands = [make_candidate(symbol="INDEP", ref=10.0, stop=9.0)]
    intents, rejections = pc.build(_book(equity=0.0), cands, corr_panel, corr_panel.dates[-1])
    assert not intents
    assert len(rejections) == 1


# -- correlation -------------------------------------------------------------------------------
def test_highly_correlated_new_symbol_rejected_against_held(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"HELD": HeldPosition(qty=10, price=100.0)}
    cand = make_candidate(symbol="CORR", ref=10.0, stop=9.0)
    intents, rejections = pc.build(_book(positions=held), [cand], corr_panel, corr_panel.dates[-1])
    assert not intents
    assert "correlation" in rejections[0].reason
    assert rejections[0].details["other"] == "HELD"


def test_anticorrelated_symbol_not_blocked(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"HELD": HeldPosition(qty=10, price=100.0)}
    cand = make_candidate(symbol="ANTI", ref=10.0, stop=9.0)
    intents, rejections = pc.build(_book(positions=held), [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    assert intents and intents[0].symbol == "ANTI"


def test_independent_symbol_not_blocked(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"HELD": HeldPosition(qty=10, price=100.0)}
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)
    intents, rejections = pc.build(_book(positions=held), [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections


def test_symbol_absent_from_panel_skips_correlation_check(config, corr_panel):
    pc = PortfolioConstructor(config)
    held = {"HELD": HeldPosition(qty=10, price=100.0)}
    cand = make_candidate(symbol="GHOST", ref=10.0, stop=9.0)
    intents, rejections = pc.build(_book(positions=held), [cand], corr_panel, corr_panel.dates[-1])
    assert not rejections
    assert intents


def test_correlation_checked_against_already_selected_symbols_in_batch(config, corr_panel):
    pc = PortfolioConstructor(config)
    first = make_candidate(symbol="HELD", ref=100.0, stop=95.0, strategy_id="s1")
    second = make_candidate(symbol="CORR", ref=10.0, stop=9.0, strategy_id="s2")
    intents, rejections = pc.build(_book(), [first, second], corr_panel, corr_panel.dates[-1])
    assert len(intents) == 1 and intents[0].symbol == "HELD"
    assert len(rejections) == 1 and rejections[0].symbol == "CORR"


# -- explainability & config validation ---------------------------------------------------------
def test_sizing_dict_is_explainable(config, corr_panel):
    pc = PortfolioConstructor(config)
    cand = make_candidate(symbol="INDEP", ref=10.0, stop=9.0)
    intents, _ = pc.build(_book(), [cand], corr_panel, corr_panel.dates[-1])
    sizing = intents[0].sizing
    for key in ("risk_per_trade", "equity", "entry_ref_price", "stop_price", "per_share_risk", "raw_qty",
               "qty", "notional", "position_weight", "gross_exposure_after", "open_risk_after",
               "cash_after", "binding_caps"):
        assert key in sizing


def test_constructor_rejects_nonpositive_config_values(config):
    with pytest.raises(ValueError):
        PortfolioConstructor(config.with_overrides({"portfolio": {"risk_per_trade": 0}}))
    with pytest.raises(ValueError):
        PortfolioConstructor(config.with_overrides({"portfolio": {"cash_buffer": 1.0}}))
    with pytest.raises(ValueError):
        PortfolioConstructor(config.with_overrides({"portfolio": {"max_pairwise_correlation": 1.5}}))
