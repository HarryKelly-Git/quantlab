"""PAPER_EXPLORATION: learn forward from the best-looking experimental setups, on PAPER only.

Separate from STRICT validation (the unchanged decision chain). See :mod:`.engine` and
:mod:`.hypotheses`.
"""
from quantlab.exploration.engine import (MODES, ExplorationOutcomeTracker, ExplorationPolicy, experiment_results,
                                         paper_mode, plan_exploration, preopen_submit)
from quantlab.exploration.hypotheses import STAGES, advance, create_hypothesis, hypotheses

__all__ = ["MODES", "STAGES", "ExplorationOutcomeTracker", "ExplorationPolicy", "advance", "create_hypothesis",
           "experiment_results", "hypotheses", "paper_mode", "plan_exploration", "preopen_submit"]
