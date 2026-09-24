"""Shadow book: every serious candidate (traded or not) and what it would have returned."""
from quantlab.shadow.book import ShadowBook, ShadowBookError, ShadowConflictError, parse_quant_reasoning
from quantlab.shadow.outcomes import OutcomeEvaluation, ShadowOutcomeTracker, panel_is_synthetic

__all__ = [
    "OutcomeEvaluation",
    "ShadowBook",
    "ShadowBookError",
    "ShadowConflictError",
    "ShadowOutcomeTracker",
    "panel_is_synthetic",
    "parse_quant_reasoning",
]
