"""AI/LLM research layer (optional, budgeted, fail-safe).

Everything an LLM produces here is an AI_OPINION. It never becomes a probability, never overrides a
hard risk constraint, and any failure (disabled, over budget, invalid output, hallucinated
reference) collapses to ``AIDecision.UNKNOWN``.

Public entry points (imported lazily so that ``quantlab.ai.providers`` can import
``quantlab.ai.schemas`` without a circular import through this package):

* :class:`quantlab.ai.service.AIReviewService`  researcher -> adversary -> judge review of a candidate
* :func:`quantlab.ai.evidence.build_evidence_packet` / :func:`persist_packet` / :func:`bundle_context`
* :class:`quantlab.ai.ideas.IdeaGenerator`         AI-proposed hypotheses (status PROPOSED only)
* :func:`quantlab.ai.contamination.is_contaminated`
* :func:`quantlab.ai.providers.build_providers`
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AIReviewService": "quantlab.ai.service",
    "ReviewResult": "quantlab.ai.service",
    "ModelComparison": "quantlab.ai.service",
    "ai_check_result": "quantlab.ai.service",
    "build_evidence_packet": "quantlab.ai.evidence",
    "persist_packet": "quantlab.ai.evidence",
    "bundle_context": "quantlab.ai.evidence",
    "IdeaGenerator": "quantlab.ai.ideas",
    "is_contaminated": "quantlab.ai.contamination",
    "build_providers": "quantlab.ai.providers",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'quantlab.ai' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
