"""Bot decision stack (ARCHITECTURE.md section 7).

Modules (import from the submodules directly; this package deliberately re-exports nothing so the
decision -> portfolio -> risk imports can never form a cycle):

  * :mod:`quantlab.decision.stats_provider` - validated per-strategy trade statistics (OOS first)
  * :mod:`quantlab.decision.ranking`        - opportunity score (a RANKING, not a probability)
  * :mod:`quantlab.decision.expected_value` - empirical EV shrunk toward a no-edge prior
  * :mod:`quantlab.decision.no_trade`       - "why should we NOT trade this?"
  * :mod:`quantlab.decision.final`          - final decision, persistence and explanation
"""
