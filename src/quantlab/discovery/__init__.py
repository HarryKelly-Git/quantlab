"""Market discovery layer (research only). See engine.py for the funnel and families.py for the
families. Discovery never permits a trade; it reports what the unchanged validation chain decided."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantlab.discovery.engine import (
    Assessment, DiscoveryEngine, DiscoverySettings, ScanResult, persist, strategy_links,
)
from quantlab.discovery.outcomes import DiscoveryOutcomeTracker
from quantlab.discovery.status import strategy_research_status

_AUTO = object()


@dataclass
class DiscoveryRun:
    discovery_run_id: str
    scan: ScanResult
    assessment: Assessment
    outcomes_written: int


def run_discovery(ctx: Any, bundle, as_of, *, run_id: str | None = None, quarantine: dict | None = None,
                  links: Any = _AUTO, persist_results: bool = True) -> DiscoveryRun:
    """Scan -> assess against the recorded decisions of the session -> persist -> update forward
    outcomes of earlier discoveries. ``links`` defaults to the strategy candidates/decisions of the
    session's pipeline run (``run_id``) or of its latest succeeded run."""
    eng = DiscoveryEngine(ctx.config)
    scan = eng.scan(bundle, as_of, quarantine=quarantine)
    if links is _AUTO:
        links = strategy_links(ctx.db, as_of, run_id, synthetic=scan.is_synthetic)
    a = eng.assess(scan, links, strategy_research_status(ctx.db, ctx.config))
    if persist_results:
        try:
            from quantlab.discovery.source_coverage import source_coverage
            cut = scan.calendar.cutoff(scan.as_of) if scan.calendar is not None else None
            scan.source_coverage = source_coverage(ctx, universe=scan.basic_symbols, as_of=cut,
                                                   synthetic=bool(scan.is_synthetic))
        except Exception as exc:          # coverage is a report: never blocks discovery
            scan.source_coverage = {"error": repr(exc)[:500]}
    disc_id = persist(ctx.db, eng, scan, a, run_id) if persist_results else ""
    n = DiscoveryOutcomeTracker(ctx.db, ctx.config).update(bundle, as_of, scan.is_synthetic) if persist_results else 0
    return DiscoveryRun(disc_id, scan, a, n)


__all__ = ["Assessment", "DiscoveryEngine", "DiscoveryOutcomeTracker", "DiscoveryRun", "DiscoverySettings",
           "ScanResult", "persist", "run_discovery", "strategy_links", "strategy_research_status"]
