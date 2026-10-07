"""Missed-opportunity analysis of an exported paper-bot database (Phase 3, Part 4).
Usage: python scripts/research/alpha/run_botdb.py path/to/quantlab_export.db
Writes research/alpha/results/botdb_missed_opportunities.json and prints the headline per horizon.
Descriptive only: never change a filter on this output alone (see alpha/botdb.py)."""
import json
import sys

from quantlab.alpha import botdb, registry

res = botdb.run(sys.argv[1], registry.DIR / "results")
print(json.dumps(res["quality"], indent=1, default=str))
for h, v in res["analysis"]["by_horizon"].items():
    t, r = v["traded"], v["rejected"]
    print(f"horizon {h}: traded n={t.get('n')} excess {t.get('mean_excess')} | rejected n={r.get('n')} excess {r.get('mean_excess')} "
          f"CI {r.get('excess_ci95_by_date')} | {v['too_conservative_verdict']}")
registry.append_run(hypothesis_id="H13", family="H13_missed_opportunities", spec={"source": "bot shadow tables", "mode": "descriptive"},
                    split="FORWARD_PAPER", metrics={"quality": res["quality"]}, data={"source": str(sys.argv[1])})
