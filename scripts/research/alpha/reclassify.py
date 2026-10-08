"""Re-derive every family's A-G class from its stored Evidence with the current decision tree."""
import json

from quantlab.alpha import registry

for f in sorted((registry.DIR / "results").glob("*.json")):
    r = json.loads(f.read_text())
    if "evidence" not in r:
        continue
    cls, why = registry.classify(registry.Evidence(**r["evidence"]))
    if (cls, why) != (r.get("classification"), r.get("classification_reason")):
        print(f"{f.stem}: {r.get('classification')} -> {cls} ({why})")
    r["classification"], r["classification_reason"] = cls, why
    f.write_text(json.dumps(r, indent=1, default=str))
