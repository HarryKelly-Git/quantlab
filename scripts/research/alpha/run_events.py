"""Event + calendar families: H20 earnings-announcement premium, H21 EPS-surprise drift, H26 pre-earnings
straddle, H09 seasonality, H23 revision momentum (cross-sectional; runs through run_family with its single
OOS look). Usage: python scripts/research/alpha/run_events.py H20 H21 H26 H09 H23 [--override="reason"]
(--override: one-time documented OOS re-evaluation after a bug fix, see registry.append_run)"""
import json
import sys
import time

from quantlab.alpha import registry, research_data
from quantlab.alpha.experiments import earnings, seasonality

override = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--override=")), None)
keys = [a for a in sys.argv[1:] if not a.startswith("--override=")]
d = research_data.get()
cal = None
for key in keys:
    t0 = time.time()
    if key in ("H20", "H21", "H26") and cal is None:
        cal = earnings.clean_calendar(d)
        print(f"clean calendar: {len(cal)} events", flush=True)
    if key == "H20":
        r = earnings.run_h20(d, cal, oos_rerun_reason=override)
    elif key == "H21":
        r = earnings.run_h21(d, cal, oos_rerun_reason=override)
    elif key == "H26":
        r = earnings.run_h26(d, cal)
    elif key == "H23":
        r = earnings.run_h23(d, oos_override=override)
    elif key == "H09":
        r = seasonality.run(d)
        (registry.DIR / "results").mkdir(parents=True, exist_ok=True)
        (registry.DIR / "results" / "H09_seasonality.json").write_text(json.dumps(registry._clean(r), indent=1, default=str))
        registry.append_run(hypothesis_id="H09", family="H09_seasonality", spec={"effects": list(r["effects"])}, split="ALL", metrics=r,
                            data=d.manifest)
    print(f"== {key} {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(registry._clean(r), default=str)[:3000], flush=True)
