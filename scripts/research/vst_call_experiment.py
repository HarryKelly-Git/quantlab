"""ONE manual PAPER experiment requested by Harry (2026-10-05): a 1-week bullish VST call on the OPT book.

Paper only (Alpaca paper broker). The options paper gate is overridden for THIS script only; the bot's
config stays ``options.paper_trading: false``. All liquidity, premium and defined-risk checks still apply:
if no call passes them, nothing is placed. Labelled MANUAL_EXPERIMENT (not a model choice).

    .venv\\Scripts\\python scripts\\research\\vst_call_experiment.py
"""
from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd

from quantlab.config import load_config
from quantlab.context import AppContext
from quantlab.options.data import OptionsDataClient
from quantlab.options.settings import OptionsSettings

SYMBOL, HORIZON, QTY = "VST", 5, 1


def main() -> None:
    ctx = AppContext.create(load_config(), init_logging=False)
    import quantlab.options.service as m
    cmp_holder: dict = {}
    orig = m.compare

    def capture(*a, **k):               # keep the SAME comparison (and quotes) the decision is made on
        c = orig(*a, **k)
        cmp_holder["cmp"] = c
        return c
    m.compare = capture
    try:
        res = m.evaluate(ctx, SYMBOL, HORIZON, "LONG")
    finally:
        m.compare = orig
    out = {"evaluation": {k: res.get(k) for k in ("ok", "evaluation_id", "spot", "expiration", "choice", "reason",
                                                  "contracts_checked", "contracts_passed", "liquidity_rejections")}}
    if not res.get("ok"):
        print(json.dumps(out, indent=1, default=str))
        return
    s = OptionsSettings.from_config(ctx.config, {"paper_trading": True})
    calls = [r for r in res["expressions"] if r["kind"] != "STOCK" and "CALL" in str(r["expression"]).upper()]
    out["calls_considered"] = calls
    eligible = [r for r in calls if r["eligible"]]
    if not eligible:
        out["paper"] = {"ok": False, "refused": "no call passed the liquidity/defined-risk checks: nothing placed"}
        print(json.dumps(out, indent=1, default=str))
        return
    best = max(eligible, key=lambda r: r["E_pnl_per_risk"] if r["E_pnl_per_risk"] is not None else -1e9)
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert("America/New_York").date()
    cmp = cmp_holder["cmp"]
    match = next((r for r in cmp.results if r.expression_id == best["expression"] and r.structure is not None), None)
    if match is None:
        out["paper"] = {"ok": False, "refused": f"{best['expression']} no longer available on re-quote"}
    else:
        from quantlab.execution.alpaca_paper import AlpacaPaperBroker
        from quantlab.options.execution import OptionsPaperExecutor
        ex = OptionsPaperExecutor(ctx.config, ctx.db, AlpacaPaperBroker(ctx.config), settings=s)
        out["paper"] = ex.open_structure(match.structure, QTY, session_date=cmp.session_date,
                                         evaluation_id=res["evaluation_id"],
                                         horizon_date=cmp.horizon_date or today + timedelta(days=7))
        out["chosen_call"] = best
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

