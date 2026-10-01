"""Learning loop: mark recorded evaluations and OPT paper structures from REAL traded prices.

Option marks come from historical daily option bars (the day's last trade). If any leg of a
structure has no bar on a session, that session's mark is ``UNKNOWN`` -- never interpolated, never
carried forward. The stock expression is marked from the underlying's RAW daily close (a split
inside the horizon would make it wrong; such marks must be reviewed). Marks are append-only; an
``UNKNOWN`` row may later be followed by a ``KNOWN`` one if the vendor backfills.

A daily close is not an executable bid/ask: marks measure what traded, labelled
``source='alpaca:option_bars:1Day close (last trade)'``.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from quantlab.data.audit import expected_sessions
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.logging_setup import get_logger, log_event

log = get_logger("options.marking")

OPT_SOURCE = "alpaca:option_bars:1Day close (last trade)"
STOCK_SOURCE = "alpaca:stock bars raw close"


def _closes(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    if df is None or not len(df):
        return {}
    return {(r.symbol, pd.Timestamp(r.date).date().isoformat()): float(r.close) for r in df.itertuples()
            if r.close is not None and r.close == r.close}


def _insert(db: Database, row: dict[str, Any]) -> int:
    row = {**row, "created_at": utcnow_iso()}
    cur = db.execute("INSERT OR IGNORE INTO options_marks (" + ",".join(row) + ") VALUES (" +
                     ",".join("?" for _ in row) + ")", [row[k] for k in row])
    return cur.rowcount


def mark_evaluations(db: Database, client: Any, as_of: date) -> dict[str, int]:
    """Mark every recorded evaluation on each session after its evaluation date up to
    min(as_of, horizon date). ``as_of`` must be a session whose daily bars are final."""
    out = {"evaluations": 0, "known": 0, "unknown": 0}
    runs = db.fetchall("SELECT * FROM options_eval_runs WHERE session_date < ? AND is_synthetic=0", (str(as_of),))
    for run in runs:
        start = pd.Timestamp(run["session_date"]) + pd.Timedelta(days=1)
        end = min(pd.Timestamp(as_of), pd.Timestamp(run["horizon_date"]))
        sessions = [d.date().isoformat() for d in expected_sessions(start, end)] if end >= start else []
        if not sessions:
            continue
        exprs = db.fetchall("SELECT * FROM options_evaluations WHERE evaluation_id=?", (run["evaluation_id"],))
        legs = {e["expression_id"]: from_json(e["legs_json"], []) for e in exprs}
        syms = sorted({lg["symbol"] for ls in legs.values() for lg in ls})
        opt = _closes(client.option_bars(syms, start.date(), end.date())) if syms else {}
        stock = _closes(client.underlying_daily_bars(run["symbol"], start.date(), end.date(), adjustment="raw"))
        sign = 1 if run["direction"] == "LONG" else -1
        out["evaluations"] += 1
        for e in exprs:
            for d in sessions:
                if e["kind"] in ("STOCK", "SHORT_STOCK"):
                    close = stock.get((run["symbol"], d))
                    row = {"ref_type": "evaluation", "ref_id": run["evaluation_id"], "expression_id": e["expression_id"],
                           "session_date": d, "source": STOCK_SOURCE}
                    if close is None:
                        row.update(status="UNKNOWN", reason=f"no stock bar for {run['symbol']} on {d}")
                    else:
                        pnl = sign * (close - run["spot"])
                        row.update(status="KNOWN", value_per_unit=close, pnl_usd=pnl,
                                   pnl_per_risk=pnl / e["risk_usd"] if e["risk_usd"] else None)
                else:
                    ls = legs.get(e["expression_id"]) or []
                    missing = [lg["symbol"] for lg in ls if (lg["symbol"], d) not in opt]
                    row = {"ref_type": "evaluation", "ref_id": run["evaluation_id"], "expression_id": e["expression_id"],
                           "session_date": d, "source": OPT_SOURCE,
                           "legs_json": to_json([{"symbol": lg["symbol"], "close": opt.get((lg["symbol"], d))} for lg in ls])}
                    if not ls or missing:
                        row.update(status="UNKNOWN", reason=f"no option bar for {missing or 'legs'} on {d} (never interpolated)")
                    else:
                        value = sum((1 if lg["side"] == "buy" else -1) * opt[(lg["symbol"], d)] for lg in ls)
                        mult = float(ls[0]["multiplier"])
                        pnl = value * mult - float(e["entry_cost"])
                        row.update(status="KNOWN", value_per_unit=value, pnl_usd=pnl,
                                   pnl_per_risk=pnl / e["risk_usd"] if e["risk_usd"] else None)
                if _insert(db, row):
                    out["known" if row["status"] == "KNOWN" else "unknown"] += 1
    log_event(log, "options evaluations marked", **out)
    return out


def mark_positions(db: Database, client: Any, as_of: date) -> dict[str, int]:
    """Mark OPT structures that hold (or held) contracts: value = sum(position x close x multiplier)
    plus the structure's net premium cash (both from broker-reported fills)."""
    out = {"structures": 0, "known": 0, "unknown": 0}
    for st in db.fetchall("SELECT * FROM options_structures WHERE status IN ('OPEN','CLOSING')"):
        pos = {r["contract_symbol"]: float(r["q"]) for r in db.fetchall(
            "SELECT contract_symbol, SUM(CASE WHEN side='buy' THEN qty ELSE -qty END) AS q FROM options_fills "
            "WHERE structure_id=? GROUP BY contract_symbol", (st["structure_id"],)) if abs(float(r["q"])) > 1e-9}
        if not pos:
            continue
        cash = float(db.fetchone("SELECT COALESCE(SUM(cash_amount),0) AS v FROM options_fills WHERE structure_id=?",
                                 (st["structure_id"],))["v"])
        first = db.fetchone("SELECT MIN(session_date) AS d FROM options_fills WHERE structure_id=?", (st["structure_id"],))["d"]
        start = pd.Timestamp(first or as_of)
        sessions = [d.date().isoformat() for d in expected_sessions(start, pd.Timestamp(as_of))]
        opt = _closes(client.option_bars(sorted(pos), start.date(), as_of))
        out["structures"] += 1
        for d in sessions:
            missing = [s for s in pos if (s, d) not in opt]
            row = {"ref_type": "position", "ref_id": st["structure_id"], "expression_id": st["kind"], "session_date": d,
                   "source": OPT_SOURCE, "legs_json": to_json([{"symbol": s, "qty": q, "close": opt.get((s, d))}
                                                               for s, q in pos.items()])}
            if missing:
                row.update(status="UNKNOWN", reason=f"no option bar for {missing} on {d} (never interpolated)")
            else:
                value = sum(q * opt[(s, d)] for s, q in pos.items()) * float(st["multiplier"])
                pnl = value + cash
                row.update(status="KNOWN", value_per_unit=value, pnl_usd=pnl,
                           pnl_per_risk=pnl / st["max_loss_usd"] if st["max_loss_usd"] else None)
            if _insert(db, row):
                out["known" if row["status"] == "KNOWN" else "unknown"] += 1
    return out


def outcomes(db: Database) -> list[dict[str, Any]]:
    """Latest KNOWN mark per (evaluation, expression): stock and options measured separately."""
    return db.fetchall(
        "SELECT m.ref_id AS evaluation_id, m.expression_id, e.kind, e.chosen, MAX(m.session_date) AS last_session, "
        "m.pnl_per_risk FROM options_marks m JOIN options_evaluations e ON e.evaluation_id=m.ref_id AND "
        "e.expression_id=m.expression_id WHERE m.ref_type='evaluation' AND m.status='KNOWN' "
        "GROUP BY m.ref_id, m.expression_id ORDER BY m.ref_id")
