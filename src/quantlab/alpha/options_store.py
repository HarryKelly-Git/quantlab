"""Historical options + earnings research store from DoltHub ``post-no-preference`` (free). PAPER research.

What the source is (profiled 2026-10-06, docs/ALPHA-DISCOVERY-PLAN.md):
  * ``options.option_chain``: end-of-day snapshots per contract: bid, ask, IV (``vol``), delta, gamma,
    theta, vega, rho. Three expiries per underlying (~2 weeks, ~1 month, ~2 months) and ~10-25
    strikes around the money. NO volume, NO open interest, NO underlying price (joined from the Alpaca
    store). Snapshot cadence: weekly (Saturday = Friday close) in 2019, Mon/Wed/Fri 2020-2024, daily
    from 2025. ~2,300 underlyings.
  * ``options.volatility_history``: per underlying and snapshot: ``iv_current`` / ``hv_current`` plus
    week/month-ago and 52-week high/low values. The vendor's definitions are undocumented; QuantLab
    recomputes what it needs from the chain and from prices, and uses this table only as a check.
  * ``earnings.earnings_calendar``: (symbol, announcement date, before/after market). History shows the
    2020-2024 rows were BACKFILLED (first seen after the event), so for 2020-2024 they are realised
    announcement dates (PIT_ASSUMED); from late 2024 rows appear ~38-41 days ahead (true PIT).
  * ``earnings.eps_estimate`` / ``sales_estimate`` (dated consensus snapshots) and ``eps_history``.

Provenance is an anonymous DoltHub publisher: every study using it runs the data-quality checks in
``alpha.quality`` first and labels results with the source. Nothing on/after HOLDOUT_START is exported.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.alpha.store import HOLDOUT_START, RESEARCH_END, store_dir

DOLT_HOST, DOLT_PORT = "127.0.0.1", 3307
SOURCE = "dolthub:post-no-preference"


def _conn(db: str):
    import pymysql
    return pymysql.connect(host=DOLT_HOST, port=DOLT_PORT, user="root", database=db, read_timeout=3600)


def options_dir() -> Path:
    d = store_dir() / "options"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_option_chain(end: str = RESEARCH_END) -> dict:
    """One parquet per year; resumable (a finished year is skipped)."""
    if pd.Timestamp(end) >= pd.Timestamp(HOLDOUT_START):
        raise ValueError("refusing to export holdout dates")
    c = _conn("options")
    cur = c.cursor()
    cur.execute("SELECT DISTINCT date FROM option_chain WHERE date <= %s ORDER BY date", (end,))
    dates = [r[0] for r in cur.fetchall()]
    by_year: dict[int, list] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    out = {}
    cols = ["date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask", "vol", "delta", "gamma",
            "theta", "vega", "rho"]
    for y, ds in sorted(by_year.items()):
        path = options_dir() / f"chain_{y}.parquet"
        if path.exists():
            out[y] = "exists"
            continue
        t0 = time.time()
        frames = []
        for d in ds:
            cur.execute("SELECT " + ",".join(f"`{x}`" for x in cols) + " FROM option_chain WHERE date = %s", (d,))
            frames.append(pd.DataFrame(cur.fetchall(), columns=cols))
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        df["act_symbol"] = df["act_symbol"].astype("category")
        df["cp"] = np.where(df.pop("call_put").str.startswith("C"), "C", "P")
        df["cp"] = df["cp"].astype("category")
        for k in ("strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"):
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("float32")
        df.to_parquet(path, index=False)
        out[y] = {"snapshots": len(ds), "rows": len(df), "seconds": round(time.time() - t0, 1)}
        print(y, out[y], flush=True)
    return out


def export_small_tables(end: str = RESEARCH_END) -> dict:
    res = {}
    c = _conn("options")
    vh = pd.read_sql("SELECT * FROM volatility_history WHERE date <= %s", c, params=(end,))
    vh.to_parquet(options_dir() / "volatility_history.parquet", index=False)
    res["volatility_history"] = len(vh)
    e = _conn("earnings")
    cal = pd.read_sql("SELECT * FROM earnings_calendar WHERE date <= %s", e, params=(end,))
    cal.to_parquet(options_dir() / "earnings_calendar.parquet", index=False)
    res["earnings_calendar"] = len(cal)
    for t in ("eps_estimate", "sales_estimate"):
        df = pd.read_sql(f"SELECT * FROM {t} WHERE date <= %s", e, params=(end,))
        df.to_parquet(options_dir() / f"{t}.parquet", index=False)
        res[t] = len(df)
    eh = pd.read_sql("SELECT * FROM eps_history WHERE period_end_date <= %s", e, params=(end,))
    eh.to_parquet(options_dir() / "eps_history.parquet", index=False)
    res["eps_history"] = len(eh)
    return res


def load_chain(years: list[int] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(options_dir().glob("chain_*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in years]
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    if "date" in df and df["date"].max() >= pd.Timestamp(HOLDOUT_START):
        raise RuntimeError("options store contains holdout dates: refusing")
    return df


if __name__ == "__main__":   # pragma: no cover
    print(export_small_tables(), flush=True)
    print(export_option_chain(), flush=True)
