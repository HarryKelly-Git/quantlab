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
    """One parquet per MONTH; resumable (a finished month is skipped). Numbers are cast to DOUBLE in SQL
    and converted to float32 per snapshot: the driver's Decimal objects would otherwise need tens of GB."""
    if pd.Timestamp(end) >= pd.Timestamp(HOLDOUT_START):
        raise ValueError("refusing to export holdout dates")
    c = _conn("options")
    cur = c.cursor()
    # snapshot dates from the (already exported) per-underlying volatility table: a DISTINCT over the
    # 100M-row chain is a full scan in Dolt, while per-date queries use the primary-key prefix
    vh = pd.read_parquet(options_dir() / "volatility_history.parquet", columns=["date"])
    dates = sorted(pd.to_datetime(vh["date"]).dt.date.unique())
    dates = [d for d in dates if pd.Timestamp(d) <= pd.Timestamp(end)]
    by_month: dict[str, list] = {}
    for d in dates:
        by_month.setdefault(f"{d.year}-{d.month:02d}", []).append(d)
    num = ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]
    sel = ("SELECT `date`, `act_symbol`, `expiration`, LEFT(`call_put`, 1), "
           + ", ".join(f"CAST(`{k}` AS DOUBLE)" for k in num) + " FROM option_chain WHERE `date` = %s")
    out = {}
    for ym, ds in sorted(by_month.items()):
        path = options_dir() / f"chain_{ym}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        frames = []
        for d in ds:
            cur.execute(sel, (d,))
            rows = cur.fetchall()
            if not rows:
                continue
            a = list(zip(*rows))
            f = pd.DataFrame({"date": pd.to_datetime(pd.Series(a[0])), "act_symbol": pd.Series(a[1], dtype="string"),
                              "expiration": pd.to_datetime(pd.Series(a[2])), "cp": pd.Series(a[3], dtype="string")})
            for i, k in enumerate(num):
                f[k] = pd.to_numeric(pd.Series(a[4 + i]), errors="coerce").astype("float32")
            frames.append(f)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["act_symbol"] = df["act_symbol"].astype("category")
        df["cp"] = df["cp"].astype("category")
        df.to_parquet(path, index=False)
        out[ym] = {"snapshots": len(ds), "rows": len(df), "seconds": round(time.time() - t0, 1)}
        print(ym, out[ym], flush=True)
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
    files = sorted(options_dir().glob("chain_*-*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.split("_")[1][:4]) in years]
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    if "date" in df and df["date"].max() >= pd.Timestamp(HOLDOUT_START):
        raise RuntimeError("options store contains holdout dates: refusing")
    return df


if __name__ == "__main__":   # pragma: no cover
    if not (options_dir() / "volatility_history.parquet").exists():
        print(export_small_tables(), flush=True)
    print(export_option_chain(), flush=True)
