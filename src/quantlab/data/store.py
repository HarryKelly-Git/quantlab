"""Versioned market-data store: immutable, content-addressed Parquet datasets + a registry table.

Every ingestion writes a NEW dataset (never overwrites). The "current" view of a kind is the union
of its datasets de-duplicated on the kind's key, keeping the most recently retrieved row. An
experiment records the exact list of dataset_ids it used, so it can be reproduced bit-for-bit even
after later ingestions revise history.

Synthetic datasets are flagged (``datasets.is_synthetic = 1``) and can never be mixed with real
market data in one bundle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.core.calendar import TradingCalendar
from quantlab.data import schemas
from quantlab.data.panel import DataBundle, build_panel
from quantlab.db.database import Database, to_json, utcnow_iso


class DataStoreError(RuntimeError):
    pass


def content_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h = hashlib.sha256(hashed.tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()


class MarketDataStore:
    def __init__(self, data_dir: str | Path, db: Database):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

    # -- write ----------------------------------------------------------------------------------
    def write(
        self,
        kind: str,
        df: pd.DataFrame,
        provider: str,
        params: dict[str, Any] | None = None,
        pit_notes: str = "",
        is_synthetic: bool = False,
    ) -> str:
        df = schemas.conform(kind, df)
        key = [k for k in schemas.KEYS[kind] if k in df.columns]
        df = df.sort_values(key, kind="mergesort").reset_index(drop=True)
        digest = content_hash(df)
        dataset_id = f"{kind}_{digest[:16]}"
        if self.db.fetchone("SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)):
            return dataset_id
        rel = Path(kind) / f"{dataset_id}.parquet"
        path = self.data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        date_col = {"bars": "date", "corporate_actions": "ex_date", "events": "reaction_date",
                    "fundamentals": "period_end", "news": "created_at"}.get(kind)
        start = end = None
        if date_col and len(df):
            start, end = str(pd.Timestamp(df[date_col].min()).date()), str(pd.Timestamp(df[date_col].max()).date())
        retrieved = df["retrieved_at"].max() if "retrieved_at" in df.columns and len(df) else None
        self.db.insert("datasets", {
            "dataset_id": dataset_id,
            "kind": kind,
            "provider": provider,
            "created_at": utcnow_iso(),
            "retrieved_at": pd.Timestamp(retrieved).isoformat() if retrieved is not None else utcnow_iso(),
            "path": rel.as_posix(),
            "content_hash": digest,
            "row_count": int(len(df)),
            "symbol_count": int(df["symbol"].nunique()) if "symbol" in df.columns else None,
            "start_date": start,
            "end_date": end,
            "params_json": to_json(params or {}),
            "pit_notes": pit_notes,
            "is_synthetic": int(bool(is_synthetic)),
        })
        return dataset_id

    # -- read -----------------------------------------------------------------------------------
    def read(self, dataset_id: str) -> pd.DataFrame:
        row = self.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (dataset_id,))
        if row is None:
            raise DataStoreError(f"unknown dataset {dataset_id}")
        return pd.read_parquet(self.data_dir / row["path"])

    def dataset_ids(self, kind: str, provider: str | None = None, synthetic: bool | None = None) -> list[str]:
        sql = "SELECT dataset_id FROM datasets WHERE kind=?"
        params: list[Any] = [kind]
        if provider:
            sql += " AND provider=?"
            params.append(provider)
        if synthetic is not None:
            sql += " AND is_synthetic=?"
            params.append(int(synthetic))
        sql += " ORDER BY created_at, dataset_id"
        return [r["dataset_id"] for r in self.db.fetchall(sql, params)]

    def is_synthetic(self, dataset_ids: list[str]) -> set[bool]:
        if not dataset_ids:
            return set()
        q = ",".join("?" for _ in dataset_ids)
        rows = self.db.fetchall(f"SELECT DISTINCT is_synthetic FROM datasets WHERE dataset_id IN ({q})", dataset_ids)
        return {bool(r["is_synthetic"]) for r in rows}

    def load(self, kind: str, dataset_ids: list[str] | None = None, synthetic: bool | None = None) -> pd.DataFrame:
        ids = dataset_ids if dataset_ids is not None else self.dataset_ids(kind, synthetic=synthetic)
        if not ids:
            return schemas.empty(kind)
        flags = self.is_synthetic(ids)
        if len(flags) > 1:
            raise DataStoreError(f"refusing to mix synthetic and real {kind} datasets")
        frames = [self.read(i) for i in ids]
        df = pd.concat(frames, ignore_index=True)
        key = [k for k in schemas.KEYS[kind] if k in df.columns]
        df = df.sort_values("retrieved_at", kind="mergesort").drop_duplicates(key, keep="last")
        return df.reset_index(drop=True)

    def snapshot(self, synthetic: bool | None = None) -> dict[str, list[str]]:
        """dataset_ids per kind — record this with an experiment to reproduce it later."""
        return {k: self.dataset_ids(k, synthetic=synthetic) for k in schemas.SCHEMAS}

    # -- bundle ---------------------------------------------------------------------------------
    def load_bundle(
        self,
        benchmarks: dict[str, Any],
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        snapshot: dict[str, list[str]] | None = None,
        synthetic: bool | None = None,
    ) -> DataBundle:
        snap = snapshot or self.snapshot(synthetic=synthetic)
        all_ids = [i for ids in snap.values() for i in ids]
        flags = self.is_synthetic(all_ids)
        if len(flags) > 1:
            raise DataStoreError("refusing to mix synthetic and real datasets in one bundle")
        bars = self.load("bars", snap.get("bars", []))
        if bars.empty:
            raise DataStoreError("no bars available — run ingestion first")
        if start:
            bars = bars[bars["date"] >= pd.Timestamp(start)]
        if end:
            bars = bars[bars["date"] <= pd.Timestamp(end)]
        market = benchmarks.get("market", "SPY")
        wanted = None
        if symbols is not None:
            wanted = sorted(set(symbols) | {market} | set(benchmarks.get("sectors", {}).keys()))
            bars = bars[bars["symbol"].isin(wanted)]
        mkt_dates = bars.loc[bars["symbol"] == market, "date"]
        cal_dates = mkt_dates if len(mkt_dates) else bars["date"]
        calendar = TradingCalendar.from_dates(sorted(cal_dates.unique()))
        actions = self.load("corporate_actions", snap.get("corporate_actions", []))
        panel = build_panel(bars, actions, calendar=calendar, symbols=wanted)
        panel.meta["dataset_ids"] = all_ids

        def _load(kind: str) -> pd.DataFrame:
            df = self.load(kind, snap.get(kind, []))
            if wanted is not None and "symbol" in df.columns:
                df = df[df["symbol"].isin(wanted)]
            return df.reset_index(drop=True)

        return DataBundle(
            panel=panel,
            calendar=calendar,
            reference=_load("reference"),
            actions=actions,
            events=_load("events"),
            fundamentals=_load("fundamentals"),
            news=_load("news"),
            benchmarks=benchmarks,
            dataset_ids=all_ids,
            is_synthetic=flags == {True},
        )
