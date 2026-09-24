"""Market regime context — for reporting and research only.

Deliberately simple (trend sign x volatility level). The system first has to show that regimes
matter (via outcome breakdowns in the shadow book / backtests) before any strategy is conditioned
on them. All inputs are trailing market-level features, so the regime at D is point-in-time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.features.base import FeatureSet

METRICS = ("market_trend_200", "market_mom_60", "market_vol_20", "market_drawdown", "breadth_50", "breadth_200",
           "sector_dispersion_63")


class RegimeEngine:
    def __init__(self, config: Config):
        self.high_vol = float(config.get("regime.high_vol_annualized", 0.25))

    def compute(self, fs: FeatureSet) -> pd.DataFrame:
        df = pd.DataFrame({m: fs.market(m) for m in METRICS}, index=fs.panel.dates)
        trend, vol = df["market_trend_200"], df["market_vol_20"]
        known = trend.notna() & vol.notna()
        label = np.where(trend > 0, "bull", "bear").astype(object)
        label = np.char.add(label.astype(str), np.where(vol > self.high_vol, "_volatile", "_calm"))
        df["label"] = pd.Series(label, index=df.index).where(known, "unknown")
        return df

    def snapshot(self, fs: FeatureSet, as_of) -> dict[str, Any]:
        row = self.compute(fs).loc[pd.Timestamp(as_of)]
        out = {k: (float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else None)
               for k, v in row.items() if k != "label"}
        out["label"] = str(row["label"])
        out["info_kind"] = "MODEL_OUTPUT"
        return out

    def persist(self, db, fs: FeatureSet, as_of, run_id: str | None = None) -> dict[str, Any]:
        from quantlab.db.database import to_json, utcnow_iso
        snap = self.snapshot(fs, as_of)
        db.insert("regime_snapshots", {"as_of_date": str(pd.Timestamp(as_of).date()), "label": snap["label"],
                                       "metrics_json": to_json(snap), "run_id": run_id, "created_at": utcnow_iso()},
                  or_ignore=True)
        return snap
