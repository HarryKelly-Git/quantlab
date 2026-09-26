"""Forward outcomes of discovered setups (traded or not), at 1/3/5/10/20 sessions.

Research only: these numbers are NEVER fed back into discovery thresholds, scores or any gate. They
exist so a discovery family can later be turned into a strategy hypothesis and validated properly.

Point-in-time: an outcome for horizon h of a setup discovered on D is written only once the view
reaches D+h (``update`` truncates first). Returns use the tri-scaled close (split/dividend-aware).
A missing end bar is never filled in: while the symbol still trades after it (a data gap that a
re-ingest may repair) the horizon is retried later; only after 5 further sessions without any bar
is it recorded, once, as DELISTED_OR_MISSING with NULL numbers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.data.panel import DataBundle
from quantlab.db.database import Database, utcnow_iso


def _f(x: Any) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


class DiscoveryOutcomeTracker:
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.horizons = tuple(int(h) for h in (config.get("discovery.outcome_horizons", [1, 3, 5, 10, 20]) or []))
        self.news_max_age_days = float(config.get("discovery.news_max_age_days", 3))
        self.missing_sessions = int(config.get("execution.delisting_missing_sessions", 5))

    def update(self, bundle: DataBundle, as_of, synthetic: bool) -> int:
        d = to_session(as_of)
        view = bundle if (bundle.as_of is not None and bundle.as_of == d) else bundle.truncate(d)
        p = view.panel
        dates = p.dates
        pos = {dt: i for i, dt in enumerate(dates)}
        last = len(dates) - 1
        rows = self.db.fetchall(
            "SELECT discovery_id, symbol, as_of_date FROM discovery_candidates WHERE is_synthetic=? AND as_of_date<? "
            "AND (SELECT COUNT(*) FROM discovery_outcomes o WHERE o.discovery_id=discovery_candidates.discovery_id) < ?",
            (int(synthetic), str(d.date()), len(self.horizons)))
        if not rows:
            return 0
        done = {(r["discovery_id"], r["horizon_sessions"]) for r in self.db.fetchall(
            "SELECT o.discovery_id, o.horizon_sessions FROM discovery_outcomes o JOIN discovery_candidates c "
            "ON c.discovery_id=o.discovery_id WHERE c.is_synthetic=? AND c.as_of_date<?", (int(synthetic), str(d.date())))}
        ac, ah, al, dv = (p.aclose.to_numpy(float), p.ahigh.to_numpy(float), p.alow.to_numpy(float),
                          p.dollar_volume.to_numpy(float))
        cols = {s: j for j, s in enumerate(p.symbols)}
        mkt = cols.get(view.market_symbol)
        news = view.news
        news_t = pd.to_datetime(news["available_at"], utc=True) if not news.empty else None
        span = (pd.DataFrame({"symbol": news["symbol"], "t": news_t}).groupby("symbol")["t"].agg(["min", "max"])
                if not news.empty else pd.DataFrame(columns=["min", "max"]))
        now = utcnow_iso()
        out = []
        for r in rows:
            i = pos.get(pd.Timestamp(r["as_of_date"]))
            c = cols.get(r["symbol"])
            if i is None or c is None:
                continue
            for h in self.horizons:
                j = i + h
                if j > last:
                    continue                                   # not matured yet
                if (r["discovery_id"], h) in done:
                    continue
                base, end = ac[i, c], ac[j, c]
                row: dict[str, Any] = {"discovery_id": r["discovery_id"], "horizon_sessions": h,
                                       "as_of_date": r["as_of_date"], "end_date": str(dates[j].date()),
                                       "is_synthetic": int(synthetic), "created_at": now, "ret": None, "spy_ret": None,
                                       "excess_ret": None, "mfe": None, "mae": None, "price_confirmed": None,
                                       "volume_confirmed": None, "catalyst_persisted": None}
                if not np.isfinite(base):
                    continue                                   # no base price: nothing honest to measure
                if not np.isfinite(end):
                    later = ac[j + 1:, c]
                    if np.isfinite(later).any() or (last - j) < self.missing_sessions:
                        continue                               # data gap or too early to tell: retry later
                    row["status"] = "DELISTED_OR_MISSING"
                    out.append(row)
                    continue
                win_h, win_l = ah[i + 1:j + 1, c], al[i + 1:j + 1, c]
                row["ret"] = float(end / base - 1)
                row["mfe"] = _f(np.nanmax(win_h) / base - 1) if np.isfinite(win_h).any() else None
                row["mae"] = _f(np.nanmin(win_l) / base - 1) if np.isfinite(win_l).any() else None
                row["price_confirmed"] = int(end > base)
                if mkt is not None and np.isfinite(ac[i, mkt]) and np.isfinite(ac[j, mkt]):
                    row["spy_ret"] = float(ac[j, mkt] / ac[i, mkt] - 1)
                    row["excess_ret"] = row["ret"] - row["spy_ret"]
                pre = dv[max(0, i - 20):i, c]
                post = dv[i + 1:j + 1, c]
                if np.isfinite(pre).any() and np.nanmean(pre) > 0 and np.isfinite(post).any():
                    row["volume_confirmed"] = int(np.nanmean(post) / np.nanmean(pre) >= 1.0)
                t0, t1 = view.calendar.cutoff(dates[i]), view.calendar.cutoff(dates[j])
                if r["symbol"] in span.index and span.at[r["symbol"], "min"] <= t0 and \
                        span.at[r["symbol"], "max"] >= t1 - pd.Timedelta(days=self.news_max_age_days):
                    # the feed demonstrably covers (t0, t1] for this symbol; otherwise NULL (UNKNOWN)
                    m = (news["symbol"] == r["symbol"]) & (news_t > t0) & (news_t <= t1)
                    row["catalyst_persisted"] = int(bool(m.any()))
                row["status"] = "MATURED"
                out.append(row)
        if not out:
            return 0
        before = self.db.conn.total_changes
        self.db.insert_many("discovery_outcomes", out, or_ignore=True)
        return int(self.db.conn.total_changes - before)             # rows actually inserted


__all__ = ["DiscoveryOutcomeTracker"]
