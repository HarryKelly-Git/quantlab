"""SYNTHETIC market generator — for tests, demos and method validation ONLY.

Everything produced here is labeled provider="synthetic" and stored with is_synthetic=1. It is
never market evidence. Its purposes:
  * exercise every code path offline (splits incl. reverse splits, dividends, delistings, late
    listings, halts, earnings events with pre-market / after-close timing, restated fundamentals,
    revised news, non-common securities that the universe must exclude);
  * METHOD VALIDATION: with all ``*_edge`` params = 0 the world is a pure null (no exploitable
    signal) and the research stack must NOT find an edge; with a planted edge it SHOULD find it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from functools import cached_property
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.data import schemas
from quantlab.data.providers.base import (
    CorporateActionProvider,
    EventProvider,
    FundamentalsProvider,
    NewsProvider,
    PriceProvider,
    ReferenceProvider,
)

ET = ZoneInfo("America/New_York")
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Information Technology", "XLP": "Consumer Staples", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary",
}
# Non-common securities that must be excluded by the universe engine.
_NON_COMMON = [
    ("SYNPA", "Synthetic Holdings 6.5% Preferred Stock Series A", "PREFERRED"),
    ("SYNWS", "Synthetic Acquisition Corp Warrant", "WARRANT"),
    ("SYNU", "Synthetic Acquisition Corp Unit", "UNIT"),
    ("SYNRT", "Synthetic Biotech Rights", "RIGHT"),
    ("SYNTT", "NASDAQ TEST STOCK", "COMMON"),   # test issue
    ("SYNETF", "Synthetic Momentum ETF", "ETF"),
]


def _et(d: pd.Timestamp, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d.date(), t)).tz_localize(ET).tz_convert("UTC")


@dataclass(frozen=True)
class SyntheticSpec:
    n_stocks: int = 120
    start: str = "2016-01-04"
    end: str = "2024-12-31"
    seed: int = 42
    # Planted edges (0 => null world). Units: daily drift added to returns.
    momentum_edge: float = 0.0     # per unit of clipped 60-session idiosyncratic z-score
    pead_edge: float = 0.0         # for 30 sessions after an earnings surprise, times sign(surprise)
    frac_dividend_payers: float = 0.4
    frac_delisted: float = 0.06
    frac_late_listing: float = 0.10
    retrieved_at: str = "2026-01-01T00:00:00+00:00"


class SyntheticMarket(
    PriceProvider, CorporateActionProvider, ReferenceProvider, FundamentalsProvider, EventProvider, NewsProvider
):
    name = "synthetic"
    is_synthetic = True

    def __init__(self, spec: SyntheticSpec | None = None):
        self.spec = spec or SyntheticSpec()
        self._retrieved = pd.Timestamp(self.spec.retrieved_at)

    # ------------------------------------------------------------------------------------------
    @cached_property
    def world(self) -> dict[str, pd.DataFrame]:
        return self._generate()

    def _generate(self) -> dict[str, pd.DataFrame]:
        s = self.spec
        rng = np.random.default_rng(s.seed)
        dates = pd.bdate_range(s.start, s.end)
        T, N = len(dates), s.n_stocks
        syms = [f"SYN{i:03d}" for i in range(N)]

        # --- market regimes & factors -------------------------------------------------------
        regime = np.zeros(T, dtype=int)
        for t in range(1, T):
            p_switch = 0.01 if regime[t - 1] == 0 else 0.03
            regime[t] = 1 - regime[t - 1] if rng.random() < p_switch else regime[t - 1]
        m_vol = np.where(regime == 0, 0.008, 0.018)
        m_mu = np.where(regime == 0, 0.0005, -0.0006)
        m = m_mu + m_vol * rng.standard_t(5, T) / np.sqrt(5 / 3)
        sector_of = rng.integers(0, len(SECTOR_ETFS), N)
        sec_f = 0.006 * rng.standard_normal((T, len(SECTOR_ETFS)))

        beta = rng.uniform(0.6, 1.6, N)
        ivol = rng.uniform(0.010, 0.035, N)
        price0 = np.clip(np.exp(rng.normal(np.log(40), 1.0, N)), 2.0, 500.0)
        base_dv = np.exp(rng.normal(np.log(2e7), 1.5, N))

        start_idx = np.zeros(N, dtype=int)
        end_idx = np.full(N, T - 1)
        late = rng.random(N) < s.frac_late_listing
        start_idx[late] = rng.integers(60, T // 2, late.sum())
        dl = rng.random(N) < s.frac_delisted
        end_idx[dl] = np.maximum(start_idx[dl] + 300, rng.integers(T // 3, T - 20, dl.sum()))
        end_idx = np.minimum(end_idx, T - 1)

        # --- earnings schedule ----------------------------------------------------------------
        ev_offset = rng.integers(0, 63, N)
        is_event = np.zeros((T, N), dtype=bool)
        for i in range(N):
            is_event[ev_offset[i] + 10 :: 63, i] = True
        surprise = np.where(is_event, rng.standard_normal((T, N)), 0.0)
        pre_market = rng.random((T, N)) < 0.5

        div_payer = rng.random(N) < s.frac_dividend_payers
        div_offset = rng.integers(0, 63, N)

        # --- simulate path ----------------------------------------------------------------------
        raw = np.full((T, N), np.nan)
        opn = np.full((T, N), np.nan)
        ret = np.full((T, N), np.nan)
        split_ratio = np.ones((T, N))
        dividend = np.zeros((T, N))
        idio_hist = np.zeros((T, N))
        post_event = np.zeros(N)            # sessions remaining in the post-event window
        post_sign = np.zeros(N)
        last_split = np.full(N, -10_000)
        pending_split = np.zeros(N)         # ratio to apply next session (0 => none)
        prev = price0.copy()

        for t in range(T):
            alive = (t >= start_idx) & (t <= end_idx)
            eps = ivol * rng.standard_t(4, N) / np.sqrt(2.0)
            idio_hist[t] = eps
            drift = np.zeros(N)
            if s.momentum_edge and t >= 61:
                z = idio_hist[t - 60 : t].sum(0) / (ivol * np.sqrt(60))
                drift += s.momentum_edge * np.clip(z, -3, 3)
            if s.pead_edge:
                drift += s.pead_edge * post_sign * (post_event > 0)
            jump = np.where(is_event[t], surprise[t] * ivol * 3.0, 0.0)
            r = beta * m[t] + sec_f[t, sector_of] + eps + jump + drift
            r = np.maximum(r, -0.85)

            d_t = np.where(div_payer & (((t - div_offset) % 63) == 0) & (t > 0), 0.004 * prev, 0.0)
            ratio = np.where(pending_split > 0, pending_split, 1.0)
            close = (prev * (1 + r) - d_t) / ratio

            first = alive & (t == start_idx)
            close = np.where(first, price0, close)
            gap = 0.4 * r + 0.003 * rng.standard_normal(N)
            o = np.where(first, price0, (prev / ratio) * (1 + gap))

            raw[t] = np.where(alive, close, np.nan)
            opn[t] = np.where(alive, o, np.nan)
            ret[t] = np.where(alive & ~first, r, np.nan)
            split_ratio[t] = np.where(alive, ratio, 1.0)
            dividend[t] = np.where(alive, d_t, 0.0)
            prev = np.where(alive, close, prev)

            # post-event drift window
            new_ev = is_event[t] & alive
            post_event = np.where(new_ev, 30, np.maximum(post_event - 1, 0))
            post_sign = np.where(new_ev, np.sign(surprise[t]), post_sign)
            # schedule splits for next session (forward 2:1 above $350, reverse 1:10 below $1)
            can = alive & (t - last_split > 60)
            pending_split = np.where(can & (close > 350), 2.0, np.where(can & (close < 1.0), 0.1, 0.0))
            last_split = np.where(pending_split > 0, t + 1, last_split)

        hi = np.maximum(opn, raw) * (1 + np.abs(rng.standard_normal((T, N))) * 0.4 * ivol)
        lo = np.minimum(opn, raw) * (1 - np.abs(rng.standard_normal((T, N))) * 0.4 * ivol)
        rel_move = np.nan_to_num(np.abs(ret) / ivol, nan=0.0)
        vol_mult = np.exp(0.3 * rng.standard_normal((T, N))) * (1 + np.minimum(rel_move, 6) * 0.5)
        vol_mult = np.where(is_event, vol_mult * 3, vol_mult)
        volume = np.round(base_dv / np.where(raw > 0, raw, np.nan) * vol_mult)

        # --- benchmark ETFs -------------------------------------------------------------------
        etf_rows = []
        spy = 200 * np.cumprod(1 + m)
        etf_rows.append(("SPY", spy))
        for k, etf in enumerate(SECTOR_ETFS):
            members = sector_of == k
            mean_idio = np.nanmean(np.where(members, idio_hist, np.nan), axis=1) if members.any() else np.zeros(T)
            r_etf = m + sec_f[:, k] + np.nan_to_num(mean_idio) * 0.3
            etf_rows.append((etf, 50 * np.cumprod(1 + r_etf)))

        frames = []
        for i, sym in enumerate(syms):
            ok = ~np.isnan(raw[:, i])
            frames.append(pd.DataFrame({
                "symbol": sym, "date": dates[ok], "open": opn[ok, i], "high": hi[ok, i], "low": lo[ok, i],
                "close": raw[ok, i], "volume": volume[ok, i],
            }))
        for sym, px in etf_rows:
            o = np.r_[px[0], px[:-1]] * (1 + 0.001 * rng.standard_normal(T))
            frames.append(pd.DataFrame({
                "symbol": sym, "date": dates, "open": o, "high": np.maximum(o, px) * 1.003,
                "low": np.minimum(o, px) * 0.997, "close": px, "volume": np.full(T, 5e7),
            }))
        for j, (sym, _, _) in enumerate(_NON_COMMON):
            px = 20 * np.cumprod(1 + 0.01 * rng.standard_normal(T))
            frames.append(pd.DataFrame({
                "symbol": sym, "date": dates, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close": px, "volume": np.full(T, 2e6),
            }))
        bars = pd.concat(frames, ignore_index=True)
        bars["vwap"] = (bars["high"] + bars["low"] + bars["close"]) / 3
        bars["trade_count"] = np.nan
        bars["provider"] = self.name
        bars["retrieved_at"] = self._retrieved

        # --- corporate actions ------------------------------------------------------------------
        acts = []
        for i, sym in enumerate(syms):
            for t in np.flatnonzero(split_ratio[:, i] != 1.0):
                acts.append((sym, dates[t], "split", split_ratio[t, i], np.nan, t))
            for t in np.flatnonzero(dividend[:, i] > 0):
                acts.append((sym, dates[t], "cash_dividend", np.nan, dividend[t, i], t))
        actions = pd.DataFrame(acts, columns=["symbol", "ex_date", "action_type", "ratio", "amount", "t"])
        decl_idx = np.maximum(actions["t"].to_numpy() - 10, 0) if len(actions) else np.array([], dtype=int)
        actions["declared_date"] = dates[decl_idx] if len(actions) else pd.Series(dtype="datetime64[ns]")
        actions["available_at"] = [_et(d, time(16, 30)) for d in actions["declared_date"]]
        actions["pit_status"] = PitStatus.PIT.value
        actions["source_id"] = [f"syn-ca-{k}" for k in range(len(actions))]
        actions["provider"] = self.name
        actions["retrieved_at"] = self._retrieved
        actions = actions.drop(columns="t")

        # --- reference ------------------------------------------------------------------------
        ref_rows = []
        for i, sym in enumerate(syms):
            etf = SECTOR_ETFS[sector_of[i]]
            ref_rows.append((sym, f"Synthetic Company {i:03d} Inc. Common Stock", "NASDAQ" if i % 2 else "NYSE",
                             "COMMON", False, False, f"{9000000 + i:010d}", None, SECTOR_NAMES[etf], etf,
                             "inactive" if dl[i] else "active"))
        ref_rows.append(("SPY", "SPDR S&P 500 ETF Trust", "NYSE_ARCA", "ETF", True, False, None, None, None, None, "active"))
        for etf in SECTOR_ETFS:
            ref_rows.append((etf, f"Sector SPDR {SECTOR_NAMES[etf]}", "NYSE_ARCA", "ETF", True, False, None, None,
                             SECTOR_NAMES[etf], etf, "active"))
        for sym, name, typ in _NON_COMMON:
            ref_rows.append((sym, name, "NASDAQ", typ, typ == "ETF", sym == "SYNTT", None, None, None, None, "active"))
        reference = pd.DataFrame(ref_rows, columns=["symbol", "name", "exchange", "security_type", "is_etf",
                                                    "is_test_issue", "cik", "sic", "sector", "industry", "status"])
        reference["source"] = self.name
        reference["retrieved_at"] = self._retrieved
        reference["pit_status"] = PitStatus.ASSUMED_STATIC.value

        # --- events (earnings releases) ---------------------------------------------------------
        ev_rows = []
        cal = pd.DatetimeIndex(dates)
        for i, sym in enumerate(syms):
            for t in np.flatnonzero(is_event[:, i] & ~np.isnan(raw[:, i])):
                react = cal[t]
                if pre_market[t, i] or t == 0:
                    et = _et(react, time(7, 0))
                else:
                    et = _et(cal[t - 1], time(16, 5))
                ev_rows.append((sym, "earnings_release", et, et, react, f"syn-8k-{sym}-{t}",
                                json.dumps({"surprise_truth": round(float(surprise[t, i]), 4), "synthetic": True})))
        events = pd.DataFrame(ev_rows, columns=["symbol", "event_type", "event_time", "available_at",
                                                "reaction_date", "source_id", "payload_json"])
        events["pit_status"] = PitStatus.PIT.value
        events["provider"] = self.name
        events["retrieved_at"] = self._retrieved

        # --- fundamentals (10-Q facts filed ~20 sessions after the release; some restated) -------
        f_rows = []
        concepts = ["Revenues", "NetIncomeLoss", "EarningsPerShareDiluted", "Assets", "StockholdersEquity", "LongTermDebt"]
        for i, sym in enumerate(syms):
            rev = np.exp(rng.normal(np.log(5e8), 1.0))
            margin = rng.uniform(-0.1, 0.25)
            shares = rng.uniform(5e7, 1e9)
            for t in np.flatnonzero(is_event[:, i] & ~np.isnan(raw[:, i])):
                rev *= 1 + rng.normal(0.02, 0.05)
                margin = float(np.clip(margin + rng.normal(0, 0.02), -0.3, 0.4))
                ni = rev * margin
                assets = rev * 3
                equity = assets * rng.uniform(0.3, 0.6)
                debt = assets * rng.uniform(0.05, 0.4)
                pend = cal[max(t - 15, 0)]
                pstart = pend - pd.Timedelta(days=91)
                ft = min(t + 20, T - 1)
                filed = cal[ft]
                acc = f"syn-10q-{sym}-{t}"
                vals = [rev, ni, ni / shares, assets, equity, debt]
                for c, v in zip(concepts, vals):
                    unit = "USD/shares" if c == "EarningsPerShareDiluted" else "USD"
                    f_rows.append((sym, f"{9000000 + i:010d}", c, unit, pstart, pend, pend.year, "Q", "10-Q",
                                   float(v), acc, filed, _et(filed, time(17, 0))))
                    # restatement: re-reported 63 sessions later with a different value
                    if c == "Revenues" and rng.random() < 0.1 and ft + 63 < T:
                        f2 = cal[ft + 63]
                        f_rows.append((sym, f"{9000000 + i:010d}", c, unit, pstart, pend, pend.year, "Q", "10-Q",
                                       float(v) * 0.9, f"syn-10q-restate-{sym}-{t}", f2, _et(f2, time(17, 0))))
        fundamentals = pd.DataFrame(f_rows, columns=["symbol", "cik", "concept", "unit", "period_start", "period_end",
                                                     "fiscal_year", "fiscal_period", "form", "value", "accession",
                                                     "filed_date", "available_at"])
        fundamentals["pit_status"] = PitStatus.PIT.value
        fundamentals["provider"] = self.name
        fundamentals["retrieved_at"] = self._retrieved

        # --- news ---------------------------------------------------------------------------------
        n_rows = []
        k = 0
        for i, sym in enumerate(syms):
            live = np.flatnonzero(~np.isnan(raw[:, i]))
            picks = live[rng.random(len(live)) < 0.08]
            ev_t = np.flatnonzero(is_event[:, i] & ~np.isnan(raw[:, i]))
            for t in np.concatenate([picks, ev_t, ev_t]):
                created = _et(cal[t], time(int(rng.integers(6, 20)), int(rng.integers(0, 60))))
                revised = rng.random() < 0.15
                updated = created + pd.Timedelta(hours=int(rng.integers(1, 48))) if revised else created
                n_rows.append((f"syn-news-{k}", sym, f"Synthetic headline {k} about {sym}", "synthetic summary",
                               "synthetic-wire", "", created, updated, created,
                               PitStatus.PIT.value if not revised else PitStatus.PIT_CONSERVATIVE.value))
                k += 1
        news = pd.DataFrame(n_rows, columns=["news_id", "symbol", "headline", "summary", "source", "url",
                                             "created_at", "updated_at", "available_at", "pit_status"])
        news["provider"] = self.name
        news["retrieved_at"] = self._retrieved

        # --- catalyst-phase additions (a SEPARATE random stream appended last, so every series above
        #     is unchanged): margins, FY cash flows, SIC header observations and material 8-Ks --------
        rng2 = np.random.default_rng(s.seed + 7919)
        sic_of_etf = {"XLB": "2800", "XLC": "4813", "XLE": "1311", "XLF": "6022", "XLI": "3560", "XLK": "3674",
                      "XLP": "2080", "XLRE": "6798", "XLU": "4911", "XLV": "2834", "XLY": "5812"}
        extra_f, extra_e = [], []
        q = fundamentals[fundamentals["concept"] == "Revenues"].drop_duplicates("accession")
        for sym, grp in q.groupby("symbol", sort=True):
            gm, om = rng2.uniform(0.2, 0.6), rng2.uniform(-0.05, 0.25)
            grp = grp.sort_values("period_end")
            for k_, r in enumerate(grp.itertuples()):
                base = (r.symbol, r.cik, None, "USD", r.period_start, r.period_end, r.fiscal_year, "Q", "10-Q", None,
                        r.accession, r.filed_date, r.available_at)
                for c, m in (("GrossProfit", gm), ("OperatingIncomeLoss", om)):
                    extra_f.append(base[:2] + (c,) + base[3:9] + (float(r.value) * m,) + base[10:])
                if k_ % 4 == 3:                          # a fiscal year every 4th quarter
                    four = grp.iloc[k_ - 3:k_ + 1]
                    fy_rev = float(four["value"].sum())
                    fy_start = pd.Timestamp(r.period_end) - pd.Timedelta(days=364)
                    for c, v in (("Revenues", fy_rev), ("OperatingCashFlow", fy_rev * (om + 0.05)),
                                 ("Capex", fy_rev * rng2.uniform(0.02, 0.08))):
                        extra_f.append((r.symbol, r.cik, c, "USD", fy_start, r.period_end, r.fiscal_year, "FY", "10-K",
                                        float(v), r.accession + "-fy", r.filed_date, r.available_at))
            # SIC as printed in the first and last filing header; a few symbols change industry
            etf = SECTOR_ETFS[sector_of[syms.index(sym)]] if sym in syms else "XLI"
            first, last = grp.iloc[0], grp.iloc[-1]
            sic_a = sic_of_etf[etf]
            sic_b = "7372" if rng2.random() < 0.05 else sic_a
            mid = grp.iloc[len(grp) // 2]
            for r, sic in ((first, sic_a), (mid, sic_b), (last, sic_b)):
                extra_e.append((sym, "sic_observation", r.available_at, r.available_at, pd.NaT, f"sic:{r.accession}",
                                json.dumps({"sic": sic, "form": "10-Q", "accession": r.accession, "synthetic": True})))
        for i, sym in enumerate(syms):
            live = np.flatnonzero(~np.isnan(raw[:, i]))
            for t in live[rng2.random(len(live)) < 0.008]:
                item = ["1.01", "5.02", "8.01", "2.01", "3.02"][int(rng2.integers(0, 5))]
                et = _et(cal[t], time(16, 20))
                extra_e.append((sym, "sec_8k", et, et, cal[t + 1] if t + 1 < T else pd.NaT, f"syn-8kx-{sym}-{t}",
                                json.dumps({"form": "8-K", "material_items": [item], "synthetic": True})))
        if extra_f:
            fundamentals = pd.concat([fundamentals, pd.DataFrame(extra_f, columns=list(fundamentals.columns[:13])).assign(
                pit_status=PitStatus.PIT.value, provider=self.name, retrieved_at=self._retrieved)], ignore_index=True)
        if extra_e:
            events = pd.concat([events, pd.DataFrame(extra_e, columns=["symbol", "event_type", "event_time", "available_at",
                                                                       "reaction_date", "source_id", "payload_json"]).assign(
                pit_status=PitStatus.PIT.value, provider=self.name, retrieved_at=self._retrieved)], ignore_index=True)

        return {
            "bars": schemas.conform("bars", bars),
            "corporate_actions": schemas.conform("corporate_actions", actions),
            "reference": schemas.conform("reference", reference),
            "events": schemas.conform("events", events),
            "fundamentals": schemas.conform("fundamentals", fundamentals),
            "news": schemas.conform("news", news),
            # ground truth for method-validation tests (never exposed through provider methods)
            "_truth_returns": pd.DataFrame(ret, index=dates, columns=syms),
        }

    # -- provider interface ---------------------------------------------------------------------
    def _slice(self, kind: str, symbols, start, end, date_col: str | None) -> pd.DataFrame:
        df = self.world[kind]
        if symbols is not None:
            df = df[df["symbol"].isin([s.upper() for s in symbols])]
        if date_col is not None:
            d = pd.to_datetime(df[date_col])
            if getattr(d.dt, "tz", None) is not None:
                d = d.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            df = df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
        return df.reset_index(drop=True)

    def get_daily_bars(self, symbols, start: date, end: date) -> pd.DataFrame:
        return self._slice("bars", symbols, start, end, "date")

    def get_corporate_actions(self, symbols, start: date, end: date) -> pd.DataFrame:
        return self._slice("corporate_actions", symbols, start, end, "ex_date")

    def get_securities(self) -> pd.DataFrame:
        return self.world["reference"].copy()

    def get_fundamentals(self, symbols, concepts=None) -> pd.DataFrame:
        df = self._slice("fundamentals", symbols, None, None, None)
        return df[df["concept"].isin(concepts)] if concepts else df

    def get_earnings_events(self, symbols, start: date, end: date) -> pd.DataFrame:
        return self._slice("events", symbols, start, end, "reaction_date")

    def get_news(self, symbols, start: date, end: date) -> pd.DataFrame:
        return self._slice("news", symbols, start, end, "created_at")

    @property
    def all_symbols(self) -> list[str]:
        return sorted(self.world["bars"]["symbol"].unique())
