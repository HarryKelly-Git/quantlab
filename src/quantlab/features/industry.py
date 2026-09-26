"""Point-in-time industry / sector features from SEC SIC observations (``events.sic_observation``).

The SIC code of a symbol at session D is the one printed in the header of its latest periodic
report whose acceptance time is <= cutoff(D) (see :mod:`quantlab.data.sec_catalysts`). Before its
first observed filing the code is UNKNOWN (NaN), never a guessed or current value.

* Industry = the 3-digit SIC industry group when it has >= MIN_MEMBERS members at D, else the
  2-digit major group when that has >= MIN_MEMBERS, else UNKNOWN. Codes are numeric: 3-digit
  groups keep their value (e.g. 283 = drugs), 2-digit fallbacks are 1000 + group (e.g. 1028).
* Industry returns are equal-weighted and LEAVE-ONE-OUT (the stock itself is excluded), so a
  stock's own move cannot make its industry look strong.
* Sector = the SIC -> sector-ETF approximation of :mod:`quantlab.sectors` (its caveats apply)
  applied to the POINT-IN-TIME code; sector performance is that ETF's own return.
* Members at D = symbols with a bar at D (and inside ``fs.universe`` when one is given).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.features.base import FEATURES, FeatureSet
from quantlab.features.price import full_like_nan, market_series, memo, ret_n

MIN_MEMBERS = 5
_SRC_I = "bundle.events sic_observation (SIC from each filing's SGML header, as of its acceptance) + panel aclose"


def sic_asof(fs: FeatureSet) -> pd.DataFrame:
    """dates x symbols SIC code (float) known at each session's cutoff; NaN = UNKNOWN."""
    def build() -> pd.DataFrame:
        p = fs.panel
        out = pd.DataFrame(np.nan, index=p.dates, columns=p.symbols)
        ev = fs.bundle.events
        if ev.empty:
            return out
        ev = ev[(ev["event_type"] == "sic_observation") & ev["symbol"].isin(p.symbols)]
        if ev.empty:
            return out
        ev = ev.sort_values("available_at", kind="mergesort")
        sess = fs.bundle.calendar.first_usable_sessions(ev["available_at"])
        # an observation known before this panel's first session (e.g. a tail window) is the state
        # AT that first session: carry it in instead of dropping it (the latest such one wins)
        sess = sess.where(sess.isna() | (sess >= p.dates[0]), p.dates[0])
        code = ev["payload_json"].map(lambda s: pd.to_numeric((json.loads(s) or {}).get("sic"), errors="coerce"))
        marks = pd.DataFrame({"s": sess.to_numpy(), "sym": ev["symbol"].to_numpy(), "sic": code.to_numpy()}).dropna()
        if marks.empty:
            return out
        marks = marks.drop_duplicates(["s", "sym"], keep="last")      # latest filing of that session wins
        wide = marks.pivot(index="s", columns="sym", values="sic")
        wide = wide.reindex(index=p.dates, columns=p.symbols)
        return wide.ffill().astype("float64")
    return memo(fs, "sic_asof", build)  # type: ignore[return-value]


def _members(fs: FeatureSet) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        m = fs.panel.close.notna() & sic_asof(fs).notna()
        if fs.universe is not None:
            m &= fs.universe.reindex_like(m).fillna(False).astype(bool)
        return m
    return memo(fs, "industry_members_mask", build)  # type: ignore[return-value]


def _long(fs: FeatureSet) -> pd.DataFrame:
    """Long frame of member cells: date, symbol, industry code, ret_20, ret_63."""
    def build() -> pd.DataFrame:
        p = fs.panel
        mem = _members(fs).to_numpy()
        di, si = np.nonzero(mem)
        sic = sic_asof(fs).to_numpy()[di, si]
        a = p.aclose
        r20, r63 = ret_n(a, 20).to_numpy()[di, si], ret_n(a, 63).to_numpy()[di, si]
        df = pd.DataFrame({"di": di, "si": si, "sic3": np.floor(sic / 10.0), "sic2": np.floor(sic / 100.0),
                           "r20": r20, "r63": r63})
        n3 = df.groupby(["di", "sic3"])["si"].transform("size")
        n2 = df.groupby(["di", "sic2"])["si"].transform("size")
        df["code"] = np.where(n3 >= MIN_MEMBERS, df["sic3"],
                              np.where(n2 >= MIN_MEMBERS, 1000.0 + df["sic2"], np.nan))
        df = df.dropna(subset=["code"])
        g = df.groupby(["di", "code"])
        df["n"] = g["si"].transform("size")
        for c in ("r20", "r63"):
            ok = df[c].notna()
            s = df[c].where(ok, 0.0).groupby([df["di"], df["code"]]).transform("sum")
            k = ok.groupby([df["di"], df["code"]]).transform("sum")
            own, own_k = df[c].where(ok, 0.0), ok.astype(float)
            loo_k = k - own_k
            df[f"loo_{c}"] = np.where(loo_k >= MIN_MEMBERS - 1, (s - own) / loo_k.where(loo_k > 0), np.nan)
            df[f"grp_{c}"] = np.where(k >= MIN_MEMBERS, s / k.where(k > 0), np.nan)
        return df
    return memo(fs, "industry_long", build)  # type: ignore[return-value]


def _wide(fs: FeatureSet, col: str) -> pd.DataFrame:
    df = _long(fs)
    out = np.full((len(fs.panel.dates), len(fs.panel.symbols)), np.nan)
    out[df["di"].to_numpy(), df["si"].to_numpy()] = df[col].to_numpy(dtype="float64")
    return pd.DataFrame(out, index=fs.panel.dates, columns=fs.panel.symbols)


@FEATURES.feature("sic_code_asof", "industry", "SIC code from the latest filing header accepted by cutoff(D); NaN = UNKNOWN",
                  _SRC_I, PitStatus.PIT, lookback=0)
def sic_code_asof(fs: FeatureSet) -> pd.DataFrame:
    return sic_asof(fs)


@FEATURES.feature("industry_code", "industry", "3-digit SIC group (>= 5 members at D) else 1000 + 2-digit group, else NaN",
                  _SRC_I, PitStatus.PIT, lookback=0)
def industry_code(fs: FeatureSet) -> pd.DataFrame:
    return _wide(fs, "code")


@FEATURES.feature("industry_members", "industry", "members of the symbol's industry group at D (incl. itself)",
                  _SRC_I, PitStatus.PIT, lookback=0)
def industry_members(fs: FeatureSet) -> pd.DataFrame:
    return _wide(fs, "n")


@FEATURES.feature("industry_ret_20", "industry", "equal-weighted leave-one-out 20-session return of the industry group",
                  _SRC_I, PitStatus.PIT, lookback=20)
def industry_ret_20(fs: FeatureSet) -> pd.DataFrame:
    return _wide(fs, "loo_r20")


@FEATURES.feature("rs_industry_20", "industry", "ret_20(stock) - leave-one-out industry ret_20",
                  _SRC_I, PitStatus.PIT, lookback=20)
def rs_industry_20(fs: FeatureSet) -> pd.DataFrame:
    return ret_n(fs.panel.aclose, 20) - _wide(fs, "loo_r20")


@FEATURES.feature("industry_rank_63", "industry",
                  "percentile rank (0-1] of the industry group's mean 63-session return among groups at D",
                  _SRC_I, PitStatus.PIT, lookback=63)
def industry_rank_63(fs: FeatureSet) -> pd.DataFrame:
    df = _long(fs)
    grp = df.drop_duplicates(["di", "code"])[["di", "code", "grp_r63"]].dropna()
    grp["rank"] = grp.groupby("di")["grp_r63"].rank(pct=True, method="average")
    m = df.merge(grp[["di", "code", "rank"]], on=["di", "code"], how="left")
    out = np.full((len(fs.panel.dates), len(fs.panel.symbols)), np.nan)
    out[m["di"].to_numpy(), m["si"].to_numpy()] = m["rank"].to_numpy(dtype="float64")
    return pd.DataFrame(out, index=fs.panel.dates, columns=fs.panel.symbols)


def _sector_etf_frame(fs: FeatureSet, field_fn) -> pd.DataFrame:
    """Per cell: ``field_fn(etf)`` for the sector ETF of the POINT-IN-TIME SIC code."""
    from quantlab.sectors import configured_sectors, sic_to_sector_etf
    sic = sic_asof(fs)
    allowed = set(configured_sectors(fs.bundle)) & set(fs.panel.symbols)
    if not allowed:
        return full_like_nan(fs)
    codes = pd.unique(sic.to_numpy().ravel())
    etf_of = {c: sic_to_sector_etf(int(c), allowed) for c in codes if np.isfinite(c)}
    out = full_like_nan(fs)
    for etf in sorted({e for e in etf_of.values() if e}):
        vals = field_fn(etf)
        mask = sic.isin([c for c, e in etf_of.items() if e == etf])
        out = out.mask(mask, np.broadcast_to(vals.to_numpy()[:, None], out.shape))
    return out


@FEATURES.feature("rs_sector_20_pit", "industry", "ret_20(stock) - ret_20(sector ETF of the point-in-time SIC code)",
                  _SRC_I + "; sector ETF bars; quantlab.sectors.SIC_SECTOR_RANGES", PitStatus.PIT, lookback=20)
def rs_sector_20_pit(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    return ret_n(a, 20) - _sector_etf_frame(fs, lambda etf: ret_n(a[etf], 20))


@FEATURES.feature("sector_rs_spy_63_pit", "industry", "ret_63(sector ETF of the point-in-time SIC code) - ret_63(SPY)",
                  _SRC_I + "; sector ETF bars; quantlab.sectors.SIC_SECTOR_RANGES", PitStatus.PIT, lookback=63)
def sector_rs_spy_63_pit(fs: FeatureSet) -> pd.DataFrame:
    a = fs.panel.aclose
    m = market_series(fs, "aclose")
    if m is None:
        return full_like_nan(fs)
    spy = ret_n(m, 63)
    return _sector_etf_frame(fs, lambda etf: ret_n(a[etf], 63) - spy)
