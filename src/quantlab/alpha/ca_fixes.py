"""Corporate-action data fixes found in the alpha sprint (panel patch "sprint-c"). PAPER research.

The sprint's master replay exposed fake moves in Alpaca's ``adjustment=all`` bars:

- **Spin-offs, stock distributions and similar actions mis-adjusted or not adjusted:**

  | Stock | Event | Adjusted return | Raw move |
  |---|---|---|---|
  | NVS 2019-04-09 | Alcon spin-off | -82% | -12% |
  | CNX 2017-11-29 | | -90% | -16% |
  | BTX 2022-10-17 | | -95% | +2% |
  | UHAL 2022-11-10 | 9-for-1 distribution of non-voting shares, not adjusted | -90% | |

  For UHAL, holders actually gained about 8%.
- **Ticker reuse after a merger:** BHVN 2022-10-04 went from $151.8 to $8.3. Pfizer bought Biohaven and
  the spin-off took over the ticker, so the series joins two companies.

The fixes use corporate-action records only to locate EVENT DATES. They are data hygiene, never a signal.

1. A day is flagged UNKNOWN (NaN returns) by either of two rules; ``flags`` has the details.
   - **(A) Price rule:** the adjustment AMPLIFIED the move. A correct adjustment only shrinks a raw move.
     Alpaca's corporate-action list misses most spin-offs (120 records in nine years), so this rule
     needs no list.
   - **(B) Event rule:** an extreme move within +/-1 session of a listed distribution. This catches
     missing adjustments, where raw = adjusted.

   De-SPAC moves of -60% or +100% are REAL price moves and stay.
   Adjusted prices are re-chained forward with a NEUTRAL (0) step on flagged days, so trailing ratios
   (momentum, volatility, distance from high) no longer see a fake jump. Whether the neutral step is
   right is unknown; the master replay reports the raw-return alternative as a sensitivity.
2. An entity is TRUNCATED after a CASH merger's effective date when its ticker keeps trading more than 5
   sessions later with a jump of more than 50% within +/-3 sessions. The later bars belong to another
   company. Stock mergers and de-SPACs are not truncated: the same security continues.

Every flag is listed in research/alpha/results/sprint/ca_fixes_report.json.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TYPES = ("spin_off", "stock_dividend", "cash_merger", "stock_merger", "stock_and_cash_merger", "unit_split",
         "reorganization", "rights_distribution", "redemption", "forward_split", "reverse_split", "worthless_removal")
FLAG_TYPES = ("spin_offs", "stock_dividends", "cash_mergers", "stock_mergers", "stock_and_cash_mergers", "unit_splits",
              "reorganizations", "rights_distributions", "redemptions")
MERGERS = ("cash_mergers", "stock_mergers", "stock_and_cash_mergers")
DISTRIBUTIONS = ("spin_offs", "stock_dividends", "unit_splits", "reorganizations", "rights_distributions", "redemptions")
JUMP = math.log(1.25)
ADJ_DIFF = math.log(1.15)
PATCH_VERSION = "sprint-c: corporate-action fake moves set UNKNOWN + adjusted prices re-chained; merger ticker reuse truncated"
# Missing adjustments absent from Alpaca's corporate-action list, found in the sprint's extreme-move review.
# (entity, date, source)
MANUAL = (("UHAL", "2022-11-10", "U-Haul Holding distributed 9 Series N non-voting shares per share; raw = adjusted = -90%, "
                                 "holders were whole (UHAL + 9 UHAL.B); not in Alpaca's corporate-action list"),)


def download(start: str = "2016-01-01", end: str = "2024-12-31", per_minute: int = 100):
    """All corporate actions of TYPES in [start, end], quarterly + paginated, saved to
    var/alpha/external/corporate_actions.parquet (one row per record; ``kind`` = response key)."""
    import requests

    from quantlab.alpha.store import DATA_URL, RateLimiter, _auth, _get, store_dir
    limiter = RateLimiter(per_minute)
    s = requests.Session()
    s.headers.update(_auth())
    rows = []
    for a in pd.date_range(start, end, freq="QS"):
        b = min(a + pd.offsets.QuarterEnd(0), pd.Timestamp(end))
        token = None
        while True:
            params = {"types": ",".join(TYPES), "start": str(a.date()), "end": str(b.date()), "limit": 1000}
            if token:
                params["page_token"] = token
            j = _get(s, f"{DATA_URL}/v1/corporate-actions", params, limiter)
            for kind, recs in (j.get("corporate_actions") or {}).items():
                for r in recs or []:
                    rows.append({"kind": kind, **r})
            token = j.get("next_page_token")
            if not token:
                break
    df = pd.DataFrame(rows).drop_duplicates("id")
    out = store_dir() / "external"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "corporate_actions.parquet", index=False)
    return df


def _event_rows(ca: pd.DataFrame) -> pd.DataFrame:
    """(ticker, date, kind) for every record that can create a fake move in the touched ticker."""
    out = []
    sym_cols = ("source_symbol", "symbol", "acquiree_symbol", "old_symbol")
    date_cols = ("ex_date", "effective_date", "process_date", "payable_date")
    for r in ca.itertuples(index=False):
        rd = r._asdict()
        if rd.get("kind") not in FLAG_TYPES:
            continue
        sym = next((rd[c] for c in sym_cols if c in rd and isinstance(rd[c], str) and rd[c]), None)
        if not sym:
            continue
        for c in date_cols:
            v = rd.get(c)
            if isinstance(v, str) and v:
                out.append({"ticker": sym, "date": pd.Timestamp(v), "kind": rd["kind"]})
    return pd.DataFrame(out).drop_duplicates()


def flags(p, resolved: pd.DataFrame, ca: pd.DataFrame, cols: pd.Index | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flagged entity-days (returns UNKNOWN) and truncations (entity, last good date).

    A (price rule, every day): the ADJUSTMENT AMPLIFIED the move. |ln(1+r_adj)| > ln(1.25),
      |ln(1+r_adj) - ln(1+r_raw)| > ln(1.15) and |ln(1+r_adj)| > |ln(1+r_raw)| + ln(1.15). A correct
      adjustment only ever SHRINKS a raw move: splits, reverse splits, special dividends, spin-offs. Example:
      NVS 2019-04-09, raw -12%, adjusted -82%. Alpaca's corporate-action list misses most spin-offs, so this
      rule needs no event list.
    B (event rule): an extreme adjusted move (|ln(1+r)| > ln(1.25)) within +/-1 session of a DISTRIBUTION in
      Alpaca's list (spin-off, stock dividend, unit split, reorganization, rights distribution, redemption).
      This catches MISSING adjustments where raw = adjusted (ZWS, VYX, LBTYB).
    Truncation: a CASH merger whose ticker keeps trading more than 5 sessions later, with a > 50% jump
      within +/-3 sessions (BHVN). De-SPAC / stock-merger moves are real and stay."""
    cols = p.symbols if cols is None else cols
    dates = p.dates
    ret = p["ret_cc"][cols]
    close = p["close"][cols]
    raw = close / close.shift(1) - 1
    la = np.log1p(ret.clip(lower=-0.9999))
    lr = np.log1p(raw.clip(lower=-0.9999))
    A = (la.abs() > JUMP) & ((la - lr).abs() > ADJ_DIFF) & (la.abs() > lr.abs() + ADJ_DIFF) & la.notna() & lr.notna()
    st = A.stack(future_stack=True)
    a = st[st].reset_index().iloc[:, :2]
    a.columns = ["date", "entity"]
    fl = [{"entity": r.entity, "date": r.date, "kind": "price_rule_adjustment_amplified",
           "ret_adj": float(ret.at[r.date, r.entity]), "ret_raw": float(raw.at[r.date, r.entity])}
          for r in a.itertuples(index=False)]
    tr = []
    ev = _event_rows(ca)
    if not ev.empty:
        pos = np.clip(dates.searchsorted(ev["date"].to_numpy()), 0, len(dates) - 1)
        ev["session"] = dates[pos]
        rmap = resolved[["ticker", "date", "entity"]].copy()
        rmap["ticker"] = rmap["ticker"].astype(str)
        rmap["entity"] = rmap["entity"].astype(str)
        ev = ev.merge(rmap.rename(columns={"date": "session"}), on=["ticker", "session"], how="left")
        ev["entity"] = ev["entity"].fillna(ev["ticker"])
        ev = ev[ev["entity"].isin(cols)]
        for e in ev.itertuples(index=False):
            i = dates.get_loc(e.session)
            col = cols.get_loc(e.entity)
            if e.kind in DISTRIBUTIONS:
                for j in range(max(i - 1, 1), min(i + 2, len(dates))):
                    v = la.iat[j, col]
                    if np.isfinite(v) and abs(v) > JUMP:
                        fl.append({"entity": e.entity, "date": dates[j], "kind": e.kind, "ret_adj": float(ret.iat[j, col]),
                                   "ret_raw": float(raw.iat[j, col])})
            if e.kind == "cash_mergers":
                later = close.iloc[i + 5:, col].notna()
                if later.any():
                    win = la.iloc[max(i - 3, 0): i + 4, col]
                    if (win.abs() > math.log(1.5)).any():
                        tr.append({"entity": e.entity, "last_date": dates[max(i - 1, 0)], "kind": e.kind})
    f = pd.DataFrame(fl).drop_duplicates(["entity", "date"]) if fl else pd.DataFrame(columns=["entity", "date", "kind"])
    t = pd.DataFrame(tr).drop_duplicates("entity") if tr else pd.DataFrame(columns=["entity", "last_date", "kind"])
    return f, t


def apply(p, fl: pd.DataFrame, tr: pd.DataFrame) -> dict[str, Any]:
    """Patch an AlphaPanel IN PLACE: truncate, set flagged returns UNKNOWN, re-chain adjusted prices
    forward (neutral step on flagged days), recompute ret_on/ret_id/ret_oo from the re-chained prices."""
    f = p.f
    dates = p.dates
    n_trunc = 0
    for r in tr.itertuples(index=False):
        if r.entity not in p.symbols:
            continue
        after = dates > pd.Timestamp(r.last_date)
        for k in f:
            if isinstance(f[k], pd.DataFrame) and r.entity in f[k].columns and f[k][r.entity].dtype.kind == "f":
                f[k].loc[after, r.entity] = np.nan
        n_trunc += 1
    touched = sorted(set(fl["entity"]) & set(p.symbols)) if len(fl) else []
    if touched:
        ac = f["adj_close"][touched]
        r = (ac / ac.shift(1) - 1)
        mask = pd.DataFrame(False, index=dates, columns=touched)
        for row in fl.itertuples(index=False):
            if row.entity in mask.columns:
                mask.at[pd.Timestamp(row.date), row.entity] = True
        step = r.where(~mask, 0.0).fillna(0.0)
        tri = (1 + step).cumprod().where(ac.notna())
        k = tri / ac
        for fld in ("adj_open", "adj_high", "adj_low", "adj_close"):
            f[fld].loc[:, touched] = f[fld][touched] * k
        a_c, a_o = f["adj_close"][touched], f["adj_open"][touched]
        rc = (a_c / a_c.shift(1) - 1).where(~mask)
        f["ret_cc"].loc[:, touched] = rc
        f["ret_on"].loc[:, touched] = (a_o / a_c.shift(1) - 1).where(~mask)
        f["ret_id"].loc[:, touched] = (a_c / a_o - 1).where(~mask)
        f["ret_oo"].loc[:, touched] = (a_o.shift(-1) / a_o - 1).where(~mask.shift(-1, fill_value=False))
    p.meta["ca_patch"] = {"version": PATCH_VERSION, "flagged_days": int(len(fl)), "entities_flagged": len(touched),
                          "truncated_entities": n_trunc}
    return p.meta["ca_patch"]


def patch_research_data(d, report_path=None) -> dict[str, Any]:
    """Load the downloaded corporate actions, flag, patch ``d.p`` in place, write the report."""
    import json

    from quantlab.alpha.store import store_dir
    path = store_dir() / "external" / "corporate_actions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run ca_fixes.download() first (data UNKNOWN, refusing to guess)")
    ca = pd.read_parquet(path)
    fl, tr = flags(d.p, d.resolved(), ca)
    man = pd.DataFrame([{"entity": e, "date": pd.Timestamp(dt), "kind": "manual: " + why} for e, dt, why in MANUAL
                        if e in d.p.symbols])
    fl = pd.concat([fl, man], ignore_index=True).drop_duplicates(["entity", "date"]) if len(man) else fl
    info = apply(d.p, fl, tr)
    d.p.meta["ca_flags"] = fl[["entity", "date"]].copy()
    if hasattr(d, "_ret_cc_pnl"):
        object.__delattr__(d, "_ret_cc_pnl")
    if report_path is not None:
        rep = {"patch": info, "flagged": fl.astype({"date": str}).to_dict("records") if len(fl) else [],
               "truncated": tr.astype({"last_date": str}).to_dict("records") if len(tr) else []}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rep, indent=1, default=str))
    d.manifest["ca_patch"] = info
    return info
