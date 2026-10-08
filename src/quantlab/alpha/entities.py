"""Ticker -> entity resolution BY DATE (audit fixes C2 and M1) and twin-entity de-duplication (M2).

The store holds one series per ENTITY. Plain keys ('CZR') are the entity known by that ticker today
(Alpaca's default ``asof`` mapping: its whole history, including periods when it traded under another
ticker, e.g. Eldorado as ERI). Keys 'TICKER@YYYY-MM-DD' are entities resolved as of the date that ticker
was last seen in the historical Nasdaq Trader directory (typically a company that later delisted and
whose ticker was then reused, e.g. old Caesars as CZR until 2020).

External data keyed by (ticker, date) - option chains, earnings calendar, consensus estimates - refers to
the company that USED the ticker on that date. Rule: among entities for that ticker with a bar on the
date, an '@' entity whose last bar is on/after the date wins (it is the one that used the ticker up to its
last_seen); if several, the one that ends first; otherwise the plain entity.
Any remaining mismatch is caught downstream (options: put-call parity vs the matched close).

TWINS (audit fix M2). Two entity keys can carry the SAME security over part of their history (a rename:
the old-ticker entity and the current entity whose asof history includes the old ticker; a merger whose
legal survivor took the partner's ticker; SPAC splices). On such days the raw close AND volume are
identical. Keeping both double-weights the company and, when the twin ends, books a rename as a delisting.
Rule: two keys with identical (date, raw close, volume > 0) on >= ``min_days`` days are twins; on each
identical day only the twin with the most bars is kept (ties: plain key, then name). A member
that loses >= 95% of its rows is dropped entirely. A member whose LAST bar was an identical day did not
delist (it continued under the kept key): it is marked ``renamed``. The dropped (key, date) pairs are
returned as ALIASES so that external (ticker, date) data still maps to the kept key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ticker(sym: pd.Index) -> pd.Index:
    return pd.Index(sym.astype(str)).str.split("@").str[0]


AT_TOLERANCE_DAYS = 7          # an '@' entity claims its ticker up to its directory last_seen date + 7 days


FORWARD_WINDOW = (30, 120)     # an '@' key's rename must fall within [last_seen - 30d, last_seen + 120d]


def rename_chains(renames: pd.DataFrame | None, keys) -> dict[str, tuple[np.ndarray, list[str]]]:
    """For each store key whose company traded under more than one ticker: the tickers it used over time,
    oldest first, with the start date of every later one (from Alpaca's symbol-change history).
      * plain key K: followed BACKWARDS from the latest change into K;
      * 'T@D' key (T last seen in the directory on D): backwards from T before D, and FORWARDS through
        the change out of T dated within [D - 30d, D + 120d] (the two sources' dates differ a little),
        e.g. CBS@2019-12-01 -> VIAC (2020) -> PARA (2022).
    Every step after the first must be the same security (CUSIP continuity); a break stops the chain, and
    the earlier/later ticker is then left unclaimed rather than guessed."""
    if renames is None or len(renames) == 0:
        return {}
    r = renames[renames["old_symbol"] != renames["new_symbol"]].sort_values("process_date")   # CUSIP-only records change no ticker
    by_new = {sym: g for sym, g in r.groupby("new_symbol")}
    by_old = {sym: g for sym, g in r.groupby("old_symbol")}
    out: dict[str, tuple[np.ndarray, list[str]]] = {}
    for k in keys:
        at = "@" in k
        t = k.split("@")[0]
        anchor = pd.Timestamp(k.split("@")[1]) if at else None
        if t not in by_new and not (at and t in by_old):
            continue
        # backwards
        cur, end, cusip, back = t, anchor, None, []
        first_cusip = None
        while len(back) < 25:
            g = by_new.get(cur)
            if g is not None and end is not None:
                g = g[g["process_date"] < end]
            if g is not None and cusip is not None:
                g = g[g["new_cusip"] == cusip]
            if g is None or g.empty:
                back.append((cur, None))
                break
            ch = g.iloc[-1]
            first_cusip = first_cusip or ch["new_cusip"]
            back.append((cur, ch["process_date"]))
            cur, end, cusip = ch["old_symbol"], ch["process_date"], ch["old_cusip"]
        segs = back[::-1]
        # forwards ('@' keys only)
        if at:
            cur, cusip, lo, hi = t, first_cusip, anchor - pd.Timedelta(days=FORWARD_WINDOW[0]), anchor + pd.Timedelta(days=FORWARD_WINDOW[1])
            while len(segs) < 50:
                g = by_old.get(cur)
                if g is None:
                    break
                g = g[g["process_date"] >= lo]
                if hi is not None:
                    g = g[g["process_date"] <= hi]
                if cusip is not None:
                    g = g[g["old_cusip"] == cusip]
                if g.empty:
                    break
                ch = g.iloc[0]
                segs.append((ch["new_symbol"], ch["process_date"]))
                cur, cusip, lo, hi = ch["new_symbol"], ch["new_cusip"], ch["process_date"], None
        if len(segs) < 2:
            continue
        starts = np.array([pd.Timestamp(d).to_datetime64() for _, d in segs[1:]], dtype="datetime64[ns]")
        out[k] = (starts, [x for x, _ in segs])
    return out


def resolve(bars: pd.DataFrame, renames: pd.DataFrame | None = None) -> pd.DataFrame:
    """(ticker, date) -> entity for every date the store has a bar. ``bars``: symbol, date (+ any cols,
    which are carried through for the chosen entity). Output: the chosen input rows plus categorical
    ``ticker`` and ``entity`` columns (``symbol`` = ``entity``).
    Which ticker a key's bar was traded under on its date:
      * 'T@D' keys: T up to D + 7 days; later bars claim nothing, unless the symbol-change history shows
        the company continuing under a new ticker (then that ticker, e.g. CBS@2019-12-01 -> VIAC -> PARA);
      * plain keys with a symbol-change history (``renames``, Alpaca corporate actions): the ticker in force
        on that date (audit follow-up to C2: a plain key's asof history includes the years before it took
        its current ticker - e.g. today's PARA, Banzai, traded as BNZI until 2026-08 - so it must never
        answer for the company that used the ticker before);
      * other plain keys: the key.
    Among keys claiming the same (ticker, date): '@' keys first (the one ending first), then plain.
    Integer codes throughout: the full store is ~19M rows."""
    sym = bars["symbol"]
    cat = sym if isinstance(sym.dtype, pd.CategoricalDtype) else sym.astype(str).astype("category")
    cats = pd.Index(cat.cat.categories.astype(str))
    codes = cat.cat.codes.to_numpy().astype(np.int64)
    dates = pd.to_datetime(bars["date"]).to_numpy().astype("datetime64[ns]")
    is_at_c = np.asarray(cats.str.contains("@"), dtype=bool)
    lim_c = (pd.to_datetime(cats.str.split("@").str[1], errors="coerce")
             + pd.Timedelta(days=AT_TOLERANCE_DAYS)).to_numpy().astype("datetime64[ns]")
    tick_c = _ticker(cats)
    chains = rename_chains(renames, list(cats))
    fwd = np.array([bool(is_at_c[i]) and k in chains and chains[k][1][-1] != tick_c[i] for i, k in enumerate(cats)], dtype=bool)
    keep = np.ones(len(codes), dtype=bool)
    at_rows = is_at_c[codes] & ~fwd[codes]                  # '@' keys without a forward rename stop at last_seen
    keep[at_rows] = ~(dates[at_rows] > lim_c[codes[at_rows]])
    del at_rows
    names = set(tick_c)
    for _, nm in chains.values():
        names.update(nm)
    tickers = pd.Index(sorted(names))
    tcode = tickers.get_indexer(tick_c)[codes]
    if chains:
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(cats) + 1))
        for k, (starts, nm) in chains.items():
            ci = cats.get_loc(k)
            rows = order[bounds[ci]:bounds[ci + 1]]
            if len(rows):
                tcode[rows] = tickers.get_indexer(pd.Index(nm))[np.searchsorted(starts, dates[rows], side="right")]
        del order
    end_c = pd.Series(dates).groupby(codes).max().reindex(range(len(cats))).to_numpy().astype("datetime64[ns]")
    idx = np.nonzero(keep)[0]
    del keep
    o = np.lexsort((end_c[codes[idx]].view("i8"), np.where(is_at_c[codes[idx]], 0, 1), dates[idx].view("i8"), tcode[idx]))
    sel = idx[o]
    del idx, o
    t_s, d_s = tcode[sel], dates[sel].view("i8")
    first = np.ones(len(sel), dtype=bool)
    first[1:] = (t_s[1:] != t_s[:-1]) | (d_s[1:] != d_s[:-1])
    sel = np.sort(sel[first])
    out = bars.iloc[sel].copy()
    out["entity"] = pd.Categorical.from_codes(codes[sel], categories=cats)
    out["symbol"] = out["entity"]
    out["ticker"] = pd.Categorical.from_codes(tcode[sel], categories=tickers)
    return out.reset_index(drop=True)


def map_events(events: pd.DataFrame, resolved: pd.DataFrame, ticker_col: str, date_col: str) -> pd.Series:
    """Entity for each (ticker, session) row of ``events``; NaN when no store entity used that ticker on
    that session."""
    tick = events[ticker_col].astype(str)
    key = resolved.loc[resolved["ticker"].isin(set(tick)), ["ticker", "date", "entity"]]
    key = pd.DataFrame({ticker_col: key["ticker"].astype(str).to_numpy(), date_col: pd.to_datetime(key["date"]).to_numpy(),
                        "entity": key["entity"].astype(str).to_numpy()})
    e = pd.DataFrame({ticker_col: tick.to_numpy(), date_col: pd.to_datetime(events[date_col]).to_numpy()}, index=events.index)
    m = e.reset_index().merge(key, on=[ticker_col, date_col], how="left").set_index("index")
    return m["entity"].reindex(events.index)


def dedupe_twins(bars: pd.DataFrame, min_days: int = 5, drop_frac: float = 0.95
                 ) -> tuple[pd.DataFrame, pd.DataFrame, set[str], pd.DataFrame]:
    """Returns (bars without twin rows, aliases [symbol, date, keeper], renamed keys, per-pair report).
    Two keys are TWINS on the days where they share (date, raw close, volume > 0) in an UNBROKEN run of
    >= ``min_days`` of their common trading days. A twin period (rename, merger survivor, SPAC splice) is
    contiguous; coincidences are scattered (verification review: two NextShares funds matched on 5 of 330
    common days). Counted per pair, so a third key matching on a couple of days does not break the pair."""
    from collections import Counter
    from itertools import combinations
    empty = pd.DataFrame(columns=["symbol", "date", "keeper"])
    v = bars.loc[bars["volume"] > 0, ["symbol", "date", "close", "volume"]]
    d = v[v.duplicated(["date", "close", "volume"], keep=False)].copy()
    if d.empty:
        return bars, empty, set(), pd.DataFrame()
    d["symbol"] = d["symbol"].astype(str)
    n_bars = bars.groupby(bars["symbol"].astype(str)).size()
    syms = pd.Index(d["symbol"].unique())
    ranked = sorted(syms, key=lambda s: (-int(n_bars.get(s, 0)), "@" in s, s))     # most bars, plain, name
    rank = {s: i for i, s in enumerate(ranked)}
    d["rank"] = d["symbol"].map(rank)
    d = d.sort_values(["date", "close", "volume", "rank"])
    # group rows by identical (date, close, volume)
    gid = d.groupby(["date", "close", "volume"], sort=False).ngroup().to_numpy()
    members: dict[int, list[str]] = {}
    dates: dict[int, pd.Timestamp] = {}
    for g, sym, dt in zip(gid, d["symbol"].to_numpy(), d["date"].to_numpy()):
        members.setdefault(g, []).append(sym)
        dates[g] = dt
    pair_dates: dict[tuple[str, str], list] = {}
    for g, mem in members.items():
        for a, b in combinations(sorted(set(mem)), 2):
            pair_dates.setdefault((a, b), []).append(dates[g])
    cand = {k: v for k, v in pair_dates.items() if len(v) >= min_days}
    if not cand:
        return bars, empty, set(), pd.DataFrame()
    in_pairs = {x for k in cand for x in k}
    vv = v[v["symbol"].astype(str).isin(in_pairs)]
    def _ns(x) -> np.ndarray:
        return np.asarray(x).astype("datetime64[ns]").astype(np.int64)
    days = {k: _ns(g.to_numpy()) for k, g in vv.groupby(vv["symbol"].astype(str))["date"]}
    twin_days: dict[tuple[str, str], set] = {}
    for k, ident in cand.items():
        common = np.intersect1d(days[k[0]], days[k[1]])
        is_id = np.isin(common, _ns(ident))
        edges = np.diff(np.r_[0, is_id.astype(np.int8), 0])            # runs of consecutive identical common days
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        keep_days = [common[a_:b_] for a_, b_ in zip(starts, ends) if b_ - a_ >= min_days]
        if keep_days:
            twin_days[k] = set(np.concatenate(keep_days).tolist())
    if not twin_days:
        return bars, empty, set(), pd.DataFrame()
    drop_rows = []
    for g, mem in members.items():              # members are in priority order (sorted by rank above)
        kept: list[str] = []
        dt_g = int(_ns([dates[g]])[0])
        for x in mem:
            hit = next((k for k in kept if dt_g in twin_days.get((min(k, x), max(k, x)), ())), None)
            if hit is None:
                kept.append(x)
            else:
                drop_rows.append((x, dates[g], hit))
    drop = pd.DataFrame(drop_rows, columns=["symbol", "date", "keeper"])
    if drop.empty:
        return bars, empty, set(), pd.DataFrame()
    drop["date"] = pd.to_datetime(drop["date"])
    # members that lose >= drop_frac of their rows go entirely (their few odd rows are the same company)
    lost = drop.groupby("symbol").size() / n_bars.reindex(drop["symbol"].unique())
    whole = set(lost[lost >= drop_frac].index)
    sym_str = bars["symbol"].astype(str)
    last_bar = bars.groupby(sym_str)["date"].max()
    dl = drop.assign(last=drop["symbol"].map(last_bar))
    renamed = set(dl.loc[dl["date"] == dl["last"], "symbol"]) - whole
    main_keeper = drop.groupby("symbol")["keeper"].agg(lambda s: s.value_counts().index[0])
    extra = bars.loc[sym_str.isin(whole), ["symbol", "date"]].assign(symbol=lambda x: x["symbol"].astype(str))
    extra["keeper"] = extra["symbol"].map(main_keeper)
    aliases = (pd.concat([drop, extra], ignore_index=True).drop_duplicates(["symbol", "date"])
               .sort_values(["symbol", "date"]).reset_index(drop=True))
    key = pd.MultiIndex.from_frame(aliases[["symbol", "date"]])
    row_key = pd.MultiIndex.from_arrays([sym_str, bars["date"]])
    out = bars[~row_key.isin(key)]
    rep = (drop.groupby(["symbol", "keeper"]).size().rename("identical_days").reset_index()
           .assign(n_bars=lambda x: x["symbol"].map(n_bars), dropped_entirely=lambda x: x["symbol"].isin(whole),
                   renamed=lambda x: x["symbol"].isin(renamed)))
    return out, aliases, renamed, rep


def apply_aliases(entity: pd.Series, date: pd.Series, aliases: pd.DataFrame | None) -> pd.Series:
    """Replace (entity, date) pairs that were removed as twins by the key that was kept. Only rows whose
    entity appears in ``aliases`` are looked up (the resolution table has ~19M rows)."""
    if aliases is None or len(aliases) == 0:
        return entity
    hit = entity.isin(set(aliases["symbol"].astype(str))).to_numpy()
    if not hit.any():
        return entity
    sub = pd.DataFrame({"_e": entity[hit].astype(str).to_numpy(), "_d": pd.to_datetime(date[hit]).to_numpy()})
    a = pd.DataFrame({"_e": aliases["symbol"].astype(str).to_numpy(), "_d": pd.to_datetime(aliases["date"]).to_numpy(),
                      "keeper": aliases["keeper"].astype(str).to_numpy()})
    k = sub.merge(a, on=["_e", "_d"], how="left")["keeper"].to_numpy()
    if isinstance(entity.dtype, pd.CategoricalDtype):
        new = sorted({x for x in k if isinstance(x, str)} - set(entity.cat.categories))
        out = entity.cat.add_categories(new) if new else entity.copy()
    else:
        out = entity.copy()
    vals = out.to_numpy(copy=True)
    pos = np.nonzero(hit)[0]
    found = pd.notna(k)
    if isinstance(out.dtype, pd.CategoricalDtype):
        out.iloc[pos[found]] = k[found]
        return out
    vals[pos[found]] = k[found]
    return pd.Series(vals, index=entity.index, name=entity.name)
