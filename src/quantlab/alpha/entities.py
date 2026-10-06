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


def resolve(bars: pd.DataFrame) -> pd.DataFrame:
    """(ticker, date) -> entity for every date the store has a bar. ``bars``: symbol, date (+ any cols,
    which are carried through for the chosen entity). Output columns: the input ones + ticker + entity."""
    b = bars.copy()
    b["symbol"] = b["symbol"].astype(str)
    syms = pd.Index(b["symbol"].unique())
    tick = pd.Series(_ticker(syms), index=syms)
    b["ticker"] = b["symbol"].map(tick)
    b["entity"] = b["symbol"]
    n_ent = tick.groupby(tick).transform("size")
    multi = set(n_ent[n_ent > 1].index)                     # entity keys whose ticker has several entities
    if not multi:
        return b
    one = b[~b["symbol"].isin(multi)]
    many = b[b["symbol"].isin(multi)].copy()
    last = many.groupby("symbol")["date"].transform("max")
    many["prio"] = np.where(many["symbol"] != many["ticker"], 0, 1)    # '@' entities first
    many["end"] = last
    many = (many.sort_values(["ticker", "date", "prio", "end"])
            .drop_duplicates(["ticker", "date"], keep="first").drop(columns=["prio", "end"]))
    return pd.concat([one, many], ignore_index=True)


def map_events(events: pd.DataFrame, resolved: pd.DataFrame, ticker_col: str, date_col: str) -> pd.Series:
    """Entity for each (ticker, session) row of ``events``; NaN when the store has no bar that session."""
    key = resolved[["ticker", "date", "entity"]].rename(columns={"ticker": ticker_col, "date": date_col})
    e = events[[ticker_col, date_col]].copy()
    e[date_col] = pd.to_datetime(e[date_col])
    m = e.reset_index().merge(key, on=[ticker_col, date_col], how="left").set_index("index")
    return m["entity"].reindex(events.index)


def dedupe_twins(bars: pd.DataFrame, min_days: int = 5, drop_frac: float = 0.95
                 ) -> tuple[pd.DataFrame, pd.DataFrame, set[str], pd.DataFrame]:
    """Returns (bars without twin rows, aliases [symbol, date, keeper], renamed keys, per-pair report).
    Two keys are TWINS when they share (date, raw close, volume > 0) on >= ``min_days`` days (counted per
    pair, so a third key matching on a couple of days by coincidence does not break the pair)."""
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
    pairs: Counter = Counter()
    for mem in members.values():
        for a, b in combinations(sorted(set(mem)), 2):
            pairs[(a, b)] += 1
    twin = {k for k, n in pairs.items() if n >= min_days}
    if not twin:
        return bars, empty, set(), pd.DataFrame()
    drop_rows = []
    for g, mem in members.items():              # members are in priority order (sorted by rank above)
        kept: list[str] = []
        for x in mem:
            hit = next((k for k in kept if (min(k, x), max(k, x)) in twin), None)
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
    """Replace (entity, date) pairs that were removed as twins by the key that was kept."""
    if aliases is None or len(aliases) == 0:
        return entity
    a = aliases.rename(columns={"symbol": "_e", "date": "_d"})
    x = pd.DataFrame({"_e": entity.astype(object), "_d": pd.to_datetime(date)}, index=entity.index).reset_index()
    m = x.merge(a, on=["_e", "_d"], how="left").set_index("index")["keeper"].reindex(entity.index)
    return m.where(m.notna(), entity)
