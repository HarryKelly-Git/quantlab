"""Point-in-time price panel and the DataBundle every research component consumes.

Design (why it is PIT-exact):
  * Raw OHLCV is kept as-is. LEVEL-based logic (price >= $5, dollar volume) uses raw fields only,
    so a future reverse split can never make a historical penny stock look like a $5 stock.
  * Daily total return is computed locally from raw data and the actions effective THAT day:
        ret_t = (close_t * split_ratio_t + dividend_t) / prev_close - 1
    It depends on nothing after t. On a recorded SPIN-OFF's effective session a negative ret_t is
    set to 0 (see :func:`build_panel`).
  * ``merger`` / ``spin_off`` (bool) mark the session a merger record takes effect / a spin-off's
    neutral step was applied. Both are dated events (row <= t), so they truncate like any field.
  * ``tri`` (total-return index) is the cumulative product of (1 + ret). Its scale is arbitrary,
    so only SCALE-INVARIANT quantities (ratios, returns, distances-from-MA in %, ATR / price) may
    be derived from tri-based fields ``aopen/ahigh/alow/aclose``. Those are exactly what a
    back-adjusted-as-of-t series would give, without using any future corporate action.
  * :meth:`DataBundle.truncate` produces exactly what was knowable at the cutoff of a session;
    :func:`quantlab.testing.pit.assert_truncation_invariant` uses it to prove features/signals
    do not peek at the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.calendar import TradingCalendar, to_session
from quantlab.data import schemas

RAW_FIELDS = ("open", "high", "low", "close", "volume")
SMALL_SPLIT_RATIO = 1.15      # |ratio| within +-15%: not confirmable from prices (stock dividends)
DERIVED_FIELDS = ("ret", "tri", "aopen", "ahigh", "alow", "aclose", "dollar_volume", "split_ratio", "dividend",
                  "merger", "spin_off")
MERGER_ACTION_TYPES = ("cash_merger", "stock_merger", "stock_and_cash_merger")
SPIN_OFF_ACTION_TYPE = "spin_off"


@dataclass
class Panel:
    """Wide (dates x symbols) frames sharing one index and column set."""

    fields: dict[str, pd.DataFrame]
    meta: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> pd.DataFrame:
        f = self.__dict__.get("fields", {})
        if name in f:
            return f[name]
        raise AttributeError(name)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def symbols(self) -> pd.Index:
        return self.fields["close"].columns

    @property
    def listed(self) -> pd.DataFrame:
        """True where the symbol has a bar on that session."""
        return self.fields["close"].notna()

    def history_length(self) -> pd.DataFrame:
        """Number of sessions with a bar up to and including each date (PIT)."""
        return self.fields["close"].notna().cumsum()

    def truncate(self, as_of) -> "Panel":
        end = to_session(as_of)
        return Panel({k: v.loc[:end] for k, v in self.fields.items()}, dict(self.meta))

    def select(self, symbols) -> "Panel":
        cols = [s for s in symbols if s in self.symbols]
        return Panel({k: v[cols] for k, v in self.fields.items()}, dict(self.meta))

    def forward_returns(self, horizon: int, entry: str = "next_open") -> pd.DataFrame:
        """LABEL helper (uses the future by definition — only for targets/outcomes, never features).

        entry='next_open': return from the next session's open to the close ``horizon`` sessions
        after the signal date (total return, split/dividend aware via tri-scaled prices).
        """
        if entry == "next_open":
            entry_px = self.fields["aopen"].shift(-1)
        elif entry == "close":
            entry_px = self.fields["aclose"]
        else:
            raise ValueError(entry)
        exit_px = self.fields["aclose"].shift(-horizon)
        return exit_px / entry_px - 1.0


def _pivot(bars: pd.DataFrame, col: str, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    wide = bars.pivot(index="date", columns="symbol", values=col)
    return wide.reindex(index=index, columns=columns).astype("float64")


def _effective_action_dates(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Map each action's ex_date to the first session >= ex_date on which the symbol traded."""
    if actions.empty:
        return actions.assign(eff_date=pd.Series(dtype="datetime64[ns]"))
    out = []
    for sym, grp in actions.groupby("symbol", sort=False):
        if sym not in close.columns:
            continue
        traded = close.index[close[sym].notna().to_numpy()]
        if len(traded) == 0:
            continue
        pos = traded.searchsorted(grp["ex_date"].to_numpy(), side="left")
        g = grp.copy()
        g["eff_date"] = [traded[p] if p < len(traded) else pd.NaT for p in pos]
        out.append(g)
    if not out:
        return actions.iloc[0:0].assign(eff_date=pd.Series(dtype="datetime64[ns]"))
    return pd.concat(out, ignore_index=True).dropna(subset=["eff_date"])


def reconcile_splits(open_: pd.DataFrame, close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Place each recorded split on the session where the RAW prices actually show it.

    Vendors sometimes record a (typically micro-cap reverse) split a session before the bars show it,
    or deliver bars that are already split-adjusted. Applying such a split blindly manufactures a fake
    -75%/+1,600% return. Rule (point-in-time: never moves a split EARLIER than its recorded ex-date):
      * candidates = the first two sessions >= ex_date on which the symbol traded;
      * a split is CONFIRMED on the first candidate where raw open / previous raw close is consistent
        with the ratio: |log(jump * ratio)| < |log(ratio)| / 2 (i.e. the jump is closer to the split
        than to no split, allowing large genuine same-day moves);
      * otherwise it is UNCONFIRMED and NOT applied (the raw series is continuous, so returns stay
        correct); it is reported so the data audit can judge whether the problem is systemic;
      * NOT_TESTABLE (not applied, cannot affect returns): the ex-date is on/before the symbol's first
        bar, so there is no earlier price to compare against;
      * small ratios (within +-15%, typically stock dividends) cannot be confirmed from prices, because
        an ordinary daily move is as large, so they are applied as recorded on the ex-date
        (APPLIED_SMALL_RATIO).
    Exact duplicate records (same symbol, ex_date, ratio) are applied once.
    Returns the splits with columns eff_date (NaT if unconfirmed) and split_status.
    """
    if splits.empty:
        return splits.assign(eff_date=pd.Series(dtype="datetime64[ns]"), split_status=pd.Series(dtype="object"))
    sp = splits.drop_duplicates(["symbol", "ex_date", "ratio"]).copy()
    eff, status = [], []
    for r in sp.itertuples():
        if r.symbol not in close.columns or not np.isfinite(r.ratio) or r.ratio <= 0 or r.ratio == 1:
            eff.append(pd.NaT), status.append("invalid")
            continue
        c = close[r.symbol]
        traded = c.index[c.notna().to_numpy()]
        i = int(traded.searchsorted(pd.Timestamp(r.ex_date), side="left"))
        if i == 0 or i >= len(traded):
            eff.append(pd.NaT), status.append("not_testable")
            continue
        if abs(np.log(r.ratio)) < np.log(SMALL_SPLIT_RATIO):
            eff.append(traded[i]), status.append("applied_small_ratio")
            continue
        placed = None
        for k in (i, i + 1):
            if k <= 0 or k >= len(traded):
                continue
            s, prev = traded[k], traded[k - 1]
            o = open_.at[s, r.symbol]
            px = o if np.isfinite(o) and o > 0 else c[s]
            jump = px / c[prev]
            if np.isfinite(jump) and jump > 0 and abs(np.log(jump * r.ratio)) < abs(np.log(r.ratio)) / 2:
                placed = s
                break
        eff.append(placed if placed is not None else pd.NaT)
        status.append("confirmed_on_ex_date" if placed is not None and placed == (traded[i] if i < len(traded) else None)
                      else ("confirmed_next_session" if placed is not None else "unconfirmed"))
    sp["eff_date"] = eff
    sp["split_status"] = status
    return sp


def build_panel(
    bars: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    calendar: TradingCalendar | None = None,
    symbols: list[str] | None = None,
) -> Panel:
    """Build a PIT panel from RAW bars + corporate actions (both in canonical schema).

    Besides splits and cash dividends:
      * SPIN-OFF (action_type ``spin_off``, placed like a dividend on the parent's first traded
        session >= ex_date): the parent's raw price drops by about the value of the distributed
        shares, which holders keep. A NEGATIVE ret on that session is set to 0, a NEUTRAL step as if
        a distribution of equal value had been paid; a positive ret is left unchanged. tri and
        aopen/ahigh/alow/aclose follow from ret. This is an approximation: the distributed shares'
        exact value is unknown (and the parent's own move that session is lost with the drop).
        Spin-offs missing from the vendor's list (Alpaca lists only some) are still booked as raw
        drops. ``spin_off`` (bool) marks the sessions where the neutral step was applied; the raw
        fields, ``dividend`` and ``split_ratio`` are untouched (the paper ledger reads those).
      * MERGER (``cash_merger`` / ``stock_merger`` / ``stock_and_cash_merger`` on the acquiree):
        ``merger`` (bool) is True on the first session of the panel >= the record's ex_date
        (effective date). The acquiree usually never trades again, so this is a CALENDAR session,
        not a traded one. Used only to price a delisting (CostModel.delisting_exit_return); a record
        effective after the panel's last session is not placed (not yet known: point-in-time).
    """
    bars = schemas.conform("bars", bars)
    if symbols is not None:
        bars = bars[bars["symbol"].isin(symbols)]
    bars = bars.drop_duplicates(schemas.BARS_KEY, keep="last")
    index = calendar.sessions if calendar is not None else pd.DatetimeIndex(sorted(bars["date"].unique()))
    columns = pd.Index(sorted(bars["symbol"].unique()) if symbols is None else [s for s in symbols if s in set(bars["symbol"])])

    f = {c: _pivot(bars, c, index, columns) for c in RAW_FIELDS}
    close = f["close"]

    split_ratio = pd.DataFrame(1.0, index=index, columns=columns)
    dividend = pd.DataFrame(0.0, index=index, columns=columns)
    merger = np.zeros(close.shape, dtype=bool)
    spin_on = np.zeros(close.shape, dtype=bool)
    if actions is not None and len(actions):
        acts = schemas.conform("corporate_actions", actions)
        acts = acts[acts["symbol"].isin(columns)]
        splits = reconcile_splits(f["open"], close, acts[acts["action_type"] == "split"]).dropna(subset=["eff_date"])
        # mergers: first CALENDAR session >= effective date (the acquiree has no later bar)
        mg = acts[acts["action_type"].isin(MERGER_ACTION_TYPES) & acts["ex_date"].notna()]
        if len(mg):
            pos = index.searchsorted(pd.DatetimeIndex(mg["ex_date"]), side="left")
            cix = columns.get_indexer(mg["symbol"])
            ok = pos < len(index)
            merger[pos[ok], cix[ok]] = True
        acts = _effective_action_dates(close, acts[acts["action_type"] != "split"])
        for (d, s), r in splits.groupby(["eff_date", "symbol"])["ratio"].prod().items():
            if d in split_ratio.index and np.isfinite(r) and r > 0:
                split_ratio.loc[d, s] *= r
        divs = acts[acts["action_type"] == "cash_dividend"]
        for (d, s), a in divs.groupby(["eff_date", "symbol"])["amount"].sum().items():
            if d in dividend.index and np.isfinite(a):
                dividend.loc[d, s] += a
        spins = acts[acts["action_type"] == SPIN_OFF_ACTION_TYPE]
        if len(spins):
            rix = index.get_indexer(pd.DatetimeIndex(spins["eff_date"]))
            cix = columns.get_indexer(spins["symbol"])
            ok = rix >= 0
            spin_on[rix[ok], cix[ok]] = True

    prev_close = close.ffill().shift(1)
    ret = (close * split_ratio + dividend) / prev_close - 1.0
    ret = ret.where(close.notna() & prev_close.notna())
    # spin-off neutral step (see docstring): only a DROP on the spin-off session is neutralized
    spin_off = pd.DataFrame(spin_on, index=index, columns=columns) & (ret < 0)
    ret = ret.mask(spin_off, 0.0)

    tri = (1.0 + ret.fillna(0.0)).cumprod()
    tri = tri.where(close.notna())
    k = tri / close
    f.update(
        ret=ret,
        tri=tri,
        aopen=f["open"] * k,
        ahigh=f["high"] * k,
        alow=f["low"] * k,
        aclose=tri,
        dollar_volume=close * f["volume"],
        split_ratio=split_ratio,
        dividend=dividend,
        merger=pd.DataFrame(merger, index=index, columns=columns),
        spin_off=spin_off,
    )
    # NOTE: never store full-history facts (e.g. a symbol's last bar = future delisting) in meta —
    # meta survives truncate(). Derive such facts from the (possibly truncated) frames instead.
    meta = {"providers": sorted(bars["provider"].dropna().unique().tolist())}
    return Panel(f, meta)


@dataclass
class DataBundle:
    """Everything a feature/strategy may look at, already restricted to what was knowable.

    ``as_of`` is None for a full-history bundle; :meth:`truncate` returns the PIT view for a session.
    """

    panel: Panel
    calendar: TradingCalendar
    reference: pd.DataFrame = field(default_factory=lambda: schemas.empty("reference"))
    actions: pd.DataFrame = field(default_factory=lambda: schemas.empty("corporate_actions"))
    events: pd.DataFrame = field(default_factory=lambda: schemas.empty("events"))
    fundamentals: pd.DataFrame = field(default_factory=lambda: schemas.empty("fundamentals"))
    news: pd.DataFrame = field(default_factory=lambda: schemas.empty("news"))
    benchmarks: dict[str, Any] = field(default_factory=dict)   # {"market": "SPY", "sectors": {...}}
    dataset_ids: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    as_of: pd.Timestamp | None = None
    # congress / insider disclosures (schema ``alt_trades``); context only, never scored
    alt_trades: pd.DataFrame = field(default_factory=lambda: schemas.empty("alt_trades"))

    def truncate(self, as_of) -> "DataBundle":
        """Point-in-time view: only information available at cutoff(as_of)."""
        d = to_session(as_of)
        cutoff = self.calendar.cutoff(d)

        def _avail(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "available_at" not in df.columns:
                return df
            return df[pd.to_datetime(df["available_at"], utc=True) <= cutoff]

        acts = self.actions
        if not acts.empty:
            acts = acts[pd.to_datetime(acts["ex_date"]) <= d]
        return replace(
            self,
            panel=self.panel.truncate(d),
            actions=acts,
            events=_avail(self.events),
            fundamentals=_avail(self.fundamentals),
            news=_avail(self.news),
            alt_trades=_avail(self.alt_trades),
            as_of=d,
        )

    @property
    def market_symbol(self) -> str:
        return self.benchmarks.get("market", "SPY")

    @property
    def sector_etfs(self) -> dict[str, str]:
        return dict(self.benchmarks.get("sectors", {}))
