"""Point-in-time price panel and the DataBundle every research component consumes.

Design (why it is PIT-exact):
  * Raw OHLCV is kept as-is. LEVEL-based logic (price >= $5, dollar volume) uses raw fields only,
    so a future reverse split can never make a historical penny stock look like a $5 stock.
  * Daily total return is computed locally from raw data and the actions effective THAT day:
        ret_t = (close_t * split_ratio_t + dividend_t) / prev_close - 1
    It depends on nothing after t.
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
DERIVED_FIELDS = ("ret", "tri", "aopen", "ahigh", "alow", "aclose", "dollar_volume", "split_ratio", "dividend")


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


def build_panel(
    bars: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    calendar: TradingCalendar | None = None,
    symbols: list[str] | None = None,
) -> Panel:
    """Build a PIT panel from RAW bars + corporate actions (both in canonical schema)."""
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
    if actions is not None and len(actions):
        acts = schemas.conform("corporate_actions", actions)
        acts = acts[acts["symbol"].isin(columns)]
        acts = _effective_action_dates(close, acts)
        splits = acts[acts["action_type"] == "split"]
        for (d, s), r in splits.groupby(["eff_date", "symbol"])["ratio"].prod().items():
            if d in split_ratio.index and np.isfinite(r) and r > 0:
                split_ratio.loc[d, s] *= r
        divs = acts[acts["action_type"] == "cash_dividend"]
        for (d, s), a in divs.groupby(["eff_date", "symbol"])["amount"].sum().items():
            if d in dividend.index and np.isfinite(a):
                dividend.loc[d, s] += a

    prev_close = close.ffill().shift(1)
    ret = (close * split_ratio + dividend) / prev_close - 1.0
    ret = ret.where(close.notna() & prev_close.notna())

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
            as_of=d,
        )

    @property
    def market_symbol(self) -> str:
        return self.benchmarks.get("market", "SPY")

    @property
    def sector_etfs(self) -> dict[str, str]:
        return dict(self.benchmarks.get("sectors", {}))
