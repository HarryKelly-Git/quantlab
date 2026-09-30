"""Event-driven daily backtester with share-based portfolio accounting.

SEMANTICS (identical to :func:`quantlab.core.tradesim.simulate_plan`, tested against it):
  * Decision at the close of signal session D, fill at the OPEN of the next session on which the
    symbol trades (at its close if that session's open is missing) via ``CostModel.fill_price``
    plus ``CostModel.commission``. The one-way cost uses the 20-session median dollar volume at D
    for BOTH legs, exactly like simulate_plan.
  * Stops/targets are compared with each held session's CLOSE in tri-scaled ("a") prices — the RAW
    stop at D is converted with aclose[D]/close[D] — and the exit fills at the next open. Time exit
    after ``plan.holding_sessions`` held sessions (the entry session counts as 1).
  * ``costs.stop_model == "intraday"`` (a broker-held stop-market order, as the paper runner places):
    a held session that OPENS through the stop exits at that open, one whose low/high touches it
    exits at the stop, both on that same session; on the entry session only the low/high counts and
    only when the entry was on the right side of the stop. Targets and time exits are unchanged.
  * A held symbol with no bar for ``costs.delisting_missing_sessions`` consecutive sessions is
    DELISTED on that session: exit value = last close x (1 + costs.delisting_return), still charged
    the sell cost. Decided only from sessions already seen; a gap still open at ``end`` is closed as
    END_OF_TEST at the last close (no haircut, no peeking).
  * Positions still open at ``end`` are closed at the ``end`` close (END_OF_TEST), sell cost
    charged, so every reported number is net of the full round trip.

ACCOUNTING: cash + positions held in SHARES. On a split's effective session shares *= ratio; a
cash dividend credits shares x amount (per pre-split share, ex-date basis) — both only for shares
held at the previous close, as in reality. Positions are marked to market on the RAW close
(last known close while a bar is missing). equity = cash + sum(shares x close) holds every day.

POINT IN TIME: signals are consumed only for the session they are dated; sizing uses equity and
cash known at the decision close; costs use trailing (<= D) dollar volume. The engine never reads
data after ``end`` and refuses periods that reach the locked holdout unless given a valid token
(:class:`quantlab.validation.holdout.HoldoutGuard`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quantlab.core.calendar import TradingCalendar, to_session
from quantlab.core.costs import CostModel
from quantlab.core.types import Candidate, Direction, ExitReason, TradePlan
from quantlab.data.panel import DataBundle, Panel
from quantlab.db.database import Database
from quantlab.logging_setup import get_logger, log_event
from quantlab.validation.holdout import HoldoutGuard

from . import metrics as M

log = get_logger(__name__)

CandidateFilter = Callable[[pd.Timestamp, list[Candidate]], list[Candidate]]

TRADE_COLUMNS = [
    "trade_id", "strategy_id", "strategy_version", "symbol", "direction", "signal_date", "entry_date",
    "exit_date", "exit_reason", "qty", "exit_qty", "entry_ref_price", "entry_price", "exit_ref_price",
    "exit_price", "gross_ret", "cost_ret", "net_ret", "gross_ret_tri", "pnl", "dividends", "costs",
    "holding_sessions", "mfe", "mae", "score", "score_pct", "stop_price", "target_price",
    "plan_holding_sessions", "benchmark_ret", "excess_ret", "sector",
]
EQUITY_COLUMNS = ["equity", "cash", "positions_value", "gross_exposure", "net_exposure", "positions", "cum_costs"]


# ------------------------------------------------------------------------------------------------
# Portfolio ledger (shared with validation.baselines)
# ------------------------------------------------------------------------------------------------
@dataclass
class Lot:
    """One open position. ``shares`` is SIGNED (long > 0, short < 0) and split-adjusted."""

    symbol: str
    col: int
    sign: int
    shares: float
    entry_qty: float
    entry_idx: int
    entry_ref: float           # raw price before costs
    entry_fill: float          # raw price after costs
    entry_a: float             # tri-scaled entry price (for excursions / tri return)
    one_way: float
    last_close: float
    last_valid: int
    flow: float = 0.0          # net cash flow of this lot so far (negative = paid out)
    costs: float = 0.0         # spread/slippage + commission cash paid
    dividends: float = 0.0     # dividend cash received (long) / paid (short, negative)
    held: int = 0
    hi_ex: float = 0.0
    lo_ex: float = 0.0
    pending_exit: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Portfolio:
    """Cash + lots. Every cash movement goes through :meth:`trade` or :meth:`apply_actions`."""

    def __init__(self, cash: float, costs: CostModel, fractional: bool = False):
        self.cash = float(cash)
        self.costs = costs
        self.fractional = fractional
        self.lots: dict[str, Lot] = {}
        self.cum_costs = 0.0

    def round_qty(self, q: float) -> float:
        if self.fractional:
            return float(q)
        return float(math.floor(q + 1e-9))

    def trade(self, lot: Lot, q: float, ref_price: float) -> float:
        """Execute ``q`` signed shares (buy > 0, sell < 0) at ``ref_price`` +- the lot's one-way
        cost. Returns the fill price."""
        buy = q > 0
        fill = ref_price * (1 + lot.one_way) if buy else ref_price * (1 - lot.one_way)
        commission = self.costs.commission(q)
        cash_delta = -(q * fill) - commission
        cost_cash = abs(q) * ref_price * lot.one_way + commission
        self.cash += cash_delta
        lot.flow += cash_delta
        lot.costs += cost_cash
        lot.shares += q
        self.cum_costs += cost_cash
        return fill

    def apply_actions(self, i: int, split_ratio: np.ndarray, dividend: np.ndarray) -> None:
        """Splits and cash dividends effective on session i, for shares held at the prior close."""
        for lot in self.lots.values():
            if lot.entry_idx >= i:
                continue
            d = dividend[i, lot.col]
            if d != 0 and np.isfinite(d):
                amt = lot.shares * d
                self.cash += amt
                lot.flow += amt
                lot.dividends += amt
            r = split_ratio[i, lot.col]
            if r != 1.0 and np.isfinite(r) and r > 0:
                lot.shares *= r

    def positions_value(self) -> tuple[float, float]:
        """(net market value, gross market value) at each lot's last known raw close."""
        net = gross = 0.0
        for lot in self.lots.values():
            v = lot.shares * lot.last_close
            net += v
            gross += abs(v)
        return net, gross

    def equity(self) -> float:
        return self.cash + self.positions_value()[0]


# ------------------------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------------------------
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict[str, Any]
    params: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    benchmark_equity: pd.DataFrame | None = None

    def daily_returns(self) -> pd.Series:
        return M.daily_returns(self.equity)

    @property
    def trade_returns(self) -> np.ndarray:
        if self.trades.empty:
            return np.zeros(0)
        return self.trades["net_ret"].dropna().to_numpy(dtype="float64")


def tri_equity(panel: Panel, symbol: str, dates: pd.DatetimeIndex, capital: float) -> pd.DataFrame | None:
    """Buy-and-hold equity of ``symbol`` (total return, marked at close) over ``dates``."""
    if symbol not in panel.symbols or len(dates) == 0:
        return None
    a = panel.aclose[symbol].reindex(dates).ffill()
    if not np.isfinite(a.iloc[0]):
        return None
    eq = capital * a / a.iloc[0]
    return pd.DataFrame({"equity": eq, "cash": 0.0, "positions_value": eq, "gross_exposure": 1.0,
                         "net_exposure": 1.0, "positions": 1, "cum_costs": 0.0})


# ------------------------------------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------------------------------------
@dataclass
class _PendingEntry:
    strategy_id: str
    symbol: str
    col: int
    signal_idx: int
    qty: float
    sign: int
    entry_ref_at_signal: float
    one_way: float
    a_stop: float | None
    a_target: float | None
    hold_limit: int
    meta: dict[str, Any]


class BacktestEngine:
    """``BacktestEngine(config, bundle_or_panel, db=None).run(signals, strategies, ...)``."""

    def __init__(self, config, data: DataBundle | Panel, db: Database | None = None,
                 costs: CostModel | None = None):
        self.config = config
        if isinstance(data, DataBundle):
            self.bundle: DataBundle | None = data
            self.panel = data.panel
            self.market_symbol = data.market_symbol
        elif isinstance(data, Panel):
            self.bundle = None
            self.panel = data
            self.market_symbol = config.get("benchmarks.market", "SPY")
        else:
            raise TypeError("BacktestEngine needs a DataBundle or Panel")
        self.db = db
        self.costs = costs or CostModel.from_config(config)
        self.guard = HoldoutGuard(config, db)
        bt = config.section("backtest")
        self.initial_capital = float(bt.get("initial_capital", 100_000))
        self.max_positions = int(bt.get("max_positions", 10))
        self.sizing = str(bt.get("sizing", "equal_risk"))
        self.risk_per_trade = float(bt.get("risk_per_trade", 0.005))
        self.max_position_weight = float(bt.get("max_position_weight", 0.10))
        self.allow_short = bool(bt.get("allow_short", False))
        self.fractional = bool(bt.get("fractional", False))
        if self.sizing not in ("equal_risk", "equal_weight"):
            raise ValueError(f"backtest.sizing must be equal_risk|equal_weight, got {self.sizing!r}")
        if bt.get("execution", "next_open") != "next_open":
            raise ValueError("only backtest.execution=next_open is supported (the shared trade semantics)")
        self._arrays: dict[str, np.ndarray] | None = None
        self._sector = self._sector_map()

    # -- data -------------------------------------------------------------------------------------
    def _sector_map(self) -> dict[str, str]:
        ref = self.bundle.reference if self.bundle is not None else None
        if ref is None or ref.empty or "sector" not in ref.columns:
            return {}
        r = ref.dropna(subset=["sector"]).drop_duplicates("symbol", keep="last")
        return dict(zip(r["symbol"], r["sector"]))   # ASSUMED_STATIC (current classification)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            p = self.panel
            dv = p.dollar_volume
            # trailing 20-session median (same window as simulate_plan: D-19..D, NaNs skipped)
            adv = dv.rolling(20, min_periods=1).median()
            self._arrays = {
                "close": p.close.to_numpy(dtype="float64"), "open": p.open.to_numpy(dtype="float64"),
                "aclose": p.aclose.to_numpy(dtype="float64"), "aopen": p.aopen.to_numpy(dtype="float64"),
                "ahigh": p.ahigh.to_numpy(dtype="float64"), "alow": p.alow.to_numpy(dtype="float64"),
                "split": p.split_ratio.to_numpy(dtype="float64"), "div": p.dividend.to_numpy(dtype="float64"),
                "adv": adv.to_numpy(dtype="float64"),
            }
        return self._arrays

    def _feature_set(self, fs):
        if fs is not None:
            return fs
        from quantlab.features.base import FeatureSet   # local: features are optional for plans
        bundle = self.bundle or DataBundle(panel=self.panel, calendar=TradingCalendar(self.panel.dates))
        return FeatureSet(bundle)

    # -- run --------------------------------------------------------------------------------------
    def run(self, signals: dict[str, pd.DataFrame], strategies: dict[str, Any],
            universe: pd.DataFrame | None = None, start=None, end=None,
            candidate_filter: CandidateFilter | None = None, holdout_token: str | None = None,
            fs=None, experiment_id: str | None = None) -> BacktestResult:
        dates = self.panel.dates
        symbols = self.panel.symbols
        s = to_session(start) if start is not None else dates[0]
        e = to_session(end) if end is not None else dates[-1]
        i0 = int(dates.searchsorted(s, side="left"))
        i1 = int(dates.searchsorted(e, side="right")) - 1
        if i0 >= len(dates) or i1 < i0:
            raise ValueError(f"no sessions in [{s.date()}, {e.date()}]")
        grant = self.guard.check(dates[i0], dates[i1], holdout_token, experiment_id)
        missing = [sid for sid in signals if sid not in strategies]
        if missing:
            raise ValueError(f"no Strategy object for signals {missing} (needed for plan())")

        A = self.arrays
        close = A["close"]
        valid_close = np.isfinite(close)
        umask = None
        if universe is not None:
            umask = universe.reindex(index=dates, columns=symbols).fillna(False).astype(bool).to_numpy()
        S: dict[str, np.ndarray] = {}
        for sid in sorted(signals):
            arr = signals[sid].reindex(index=dates, columns=symbols).to_numpy(dtype="float64", copy=True)
            if umask is not None:
                arr[~umask] = np.nan
            arr[~valid_close] = np.nan
            S[sid] = arr
        fs_ = self._feature_set(fs) if S else None

        pf = Portfolio(self.initial_capital, self.costs, self.fractional)
        pending: list[_PendingEntry] = []
        trades: list[dict[str, Any]] = []
        eq_rows: list[dict[str, Any]] = []
        diag = {k: 0 for k in ("signals_seen", "no_free_slot", "already_held", "filtered_out", "plan_error",
                               "no_valid_stop", "size_zero", "cash_capped", "cancelled_cash_at_fill",
                               "cancelled_end_of_test", "short_not_allowed", "entries", "delistings")}
        dr = self.costs.delisting_return

        for i in range(i0, i1 + 1):
            # (1) corporate actions for shares held at the previous close; resize pending orders
            pf.apply_actions(i, A["split"], A["div"])
            for pe in pending:
                r = A["split"][i, pe.col]
                if r != 1.0 and np.isfinite(r) and r > 0:
                    pe.qty = pf.round_qty(pe.qty * r) if not self.fractional else pe.qty * r
            # (2) exits triggered at earlier closes fill at this session's open (close if no open)
            for sym in sorted(pf.lots):
                lot = pf.lots[sym]
                if lot.pending_exit is None:
                    continue
                px, apx = self._exec_price(A, i, lot.col)
                if px is None:
                    continue
                trades.append(self._close(pf, lot, i, px, apx, lot.pending_exit, dates, i))
            # (3) entries decided at the previous close
            still: list[_PendingEntry] = []
            for pe in pending:
                px, apx = self._exec_price(A, i, pe.col)
                if px is None:
                    still.append(pe)
                    continue
                if pe.symbol in pf.lots or len(pf.lots) >= self.max_positions:
                    diag["no_free_slot"] += 1
                    continue
                lot = self._open(pf, pe, i, px, apx, diag)
                if lot is not None:
                    diag["entries"] += 1
            pending = still
            # (4) close: excursions, triggers, delistings
            for sym in sorted(pf.lots):
                lot = pf.lots[sym]
                c = close[i, lot.col]
                if np.isfinite(c):
                    lot.last_close, lot.last_valid = float(c), i
                    if lot.pending_exit is None:
                        hit = self._on_close(A, lot, i)
                        if hit is not None:              # intraday stop: filled on this session
                            ref, a_px, at_open = hit
                            trades.append(self._close(pf, lot, i, ref, a_px, ExitReason.STOP.value, dates, i,
                                                      bench_at_close=not at_open))
                elif i - lot.last_valid >= self.costs.delisting_missing_sessions:
                    diag["delistings"] += 1
                    lv = lot.last_valid
                    # booked on THIS session (when the rule fires), valued at the last close x (1+dr)
                    trades.append(self._close(pf, lot, i, lot.last_close * (1 + dr),
                                              A["aclose"][lv, lot.col] * (1 + dr), ExitReason.DELISTED.value, dates, i))
            # (5) end of test: liquidate at the close
            if i == i1:
                for sym in sorted(pf.lots):
                    lot = pf.lots[sym]
                    trades.append(self._close(pf, lot, lot.last_valid, lot.last_close,
                                              A["aclose"][lot.last_valid, lot.col], ExitReason.END_OF_TEST.value, dates, i))
                diag["cancelled_end_of_test"] += len(pending)
                pending = []
            net_v, gross_v = pf.positions_value()
            eqv = pf.cash + net_v
            eq_rows.append({"date": dates[i], "equity": eqv, "cash": pf.cash, "positions_value": net_v,
                            "gross_exposure": gross_v / eqv if eqv > 0 else np.nan,
                            "net_exposure": net_v / eqv if eqv > 0 else np.nan,
                            "positions": len(pf.lots), "cum_costs": pf.cum_costs})
            # (6) decisions at this close for the next session
            if i < i1 and S:
                pending.extend(self._decide(i, S, strategies, fs_, pf, pending, candidate_filter, diag, eqv))

        equity = pd.DataFrame(eq_rows).set_index("date")
        equity.index.name = "date"
        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        if len(trades_df):
            trades_df = trades_df.sort_values(["entry_date", "symbol", "strategy_id"], kind="mergesort").reset_index(drop=True)
        bench = tri_equity(self.panel, self.market_symbol, dates[i0: i1 + 1], self.initial_capital)
        met = M.summary(equity, trades_df, bench,
                        min_trades=int(self.config.get("validation.min_trades_for_conclusion", 100)))
        labels = ["SYNTHETIC"] if (self.bundle is not None and self.bundle.is_synthetic) else []
        if grant is not None:
            labels.append("HOLDOUT")
        params = {
            "start": str(dates[i0].date()), "end": str(dates[i1].date()),
            "backtest": self.config.section("backtest"), "costs": self.config.section("costs"),
            "fractional": self.fractional,
            "strategies": {sid: _describe(strategies[sid]) for sid in sorted(signals)},
            "candidate_filter": getattr(candidate_filter, "__name__", type(candidate_filter).__name__)
            if candidate_filter is not None else None,
            "universe": "mask" if universe is not None else "all_symbols_with_data",
            "holdout_access_log_id": grant.access_log_id if grant is not None else None,
            "market_symbol": self.market_symbol,
        }
        log_event(log, "backtest finished", start=params["start"], end=params["end"], trades=len(trades_df),
                  final_equity=float(equity["equity"].iloc[-1]), strategies=sorted(signals))
        return BacktestResult(trades_df, equity, met, params, diag, labels, bench)

    # -- mechanics --------------------------------------------------------------------------------
    @staticmethod
    def _exec_price(A, i: int, col: int) -> tuple[float | None, float | None]:
        o = A["open"][i, col]
        if np.isfinite(o) and np.isfinite(A["aopen"][i, col]):
            return float(o), float(A["aopen"][i, col])
        c = A["close"][i, col]
        if np.isfinite(c) and np.isfinite(A["aclose"][i, col]):
            return float(c), float(A["aclose"][i, col])
        return None, None

    def _open(self, pf: Portfolio, pe: _PendingEntry, i: int, px: float, apx: float, diag) -> Lot | None:
        q = pe.qty
        fill = px * (1 + pe.one_way) if pe.sign > 0 else px * (1 - pe.one_way)
        # cash cap at the fill (longs pay cash; shorts must hold 100% collateral)
        need = q * fill + self.costs.commission(q)
        if need > pf.cash:
            q = pf.round_qty(max(pf.cash - self.costs.commission_min_per_order, 0.0)
                             / (fill + self.costs.commission_per_share))
            while q > 0 and q * fill + self.costs.commission(q) > pf.cash:
                q = pf.round_qty(q - 1) if not self.fractional else q * (1 - 1e-9)
        if q <= 0:
            diag["cancelled_cash_at_fill"] += 1
            return None
        lot = Lot(pe.symbol, pe.col, pe.sign, 0.0, q, i, px, fill, apx, pe.one_way, px, i,
                  meta={**pe.meta, "signal_idx": pe.signal_idx, "a_stop": pe.a_stop, "a_target": pe.a_target,
                        "hold_limit": pe.hold_limit})
        pf.trade(lot, pe.sign * q, px)
        pf.lots[pe.symbol] = lot
        return lot

    def _on_close(self, A, lot: Lot, i: int) -> tuple[float, float, bool] | None:
        """simulate_plan's per-session loop: count the session, update excursions, check exits.
        Returns (raw price, tri price, filled at the open) when an intraday stop fills on this
        session; otherwise sets ``pending_exit`` for a close-triggered exit (or nothing)."""
        lot.held += 1
        h, l = A["ahigh"][i, lot.col], A["alow"][i, lot.col]
        if np.isfinite(h):
            lot.hi_ex = max(lot.hi_ex, h / lot.entry_a - 1)
        if np.isfinite(l):
            lot.lo_ex = min(lot.lo_ex, l / lot.entry_a - 1)
        c = A["aclose"][i, lot.col]
        m = lot.meta
        a_stop = m["a_stop"]
        if (self.costs.stop_model == "intraday" and a_stop is not None
                and (i > lot.entry_idx or (lot.entry_a - a_stop) * lot.sign > 0)):
            ao = A["aopen"][i, lot.col]
            ratio = A["close"][i, lot.col] / c           # raw / tri at this session
            if i > lot.entry_idx and np.isfinite(ao) and (ao - a_stop) * lot.sign <= 0:
                return float(A["open"][i, lot.col]), float(ao), True
            extreme = l if lot.sign > 0 else h
            if np.isfinite(extreme) and (extreme - a_stop) * lot.sign <= 0:
                return float(a_stop * ratio), float(a_stop), False
        if m["a_stop"] is not None and (c - m["a_stop"]) * lot.sign <= 0:
            lot.pending_exit = ExitReason.STOP.value
        elif m["a_target"] is not None and (c - m["a_target"]) * lot.sign >= 0:
            lot.pending_exit = ExitReason.TARGET.value
        elif lot.held >= m["hold_limit"]:
            lot.pending_exit = ExitReason.TIME.value

    def _close(self, pf: Portfolio, lot: Lot, exit_idx: int, ref: float, a_exit: float, reason: str,
               dates: pd.DatetimeIndex, book_idx: int, bench_at_close: bool = False) -> dict[str, Any]:
        exit_qty = abs(lot.shares)
        fill = pf.trade(lot, -lot.shares, ref)
        del pf.lots[lot.symbol]
        basis = abs(lot.entry_qty) * lot.entry_ref
        pnl = lot.flow
        gross_pnl = pnl + lot.costs
        m = lot.meta
        mfe, mae = (lot.hi_ex, lot.lo_ex) if lot.sign > 0 else (-lot.lo_ex, -lot.hi_ex)
        bench_ret = None
        if self.market_symbol in self.panel.symbols:
            bc = self.panel.symbols.get_loc(self.market_symbol)
            A = self.arrays
            b0 = A["aopen"][lot.entry_idx, bc]
            b1 = A["aclose"][exit_idx, bc] if (bench_at_close or reason in (ExitReason.DELISTED.value,
                                                                              ExitReason.END_OF_TEST.value)) \
                else A["aopen"][exit_idx, bc]
            if np.isfinite(b0) and np.isfinite(b1) and b0 > 0:
                bench_ret = float(b1 / b0 - 1)
        net = pnl / basis if basis > 0 else np.nan
        return {
            "trade_id": f"{lot.symbol}-{dates[m['signal_idx']]:%Y%m%d}-{m['strategy_id']}",
            "strategy_id": m["strategy_id"], "strategy_version": m["strategy_version"], "symbol": lot.symbol,
            "direction": "LONG" if lot.sign > 0 else "SHORT",
            "signal_date": dates[m["signal_idx"]], "entry_date": dates[lot.entry_idx], "exit_date": dates[exit_idx],
            "exit_reason": reason, "qty": abs(lot.entry_qty), "exit_qty": exit_qty,
            "entry_ref_price": lot.entry_ref, "entry_price": lot.entry_fill, "exit_ref_price": ref, "exit_price": fill,
            "gross_ret": gross_pnl / basis if basis > 0 else np.nan, "cost_ret": lot.costs / basis if basis > 0 else np.nan,
            "net_ret": net, "gross_ret_tri": (a_exit / lot.entry_a - 1) * lot.sign,
            "pnl": pnl, "dividends": lot.dividends, "costs": lot.costs, "holding_sessions": lot.held,
            "mfe": mfe, "mae": mae, "score": m["score"], "score_pct": m["score_pct"],
            "stop_price": m["stop_price"], "target_price": m["target_price"], "plan_holding_sessions": m["hold_limit"],
            "benchmark_ret": bench_ret, "excess_ret": (net - bench_ret) if bench_ret is not None else None,
            "sector": self._sector.get(lot.symbol),
        }

    def _decide(self, i: int, S: dict[str, np.ndarray], strategies, fs, pf: Portfolio,
                pending: list[_PendingEntry], candidate_filter, diag, equity: float) -> list[_PendingEntry]:
        A = self.arrays
        dates, symbols = self.panel.dates, self.panel.symbols
        ranked: list[tuple[float, str, str, int, float]] = []
        for sid, arr in S.items():
            row = arr[i]
            cols = np.flatnonzero(np.isfinite(row))
            if len(cols) == 0:
                continue
            diag["signals_seen"] += len(cols)
            strat = strategies[sid]
            if getattr(strat, "direction", Direction.LONG) is Direction.SHORT and not self.allow_short:
                diag["short_not_allowed"] += len(cols)
                continue
            pct = rankdata(row[cols], method="average") / len(cols)
            for c, p in zip(cols, pct):
                ranked.append((-float(p), str(symbols[c]), sid, int(c), float(row[c])))
        if not ranked:
            return []
        ranked.sort()
        busy = set(pf.lots) | {pe.symbol for pe in pending}
        seen: set[str] = set()
        cands = []
        for neg_p, sym, sid, c, sc in ranked:
            if sym in seen:
                continue                        # one position per symbol: best-ranked strategy wins
            seen.add(sym)
            if sym in busy:
                diag["already_held"] += 1
                continue
            cands.append((neg_p, sym, sid, c, sc))
        occupied = sum(1 for lot in pf.lots.values() if lot.pending_exit is None) + len(pending)
        free = self.max_positions - occupied
        if free <= 0:
            diag["no_free_slot"] += len(cands)
            return []
        d = dates[i]
        plans: dict[tuple[str, str], TradePlan] = {}

        def plan_for(sym: str, sid: str) -> TradePlan | None:
            key = (sym, sid)
            if key not in plans:
                try:
                    plans[key] = strategies[sid].plan(fs, sym, d)
                except Exception as exc:  # a data problem for one candidate => NO TRADE for it
                    diag["plan_error"] += 1
                    log_event(log, "plan() failed; candidate skipped", level=30, symbol=sym, strategy=sid,
                              date=str(d.date()), error=repr(exc))
                    plans[key] = None
            return plans[key]

        if candidate_filter is not None:
            objs = []
            for neg_p, sym, sid, c, sc in cands:
                pl = plan_for(sym, sid)
                if pl is None:
                    continue
                st = strategies[sid]
                adv = A["adv"][i, c]
                objs.append(Candidate(symbol=sym, as_of_date=d.date(), strategy_id=sid,
                                      strategy_version=str(getattr(st, "version", "")), score=sc,
                                      direction=getattr(st, "direction", Direction.LONG),
                                      features={"score_pct": -neg_p}, plan=pl,
                                      risk={"adv20": float(adv) if np.isfinite(adv) else None}))
            kept = candidate_filter(d, objs)
            keep = {(k.symbol, k.strategy_id) for k in kept}
            before = len(cands)
            cands = [x for x in cands if (x[1], x[2]) in keep]
            diag["filtered_out"] += before - len(cands)

        # cash available at the decision: cash + expected proceeds of pending exits - reserved orders
        avail = pf.cash
        for lot in pf.lots.values():
            if lot.pending_exit is not None:
                avail += lot.shares * lot.last_close - abs(lot.shares) * lot.last_close * lot.one_way
        avail -= sum(abs(pe.qty) * pe.entry_ref_at_signal * (1 + pe.one_way) for pe in pending)

        out: list[_PendingEntry] = []
        for neg_p, sym, sid, c, sc in cands:
            if len(out) >= free:
                diag["no_free_slot"] += 1
                continue
            pl = plan_for(sym, sid)
            if pl is None:
                continue
            st = strategies[sid]
            sign = getattr(st, "direction", Direction.LONG).sign
            cl, acl = A["close"][i, c], A["aclose"][i, c]
            ref = pl.entry_ref_price if pl.entry_ref_price and np.isfinite(pl.entry_ref_price) and pl.entry_ref_price > 0 else cl
            adv = A["adv"][i, c]
            one_way = self.costs.one_way_cost_frac(float(adv) if np.isfinite(adv) else None)
            stop = pl.stop_price if pl.stop_price else None
            target = pl.target_price if pl.target_price else None
            if self.sizing == "equal_risk":
                risk_ps = (ref - stop) * sign if stop is not None else np.nan
                if not (np.isfinite(risk_ps) and risk_ps > 0):
                    diag["no_valid_stop"] += 1
                    continue
                q = self.risk_per_trade * equity / risk_ps
            else:
                q = min(1.0 / self.max_positions, self.max_position_weight) * equity / ref
            q = min(q, self.max_position_weight * equity / ref)
            cap = max(avail, 0.0) / (ref * (1 + one_way))
            if q > cap:
                diag["cash_capped"] += 1
                q = cap
            q = pf.round_qty(q)
            if not q > 0:
                diag["size_zero"] += 1
                continue
            avail -= q * ref * (1 + one_way)
            scale = acl / cl
            out.append(_PendingEntry(
                sid, sym, c, i, q, sign, float(ref), one_way,
                stop * scale if stop is not None else None, target * scale if target is not None else None,
                int(pl.holding_sessions),
                {"strategy_id": sid, "strategy_version": str(getattr(st, "version", "")), "score": sc,
                 "score_pct": -neg_p, "stop_price": stop, "target_price": target}))
        return out


def _describe(strategy) -> dict[str, Any]:
    try:
        return strategy.describe()
    except Exception:  # describe() is informational only
        return {"strategy_id": getattr(strategy, "strategy_id", None), "version": getattr(strategy, "version", None)}


# ------------------------------------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------------------------------------
def _num(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _date(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or x is pd.NaT:
        return None
    return str(pd.Timestamp(x).date())


def save_to_db(db: Database, experiment_id: str, result: BacktestResult, segment: str = "full") -> dict[str, int]:
    """Append trades and the equity curve to backtest_trades / backtest_equity (tables are
    append-only). The experiment row must already exist (experiments.registry.start)."""
    t = result.trades
    rows = []
    for r in t.to_dict("records"):
        rows.append({
            "experiment_id": experiment_id, "trade_id": f"{segment}|{r['trade_id']}", "segment": segment,
            "strategy_id": r["strategy_id"], "strategy_version": r["strategy_version"] or "",
            "symbol": r["symbol"], "signal_date": _date(r["signal_date"]), "entry_date": _date(r["entry_date"]),
            "exit_date": _date(r["exit_date"]), "exit_reason": r["exit_reason"], "qty": _num(r["qty"]),
            "entry_price": _num(r["entry_price"]), "exit_price": _num(r["exit_price"]),
            "gross_ret": _num(r["gross_ret"]), "cost_ret": _num(r["cost_ret"]), "net_ret": _num(r["net_ret"]),
            "pnl": _num(r["pnl"]), "holding_sessions": _num(r["holding_sessions"]), "mfe": _num(r["mfe"]),
            "mae": _num(r["mae"]), "score": _num(r["score"]), "regime": r.get("regime"), "sector": r.get("sector"),
        })
    eq = result.equity
    erows = [{"experiment_id": experiment_id, "segment": segment, "date": _date(d),
              "equity": float(row["equity"]), "cash": float(row["cash"]),
              "gross_exposure": float(row["gross_exposure"]) if np.isfinite(row["gross_exposure"]) else 0.0,
              "positions": int(row["positions"])}
             for d, row in eq.iterrows()]
    with db.transaction():
        n_t = db.insert_many("backtest_trades", rows)
        n_e = db.insert_many("backtest_equity", erows)
    return {"trades": n_t, "equity_rows": n_e}
