"""Validated per-strategy trade statistics for the EV engine, the ranker and the no-trade engine.

Where the numbers come from (strongest evidence first):
  1. ``walk_forward_oos``: ``backtest_trades`` rows in ``oos:*`` segments (walk-forward test
     windows). Only the most recent qualifying experiment per strategy version is used, so
     re-running a walk-forward never double-counts trades.
  2. ``shadow_forward``: matured ``shadow_outcomes`` of this strategy's ``shadow_opportunities``.
     This is genuine forward evidence. It is combined with (1) when both exist, de-duplicated on
     (symbol, signal date).
  3. ``in_sample``: ``in_sample`` / ``full`` backtest segments. Used only when there is no
     out-of-sample or forward evidence. It is optimistic by construction, so ``validated`` is
     False and consumers discount it (EV weight, "unproven" warning).
  4. ``none``: n = 0.

Point-in-time: a trade is evidence at session D only if it had EXITED by D (``exit_date <= D``) and,
for shadow outcomes, was measured by cutoff(D). ``get(..., as_of=D)`` therefore returns the same
answer whether or not later trades exist in the database (tests prove this). ``as_of=None`` means
"everything matured so far" (the live pipeline passes the decision date anyway).

Exclusions (fail-safe defaults):
  * experiments with ``uses_synthetic_data = 1`` and synthetic shadow rows. Synthetic data is never
    market evidence, unless ``allow_synthetic=True`` (method validation / tests).
  * experiments without a ``succeeded`` result, or with a ``failed`` result (partial runs).
  * the locked holdout: segment ``holdout`` or signal dates >= ``validation.holdout.start``. Feeding
    holdout trades into daily decision parameters would quietly turn the holdout into training data.

Return semantics: win/loss partitioning and ``avg_win``/``avg_loss`` use GROSS returns (before
costs). The EV engine then charges the candidate's own CostModel cost once. ``expectancy`` and the
t-statistics use NET returns, which is what matters for "is this strategy losing money?". Shadow
outcomes only carry ``ret``. It is treated as net of costs, and gross is conservatively set equal
to it (costs are not added back), so shadow evidence can only understate the gross edge.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import MARKET_TZ, to_session
from quantlab.core.types import PitStatus
from quantlab.db.database import Database
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

SOURCE_OOS = "walk_forward_oos"
SOURCE_SHADOW = "shadow_forward"
SOURCE_OOS_SHADOW = "walk_forward_oos+shadow_forward"
SOURCE_IN_SAMPLE = "in_sample"
SOURCE_NONE = "none"
VALIDATED_SOURCES = frozenset({SOURCE_OOS, SOURCE_SHADOW, SOURCE_OOS_SHADOW})

_OOS_PREFIX = "oos:"
_IN_SAMPLE_SEGMENTS = ("in_sample", "full")
_MATURED_SHADOW_STATUSES = ("complete", "delisted")


@dataclass(frozen=True)
class RegimeStats:
    """Net-return statistics of one strategy restricted to trades signalled in one regime."""

    n: int
    win_rate: float | None
    expectancy: float | None
    std: float | None
    t_stat: float | None


@dataclass
class StrategyStats:
    """Summary of a strategy version's matured trades. MODEL_OUTPUT with a stated evidence source."""

    strategy_id: str
    version: str
    n: int = 0
    win_rate: float | None = None          # share of trades with gross return > 0
    avg_win: float | None = None           # mean GROSS return of winners (> 0)
    avg_loss: float | None = None          # mean GROSS return of losers (<= 0)
    gross_expectancy: float | None = None  # mean GROSS return per trade
    expectancy: float | None = None        # mean NET return per trade (after the costs charged at the time)
    std: float | None = None               # std of NET returns per trade (ddof=1)
    sharpe_like: float | None = None       # expectancy / std, per trade, NOT annualized
    t_stat: float | None = None            # expectancy / (std / sqrt(n))
    by_regime: dict[str, RegimeStats] = field(default_factory=dict)
    source: str = SOURCE_NONE
    experiment_ids: list[str] = field(default_factory=list)
    n_backtest: int = 0
    n_shadow: int = 0
    as_of: date | None = None
    regime_filter: str | None = None
    uses_synthetic: bool = False
    pit_status: PitStatus = PitStatus.PIT
    notes: list[str] = field(default_factory=list)

    @property
    def validated(self) -> bool:
        """True only for out-of-sample / forward evidence. In-sample never counts as validation."""
        return self.source in VALIDATED_SOURCES and self.n > 0

    @classmethod
    def empty(cls, strategy_id: str, version: str, as_of: date | None = None,
              notes: list[str] | None = None) -> "StrategyStats":
        return cls(strategy_id=strategy_id, version=version, as_of=as_of, notes=list(notes or []))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat() if self.as_of else None
        d["pit_status"] = self.pit_status.value
        d["validated"] = self.validated
        return d


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _net_moments(net: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    """(mean, std, sharpe_like, t_stat) of net returns. std needs n >= 2 and is None when zero."""
    n = len(net)
    if n == 0:
        return None, None, None, None
    mean = float(net.mean())
    if n < 2:
        return mean, None, None, None
    std = float(net.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return mean, None, None, None
    return mean, std, mean / std, mean / (std / math.sqrt(n))


def summarize_returns(gross: np.ndarray, net: np.ndarray, regimes: list[str | None] | None = None) -> dict[str, Any]:
    """Pure summary used by the provider (and directly testable with hand-computed values)."""
    gross = np.asarray(gross, dtype=float)
    net = np.asarray(net, dtype=float)
    if gross.shape != net.shape:
        raise ValueError("gross and net return arrays must have equal length")
    n = int(len(gross))
    out: dict[str, Any] = {"n": n, "win_rate": None, "avg_win": None, "avg_loss": None,
                           "gross_expectancy": None, "expectancy": None, "std": None,
                           "sharpe_like": None, "t_stat": None, "by_regime": {}}
    if n == 0:
        return out
    wins = gross > 0
    out["win_rate"] = float(wins.mean())
    out["avg_win"] = float(gross[wins].mean()) if wins.any() else None
    out["avg_loss"] = float(gross[~wins].mean()) if (~wins).any() else None
    out["gross_expectancy"] = float(gross.mean())
    out["expectancy"], out["std"], out["sharpe_like"], out["t_stat"] = _net_moments(net)
    if regimes is not None:
        labels = pd.Series(list(regimes), dtype="object")
        for label in sorted({r for r in labels.dropna().unique()}):
            mask = (labels == label).to_numpy()
            sub_net, sub_gross = net[mask], gross[mask]
            mean, std, _, t = _net_moments(sub_net)
            out["by_regime"][str(label)] = RegimeStats(
                n=int(mask.sum()), win_rate=float((sub_gross > 0).mean()), expectancy=mean, std=std, t_stat=t)
    return out


class StrategyStatsProvider:
    """Reads strategy evidence from the audit database. Create one instance per run: results are
    cached per (strategy, version, regime, as_of) for the life of the instance."""

    def __init__(self, db: Database, config: Config | None = None, allow_synthetic: bool | None = None,
                 include_holdout: bool | None = None, require_succeeded_experiment: bool | None = None):
        self.db = db
        get = config.get if config is not None else (lambda _k, d=None: d)
        self.allow_synthetic = bool(get("decision.stats.allow_synthetic", False)
                                    if allow_synthetic is None else allow_synthetic)
        self.include_holdout = bool(get("decision.stats.include_holdout", False)
                                    if include_holdout is None else include_holdout)
        self.require_succeeded = bool(get("decision.stats.require_succeeded_experiment", True)
                                      if require_succeeded_experiment is None else require_succeeded_experiment)
        holdout = get("validation.holdout.start", None)
        self.holdout_start = to_session(holdout) if holdout else None
        hh, mm = str(get("project.info_cutoff_local_time", "16:00")).split(":")
        self._cutoff_time = time(int(hh), int(mm))
        self._cache: dict[tuple, StrategyStats] = {}
        self._bt_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._sh_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # -- public ---------------------------------------------------------------------------------
    def clear_cache(self) -> None:
        self._cache.clear()
        self._bt_cache.clear()
        self._sh_cache.clear()

    def get(self, strategy_id: str, version: str, regime: str | None = None,
            as_of: date | str | pd.Timestamp | None = None) -> StrategyStats:
        d = to_session(as_of) if as_of is not None else None
        key = (strategy_id, version, regime, d)
        if key not in self._cache:
            self._cache[key] = self._compute(strategy_id, version, regime, d)
        return self._cache[key]

    # -- internals ------------------------------------------------------------------------------
    def _cutoff_utc(self, d: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(datetime.combine(d.date(), self._cutoff_time)).tz_localize(MARKET_TZ).tz_convert("UTC")

    def _compute(self, strategy_id: str, version: str, regime: str | None, d: pd.Timestamp | None) -> StrategyStats:
        notes: list[str] = []
        bt = self._backtest_rows(strategy_id, version)
        bt = self._filter_backtest(bt, d, notes)
        oos, oos_exp = self._latest_experiment(bt[bt["segment"].str.startswith(_OOS_PREFIX)])
        ins, ins_exp = self._latest_experiment(bt[bt["segment"].isin(_IN_SAMPLE_SEGMENTS)])
        sh = self._filter_shadow(self._shadow_rows(strategy_id, version), d, notes)

        if len(oos) and len(sh):
            source, parts, exps = SOURCE_OOS_SHADOW, [oos, sh], oos_exp
        elif len(oos):
            source, parts, exps = SOURCE_OOS, [oos], oos_exp
        elif len(sh):
            source, parts, exps = SOURCE_SHADOW, [sh], []
            if len(ins):
                notes.append("in-sample backtest trades ignored: forward shadow evidence exists")
        elif len(ins):
            source, parts, exps = SOURCE_IN_SAMPLE, [ins], ins_exp
            notes.append("IN-SAMPLE evidence only: optimistic, not validated out-of-sample")
        else:
            notes.append("no matured trades for this strategy version")
            st = StrategyStats.empty(strategy_id, version, d.date() if d is not None else None, notes)
            st.regime_filter = regime
            return st

        rows = pd.concat(parts, ignore_index=True)
        before = len(rows)
        rows = rows.drop_duplicates(["symbol", "signal_date"], keep="first")
        if len(rows) < before:
            notes.append(f"{before - len(rows)} duplicate (symbol, signal_date) trades removed")
        by_regime = summarize_returns(rows["gross"].to_numpy(), rows["net"].to_numpy(),
                                      rows["regime"].tolist())["by_regime"]
        if regime is not None:
            rows = rows[rows["regime"] == regime]
            notes.append(f"restricted to regime={regime}")
        s = summarize_returns(rows["gross"].to_numpy(), rows["net"].to_numpy())
        synthetic = bool(rows["synthetic"].any()) if len(rows) else False
        if synthetic:
            notes.append("SYNTHETIC evidence included (allow_synthetic=True): not market evidence")
        if (rows["origin"] == "shadow").any():
            notes.append("shadow ret treated as net; gross set equal to it (conservative)")
        stats = StrategyStats(
            strategy_id=strategy_id, version=version, n=s["n"], win_rate=s["win_rate"], avg_win=s["avg_win"],
            avg_loss=s["avg_loss"], gross_expectancy=s["gross_expectancy"], expectancy=s["expectancy"],
            std=s["std"], sharpe_like=s["sharpe_like"], t_stat=s["t_stat"], by_regime=by_regime,
            source=source, experiment_ids=exps, n_backtest=int((rows["origin"] == "backtest").sum()),
            n_shadow=int((rows["origin"] == "shadow").sum()), as_of=d.date() if d is not None else None,
            regime_filter=regime, uses_synthetic=synthetic, pit_status=PitStatus.PIT, notes=notes,
        )
        log_event(log, "strategy stats", strategy_id=strategy_id, version=version, source=source, n=stats.n,
                  as_of=str(stats.as_of), regime=regime)
        return stats

    def _backtest_rows(self, strategy_id: str, version: str) -> pd.DataFrame:
        key = (strategy_id, version)
        if key not in self._bt_cache:
            df = self.db.query_df(
                """
                SELECT bt.experiment_id, bt.trade_id, bt.segment, bt.symbol, bt.signal_date, bt.exit_date,
                       bt.gross_ret, bt.cost_ret, bt.net_ret, bt.regime,
                       e.kind AS exp_kind, e.created_at AS exp_created_at,
                       e.uses_synthetic_data AS synthetic,
                       (SELECT COUNT(*) FROM experiment_results r
                         WHERE r.experiment_id = e.experiment_id AND r.status = 'succeeded') AS n_ok,
                       (SELECT COUNT(*) FROM experiment_results r
                         WHERE r.experiment_id = e.experiment_id AND r.status = 'failed') AS n_failed
                FROM backtest_trades bt JOIN experiments e ON e.experiment_id = bt.experiment_id
                WHERE bt.strategy_id = ? AND bt.strategy_version = ?
                """,
                (strategy_id, version),
            )
            self._bt_cache[key] = df
        return self._bt_cache[key]

    def _filter_backtest(self, df: pd.DataFrame, d: pd.Timestamp | None, notes: list[str]) -> pd.DataFrame:
        cols = ["experiment_id", "exp_created_at", "segment", "symbol", "signal_date", "gross", "net",
                "regime", "synthetic", "origin"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df.copy()
        df["segment"] = df["segment"].fillna("full").astype(str)
        df["synthetic"] = df["synthetic"].fillna(0).astype(int).astype(bool)
        if not self.allow_synthetic:
            n_syn = int(df["synthetic"].sum())
            if n_syn:
                notes.append(f"{n_syn} backtest trades from SYNTHETIC experiments excluded")
            df = df[~df["synthetic"]]
        if self.require_succeeded:
            ok = (df["n_ok"] > 0) & (df["n_failed"] == 0)
            if (~ok).any():
                notes.append(f"{int((~ok).sum())} trades from experiments without a clean 'succeeded' result excluded")
            df = df[ok]
        sig = pd.to_datetime(df["signal_date"], errors="coerce")
        ext = pd.to_datetime(df["exit_date"], errors="coerce")
        if not self.include_holdout:
            in_holdout = df["segment"].eq("holdout")
            if self.holdout_start is not None:
                in_holdout = in_holdout | (sig >= self.holdout_start)
            if in_holdout.any():
                notes.append(f"{int(in_holdout.sum())} trades in the locked holdout excluded")
            df, sig, ext = df[~in_holdout], sig[~in_holdout], ext[~in_holdout]
        matured = ext.notna() & sig.notna()
        if d is not None:
            matured &= ext <= d
        df = df[matured]
        gross = pd.to_numeric(df["gross_ret"], errors="coerce")
        net = pd.to_numeric(df["net_ret"], errors="coerce")
        cost = pd.to_numeric(df["cost_ret"], errors="coerce")
        gross = gross.where(gross.notna(), net + cost)
        net = net.where(net.notna(), gross - cost)
        df = df.assign(gross=gross, net=net, origin="backtest",
                       signal_date=pd.to_datetime(df["signal_date"]).dt.normalize())
        df = df[np.isfinite(df["gross"].astype(float)) & np.isfinite(df["net"].astype(float))]
        return df[cols]

    @staticmethod
    def _latest_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Only the most recent experiment's trades: re-runs of the same walk-forward must not be
        stacked on top of each other, and "latest" is a rule that cannot be cherry-picked."""
        if df.empty:
            return df, []
        exps = df[["experiment_id", "exp_created_at"]].drop_duplicates().sort_values(
            ["exp_created_at", "experiment_id"])
        latest = str(exps.iloc[-1]["experiment_id"])
        return df[df["experiment_id"] == latest], [latest]

    def _shadow_rows(self, strategy_id: str, version: str) -> pd.DataFrame:
        key = (strategy_id, version)
        if key not in self._sh_cache:
            q = ",".join("?" for _ in _MATURED_SHADOW_STATUSES)
            self._sh_cache[key] = self.db.query_df(
                f"""
                SELECT so.opportunity_id, so.as_of_date, so.symbol, so.holding_sessions, so.is_synthetic,
                       o.horizon_sessions, o.ret, o.exit_date, o.measured_at, o.status, rs.label AS regime
                FROM shadow_opportunities so
                JOIN shadow_outcomes o ON o.opportunity_id = so.opportunity_id
                LEFT JOIN regime_snapshots rs ON rs.as_of_date = so.as_of_date
                WHERE so.strategy_id = ? AND so.strategy_version = ? AND o.status IN ({q})
                """,
                (strategy_id, version, *_MATURED_SHADOW_STATUSES),
            )
        return self._sh_cache[key]

    def _filter_shadow(self, df: pd.DataFrame, d: pd.Timestamp | None, notes: list[str]) -> pd.DataFrame:
        cols = ["experiment_id", "exp_created_at", "segment", "symbol", "signal_date", "gross", "net",
                "regime", "synthetic", "origin"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df.copy()
        df["synthetic"] = df["is_synthetic"].fillna(0).astype(int).astype(bool)
        if not self.allow_synthetic:
            if df["synthetic"].any():
                notes.append(f"{int(df['synthetic'].sum())} SYNTHETIC shadow outcomes excluded")
            df = df[~df["synthetic"]]
        # Same trade semantics as the plan: prefer the outcome measured at the plan's own holding
        # period; otherwise the longest horizon (the plan's final outcome).
        df = df.assign(_exact=(df["horizon_sessions"] == df["holding_sessions"]).astype(int))
        df = df.sort_values(["opportunity_id", "_exact", "horizon_sessions"]).groupby("opportunity_id").tail(1)
        if (df["_exact"] == 0).any():
            notes.append(f"{int((df['_exact'] == 0).sum())} shadow outcomes used their longest horizon "
                         "(no horizon equal to the plan's holding_sessions)")
        ext = pd.to_datetime(df["exit_date"], errors="coerce")
        measured = pd.to_datetime(df["measured_at"], errors="coerce", utc=True)
        ok = ext.notna() & pd.to_numeric(df["ret"], errors="coerce").notna()
        if d is not None:
            ok &= (ext <= d) & measured.notna() & (measured <= self._cutoff_utc(d))
        df = df[ok]
        r = pd.to_numeric(df["ret"], errors="coerce").astype(float)
        df = df.assign(experiment_id=None, exp_created_at=None, segment="shadow", gross=r, net=r,
                       origin="shadow", signal_date=pd.to_datetime(df["as_of_date"]).dt.normalize())
        df = df[np.isfinite(df["gross"])]
        return df[cols]
