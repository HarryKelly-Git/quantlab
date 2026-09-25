"""Universe engine: which symbols are tradable research candidates on each session.

Point-in-time reasoning:
  * Price and liquidity rules use RAW close and RAW dollar volume on or before D only, so a later
    reverse split cannot make a historical penny stock look investable.
  * History length counts bars up to D.
  * Security type / exchange come from the CURRENT reference snapshot (ASSUMED_STATIC). Symbols
    with an unknown type are EXCLUDED, never assumed to be common stock.
  * Survivorship: the universe can only contain symbols that have data. Whether delisted names
    are covered depends on the provider; :meth:`survivorship_status` reports what is known.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.types import PitStatus
from quantlab.data.panel import DataBundle

UNIVERSE_PIT_STATUS = PitStatus.ASSUMED_STATIC


def normalize_exclude(exclude) -> dict[str, pd.Timestamp | None]:
    """set -> excluded for all history; dict -> excluded from the given session on (None = all)."""
    if not exclude:
        return {}
    if isinstance(exclude, dict):
        return {k: (pd.Timestamp(v) if v is not None else None) for k, v in exclude.items()}
    return {k: None for k in exclude}

_TYPE_FLAGS = {
    "ETF": "exclude_etfs",
    "FUND": "exclude_funds_trusts",
    "PREFERRED": "exclude_preferred",
    "WARRANT": "exclude_warrants",
    "RIGHT": "exclude_rights",
    "UNIT": "exclude_units",
}


@dataclass(frozen=True)
class UniverseRules:
    allowed_exchanges: frozenset[str]
    min_price: float
    min_median_dollar_volume: float
    dollar_volume_window: int
    min_history_sessions: int
    max_symbols: int | None
    exclude_test_issues: bool
    excluded_types: frozenset[str]

    @classmethod
    def from_config(cls, config: Config) -> "UniverseRules":
        u = config.section("universe")
        excluded = {t for t, flag in _TYPE_FLAGS.items() if u.get(flag, True)}
        return cls(
            allowed_exchanges=frozenset(u.get("allowed_exchanges", [])),
            min_price=float(u["min_price"]),
            min_median_dollar_volume=float(u["min_median_dollar_volume"]),
            dollar_volume_window=int(u["dollar_volume_window"]),
            min_history_sessions=int(u["min_history_sessions"]),
            max_symbols=int(u["max_symbols"]) if u.get("max_symbols") else None,
            exclude_test_issues=bool(u.get("exclude_test_issues", True)),
            excluded_types=frozenset(excluded),
        )


class UniverseEngine:
    def __init__(self, config: Config):
        self.config = config
        self.rules = UniverseRules.from_config(config)

    # -- static (reference-based) eligibility -----------------------------------------------------
    def static_reasons(self, bundle: DataBundle, exclude=None) -> pd.Series:
        """symbol -> reason it can NEVER be in the universe ('' when statically eligible)."""
        r = self.rules
        exclude = {k for k, v in normalize_exclude(exclude).items() if v is None}
        benchmarks = {bundle.market_symbol, *bundle.sector_etfs.keys()}
        ref = bundle.reference
        if not ref.empty:
            ref = ref.sort_values("retrieved_at").drop_duplicates("symbol", keep="last").set_index("symbol")
        reasons = {}
        for sym in bundle.panel.symbols:
            if sym in benchmarks:
                reasons[sym] = "benchmark"
            elif exclude and sym in exclude:
                reasons[sym] = "data quarantine"
            elif ref.empty or sym not in ref.index:
                reasons[sym] = "no reference data (security type UNKNOWN)"
            else:
                row = ref.loc[sym]
                typ = str(row.get("security_type") or "UNKNOWN").upper()
                if bool(row.get("is_etf")) and "ETF" in r.excluded_types:
                    reasons[sym] = "ETF"
                elif r.exclude_test_issues and bool(row.get("is_test_issue")):
                    reasons[sym] = "test issue"
                elif typ in r.excluded_types:
                    reasons[sym] = f"security type {typ}"
                elif typ != "COMMON":
                    reasons[sym] = f"security type {typ} (only COMMON allowed)"
                elif r.allowed_exchanges and str(row.get("exchange")) not in r.allowed_exchanges:
                    reasons[sym] = f"exchange {row.get('exchange')} not allowed"
                else:
                    reasons[sym] = ""
        return pd.Series(reasons, dtype="object")

    # -- per-session rules ------------------------------------------------------------------------
    def _dynamic_masks(self, bundle: DataBundle) -> dict[str, pd.DataFrame]:
        r = self.rules
        p = bundle.panel
        mdv = p.dollar_volume.rolling(r.dollar_volume_window, min_periods=r.dollar_volume_window).median()
        return {
            "no bar on session": p.close.notna(),
            f"history < {r.min_history_sessions} sessions": p.history_length() >= r.min_history_sessions,
            f"raw close < {r.min_price:g}": p.close >= r.min_price,
            f"median dollar volume < {r.min_median_dollar_volume:,.0f}": mdv >= r.min_median_dollar_volume,
            "_mdv": mdv,  # carried for the max_symbols cap
        }

    def membership(self, bundle: DataBundle, exclude=None) -> pd.DataFrame:
        """Boolean (sessions x symbols) universe mask. Dated exclusions apply from their session on."""
        static_ok = self.static_reasons(bundle, exclude) == ""
        masks = self._dynamic_masks(bundle)
        mdv = masks.pop("_mdv")
        member = pd.DataFrame(True, index=bundle.panel.dates, columns=bundle.panel.symbols)
        for m in masks.values():
            member &= m.fillna(False)
        member &= pd.DataFrame(np.broadcast_to(static_ok.reindex(member.columns).fillna(False).to_numpy(), member.shape),
                               index=member.index, columns=member.columns)
        for sym, since in normalize_exclude(exclude).items():
            if since is not None and sym in member.columns:
                member.loc[member.index >= since, sym] = False
        if self.rules.max_symbols:
            rank = mdv.where(member).rank(axis=1, ascending=False, method="first")
            member &= rank <= self.rules.max_symbols
        return member

    def explain(self, bundle: DataBundle, as_of, exclude=None) -> pd.DataFrame:
        """One row per symbol: included flag and the first failing rule (human-readable)."""
        d = pd.Timestamp(as_of)
        static = self.static_reasons(bundle, exclude)
        masks = self._dynamic_masks(bundle)
        masks.pop("_mdv")
        member_row = self.membership(bundle, exclude).loc[d]
        rows = []
        for sym in bundle.panel.symbols:
            reason = static.get(sym, "")
            if not reason:
                for name, m in masks.items():
                    if not bool(m.at[d, sym]):
                        reason = name
                        break
            since = normalize_exclude(exclude).get(sym)
            if not reason and since is not None and d >= since:
                reason = f"data quarantine from {since.date()}"
            if not reason and not member_row[sym]:
                reason = f"outside top {self.rules.max_symbols} by liquidity"
            rows.append({"symbol": sym, "included": bool(member_row[sym]), "reason": reason or "included"})
        return pd.DataFrame(rows)

    def survivorship_status(self, bundle: DataBundle) -> dict:
        """What we can say about survivorship bias in this bundle (never claims 'free' without proof)."""
        close = bundle.panel.close
        last = close.apply(lambda s: s.last_valid_index())
        ended_early = int((last < close.index[-1]).sum())
        inactive = 0
        if not bundle.reference.empty and "status" in bundle.reference.columns:
            inactive = int((bundle.reference["status"].astype(str).str.lower() == "inactive").sum())
        status = "PARTIAL" if (ended_early or inactive) else "UNKNOWN"
        return {"status": status, "symbols_ending_before_panel_end": ended_early,
                "inactive_in_reference": inactive,
                "note": "Coverage of delisted securities is provider-dependent; results may be survivorship-biased."}

    def snapshot(self, bundle: DataBundle, as_of, db=None, run_id: str | None = None,
                 exclude=None) -> pd.DataFrame:
        """explain() for one session, optionally persisted to universe_snapshots."""
        df = self.explain(bundle, as_of, exclude)
        if db is not None:
            d = pd.Timestamp(as_of).date().isoformat()
            db.insert_many("universe_snapshots", [
                {"as_of_date": d, "symbol": r.symbol, "included": int(r.included), "reason": r.reason, "run_id": run_id}
                for r in df.itertuples()], or_ignore=True)
        return df
