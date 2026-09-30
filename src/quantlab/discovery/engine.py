"""Market discovery engine: FULL UNIVERSE -> BASIC DATA/LIQUIDITY -> DISCOVERY FEATURES -> DISCOVERY
CANDIDATES -> TOP-N RANKING -> WATCHLIST -> VALIDATION -> PAPER ELIGIBLE -> PAPER TRADE.

Discovery asks "what looks interesting in today's market?". Validation asks "does this setup have
enough evidence to permit a paper trade?". This module only ANSWERS THE FIRST QUESTION and then
REPORTS what the existing, unchanged validation chain (research universe, strategy validation, the
decision chain's no-trade/EV/portfolio/risk/execution gates, the kill switch) decided. It never
creates, sizes or submits an order, and never changes a gate.

Point-in-time: :meth:`DiscoveryEngine.scan` works on ``bundle.truncate(as_of)`` only (asserted by
a truncation-invariance test). Features come from the existing FeatureSet on the last
``discovery.lookback_sessions`` sessions of that view (every feature used needs <= 273 sessions).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import Config
from quantlab.core.calendar import to_session
from quantlab.core.types import PitStatus, new_id
from quantlab.data.panel import DataBundle, Panel
from quantlab.db.database import Database, from_json, to_json, utcnow_iso
from quantlab.discovery.families import (
    AUX_FEATURES, CONTEXT, CONTEXT_FEATURES, FIRE, LABELS, POINTS, SCORED, SCORED_FEATURES, SCORED_POINTS, SOURCES,
    Triggers, fire_context, fired_masks, pct_rank,
)
from quantlab.discovery.status import describe_strategy
from quantlab.features.base import FeatureSet
from quantlab.logging_setup import get_logger, log_event

log = get_logger(__name__)

VALID, UNKNOWN, INVALID = "VALID", "UNKNOWN", "INVALID"
# block categories for "why no paper trades"
BLOCKERS = {
    "DATA": "data-quality issue",
    "UNIVERSE": "fails the research universe / liquidity rules",
    "NO_STRATEGY_COVERAGE": "no strategy signals this setup (discovery only)",
    "STRATEGY_VALIDATION": "strategy not validated or promoted for paper trading",
    "DECISION_GATE": "rejected by the decision chain (no-trade rules, EV gate, portfolio, risk)",
    "SYSTEM_PAUSED": "system paused (kill switch)",
    "EXECUTION": "order not placed (session window, broker, risk limits)",
    "VALIDATION_NOT_RUN": "no pipeline decisions for this session yet",
}


@dataclass
class DiscoverySettings:
    lookback_sessions: int = 300
    min_price: float = 1.0
    min_median_dollar_volume: float = 1_000_000.0
    min_history_sessions: int = 60
    high_rank_score: float = 70.0
    min_score_coverage: float = 0.8
    watchlist_size: int = 25
    near_miss_count: int = 15
    context_min_coverage: float = 0.8
    news_max_age_days: float = 3.0
    min_discovery_rate: float = 0.005
    max_discovery_rate: float = 0.5
    one_rule_share: float = 0.95
    min_feature_coverage: float = 0.8
    outcome_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    triggers: Triggers = field(default_factory=Triggers)

    @classmethod
    def from_config(cls, config: Config) -> "DiscoverySettings":
        c = config.section("discovery") or {}
        b, dg = c.get("basic_filter", {}) or {}, c.get("diagnostics", {}) or {}
        return cls(int(c.get("lookback_sessions", 300)), float(b.get("min_price", 1.0)),
                   float(b.get("min_median_dollar_volume", 1e6)), int(b.get("min_history_sessions", 60)),
                   float(c.get("high_rank_score", 70)), float(c.get("min_score_coverage", 0.8)),
                   int(c.get("watchlist_size", 25)), int(c.get("near_miss_count", 15)),
                   float(c.get("context_min_coverage", 0.8)), float(c.get("news_max_age_days", 3)),
                   float(dg.get("min_discovery_rate", 0.005)), float(dg.get("max_discovery_rate", 0.5)),
                   float(dg.get("one_rule_share", 0.95)), float(dg.get("min_feature_coverage", 0.8)),
                   tuple(int(h) for h in c.get("outcome_horizons", (1, 3, 5, 10, 20))), Triggers.from_config(c))

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["triggers"] = dict(self.triggers.__dict__)
        return d


@dataclass
class ScanResult:
    as_of: pd.Timestamp
    table: pd.DataFrame                      # one row per basic-filter symbol
    counts: dict[str, int]
    coverage: list[dict[str, Any]]           # feature-coverage panel
    market_context: dict[str, Any]
    is_synthetic: bool
    dataset_ids: list[str] = field(default_factory=list)
    not_scanned: dict[str, str] = field(default_factory=dict)   # symbol -> why it failed the basic filter
    calendar: Any = None
    catalyst_panel: dict[str, Any] = field(default_factory=dict)  # CATALYSTS panel at D (dashboard)
    basic_symbols: list[str] = field(default_factory=list)        # symbols that passed the basic filter
    source_coverage: dict[str, Any] | None = None                 # catalyst data-source coverage at scan time


@dataclass
class Assessment:
    candidates: list[dict[str, Any]]
    funnel: dict[str, Any]
    blockers: dict[str, Any]
    near_misses: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


def tail_bundle(view: DataBundle, n: int) -> DataBundle:
    """The last ``n`` sessions of an already-truncated view (features only look back <= n)."""
    if len(view.panel.dates) <= n:
        return view
    p = Panel({k: v.iloc[-n:] for k, v in view.panel.fields.items()}, dict(view.panel.meta))
    return replace(view, panel=p)


def _finite(x: Any) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _num(x: Any) -> float | None:
    return float(x) if _finite(x) else None


def _levels(r: pd.Series, close: Any) -> dict[str, Any]:
    """Raw-price reference levels at D for conditional next-session setups (derived from ratios at D,
    so they are in D's raw price units). Missing inputs stay None (UNKNOWN)."""
    c = _num(close)
    if c is None:
        return {}
    def lvl(ratio_name: str) -> float | None:
        v = _num(r.get(ratio_name))
        return round(c / (1 + v), 4) if v is not None and v > -1 else None
    atr = _num(r.get("atr14_pct"))
    loc = _num(r.get("close_location"))
    return {"close": round(c, 4), "atr": round(atr * c, 4) if atr is not None else None,
            "ma20": lvl("dist_ma20"), "ma50": lvl("dist_ma50"), "high_55_prior": lvl("breakout_55"),
            "close_location": loc}


class DiscoveryEngine:
    def __init__(self, config: Config):
        self.config = config
        self.s = DiscoverySettings.from_config(config)

    # ------------------------------------------------------------------------------------------
    # scan (pure, point-in-time)
    # ------------------------------------------------------------------------------------------
    def core(self, panel: Panel, fs: FeatureSet, d, benches: set) -> dict[str, Any]:
        """Scores and fired families at session ``d`` (shared by the live scan and the research
        replay). ``fs`` may span sessions after ``d`` (features are causal by contract); everything
        derived here reads only rows <= d."""
        p = panel.truncate(d)
        close = p.close.iloc[-1]
        hist = p.history_length().iloc[-1]
        adv20 = fs.get("adv20").loc[d]
        basic = (close.notna() & (close >= self.s.min_price) & (adv20 >= self.s.min_median_dollar_volume)
                 & (hist >= self.s.min_history_sessions) & ~close.index.isin(list(benches)))
        syms = close.index[basic.fillna(False).to_numpy()]
        scored_feats = [f for fam in SCORED for f, _ in SCORED_FEATURES[fam]]
        need = list(dict.fromkeys(scored_feats + list(AUX_FEATURES)))
        xs = fs.cross_section(d, need).loc[syms]
        ac, ah, al = p.aclose, p.ahigh, p.alow

        def _new_high(n: int) -> pd.Series:
            w = ah.iloc[-(n + 1):-1]
            full = (w.notna().sum() == n) & ac.iloc[-1].notna() if len(w) == n else pd.Series(False, index=ac.columns)
            return (ac.iloc[-1] >= w.max()).astype(float).where(full).reindex(syms)
        xs["new_high_20"] = _new_high(20)
        xs["new_high_50"] = _new_high(50)
        two = len(ac) >= 2
        prev_c = ac.iloc[-2] if two else pd.Series(np.nan, index=ac.columns)
        tr = pd.concat([ah.iloc[-1], prev_c], axis=1).max(axis=1) - pd.concat([al.iloc[-1], prev_c], axis=1).min(axis=1)
        prev_d = p.dates[-2] if two else None
        atr_prev = fs.get("atr14_pct").loc[prev_d] * prev_c if two else np.nan
        xs["range_expansion"] = (tr / atr_prev).replace([np.inf, -np.inf], np.nan).reindex(syms)
        xs["prev_contraction"] = (fs.get("range_contraction_20_60").loc[prev_d] if two
                                  else pd.Series(np.nan, index=ac.columns)).reindex(syms)
        rng = (ah.iloc[-1] - al.iloc[-1])
        xs["close_location"] = ((ac.iloc[-1] - al.iloc[-1]) / rng.where(rng > 0)).reindex(syms)
        xs["accel"] = xs["ret_20d"] - xs["ret_60d"] / 3.0

        states = pd.DataFrame(index=syms)
        for f in scored_feats:
            lb = fs.registry.spec(f).lookback
            states[f] = np.where(np.isfinite(xs[f].to_numpy(dtype=float)), VALID,
                                 np.where(hist.reindex(syms).to_numpy() <= lb, UNKNOWN, INVALID))
        pct = pd.DataFrame(index=syms)
        for fam in SCORED:
            for f, sign in SCORED_FEATURES[fam]:
                pct[f] = pct_rank(xs[f] * sign)
        comp = pd.DataFrame(index=syms)
        for fam in SCORED:
            comp[fam] = POINTS * pct[[f for f, _ in SCORED_FEATURES[fam]]].mean(axis=1, skipna=True)
        # coverage counts data availability across ALL scored families; the score is the mean of the
        # families that contribute points, which excludes relative_strength because its percentile
        # rank is momentum's (see families.SCORED_POINTS)
        n_known = comp.notna().sum(axis=1)
        pts = comp[list(SCORED_POINTS)]
        n_pts = pts.notna().sum(axis=1)
        score = 100.0 * pts.sum(axis=1, min_count=1) / (POINTS * n_pts.replace(0, np.nan))
        pct_raw = pd.DataFrame({f: pct_rank(xs[f]) for f in scored_feats}, index=syms)
        masks = fired_masks(xs, pct_raw, comp, self.s.triggers)
        return {"syms": syms, "xs": xs, "states": states, "comp": comp, "score": score,
                "coverage": n_known / len(SCORED), "pct_raw": pct_raw, "masks": masks, "close": close, "hist": hist,
                "adv20": adv20, "p": p, "scored_feats": scored_feats,
                "counts": {"bar_on_session": int(close.notna().sum()), "basic": int(len(syms))}}

    def scan(self, bundle: DataBundle, as_of, quarantine: dict | None = None) -> ScanResult:
        from quantlab.universe import UniverseEngine

        d = to_session(as_of)
        view = bundle if (bundle.as_of is not None and bundle.as_of == d) else bundle.truncate(d)
        if len(view.panel.dates) == 0 or view.panel.dates[-1] != d:
            raise ValueError(f"{d.date()} is not a session in the data")
        tb = tail_bundle(view, self.s.lookback_sessions)
        fs = FeatureSet(tb, dtype="float32")        # ~6 MB per feature frame on the real universe
        c = self.core(tb.panel, fs, d, {view.market_symbol, *view.sector_etfs})
        syms, xs, states, comp, pct_raw, masks = c["syms"], c["xs"], c["states"], c["comp"], c["pct_raw"], c["masks"]
        counts = {"full_universe": int(len(view.panel.symbols)), **c["counts"]}

        # reason text + descriptive bias (firing itself comes from the vectorised masks)
        fired, reasons, bias = {}, {}, {}
        for s in syms:
            r, pr = xs.loc[s], pct_raw.loc[s]
            fired[s] = [fam for fam in SCORED if masks.at[s, fam]]
            reasons[s], votes = {}, collections.Counter()
            for fam in SCORED:
                if pd.isna(comp.at[s, fam]):
                    continue
                _, why, b = FIRE[fam](r, pr, self.s.triggers)
                if why:
                    reasons[s][fam] = why
                if fam in fired[s]:
                    votes[b] += 1
            bias[s] = ("NEUTRAL" if not votes or (votes["BULLISH"] == votes["BEARISH"])
                       else ("BULLISH" if votes["BULLISH"] > votes["BEARISH"] else "BEARISH"))

        context, ctx_cov = self._context(view, tb, fs, d, syms)
        from quantlab.discovery.catalysts import CatalystEngine
        cat_eng = CatalystEngine(self.config)
        cats = cat_eng.evaluate(view, fs, d, syms, technical={s: {"fired": fired[s], "bias": bias[s]} for s in syms})
        uni = UniverseEngine(self.config).explain(view, d, exclude=quarantine).set_index("symbol")["reason"]
        table = pd.DataFrame(index=syms)
        table["score"] = c["score"]
        table["coverage"] = c["coverage"]
        for fam in SCORED:
            table[f"c_{fam}"] = comp[fam]
        table["fired"] = pd.Series(fired, dtype=object)
        table["reasons"] = pd.Series(reasons, dtype=object)
        table["bias"] = pd.Series(bias, dtype=object)
        table["context"] = pd.Series(context, dtype=object)
        table["universe_reason"] = uni.reindex(syms).fillna("no universe row")
        table["close"] = c["close"].reindex(syms)
        table["adv20"] = c["adv20"].reindex(syms)
        table["history"] = c["hist"].reindex(syms)
        table["factors"] = pd.Series({s: self._factors(s, xs, states, pct_raw, d, fs) for s in syms}, dtype=object)
        table["invalid"] = pd.Series({s: [f for f in c["scored_feats"] if states.at[s, f] == INVALID] for s in syms},
                                     dtype=object)
        table["unknown"] = pd.Series({s: [f for f in c["scored_feats"] if states.at[s, f] == UNKNOWN] for s in syms},
                                     dtype=object)
        table["levels"] = pd.Series({s: _levels(xs.loc[s], c["close"].get(s)) for s in syms}, dtype=object)
        table["catalyst"] = pd.Series(cats, dtype=object)
        table["catalyst_fired"] = pd.Series({s: list(cats[s]["families"]) if s in cats else [] for s in syms}, dtype=object)
        # why symbols outside the basic filter were not scanned (a strategy signal can still enter the pool)
        p = c["p"]
        not_scanned: dict[str, str] = {}
        for s in view.panel.symbols:
            if s in table.index:
                continue
            cl = c["close"].get(s)
            if s in {view.market_symbol, *view.sector_etfs}:
                not_scanned[s] = "benchmark ETF"
            elif cl is None or not _finite(cl):
                not_scanned[s] = "no bar on session"
            elif cl < self.s.min_price:
                not_scanned[s] = f"price < ${self.s.min_price:g}"
            elif not _finite(c["adv20"].get(s)) or c["adv20"].get(s) < self.s.min_median_dollar_volume:
                not_scanned[s] = f"median dollar volume < ${self.s.min_median_dollar_volume:,.0f}"
            else:
                not_scanned[s] = f"history < {self.s.min_history_sessions} sessions"
        cov_rows = self._coverage(counts, table, states, fs, ctx_cov, xs)
        res = ScanResult(d, table, counts, cov_rows, self._market_context(p, view), bool(view.is_synthetic),
                         list(view.dataset_ids))
        res.not_scanned = not_scanned
        res.calendar = view.calendar
        res.catalyst_panel = cat_eng.panel(view, fs, d, syms, cats)
        res.basic_symbols = list(syms)
        return res

    def _factors(self, s: str, xs: pd.DataFrame, states: pd.DataFrame, pct: pd.DataFrame, d, fs) -> dict[str, Any]:
        out = {}
        for fam in SCORED:
            for f, _ in SCORED_FEATURES[fam]:
                spec = fs.registry.spec(f)
                v = xs.at[s, f]
                out[f] = {"value": _num(v), "state": states.at[s, f], "pct": _num(pct.at[s, f]),
                          "family": fam, "source": spec.source, "pit_status": spec.pit_status.value,
                          "as_of": str(d.date())}
        for f in ("new_high_20", "new_high_50", "range_expansion", "prev_contraction", "close_location", "ret_1d",
                  "dist_ma50", "ma50_over_ma200", "ret_z_1d", "adv20", "atr14_pct"):
            v = xs.at[s, f]
            val = bool(v) if isinstance(v, (bool, np.bool_)) else _num(v)
            out[f] = {"value": val, "state": VALID if val is not None else UNKNOWN, "source": "derived from panel a-fields",
                      "pit_status": PitStatus.PIT.value, "as_of": str(d.date())}
        return out

    def _context(self, view: DataBundle, tb: DataBundle, fs: FeatureSet, d, syms) -> tuple[dict, dict]:
        """Context families: KNOWN only where the source covers the symbol. Never 0 for missing."""
        cutoff = view.calendar.cutoff(d)
        dates = tb.panel.dates
        base_start = view.calendar.cutoff(dates[max(0, len(dates) - 61)])
        covered: dict[str, set] = {}
        why: dict[str, str] = {}
        why_sym: dict[tuple[str, str], str] = {}
        ev = view.events
        covered["earnings"] = set()
        if not ev.empty:
            recent = ((pd.to_datetime(ev["available_at"], utc=True) >= view.calendar.cutoff(dates[max(0, len(dates) - 101)]))
                      & (ev["event_type"] == "earnings_release"))
            covered["earnings"] = set(ev.loc[recent, "symbol"])
            for s_ in set(ev["symbol"]) - covered["earnings"]:
                why_sym[("earnings", s_)] = "no earnings event in the last 100 sessions: coverage unknown"
        why["earnings"] = "no SEC 8-K earnings events ingested for this symbol"
        nw = view.news
        covered["news"] = set()
        why["news"] = "no news ingested for this symbol" if not nw.empty else "no news dataset"
        if not nw.empty:
            cov = news_coverage(nw, base_start, cutoff, self.s.news_max_age_days)
            covered["news"] = {s_ for s_, (ok, _) in cov.items() if ok}
            for s_, (ok, w) in cov.items():
                if not ok:
                    why_sym[("news", s_)] = w
        covered["fundamentals"] = set(view.fundamentals["symbol"]) if not view.fundamentals.empty else set()
        why["fundamentals"] = "no fundamentals dataset for this symbol"
        sic_now = fs.cross_section(d, ["sic_code_asof"])["sic_code_asof"] if "sic_code_asof" in fs.registry else None
        covered["sector"] = set(sic_now.index[sic_now.notna()]) if sic_now is not None else set()
        why["sector"] = "no SIC observed in a filing header by the decision time"

        vals: dict[str, pd.DataFrame] = {}
        for fam in CONTEXT:
            cov_syms = [s for s in syms if s in covered[fam]]
            if cov_syms:
                vals[fam] = fs.cross_section(d, list(CONTEXT_FEATURES[fam])).loc[cov_syms]
        out: dict[str, dict] = {}
        items_by_sym = self._news_items(nw, cutoff) if covered["news"] else {}
        for s in syms:
            c = {}
            for fam in CONTEXT:
                if fam in vals and s in vals[fam].index:
                    r = vals[fam].loc[s]
                    known = any(_finite(r[f]) for f in CONTEXT_FEATURES[fam])
                    if not known:
                        c[fam] = {"state": UNKNOWN, "why": "source covers the symbol but the feature is not computable yet"}
                        continue
                    ok, rs = fire_context(fam, r, self.s.triggers)
                    c[fam] = {"state": "KNOWN", "fired": ok, "reasons": rs,
                              "values": {f: _num(r[f]) for f in CONTEXT_FEATURES[fam]}}
                    if fam == "news" and ok:
                        c[fam]["items"] = items_by_sym.get(s, [])
                else:
                    c[fam] = {"state": UNKNOWN, "why": why_sym.get((fam, s), why[fam])}
            out[s] = c
        cov = {fam: {"known": sum(1 for s in syms if out[s][fam]["state"] == "KNOWN"), "n": len(syms)} for fam in CONTEXT}
        return out, cov

    @staticmethod
    def _news_items(nw: pd.DataFrame, cutoff) -> dict[str, list[dict[str, Any]]]:
        """Recent items with provenance. A headline revised after the cutoff is withheld: the stored
        text is the provider's latest version, which was not knowable at decision time."""
        recent = nw[pd.to_datetime(nw["available_at"], utc=True) > cutoff - pd.Timedelta(days=2)]
        out: dict[str, list] = {}
        for r in recent.sort_values("available_at", ascending=False).itertuples():
            lst = out.setdefault(r.symbol, [])
            if len(lst) < 3:
                upd = pd.to_datetime(getattr(r, "updated_at", None), utc=True, errors="coerce")
                revised = upd is not None and not pd.isna(upd) and upd > cutoff
                lst.append({"headline": None if revised else getattr(r, "headline", None),
                            "headline_note": "revised after the cutoff: text withheld" if revised else None,
                            "available_at": str(r.available_at), "updated_at": str(getattr(r, "updated_at", "")),
                            "pit_status": getattr(r, "pit_status", None), "source": getattr(r, "source", None),
                            "news_id": str(getattr(r, "news_id", ""))})
        return out

    def _coverage(self, counts, table, states, fs, ctx_cov, xs) -> list[dict[str, Any]]:
        n = max(counts["basic"], 1)
        rows = [{"key": "price_volume", "label": "Price/volume",
                 "coverage": counts["bar_on_session"] / max(counts["full_universe"], 1),
                 "known": counts["bar_on_session"], "n": counts["full_universe"], "invalid": 0,
                 "pit_status": PitStatus.PIT.value, "source": SOURCES["price_volume"], "scored": False}]
        for fam in SCORED:
            feats = [f for f, _ in SCORED_FEATURES[fam]]
            known = int(table[f"c_{fam}"].notna().sum())
            invalid = int((states[feats] == INVALID).any(axis=1).sum())
            rows.append({"key": fam, "label": LABELS[fam], "coverage": known / n, "known": known, "n": n,
                         "invalid": invalid, "pit_status": fs.pit_status(feats).value, "source": SOURCES[fam],
                         "scored": True})
        for fam in CONTEXT:
            c = ctx_cov[fam]
            rows.append({"key": fam, "label": LABELS[fam], "coverage": c["known"] / n, "known": c["known"], "n": n,
                         "invalid": 0, "pit_status": fs.pit_status(list(CONTEXT_FEATURES[fam])).value,
                         "source": SOURCES[fam], "scored": False})
        for r in rows:
            r["usable"] = bool(r["coverage"] >= (self.s.min_feature_coverage if r["key"] not in CONTEXT
                                                  else self.s.context_min_coverage))
            r["role"] = ("scored" if r["scored"] else ("context only (never scored)" if r["key"] in CONTEXT
                                                       else "input"))
        return rows

    @staticmethod
    def _market_context(p: Panel, view: DataBundle) -> dict[str, Any]:
        """Sector ETF 63-session returns vs SPY. Market context only: stocks are NOT mapped to
        sectors in the real data, so stock-level sector leadership stays UNKNOWN."""
        mkt = view.market_symbol
        ac = p.aclose
        if mkt not in ac.columns or len(ac) < 64:
            return {"sector_etfs": [], "note": "not enough history"}
        spy = ac[mkt].iloc[-1] / ac[mkt].iloc[-64] - 1
        if not _finite(spy):
            return {"sector_etfs": [], "note": "SPY 63-session return unavailable (missing bar)"}
        rows = []
        for etf, name in view.sector_etfs.items():
            if etf in ac.columns and _finite(ac[etf].iloc[-1]) and _finite(ac[etf].iloc[-64]):
                r = ac[etf].iloc[-1] / ac[etf].iloc[-64] - 1
                rows.append({"etf": etf, "sector": name, "ret_63": float(r), "vs_spy": float(r - spy)})
        rows.sort(key=lambda x: -x["vs_spy"])
        return {"sector_etfs": rows, "spy_ret_63": float(spy),
                "note": "ETF-level context only; stock-to-sector mapping is not available"}

    # ------------------------------------------------------------------------------------------
    # assessment: what the unchanged validation chain decided for each discovered setup
    # ------------------------------------------------------------------------------------------
    def assess(self, scan: ScanResult, links: dict[str, list[dict[str, Any]]] | None,
               strategies: dict[str, dict[str, Any]]) -> Assessment:
        """MASTER CANDIDATE POOL = discovery setups UNION strategy signals. A strategy signal enters
        the pool (and the unchanged validation chain decides it) whatever its discovery score:
        discovery ranking never suppresses a strategy signal."""
        from quantlab.discovery.catalysts import CatalystTriggers, catalyst_text, evidence_chain, setup_class
        t = scan.table
        fired_syms = set(t.index[t["fired"].map(len) > 0])
        cat_syms = set(t.index[t["catalyst_fired"].map(len) > 0]) if "catalyst_fired" in t.columns else set()
        disc_syms = fired_syms | cat_syms
        linked = set(links or {})
        rows = []
        for sym in disc_syms | linked:
            r = t.loc[sym].copy() if sym in t.index else self._unscanned_row(sym, scan)
            r["origin"] = "BOTH" if (sym in disc_syms and sym in linked) else ("DISCOVERY" if sym in disc_syms
                                                                                else "STRATEGY")
            rows.append((sym, r))
        rows.sort(key=lambda x: (-(x[1]["score"]) if _finite(x[1]["score"]) else float("inf"), x[0]))
        cands: list[dict[str, Any]] = []
        for rank, (sym, r) in enumerate(rows, start=1):
            r["rank"] = rank
            high = bool(r["origin"] != "STRATEGY" and _finite(r["score"]) and r["score"] >= self.s.high_rank_score
                        and _finite(r["coverage"]) and r["coverage"] >= self.s.min_score_coverage)
            c = self._assess_one(sym, r, high, links, strategies)
            c["origin"] = r["origin"]
            c["scanned"] = bool(r.get("scanned", True))
            c["levels"] = r.get("levels") if isinstance(r.get("levels"), dict) else {}
            cat = r.get("catalyst") if isinstance(r.get("catalyst"), dict) else {}
            c["catalyst"] = cat
            c["catalyst_fired"] = list(cat.get("families") or [])
            c["setup_class"] = setup_class(bool(c["fired"] or c["links"]), bool(c["catalyst_fired"]),
                                           bool(cat.get("catalyst_known")))
            cands.append(c)
        # watchlist: top-N high-ranked setups that passed DATA and UNIVERSE validation
        wl = 0
        for c in cands:
            if c["high_quality"] and not c["data_or_universe_failed"] and wl < self.s.watchlist_size:
                c["on_watchlist"] = True
                wl += 1
            else:
                c["on_watchlist"] = False
        # market-confirmed catalyst setups with a POSITIVE price response (QuantLab is long-only) get a
        # separate, capped allowance; ordered by independent evidence agreement, then discovery score
        cap = CatalystTriggers.from_config(self.config).watchlist_size
        cw = [c for c in cands if c["catalyst_fired"] and not c["on_watchlist"] and not c["data_or_universe_failed"]
              and _cat_direction(c["catalyst"]) == "POSITIVE"]
        cw.sort(key=lambda c: (-(c["catalyst"]["agreement"]["n_supporting"] - c["catalyst"]["agreement"]["n_contradicting"]),
                               -(c["score"] or 0.0), c["symbol"]))
        for c in cw[:cap]:
            c["on_watchlist"] = True
            c["catalyst_watch"] = True
        for c in cands:
            c["status"] = self._status(c)
            c["setup"] = setup_record(c)
            if c["catalyst"]:
                extra = catalyst_text(c["catalyst"], c["levels"])
                for k in ("why", "confirm", "invalidate"):
                    c["setup"][k] = extra[k] + c["setup"][k]
                c["setup"]["missing"] = extra["missing"] + [m for m in c["setup"]["missing"] if m not in extra["missing"]]
            c["setup"]["setup_class"] = c["setup_class"]
            c["chain"] = evidence_chain(c)
        return Assessment(cands, self._funnel(scan, cands, links), self._blockers(cands),
                          self._near_misses(cands), self._diagnostics(scan, cands))

    def _unscanned_row(self, sym: str, scan: ScanResult) -> pd.Series:
        """A strategy signal on a symbol outside the discovery scan: no score (UNKNOWN), and the
        reason it was not scanned. It still enters the pool."""
        why = scan.not_scanned.get(sym, "not in the stored data for this session")
        r = {"score": np.nan, "coverage": 0.0, **{f"c_{f}": np.nan for f in SCORED}, "fired": [], "reasons": {},
             "bias": "NEUTRAL", "context": {f: {"state": UNKNOWN, "why": f"not in the discovery scan ({why})"}
                                            for f in CONTEXT},
             "universe_reason": f"not scanned: {why}", "close": np.nan, "adv20": np.nan, "history": np.nan,
             "factors": {}, "invalid": [], "unknown": [], "levels": {}, "scanned": False, "not_scanned_reason": why,
             "catalyst": {}, "catalyst_fired": []}
        return pd.Series(r, dtype=object)

    def _assess_one(self, s, r, high, links, strategies) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def check(stage, name, passed, reason, **extra):
            checks.append({"stage": stage, "name": name, "passed": bool(passed), "reason": reason, **extra})

        for fam in r["fired"]:
            check("DISCOVERY", fam, True, "; ".join(r["reasons"].get(fam, [])) or LABELS[fam])
        for fam in (r.get("catalyst_fired") or []):
            check("DISCOVERY", fam, True, _cat_reason(r.get("catalyst") or {}, fam))
        inv = list(r["invalid"])
        if r.get("scanned", True) is False:
            # the decision chain already applied its own data/universe checks to this strategy signal
            check("DISCOVERY", "discovery_scan", True, f"not in the discovery scan ({r.get('not_scanned_reason')}); "
                  "entered the pool as a strategy signal")
        else:
            check("DATA", "feature_quality", not inv,
                  "all available features valid" if not inv else f"data-quality issue: {', '.join(inv)} not finite "
                  f"despite {int(r['history'])} sessions of history")
            ur = str(r["universe_reason"])
            check("UNIVERSE", "research_universe", ur == "included",
                  "passes the research universe (price, liquidity, history, security type)" if ur == "included" else ur)
        lk = (links or {}).get(s, []) if links is not None else None
        if lk is None:
            check("STRATEGY_COVERAGE", "strategy_coverage", False, "no pipeline decisions for this session yet",
                  block="VALIDATION_NOT_RUN")
        elif not lk:
            check("STRATEGY_COVERAGE", "strategy_coverage", False, BLOCKERS["NO_STRATEGY_COVERAGE"],
                  block="NO_STRATEGY_COVERAGE")
        else:
            check("STRATEGY_COVERAGE", "strategy_coverage", True,
                  "signalled by " + ", ".join(sorted({x["strategy_id"] for x in lk})))
            best = max(lk, key=lambda x: _progress(x))
            info = _strategy_info(strategies, best["strategy_id"], best.get("strategy_version"))
            st = (info or {}).get("status", "UNKNOWN")
            check("STRATEGY_VALIDATION", "strategy_status", st == "PAPER_ELIGIBLE", describe_strategy(info),
                  strategy_id=best["strategy_id"], strategy_status=st)
            dec, stage = best.get("decision"), best.get("reject_stage")
            ev = best.get("ev_bps")
            ev_txt = f"EV {ev:+.0f} bps after costs" if ev is not None else "EV unknown"
            if dec == "TRADE":
                check("DECISION_GATE", "decision", True, f"TRADE ({ev_txt})")
                os_ = best.get("order_status")
                if os_ in PLACED_ORDER_STATUSES:
                    check("EXECUTION", "order", True, f"paper order {os_}")
                else:
                    why_ = best.get("refusal") or (f"paper order {os_}" if os_ else "no order placed")
                    check("EXECUTION", "order", False, why_, block="EXECUTION", rule=f"order:{os_ or 'none'}")
            elif dec:
                reasons_ = best.get("reasons") or ""
                rule = reasons_.split(":")[0].strip() if ":" in reasons_ else (stage or "")
                if stage == "STRATEGY":
                    gate = "STRATEGY_VALIDATION"
                elif stage == "EXECUTION" and "risk.system_state" in reasons_:
                    gate = "SYSTEM_PAUSED"
                elif stage == "EXECUTION":
                    gate = "EXECUTION"
                else:
                    gate = "DECISION_GATE"
                check("DECISION_GATE", "decision", False, f"{dec} at {stage}: {reasons_} ({ev_txt})".strip(),
                      reject_stage=stage, block=gate, rule=rule)
        failed = [c for c in checks if not c["passed"]]
        first = failed[0] if failed else None
        block = None
        if first:
            block = first.get("block") or {"DATA": "DATA", "UNIVERSE": "UNIVERSE",
                                           "STRATEGY_VALIDATION": "STRATEGY_VALIDATION",
                                           "EXECUTION": "EXECUTION"}.get(first["stage"], "DECISION_GATE")
        ctx = r["context"]
        block_key = None
        if first:
            block_key = f"{block}|{first.get('rule') or first['name']}|{first.get('reject_stage') or ''}"
            if first["stage"] == "UNIVERSE":
                block_key = f"{block}|{first['reason']}"          # universe reasons are already rule names
        return {
            "symbol": s, "score": _num(r["score"]), "coverage": _num(r["coverage"]), "rank": int(r["rank"]),
            "components": {fam: _num(r[f"c_{fam}"]) for fam in SCORED}, "fired": list(r["fired"]),
            "reasons": r["reasons"], "bias": r["bias"], "context": ctx, "factors": r["factors"],
            "unknown_features": r["unknown"], "invalid_features": inv, "high_quality": high,
            "data_or_universe_failed": any(not c["passed"] and c["stage"] in ("DATA", "UNIVERSE") for c in checks),
            "checks": checks, "links": lk or [], "block_stage": block,
            "block_reason": (first["reason"] if first else None), "block_key": block_key,
            "unknown_context": [f for f in CONTEXT if (ctx.get(f) or {}).get("state") != "KNOWN"],
        }

    @staticmethod
    def _status(c: dict[str, Any]) -> str:
        checks = {x["stage"]: x for x in c["checks"]}
        ex = checks.get("EXECUTION")
        dg = checks.get("DECISION_GATE")
        if ex and ex["passed"]:
            return "TRADED"
        if dg and dg["passed"]:
            return "PAPER_ELIGIBLE"
        if c["data_or_universe_failed"]:
            return "REJECTED"
        if dg and not dg["passed"]:
            # the strategy is not validated/promoted -> awaiting validation; any other gate -> rejected
            return "VALIDATION_PENDING" if dg.get("reject_stage") == "STRATEGY" else "REJECTED"
        if checks.get("STRATEGY_COVERAGE", {}).get("passed"):
            return "VALIDATION_PENDING"
        return "WATCH" if c["on_watchlist"] else "DISCOVERED"

    def _funnel(self, scan: ScanResult, cands: list[dict[str, Any]], links) -> dict[str, Any]:
        st = collections.Counter(c["status"] for c in cands)
        in_val = sum(1 for c in cands if any(x["stage"] == "STRATEGY_COVERAGE" and x["passed"] for x in c["checks"]))
        strat_syms = set(links or {})
        disc_syms = {c["symbol"] for c in cands if c.get("origin", "DISCOVERY") != "STRATEGY"}   # fired a family
        n_links = sum(len(v) for v in (links or {}).values())
        all_links = [x for v in (links or {}).values() for x in v]
        session_trade = sum(1 for x in all_links if x.get("decision") == "TRADE")
        session_orders = sum(1 for x in all_links if x.get("order_status") in PLACED_ORDER_STATUSES)
        origin = collections.Counter(c.get("origin", "DISCOVERY") for c in cands)
        return {
            **scan.counts,
            "with_features": int((scan.table["coverage"] >= self.s.min_score_coverage).sum()),
            "discovered": origin.get("DISCOVERY", 0) + origin.get("BOTH", 0),     # at least one family fired
            "strategy_signals": origin.get("STRATEGY", 0) + origin.get("BOTH", 0),
            "both": origin.get("BOTH", 0), "discovery_only": origin.get("DISCOVERY", 0),
            "strategy_only": origin.get("STRATEGY", 0),
            "missed_discovery_signals": origin.get("STRATEGY", 0),   # strategy signals discovery did not select
            "pool": len(cands),
            "technical_setups": sum(1 for c in cands if c["fired"]),
            "catalyst_setups": sum(1 for c in cands if c.get("catalyst_fired")),
            "post_earnings": sum(1 for c in cands if "post_earnings" in (c.get("catalyst_fired") or [])),
            "material_event": sum(1 for c in cands if "material_event" in (c.get("catalyst_fired") or [])),
            "technical_plus_catalyst": sum(1 for c in cands if c.get("setup_class") == "TECHNICAL + CATALYST"),
            "catalyst_driven": sum(1 for c in cands if c.get("setup_class") == "CATALYST-DRIVEN"),
            "technical_only": sum(1 for c in cands if c.get("setup_class") == "TECHNICAL-ONLY"),
            "class_unknown": sum(1 for c in cands if c.get("setup_class") == "UNKNOWN"),
            "catalyst_watch": sum(1 for c in cands if c.get("catalyst_watch")),
            "high_ranked": sum(1 for c in cands if c["high_quality"]),
            "watchlist": sum(1 for c in cands if c["on_watchlist"]),
            "in_validation": in_val,
            "validation_pending": st.get("VALIDATION_PENDING", 0),
            "paper_eligible": st.get("PAPER_ELIGIBLE", 0) + st.get("TRADED", 0),
            "paper_trades": st.get("TRADED", 0),
            "status_counts": dict(st),
            "strategy_candidates": n_links,
            "strategy_symbols_not_discovered": sorted(strat_syms - disc_syms)[:50],
            "strategy_symbols_not_discovered_count": len(strat_syms - disc_syms),
            "validation_ran": links is not None,
            "session_trade_decisions": session_trade,     # all strategy candidates of the session
            "session_orders_placed": session_orders,
        }

    @staticmethod
    def _blockers(cands: list[dict[str, Any]]) -> dict[str, Any]:
        blocked = [c for c in cands if c["status"] != "TRADED"]
        by = collections.Counter(c["block_stage"] for c in blocked if c["block_stage"])
        hq = collections.Counter(c["block_stage"] for c in blocked if c["block_stage"] and c["high_quality"])
        main = by.most_common(1)[0][0] if by else None
        return {"by_stage": dict(by), "high_ranked_by_stage": dict(hq), "main": main,
                "main_text": BLOCKERS.get(main) if main else None,
                "descriptions": BLOCKERS, "n_discovered": len(cands),
                "n_eligible": sum(1 for c in cands if c["status"] in ("PAPER_ELIGIBLE", "TRADED"))}

    def _near_misses(self, cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for c in [x for x in cands if x["status"] != "TRADED"][: self.s.near_miss_count]:
            out.append({
                "symbol": c["symbol"], "score": c["score"], "coverage": c["coverage"], "status": c["status"],
                "origin": c.get("origin"),
                "passed": [f"{LABELS[f]}" for f in c["fired"]] +
                          [x["name"] for x in c["checks"] if x["passed"] and x["stage"] != "DISCOVERY"],
                "failed": [{"stage": x["stage"], "reason": x["reason"]} for x in c["checks"] if not x["passed"]],
                "unknown": [f"{LABELS[f]}: {c['context'][f].get('why', 'UNKNOWN')}" for f in c["unknown_context"]],
                "block_stage": c["block_stage"], "block_reason": c["block_reason"],
            })
        return out

    def _diagnostics(self, scan: ScanResult, cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dg: list[dict[str, Any]] = []

        def add(level, code, msg, **details):
            dg.append({"level": level, "code": code, "message": msg, "details": details})

        n_basic = scan.counts["basic"]
        if scan.counts["bar_on_session"] == 0:
            add("CRITICAL", "NO_BARS", "no symbol has a bar on this session: data source failure or wrong date")
        elif n_basic == 0:
            add("CRITICAL", "EMPTY_SCAN", "no symbol passed the basic data/liquidity filter",
                bar_on_session=scan.counts["bar_on_session"])
        rate = len(cands) / n_basic if n_basic else 0.0
        if n_basic and not cands:
            add("CRITICAL", "ZERO_DISCOVERIES", f"0 setups discovered among {n_basic} scanned symbols: check data and "
                "feature health, not 'no opportunities'", scanned=n_basic)
        elif n_basic and rate < self.s.min_discovery_rate:
            add("WARN", "LOW_DISCOVERY_RATE", f"only {len(cands)} setups among {n_basic} symbols ({rate:.2%})",
                rate=rate)
        elif n_basic and rate > self.s.max_discovery_rate:
            add("WARN", "HIGH_DISCOVERY_RATE", f"{rate:.0%} of symbols fired a family: thresholds or data suspect",
                rate=rate)
        for row in scan.coverage:
            if row["key"] in SCORED and row["coverage"] < self.s.min_feature_coverage and n_basic:
                add("WARN", "FEATURE_UNAVAILABLE", f"{row['label']} is computable for only {row['coverage']:.0%} of "
                    "scanned symbols", family=row["key"], coverage=row["coverage"])
            if row.get("invalid"):
                add("WARN", "DATA_QUALITY", f"{row['label']}: {row['invalid']} symbol(s) with invalid (non-finite) "
                    "values despite enough history", family=row["key"], n=row["invalid"])
            if row["key"] in CONTEXT and not row["usable"]:
                add("INFO", "CONTEXT_UNAVAILABLE", f"{row['label']} data covers {row['known']} of {row['n']} scanned "
                    f"symbols ({row['coverage']:.1%}): treated as UNKNOWN and excluded from the score",
                    family=row["key"], coverage=row["coverage"])
        fired = collections.Counter(f for c in cands for f in c["fired"])
        for fam in SCORED:
            row = next(r for r in scan.coverage if r["key"] == fam)
            if n_basic >= 200 and row["coverage"] >= self.s.min_feature_coverage and fired.get(fam, 0) == 0:
                add("WARN", "FAMILY_SILENT", f"{LABELS[fam]} data is available but the family fired for no symbol",
                    family=fam)
        if n_basic and len(scan.table) and scan.table["score"].isna().all():
            add("CRITICAL", "ALL_UNKNOWN", "every scanned symbol has an UNKNOWN discovery score: price/volume "
                "features are not computable (data source failure?)")
        blocked = [c for c in cands if c["block_stage"]]
        if len(blocked) >= 10:
            keys = collections.Counter(c.get("block_key") or c["block_stage"] for c in blocked)
            key, n = keys.most_common(1)[0]
            if n / len(blocked) >= self.s.one_rule_share:
                example = next(c["block_reason"] for c in blocked if (c.get("block_key") or c["block_stage"]) == key)
                add("WARN", "ONE_RULE_BLOCKS_ALL", f"{n} of {len(blocked)} discovered setups are blocked by the same "
                    f"rule ({key.split('|')[0]}), e.g. {example}", rule=key, share=n / len(blocked))
        return dg


PLACED_ORDER_STATUSES = ("new", "accepted", "partially_filled", "filled")
_STAGE_PROGRESS = ["DATA", "SIGNAL", "NO_TRADE", "STRATEGY", "EV", "PORTFOLIO", "RISK", "EXECUTION", "AI"]


def _cat_direction(cat: dict[str, Any]) -> str:
    pe, me = cat.get("post_earnings") or {}, cat.get("material_event") or {}
    if pe.get("state") == "FIRED":
        return pe.get("reaction_direction", "UNKNOWN")
    if me.get("state") == "FIRED":
        return me.get("direction", "UNKNOWN")
    return "NONE"


def _cat_reason(cat: dict[str, Any], fam: str) -> str:
    if fam == "post_earnings":
        pe = cat.get("post_earnings") or {}
        z = pe.get("reaction_z")
        return (f"earnings reaction {pe.get('sessions_since_reaction')} session(s) ago: {str(pe.get('reaction_direction')).lower()}"
                + (f" (z {z:+.1f})" if z is not None else "") + f", volume {str(pe.get('volume')).lower()}")
    me = cat.get("material_event") or {}
    return (f"company event today ({', '.join(me.get('categories') or [])}): move {str(me.get('direction')).lower()}, "
            f"volume {str(me.get('volume')).lower()}")


def next_session_info(d, calendar=None) -> dict[str, Any]:
    """The exchange-calendar session AFTER decision session ``d`` and the information cutoff of ``d``
    (its regular close). The stored bundle calendar ends at ``d``, so the forward date comes from the
    rule-based NYSE calendar (holidays + special closures; validated against 2019-2024 counts)."""
    from quantlab.data.audit import expected_sessions
    d = to_session(d)
    try:
        nxt = [x for x in expected_sessions(str((d + pd.Timedelta(days=1)).date()),
                                            str((d + pd.Timedelta(days=14)).date())) if x > d]
    except Exception:       # pragma: no cover - defensive: calendar failure must be visible, not guessed
        nxt = []
    cutoff = calendar.cutoff(d) if calendar is not None else None
    if cutoff is None:
        from zoneinfo import ZoneInfo
        cutoff = pd.Timestamp.combine(d.date(), pd.Timestamp("16:00").time()).tz_localize(ZoneInfo("America/New_York"))
    return {"next_session": str(nxt[0].date()) if nxt else None,
            "info_cutoff_at": pd.Timestamp(cutoff).tz_convert("UTC").isoformat(),
            "calendar_source": "NYSE rules (data/audit.py)" if nxt else "MISSING"}


def _setup_type(fired: list[str], links: list[dict[str, Any]], catalyst: list[str] | None = None) -> str:
    f = set(fired)
    cat = [{"post_earnings": "Post-earnings", "material_event": "Material company event"}.get(x, x) for x in (catalyst or [])]
    if not f and cat:
        return " + ".join(cat)
    if not f and links:
        return "Strategy signal (" + ", ".join(sorted({x["strategy_id"] for x in links})) + ")"
    parts = []
    if "breakout_compression" in f:
        parts.append("Breakout")
    if "momentum" in f:
        parts.append("Momentum")
    if "relative_strength" in f:
        parts.append("Relative strength")
    if "volume_activity" in f:
        parts.append("Volume expansion" if parts else "Unusual activity")
    if "mean_reversion" in f:
        parts.append("Oversold (mean reversion)")
    return " + ".join(parts) or "No family fired"


def setup_record(c: dict[str, Any]) -> dict[str, Any]:
    """A CONDITIONAL next-session setup. It never predicts a price: it states why the setup is
    interesting, what would confirm or invalidate it (levels from D's data), what is missing, and
    why it is or is not paper eligible. Confirmation always requires next-session price/volume."""
    lv, fired = c.get("levels") or {}, list(c.get("fired") or [])
    close, atr = lv.get("close"), lv.get("atr")
    fmt = lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else "UNKNOWN"   # noqa: E731
    confirm, invalidate = [], []
    gap = (f"opens within 1 ATR ({fmt(atr)}) of today's close {fmt(close)}"
           if atr and close else "opening gap within 1 ATR of today's close (ATR UNKNOWN)")
    # Every condition must still be in the future at D: a level the close is already beyond is
    # stated as something to reclaim, never as an invalidation that has in fact already happened.
    hi55, ma50 = lv.get("high_55_prior"), lv.get("ma50")
    if "breakout_compression" in fired:
        if close is None or hi55 is None:
            confirm.append("breaks out of the compressed range with relative volume >= 1.5 (55-day high UNKNOWN)")
            invalidate.append("falls back into the compressed range")
        elif close >= hi55:
            confirm.append(f"holds above the prior 55-day high {fmt(hi55)} with relative volume >= 1.5")
            invalidate.append(f"closes back below the breakout level {fmt(hi55)}")
        else:
            confirm.append(f"today's range expansion continues through the prior 55-day high {fmt(hi55)} "
                           f"({(hi55 / close - 1):+.1%} from today's close) with relative volume >= 1.5")
            invalidate.append("today's range expansion reverses back into the compressed range")
    if "momentum" in fired or "relative_strength" in fired:
        if close is None or ma50 is None:
            confirm.append("keeps outperforming SPY (50-day average UNKNOWN)")
            invalidate.append("starts underperforming SPY")
        elif close >= ma50:
            confirm.append(f"keeps outperforming SPY and holds above its 50-day average {fmt(ma50)}")
            invalidate.append(f"closes below the 50-day average {fmt(ma50)}")
        else:
            confirm.append(f"keeps outperforming SPY and reclaims its 50-day average {fmt(ma50)} "
                           f"(today's close {fmt(close)} is below it)")
            invalidate.append(f"underperforms SPY while staying below its 50-day average {fmt(ma50)}")
    if "volume_activity" in fired:
        confirm.append("follow-through: relative volume stays >= 1.2 in the direction of today's move")
        invalidate.append("reverses today's move on heavy volume")
    if "mean_reversion" in fired:
        confirm.append("stabilisation: closes up without making a new low")
        invalidate.append(f"falls more than 1 ATR below today's close (below about {fmt(close - atr)})"
                          if close and atr else "keeps falling to a new low (ATR UNKNOWN)")
    if not fired and c.get("links"):
        confirm.append("the strategy's own rule: next-open execution after a TRADE decision")
    confirm.append(gap)
    invalidate.append("a gap larger than 1 ATR, a trading halt, or a corporate action at the open")
    missing = [f"{LABELS.get(f, f)}: UNKNOWN" for f in c.get("unknown_context", [])]
    missing += [f"{f}: UNKNOWN (history)" for f in c.get("unknown_features", [])]
    missing += [f"{f}: INVALID (data quality)" for f in c.get("invalid_features", [])]
    missing.append("pre-market price: UNKNOWN (no pre-market data source configured)")
    st = c.get("status")
    paper = ("Paper eligible: TRADE decision from the unchanged gates; still needs the next-open execution window "
             "and the confirmation above" if st in ("PAPER_ELIGIBLE", "TRADED")
             else f"Not paper eligible: {c.get('block_reason') or 'no validated strategy covers this setup'}")
    reasons = [x for f in fired for x in (c.get("reasons") or {}).get(f, [])][:4]
    if c.get("links"):
        reasons.append("signalled by " + ", ".join(sorted({x["strategy_id"] for x in c["links"]})))
    return {"relevance": "NEXT_SESSION", "setup_type": _setup_type(fired, c.get("links") or [], c.get("catalyst_fired")),
            "why": reasons,
            "confirm": confirm, "invalidate": invalidate, "missing": missing, "paper": paper, "conditional": True,
            "condition": "Requires next-session price/volume confirmation. Not an order.", "levels": lv}


def _progress(link: dict[str, Any]) -> float:
    """How far a strategy candidate got (used to pick the most informative link)."""
    if link.get("order_status") in PLACED_ORDER_STATUSES:
        return 5
    if link.get("decision") == "TRADE":
        return 4
    st = link.get("reject_stage")
    return 1 + ((_STAGE_PROGRESS.index(st) + 1) / (len(_STAGE_PROGRESS) + 1) if st in _STAGE_PROGRESS else 0)


def _strategy_info(strategies: dict[str, dict[str, Any]], sid: str, version: str | None) -> dict[str, Any] | None:
    """Status of the strategy VERSION the decision chain used (falls back to the configured one)."""
    info = strategies.get(sid)
    if info and version and version in (info.get("versions") or {}):
        return {**info, **info["versions"][version]}
    return info


def news_coverage(news: pd.DataFrame, baseline_start, cutoff, max_age_days: float) -> dict[str, tuple[bool, str]]:
    """Per symbol: is the news feed known to cover [baseline_start, cutoff]? A symbol counts as
    covered only if its items start before the baseline window and are current at the cutoff;
    otherwise its news features are UNKNOWN (a zero count would be fabricated)."""
    t = pd.to_datetime(news["available_at"], utc=True)
    g = pd.DataFrame({"symbol": news["symbol"], "t": t}).groupby("symbol")["t"].agg(["min", "max"])
    out: dict[str, tuple[bool, str]] = {}
    fresh_after = cutoff - pd.Timedelta(days=max_age_days)
    for s, row in g.iterrows():
        if row["min"] > baseline_start:
            out[s] = (False, f"news history starts {row['min'].date()}: shorter than the 60-session baseline")
        elif row["max"] < fresh_after:
            out[s] = (False, f"news for this symbol ends {row['max'].date()}: not current at the cutoff")
        else:
            out[s] = (True, "")
    return out


# --------------------------------------------------------------------------------------------------
# linking to the unchanged decision chain + persistence
# --------------------------------------------------------------------------------------------------
def strategy_links(db: Database, as_of, run_id: str | None = None,
                   synthetic: bool | None = None) -> dict[str, list[dict[str, Any]]] | None:
    """symbol -> strategy candidates of session ``as_of`` with their latest decision, EV and latest
    entry-order state. ``{}`` when the session was decided but no strategy produced a candidate;
    ``None`` when no pipeline run has decided that session (validation not run yet)."""
    d = str(to_session(as_of).date())
    if run_id is None:
        row = db.fetchone(
            "SELECT r.run_id FROM runs r JOIN pipeline_steps ps ON ps.run_id=r.run_id AND ps.step='decide' "
            "AND ps.status='succeeded' LEFT JOIN pipeline_steps pd_ ON pd_.run_id=r.run_id AND pd_.step='data' "
            "WHERE r.kind='pipeline' AND r.as_of_date=? AND (? IS NULL OR json_extract(pd_.output_json, '$.synthetic')=?) "
            "ORDER BY r.started_at DESC LIMIT 1",
            (d, None if synthetic is None else int(synthetic), None if synthetic is None else int(synthetic)))
        if row is None:
            return None
        run_id = row["run_id"]
    rows = db.fetchall(
        "SELECT c.candidate_id, c.symbol, c.strategy_id, c.strategy_version, c.score, d.decision, d.reject_stage, "
        "d.reasons_json, d.ev_json, (SELECT o.status FROM orders o WHERE o.candidate_id=c.candidate_id AND o.purpose='entry' "
        "ORDER BY o.created_at DESC LIMIT 1) AS order_status, "
        "(SELECT e.reason FROM execution_refusals e WHERE e.candidate_id=c.candidate_id ORDER BY e.id DESC LIMIT 1) AS refusal "
        "FROM candidates c LEFT JOIN decisions d ON d.decision_id = (SELECT x.decision_id FROM decisions x "
        "WHERE x.candidate_id=c.candidate_id ORDER BY x.created_at DESC LIMIT 1) "
        "WHERE c.run_id=? AND c.as_of_date=? AND (? IS NULL OR c.is_synthetic=?)",
        (run_id, d, None if synthetic is None else int(synthetic), None if synthetic is None else int(synthetic)))
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ev = from_json(r["ev_json"], {}) or {}
        out.setdefault(r["symbol"], []).append({
            "candidate_id": r["candidate_id"], "strategy_id": r["strategy_id"], "strategy_version": r["strategy_version"],
            "decision": r["decision"], "reject_stage": r["reject_stage"],
            "reasons": "; ".join(from_json(r["reasons_json"], []) or [])[:400],
            "ev_bps": (float(ev["ev"]) * 1e4) if _finite(ev.get("ev")) else None,
            "order_status": r["order_status"], "refusal": r["refusal"], "run_id": run_id})
    return out


def persist(db: Database, engine: DiscoveryEngine, scan: ScanResult, a: Assessment, run_id: str | None = None) -> str:
    """Append the run, its discovered setups and diagnostics (one transaction). Idempotent per
    pipeline run: a second call for the same run_id returns the existing discovery run."""
    disc_run = f"disc_{run_id}" if run_id else new_id("disc")
    if run_id and db.fetchone("SELECT 1 FROM discovery_runs WHERE discovery_run_id=?", (disc_run,)):
        return disc_run
    now = utcnow_iso()
    d = str(scan.as_of.date())
    ns = next_session_info(scan.as_of, scan.calendar)
    fams = {"coverage": scan.coverage, "market_context": scan.market_context,
            "fired": dict(collections.Counter(f for c in a.candidates for f in c["fired"]))}
    with db.transaction():
        db.insert("discovery_runs", {
            "discovery_run_id": disc_run, "run_id": run_id, "as_of_date": d, "created_at": now,
            "is_synthetic": int(scan.is_synthetic), "funnel_json": to_json(a.funnel),
            "blockers_json": to_json({**a.blockers, "near_misses": a.near_misses}), "families_json": to_json(fams),
            "config_json": to_json(engine.s.as_dict()), "dataset_ids_json": to_json(scan.dataset_ids),
            "next_session": ns.get("next_session"), "info_cutoff_at": ns.get("info_cutoff_at"),
            "calendar_source": ns.get("calendar_source"),
            "catalysts_json": to_json(getattr(scan, "catalyst_panel", None)),
            "source_coverage_json": to_json(getattr(scan, "source_coverage", None)),
            "basic_symbols_json": to_json(getattr(scan, "basic_symbols", None))})
        db.insert_many("discovery_candidates", [{
            "discovery_id": f"{disc_run}:{c['symbol']}", "discovery_run_id": disc_run, "as_of_date": d,
            "symbol": c["symbol"], "discovery_score": c["score"],
            "score_coverage": c["coverage"] or 0.0, "rank": c["rank"],
            "families_json": to_json({"fired": c["fired"], "reasons": c["reasons"], "components": c["components"]}),
            "dimensions_json": to_json(c["components"]),
            "factors_json": to_json({"features": c["factors"], "unknown": c["unknown_features"],
                                     "invalid": c["invalid_features"]}),
            "catalyst_json": to_json(c["context"]), "direction_bias": c["bias"], "status": c["status"],
            "high_quality": int(c["high_quality"]), "on_watchlist": int(c["on_watchlist"]),
            "block_stage": c["block_stage"], "block_reason": c["block_reason"], "checks_json": to_json(c["checks"]),
            "strategy_links_json": to_json(c["links"]), "is_synthetic": int(scan.is_synthetic), "created_at": now,
            "origin": c.get("origin"), "relevance": "NEXT_SESSION", "next_session": ns.get("next_session"),
            "info_cutoff_at": ns.get("info_cutoff_at"), "discovered_at": now, "setup_json": to_json(c.get("setup")),
            "setup_class": c.get("setup_class"), "catalyst_families": ",".join(c.get("catalyst_fired") or []) or None,
            "catalyst_record_json": to_json(c.get("catalyst") or None),
            "evidence_chain_json": to_json(c.get("chain")), "created_by": "EOD_SCAN",
        } for c in a.candidates])
        db.insert_many("discovery_diagnostics", [{
            "discovery_run_id": disc_run, "as_of_date": d, "level": x["level"], "code": x["code"],
            "message": x["message"], "details_json": to_json(x["details"]), "created_at": now} for x in a.diagnostics])
    log_event(log, "discovery persisted", discovery_run_id=disc_run, discovered=len(a.candidates),
              watchlist=a.funnel["watchlist"], paper_eligible=a.funnel["paper_eligible"])
    return disc_run


__all__ = ["Assessment", "BLOCKERS", "DiscoveryEngine", "DiscoverySettings", "ScanResult", "persist",
           "strategy_links", "tail_bundle"]
