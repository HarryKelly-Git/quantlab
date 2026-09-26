"""Catalyst discovery (research only): POST-EARNINGS and MATERIAL COMPANY EVENT, the catalyst
evidence chain and the setup class of a candidate.

Separate from both the technical score and trade eligibility:
  * catalysts never change the 0-100 price/volume discovery score;
  * a catalyst never makes anything paper eligible: eligibility is decided by the unchanged chain
    (strategy validation -> EV -> risk -> execution) and read back from its recorded decisions.

Families (thresholds fixed a priori in ``discovery.catalyst``; never tuned on outcomes):
  post_earnings   an 8-K item 2.02 whose reaction session r is within the last N sessions AND the
                  market responded: |abnormal reaction z| >= min OR dollar volume >= min x normal.
  material_event  a company-specific material news item (headline category, data/news_classify.py)
                  or a material 8-K usable at D AND today's |move z| >= min OR volume >= min x.

Direction is recorded (POSITIVE / NEGATIVE / MUTED) and never assumed: an event is not bullish.
Analyst consensus surprise (EPS or revenue) has no point-in-time source here: always UNKNOWN.
Everything is read from a view truncated at cutoff(D); provenance (source, id, availability time)
is stored with each input so the decision state can be reconstructed exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.news_classify import classify_frame

CATALYST_FAMILIES = ("post_earnings", "material_event")
CATALYST_LABEL = {"post_earnings": "Post-earnings", "material_event": "Material company event"}
SETUP_CLASSES = ("TECHNICAL + CATALYST", "CATALYST-DRIVEN", "TECHNICAL-ONLY", "UNKNOWN")
CAT_FEATURES = ("days_since_earnings", "reaction_ret_1d", "reaction_z_1d", "event_rel_volume", "abn_ret_since_reaction",
                "sec_material_1d", "news_company_1d", "news_material_1d", "ret_1d", "ret_z_1d", "rel_volume_1d",
                "rs_spy_20", "industry_code", "industry_members", "industry_ret_20", "rs_industry_20",
                "industry_rank_63", "rs_sector_20_pit", "sector_rs_spy_63_pit", "rev_growth_yoy", "eps_growth_yoy",
                "gross_margin", "op_margin", "fcf_margin", "sue", "fundamental_age", "atr14_pct", "sic_code_asof")
NO_CONSENSUS = "UNKNOWN (no point-in-time analyst consensus source)"


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pct(x: float | None, digits: int = 1) -> str:
    return "UNKNOWN" if x is None else f"{x * 100:+.{digits}f}%"


@dataclass(frozen=True)
class CatalystTriggers:
    max_days_since_earnings: int = 3
    min_abs_reaction_z: float = 1.5
    min_rel_volume: float = 2.0
    min_abs_move_z: float = 1.5
    strong_industry_rank: float = 0.7
    weak_industry_rank: float = 0.3
    watchlist_size: int = 10
    earnings_coverage_days: int = 400
    news_coverage_days: int = 90

    @classmethod
    def from_config(cls, config) -> "CatalystTriggers":
        c = dict(config.get("discovery.catalyst", {}) or {})
        return cls(**{k: type(getattr(cls, k))(v) for k, v in c.items() if hasattr(cls, k)})


def direction(z: float | None, t: float) -> str:
    if z is None:
        return "UNKNOWN"
    return "POSITIVE" if z >= t else ("NEGATIVE" if z <= -t else "MUTED")


def industry_state(rank: float | None, t: CatalystTriggers) -> str:
    if rank is None:
        return "UNKNOWN"
    return "STRONG" if rank >= t.strong_industry_rank else ("WEAK" if rank <= t.weak_industry_rank else "NEUTRAL")


def fundamental_state(rev: float | None, eps: float | None) -> str:
    if rev is None and eps is None:
        return "UNKNOWN"
    ups = [v > 0 for v in (rev, eps) if v is not None]
    if all(ups) and len(ups) == 2:
        return "SUPPORTIVE"
    if not any(ups):
        return "WEAK"
    return "MIXED" if len(ups) == 2 else ("SUPPORTIVE" if ups[0] else "WEAK") + " (partial)"


class CatalystEngine:
    def __init__(self, config):
        self.config = config
        self.t = CatalystTriggers.from_config(config)

    # -- source coverage per symbol (point in time) ----------------------------------------------
    def coverage(self, view, cutoff: pd.Timestamp, syms) -> dict[str, dict[str, tuple[bool, str]]]:
        ev = view.events
        out: dict[str, dict[str, tuple[bool, str]]] = {s: {} for s in syms}
        t_ev = pd.to_datetime(ev["available_at"], utc=True) if len(ev) else pd.Series(dtype="datetime64[ns, UTC]")
        win = t_ev >= cutoff - pd.Timedelta(days=self.t.earnings_coverage_days) if len(ev) else t_ev
        er = set(ev.loc[win & (ev["event_type"] == "earnings_release"), "symbol"]) if len(ev) else set()
        sec = set(ev.loc[win & ev["event_type"].isin(["earnings_release", "periodic_report", "sec_8k",
                                                      "foreign_report"]), "symbol"]) if len(ev) else set()
        foreign = set(ev.loc[win & (ev["event_type"] == "foreign_report"), "symbol"]) if len(ev) else set()
        nw = view.news
        feed_current = False
        n_syms: set[str] = set()
        if len(nw):
            t_n = pd.to_datetime(nw["available_at"], utc=True)
            feed_current = bool(t_n.max() >= cutoff - pd.Timedelta(days=3))
            n_syms = set(nw.loc[t_n >= cutoff - pd.Timedelta(days=self.t.news_coverage_days), "symbol"])
        for s in syms:
            out[s]["earnings"] = ((True, "8-K 2.02 filer") if s in er else
                                  (False, "foreign filer: no 8-K item codes (earnings timing UNKNOWN)") if s in foreign else
                                  (False, f"no 8-K 2.02 in the last {self.t.earnings_coverage_days} days "
                                          "(non-reporting, SPAC, fund or unmapped ticker)"))
            out[s]["sec_events"] = (True, "SEC filer") if s in sec else (False, "no SEC filing in the window (unmapped?)")
            out[s]["news"] = ((True, "news feed covers the symbol") if (feed_current and s in n_syms) else
                              (False, "news feed not current at the decision time" if not feed_current else
                               f"no article tagged with the symbol in the last {self.t.news_coverage_days} days"))
        return out

    # -- provenance items at D --------------------------------------------------------------------
    def items(self, view, d: pd.Timestamp, cutoff: pd.Timestamp, syms) -> dict[str, dict[str, Any]]:
        cal = view.calendar
        prev = cal.prev_session(d)
        prev_cut = cal.cutoff(prev) if prev is not None else cutoff - pd.Timedelta(days=1)
        S = set(syms)
        out: dict[str, dict[str, Any]] = {s: {} for s in syms}
        ev = view.events
        if len(ev):
            ev = ev[ev["symbol"].isin(S)]
            t = pd.to_datetime(ev["available_at"], utc=True)
            ev = ev.assign(t_avail=t)
            er = ev[(ev["event_type"] == "earnings_release")
                    & ~ev["payload_json"].astype(str).str.contains('"form": "8-K/A"', regex=False)]
            for r in er.sort_values("t_avail").groupby("symbol").tail(1).itertuples():
                pl = json.loads(r.payload_json or "{}")
                react = cal.reaction_session(r.t_avail)
                timing = pl.get("timing") or "UNKNOWN"
                if timing == "UNKNOWN":            # ingested beyond its calendar: the rule is deterministic
                    from quantlab.data.sec_catalysts import release_timing
                    timing = release_timing(r.t_avail, cal)
                out[r.symbol]["earnings"] = {
                    "source": "SEC EDGAR 8-K item 2.02", "accession": r.source_id, "form": pl.get("form"),
                    "accepted_at": str(r.t_avail), "available_at": str(r.t_avail), "timing": timing,
                    "reaction_session": str(react.date()) if react is not None else None, "items": pl.get("items")}
            pr = ev[ev["event_type"] == "periodic_report"]
            for r in pr.sort_values("t_avail").groupby("symbol").tail(1).itertuples():
                pl = json.loads(r.payload_json or "{}")
                out[r.symbol]["periodic"] = {"form": pl.get("form"), "accession": r.source_id, "accepted_at": str(r.t_avail),
                                             "report_date": pl.get("report_date")}
            k8 = ev[(ev["event_type"] == "sec_8k") & (ev["t_avail"] > prev_cut)]
            for r in k8.itertuples():
                pl = json.loads(r.payload_json or "{}")
                out[r.symbol].setdefault("sec_8k", []).append({
                    "source": "SEC EDGAR 8-K", "accession": r.source_id, "accepted_at": str(r.t_avail),
                    "items": pl.get("material_items"), "categories": pl.get("categories"), "labels": pl.get("labels"),
                    "timing": pl.get("timing")})
            sic = ev[ev["event_type"] == "sic_observation"]
            for r in sic.sort_values("t_avail").groupby("symbol").tail(1).itertuples():
                pl = json.loads(r.payload_json or "{}")
                out[r.symbol]["sic"] = {"sic": pl.get("sic"), "title": pl.get("sic_title"), "observed_at": str(r.t_avail),
                                        "accession": pl.get("accession")}
        nw = view.news
        if len(nw):
            t = pd.to_datetime(nw["available_at"], utc=True)
            recent = nw[(t > cutoff - pd.Timedelta(days=7)) & (t <= cutoff)]
            if len(recent):
                recent = classify_frame(recent if "n_tags" in recent.columns else
                                        recent.assign(n_tags=nw.groupby("news_id")["symbol"].transform("size")
                                                      .reindex(recent.index)))
                recent = recent[recent["symbol"].isin(S) & recent["company_specific"]]
                tr = pd.to_datetime(recent["available_at"], utc=True)
                upd = pd.to_datetime(recent["updated_at"], utc=True)
                for r, ta, tu in zip(recent.itertuples(), tr, upd):
                    revised_later = bool(tu > cutoff + pd.Timedelta(seconds=60))
                    item = {"source": f"Alpaca news ({r.source})", "news_id": r.news_id, "created_at": str(ta),
                            "category": r.category, "material": bool(r.material), "guidance_dir": r.guidance_dir,
                            "headline": None if revised_later else r.headline, "url": r.url or None,
                            "headline_withheld": revised_later, "today": bool(ta > prev_cut)}
                    out[r.symbol].setdefault("news", []).append(item)
        return out

    # -- evaluation ---------------------------------------------------------------------------------
    def evaluate(self, view, fs, d, syms, technical: dict[str, dict] | None = None) -> dict[str, dict[str, Any]]:
        """Catalyst record per symbol at session ``d`` (``view`` truncated at d, ``fs`` over it)."""
        d = pd.Timestamp(d)
        cutoff = view.calendar.cutoff(d)
        syms = list(syms)
        if not syms:
            return {}
        names = [n for n in CAT_FEATURES if n in fs.registry]
        xs = fs.cross_section(d, names).reindex(syms)
        cov = self.coverage(view, cutoff, syms)
        its = self.items(view, d, cutoff, syms)
        return {s: self.one(s, xs.loc[s], cov[s], its.get(s, {}), (technical or {}).get(s, {})) for s in syms}

    def one(self, s: str, x: pd.Series, cov: dict[str, tuple[bool, str]], it: dict[str, Any],
            tech: dict[str, Any]) -> dict[str, Any]:
        t = self.t
        g = {k: _f(x.get(k)) for k in CAT_FEATURES}
        fams: list[str] = []
        # ---- post-earnings ------------------------------------------------------------------
        e = it.get("earnings")
        dse = g["days_since_earnings"]
        pe: dict[str, Any] = {"state": "UNKNOWN" if not cov["earnings"][0] else "NONE", "why": cov["earnings"][1]}
        if e is not None and dse is not None and dse <= t.max_days_since_earnings:
            z, rv = g["reaction_z_1d"], g["event_rel_volume"]
            dirn = direction(z, t.min_abs_reaction_z)
            vol = "UNKNOWN" if rv is None else ("CONFIRMED" if rv >= t.min_rel_volume else "NOT CONFIRMED")
            since = g["abn_ret_since_reaction"]
            cont = ("REACTION DAY" if dse == 0 else "UNKNOWN" if since is None or g["reaction_ret_1d"] is None else
                    ("CONTINUING" if np.sign(since) == np.sign(g["reaction_ret_1d"]) and since != 0 else "REVERSING"))
            per = it.get("periodic")
            filed_after = bool(per and pd.Timestamp(per["accepted_at"]) >= pd.Timestamp(e["accepted_at"]) - pd.Timedelta(days=1))
            fired = dirn in ("POSITIVE", "NEGATIVE") or vol == "CONFIRMED"
            pe = {"state": "FIRED" if fired else "PRESENT", "event": e, "sessions_since_reaction": int(dse),
                  "reaction_abnormal_return": g["reaction_ret_1d"], "reaction_z": z, "reaction_direction": dirn,
                  "event_rel_volume": rv, "volume": vol, "since_reaction_abnormal_return": since,
                  "continuation": cont, "eps_surprise": NO_CONSENSUS, "revenue_surprise": NO_CONSENSUS,
                  "quarter_numbers": ({"state": "FILED", "periodic": per,
                                       "text": f"{per['form']} for the quarter accepted {per['accepted_at'][:16]} UTC"}
                                      if filed_after else
                                      {"state": "UNKNOWN", "text": "the quarter's 10-Q/10-K was not filed by the "
                                       "decision time: reported EPS/revenue are not machine-readable yet"}),
                  "guidance": self._guidance(it, e)}
            if fired:
                fams.append("post_earnings")
        # ---- material company event ------------------------------------------------------------
        news_today = [n for n in it.get("news", []) if n["today"] and n["material"]]
        k8_today = it.get("sec_8k", [])
        me: dict[str, Any] = {"state": "UNKNOWN" if not (cov["news"][0] or cov["sec_events"][0]) else "NONE",
                              "why": f"news: {cov['news'][1]}; SEC: {cov['sec_events'][1]}"}
        if news_today or k8_today:
            mz, rv1 = g["ret_z_1d"], g["rel_volume_1d"]
            dirn = direction(mz, t.min_abs_move_z)
            vol = "UNKNOWN" if rv1 is None else ("CONFIRMED" if rv1 >= t.min_rel_volume else "NOT CONFIRMED")
            fired = dirn in ("POSITIVE", "NEGATIVE") or vol == "CONFIRMED"
            cats = sorted({n["category"] for n in news_today} | {c for k in k8_today for c in (k.get("categories") or [])})
            me = {"state": "FIRED" if fired else "PRESENT", "categories": cats, "news": news_today[:5],
                  "sec_8k": k8_today[:5], "move_z": mz, "move_return": g["ret_1d"], "direction": dirn,
                  "rel_volume": rv1, "volume": vol}
            if fired:
                fams.append("material_event")
        # ---- context ------------------------------------------------------------------------------
        ind = {"state": industry_state(g["industry_rank_63"], t), "industry_rank_63": g["industry_rank_63"],
               "rs_industry_20": g["rs_industry_20"], "industry_ret_20": g["industry_ret_20"],
               "members": g["industry_members"], "code": g["industry_code"], "sic": it.get("sic"),
               "rs_sector_20": g["rs_sector_20_pit"], "sector_rs_spy_63": g["sector_rs_spy_63_pit"],
               "pit": "SIC from the latest filing header accepted by the decision time"}
        if g["sic_code_asof"] is None:
            ind["state"], ind["why"] = "UNKNOWN", "no SIC observed in a filing header by the decision time"
        fun = {"state": fundamental_state(g["rev_growth_yoy"], g["eps_growth_yoy"]),
               "rev_growth_yoy": g["rev_growth_yoy"], "eps_growth_yoy": g["eps_growth_yoy"],
               "gross_margin": g["gross_margin"], "op_margin": g["op_margin"], "fcf_margin": g["fcf_margin"],
               "sue": g["sue"], "age_sessions": g["fundamental_age"],
               "pit": "as-of replay: latest filing accepted by the decision time (restatements only from their own filing)"}
        # ---- agreement (a count of independent evidence types, never a probability) ------------------
        sup, con = [], []
        rdir = pe.get("reaction_direction") if "reaction_direction" in pe else me.get("direction")
        if rdir == "POSITIVE":
            sup.append("positive price response")
        elif rdir == "NEGATIVE":
            con.append("negative price response")
        if "volume" in pe and pe["volume"] == "CONFIRMED" or me.get("volume") == "CONFIRMED":
            sup.append("abnormal volume")
        if pe.get("continuation") == "CONTINUING":
            (sup if rdir == "POSITIVE" else con).append("move continuing after the reaction")
        elif pe.get("continuation") == "REVERSING":
            (con if rdir == "POSITIVE" else sup).append("move reversing after the reaction")
        if ind["state"] == "STRONG":
            sup.append("strong industry")
        elif ind["state"] == "WEAK":
            con.append("weak industry")
        if fun["state"] == "SUPPORTIVE":
            sup.append("revenue and EPS growing y/y")
        elif fun["state"] == "WEAK":
            con.append("revenue and EPS shrinking y/y")
        gd = (pe.get("guidance") or {}).get("direction")
        if gd == "UP":
            sup.append("guidance raised (headline)")
        elif gd == "DOWN":
            con.append("guidance lowered (headline)")
        if tech.get("bias") == "BULLISH" and tech.get("fired"):
            sup.append("technical families bullish")
        elif tech.get("bias") == "BEARISH" and tech.get("fired"):
            con.append("technical families bearish")
        known = cov["earnings"][0] or cov["news"][0]
        return {"families": fams, "post_earnings": pe, "material_event": me, "industry": ind, "fundamentals": fun,
                "coverage": {k: {"known": v[0], "why": v[1]} for k, v in cov.items()}, "catalyst_known": bool(known),
                "agreement": {"supporting": sup, "contradicting": con, "n_supporting": len(sup), "n_contradicting": len(con),
                              "note": "count of independent evidence types; not a probability"},
                "provenance": {k: v for k, v in it.items() if k in ("earnings", "periodic", "sic")},
                "features": g}

    def panel(self, view, fs, d, syms, cats: dict[str, dict[str, Any]], top: int = 12) -> dict[str, Any]:
        """CATALYSTS panel at D over the basic-filter symbols: new earnings reactions, company news,
        material 8-Ks, unusual price reactions, abnormal volume and industry leadership."""
        d = pd.Timestamp(d)
        names = [n for n in ("ret_1d", "ret_z_1d", "rel_volume_1d", "industry_rank_63", "industry_code",
                             "industry_members", "industry_ret_20") if n in fs.registry]
        xs = fs.cross_section(d, names).reindex(list(syms))
        earn, news, k8 = [], [], []
        for s in syms:
            c = cats.get(s) or {}
            pe, me = c.get("post_earnings") or {}, c.get("material_event") or {}
            if pe.get("state") in ("FIRED", "PRESENT") and pe.get("sessions_since_reaction") == 0:
                earn.append({"symbol": s, "accepted_at": pe["event"]["accepted_at"], "timing": pe["event"]["timing"],
                             "reaction": pe.get("reaction_abnormal_return"), "z": pe.get("reaction_z"),
                             "direction": pe.get("reaction_direction"), "rel_volume": pe.get("event_rel_volume"),
                             "fired": pe["state"] == "FIRED"})
            for n in me.get("news") or []:
                news.append({"symbol": s, "category": n["category"], "headline": n.get("headline"),
                             "created_at": n["created_at"], "source": n["source"], "url": n.get("url"),
                             "move_z": me.get("move_z"), "fired": me.get("state") == "FIRED"})
            for k in me.get("sec_8k") or []:
                k8.append({"symbol": s, "categories": k.get("categories"), "labels": k.get("labels"),
                           "accepted_at": k["accepted_at"], "accession": k["accession"], "fired": me.get("state") == "FIRED"})
        earn.sort(key=lambda r: -abs(r["z"] or 0.0))
        news.sort(key=lambda r: (not r["fired"], -abs(r["move_z"] or 0.0)))
        z = xs["ret_z_1d"] if "ret_z_1d" in xs else pd.Series(dtype=float)
        unusual = [{"symbol": s, "ret_1d": _f(xs.at[s, "ret_1d"]), "z": _f(v),
                    "catalyst": ", ".join((cats.get(s) or {}).get("families") or []) or "none found"}
                   for s, v in z[z.abs() >= 3].sort_values(key=abs, ascending=False).head(top).items()]
        rv = xs["rel_volume_1d"] if "rel_volume_1d" in xs else pd.Series(dtype=float)
        volume = [{"symbol": s, "rel_volume": _f(v), "ret_1d": _f(xs.at[s, "ret_1d"]),
                   "catalyst": ", ".join((cats.get(s) or {}).get("families") or []) or "none found"}
                  for s, v in rv[rv >= 3].sort_values(ascending=False).head(top).items()]
        groups = []
        if "industry_code" in xs and "industry_rank_63" in xs:
            g = xs.dropna(subset=["industry_code", "industry_rank_63"]).copy()
            titles = {s: ((cats.get(s) or {}).get("provenance", {}).get("sic") or {}).get("title") for s in g.index}
            g["title"] = pd.Series(titles)
            import html
            for code, gg in g.groupby("industry_code"):
                n = _f(gg["industry_members"].iloc[0]) if "industry_members" in gg else None
                groups.append({"code": int(code), "rank_63": _f(gg["industry_rank_63"].iloc[0]),
                               "members": int(n) if n is not None else int(len(gg)), "scanned": int(len(gg)),
                               "ret_20": _f(gg["industry_ret_20"].mean()) if "industry_ret_20" in gg else None,
                               "name": (html.unescape(gg["title"].dropna().mode().iloc[0]).title()
                                        if gg["title"].notna().any() else None)})
            groups.sort(key=lambda r: -(r["rank_63"] or 0.0))
        return {"as_of": str(d.date()), "earnings_reactions_today": earn[:top], "n_earnings_reactions_today": len(earn),
                "company_news_today": news[:top * 2], "n_company_news_today": len(news),
                "material_8k_today": k8[:top], "n_material_8k_today": len(k8),
                "unusual_price_reactions": unusual, "n_unusual_price_reactions": int((z.abs() >= 3).sum()),
                "abnormal_volume": volume, "n_abnormal_volume": int((rv >= 3).sum()),
                "industry_leaders": groups[:5], "industry_laggards": groups[-5:][::-1] if len(groups) > 5 else [],
                "notes": ["catalysts are discovery evidence, not trade signals; direction is never assumed",
                          "EPS/revenue surprise vs consensus: UNKNOWN (no point-in-time consensus source)"]}

    @staticmethod
    def _guidance(it: dict[str, Any], e: dict[str, Any]) -> dict[str, Any]:
        t0 = pd.Timestamp(e["accepted_at"]) - pd.Timedelta(days=1)
        g = [n for n in it.get("news", []) if n["category"] == "guidance" and pd.Timestamp(n["created_at"]) >= t0]
        if not g:
            return {"direction": "UNKNOWN", "text": "no guidance headline found (headline rules only)"}
        dirs = {n["guidance_dir"] for n in g}
        d_ = "DOWN" if "DOWN" in dirs else "UP" if "UP" in dirs else "REAFFIRM" if "REAFFIRM" in dirs else "UNKNOWN"
        return {"direction": d_, "items": g[:3], "text": "from headline wording (PIT_CONSERVATIVE; not a forecast)"}


def setup_class(technical: bool, catalyst_fired: bool, catalyst_known: bool) -> str:
    if catalyst_fired:
        return "TECHNICAL + CATALYST" if technical else "CATALYST-DRIVEN"
    return "TECHNICAL-ONLY" if catalyst_known else "UNKNOWN"


def catalyst_text(cat: dict[str, Any], levels: dict[str, Any]) -> dict[str, list[str]]:
    """Why / confirm / invalidate / missing lines for the catalyst part of a conditional setup."""
    why, confirm, invalidate, missing = [], [], [], []
    pe, me = cat.get("post_earnings") or {}, cat.get("material_event") or {}
    close = levels.get("close")
    if pe.get("state") in ("FIRED", "PRESENT"):
        e = pe["event"]
        why.append(f"earnings (8-K 2.02) accepted {e['accepted_at'][:16]} UTC ({e['timing'].replace('_', ' ').lower()}); "
                   f"reaction {pe['sessions_since_reaction']} session(s) ago: abnormal {_pct(pe['reaction_abnormal_return'])}"
                   f" (z {pe['reaction_z']:+.1f})" if pe.get("reaction_z") is not None else
                   f"earnings (8-K 2.02) accepted {e['accepted_at'][:16]} UTC")
        if pe.get("event_rel_volume") is not None:
            why.append(f"reaction-day dollar volume {pe['event_rel_volume']:.1f}x normal")
        if pe.get("reaction_direction") == "POSITIVE":
            confirm.append("the reaction holds: no more than half of the reaction move is given back")
            invalidate.append("gives back more than half of the reaction move, or closes below the pre-announcement close")
        elif pe.get("reaction_direction") == "NEGATIVE":
            confirm.append("a negative reaction: long-only QuantLab only watches; stabilisation would be needed")
            invalidate.append("new lows below the reaction-day low")
        missing += [f"EPS surprise: {pe['eps_surprise']}", f"revenue surprise: {pe['revenue_surprise']}"]
        if (pe.get("quarter_numbers") or {}).get("state") == "UNKNOWN":
            missing.append("reported EPS/revenue: UNKNOWN (10-Q/10-K not filed yet)")
        if (pe.get("guidance") or {}).get("direction") == "UNKNOWN":
            missing.append("guidance: UNKNOWN (no guidance headline)")
    if me.get("state") in ("FIRED", "PRESENT"):
        heads = [n["headline"] for n in me.get("news", []) if n.get("headline")]
        why.append("company event today: " + ", ".join(me.get("categories") or []) +
                   (f" ({heads[0][:80]})" if heads else ""))
        if me.get("move_z") is not None:
            why.append(f"today's move {_pct(me.get('move_return'))} (z {me['move_z']:+.1f}), "
                       f"volume {me['rel_volume']:.1f}x" if me.get("rel_volume") is not None else
                       f"today's move z {me['move_z']:+.1f}")
        confirm.append("follow-through next session with relative volume >= 1.5 in the direction of the event move")
        invalidate.append("the event move fully reverses" + (f" (back through today's open area; close {close:,.2f})"
                                                             if isinstance(close, (int, float)) else ""))
    ind, fun = cat.get("industry") or {}, cat.get("fundamentals") or {}
    if ind.get("state") == "UNKNOWN":
        missing.append("industry: UNKNOWN (no SIC in a filing header by the decision time)")
    if fun.get("state") == "UNKNOWN":
        missing.append("fundamentals: UNKNOWN (no as-of XBRL facts)")
    for k, v in (cat.get("coverage") or {}).items():
        if not v.get("known"):
            missing.append(f"{k.replace('_', ' ')}: UNKNOWN ({v.get('why')})")
    return {"why": why, "confirm": confirm, "invalidate": invalidate, "missing": missing}


def catalyst_summary(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    """Flat, display-ready view of a stored catalyst record (event, when, reaction, volume, context).
    UNKNOWN stays UNKNOWN; PENDING means not observable yet (overnight events)."""
    if not rec:
        return None
    pe, me, ov = rec.get("post_earnings") or {}, rec.get("material_event") or {}, rec.get("overnight_event")
    ind, fu = rec.get("industry") or {}, rec.get("fundamentals") or {}
    out: dict[str, Any] = {"families": rec.get("families") or [], "event": None, "event_at": None, "timing": None,
                           "reaction": None, "volume": None, "direction": None,
                           "agreement": rec.get("agreement") or {}}
    if pe.get("state") in ("FIRED", "PRESENT"):
        e = pe["event"]
        z = pe.get("reaction_z")
        out.update(event="Earnings release (8-K 2.02)", event_at=e.get("accepted_at"), timing=e.get("timing"),
                   direction=pe.get("reaction_direction"),
                   reaction=(str(pe.get("reaction_direction")).lower() + " " + _pct(pe.get("reaction_abnormal_return"))
                             + " vs SPY" + ("" if z is None else f" (z {z:+.1f})")
                             + f"; {pe.get('sessions_since_reaction')} session(s) ago, "
                             + str(pe.get("continuation")).lower()),
                   volume=("UNKNOWN" if pe.get("event_rel_volume") is None else
                           f"{pe['event_rel_volume']:.1f}x normal ({str(pe.get('volume')).lower()})"),
                   numbers=(pe.get("quarter_numbers") or {}).get("text"),
                   guidance=(pe.get("guidance") or {}).get("direction"), surprise=pe.get("eps_surprise"))
    elif me.get("state") in ("FIRED", "PRESENT"):
        src = (me.get("news") or [None])[0] or (me.get("sec_8k") or [None])[0] or {}
        mz = me.get("move_z")
        out.update(event="Company event: " + ", ".join(me.get("categories") or []),
                   event_at=src.get("created_at") or src.get("accepted_at"), direction=me.get("direction"),
                   headline=src.get("headline"), source=src.get("source"),
                   reaction=(str(me.get("direction")).lower() + " " + _pct(me.get("move_return")) + " today"
                             + ("" if mz is None else f" (z {mz:+.1f})")),
                   volume=("UNKNOWN" if me.get("rel_volume") is None else
                           f"{me['rel_volume']:.1f}x normal ({str(me.get('volume')).lower()})"))
    elif ov:
        out.update(event="Earnings release after the close (reaction pending)", event_at=ov.get("available_at"),
                   timing=ov.get("phase"), reaction="PENDING (next session)", volume="PENDING (next session)",
                   direction="PENDING")
    if not ind or ind.get("state") == "UNKNOWN":
        out["industry"] = "UNKNOWN"
    elif ind.get("industry_rank_63") is not None:
        out["industry"] = (f"{ind.get('state')}: industry rank {ind['industry_rank_63']:.2f}, vs industry "
                           + _pct(ind.get("rs_industry_20")) + " (20d)")
    else:
        out["industry"] = str(ind.get("state", "UNKNOWN"))
    if not fu or fu.get("state") == "UNKNOWN":
        out["fundamentals"] = "UNKNOWN"
    else:
        out["fundamentals"] = (f"{fu.get('state')}: revenue " + _pct(fu.get("rev_growth_yoy"), 0) + " y/y, EPS "
                               + _pct(fu.get("eps_growth_yoy"), 0) + " y/y, op margin " + _pct(fu.get("op_margin"), 0))
    return out


CHAIN_STAGES = ("EVENT", "WHEN KNOWN", "PRICE RESPONSE", "VOLUME RESPONSE", "SECTOR/INDUSTRY", "FUNDAMENTALS",
                "VALIDATION", "RISK", "EV", "PAPER ELIGIBILITY")


def evidence_chain(c: dict[str, Any]) -> list[dict[str, Any]]:
    """EVENT -> WHEN KNOWN -> PRICE -> VOLUME -> SECTOR/INDUSTRY -> FUNDAMENTALS -> VALIDATION -> RISK
    -> EV -> PAPER ELIGIBILITY for one candidate. States: PASS / FAIL / PRESENT / NONE / UNKNOWN /
    NOT_REACHED / NOT_APPLICABLE. Stages 7-10 are read from the unchanged decision chain."""
    cat = c.get("catalyst") or {}
    pe, me = cat.get("post_earnings") or {}, cat.get("material_event") or {}
    rows: list[dict[str, Any]] = []

    def add(stage, state, text, provenance=None):
        rows.append({"stage": stage, "state": state, "text": text, "provenance": provenance})
    if pe.get("state") in ("FIRED", "PRESENT"):
        e = pe["event"]
        add("EVENT", "PRESENT", f"earnings release (8-K item 2.02, {e.get('form')})",
            {"source": e["source"], "id": e["accession"]})
        add("WHEN KNOWN", "PASS", f"accepted {e['accepted_at'][:19]} UTC ({e['timing']}); reaction session {e['reaction_session']}",
            {"available_at": e["available_at"]})
        add("PRICE RESPONSE", "PASS" if pe["reaction_direction"] == "POSITIVE" else
            ("FAIL" if pe["reaction_direction"] == "NEGATIVE" else "UNKNOWN" if pe["reaction_direction"] == "UNKNOWN" else "NONE"),
            f"{pe['reaction_direction'].lower()}: abnormal {_pct(pe['reaction_abnormal_return'])}; "
            f"since reaction {_pct(pe['since_reaction_abnormal_return'])} ({pe['continuation'].lower()})")
        add("VOLUME RESPONSE", "PASS" if pe["volume"] == "CONFIRMED" else ("UNKNOWN" if pe["volume"] == "UNKNOWN" else "FAIL"),
            f"reaction-day dollar volume {pe['event_rel_volume']:.1f}x normal" if pe.get("event_rel_volume") else "UNKNOWN")
    elif me.get("state") in ("FIRED", "PRESENT"):
        src = (me.get("news") or [{}])[0] if me.get("news") else (me.get("sec_8k") or [{}])[0]
        add("EVENT", "PRESENT", "company event: " + ", ".join(me.get("categories") or []),
            {"source": src.get("source"), "id": src.get("news_id") or src.get("accession")})
        add("WHEN KNOWN", "PASS", f"available {str(src.get('created_at') or src.get('accepted_at'))[:19]} UTC",
            {"available_at": src.get("created_at") or src.get("accepted_at")})
        add("PRICE RESPONSE", {"POSITIVE": "PASS", "NEGATIVE": "FAIL", "MUTED": "NONE"}.get(me["direction"], "UNKNOWN"),
            f"{me['direction'].lower()}: today {_pct(me.get('move_return'))}")
        add("VOLUME RESPONSE", "PASS" if me["volume"] == "CONFIRMED" else ("UNKNOWN" if me["volume"] == "UNKNOWN" else "FAIL"),
            f"dollar volume {me['rel_volume']:.1f}x normal" if me.get("rel_volume") is not None else "UNKNOWN")
    else:
        known = cat.get("catalyst_known", False)
        add("EVENT", "NONE" if known else "UNKNOWN", "no catalyst found" if known else
            "catalyst sources do not cover this symbol (not evidence of no event)")
        add("WHEN KNOWN", "NOT_APPLICABLE", "no event")
        ret = (cat.get("features") or {}).get("ret_1d")
        add("PRICE RESPONSE", "NOT_APPLICABLE", f"technical setup; today {_pct(ret)}")
        rv = (cat.get("features") or {}).get("rel_volume_1d")
        add("VOLUME RESPONSE", "NOT_APPLICABLE", f"dollar volume {rv:.1f}x normal" if rv is not None else "UNKNOWN")
    ind = cat.get("industry") or {}
    st = ind.get("state", "UNKNOWN")
    sic = (ind.get("sic") or {})
    add("SECTOR/INDUSTRY", {"STRONG": "PASS", "WEAK": "FAIL", "NEUTRAL": "NONE"}.get(st, "UNKNOWN"),
        (f"SIC {sic.get('sic')} {str(sic.get('title') or '').title()}: industry rank {ind['industry_rank_63']:.2f}, "
         f"vs industry {_pct(ind.get('rs_industry_20'))} (20d)") if ind.get("industry_rank_63") is not None else
        ind.get("why", "UNKNOWN"), {"observed_at": sic.get("observed_at"), "accession": sic.get("accession")} if sic else None)
    fu = cat.get("fundamentals") or {}
    fst = fu.get("state", "UNKNOWN")
    add("FUNDAMENTALS", "PASS" if fst == "SUPPORTIVE" else ("FAIL" if fst == "WEAK" else "UNKNOWN" if fst == "UNKNOWN" else "NONE"),
        f"revenue {_pct(fu.get('rev_growth_yoy'), 0)} y/y, EPS {_pct(fu.get('eps_growth_yoy'), 0)} y/y, "
        f"op margin {_pct(fu.get('op_margin'), 0)} (latest filed, as-of)")
    checks = {x["stage"]: x for x in c.get("checks", [])}
    cov, sv, dg = checks.get("STRATEGY_COVERAGE"), checks.get("STRATEGY_VALIDATION"), checks.get("DECISION_GATE")
    if cov is None or not cov["passed"]:
        add("VALIDATION", "FAIL", (cov or {}).get("reason") or "no strategy covers this setup")
        add("RISK", "NOT_REACHED", "no strategy decision")
        add("EV", "NOT_REACHED", "no strategy decision")
    else:
        add("VALIDATION", "PASS" if sv and sv["passed"] else "FAIL", (sv or cov)["reason"])
        if dg is None:
            add("RISK", "NOT_REACHED", "no decision recorded")
            add("EV", "NOT_REACHED", "no decision recorded")
        elif dg["passed"]:
            add("RISK", "PASS", "risk chain passed")
            add("EV", "PASS", dg["reason"])
        else:
            stage = str(dg.get("reject_stage") or "")
            reason = dg["reason"]
            is_ev = "ev" in reason.lower() or "expectancy" in reason.lower()
            add("RISK", "FAIL" if stage in ("RISK", "PORTFOLIO", "EXECUTION") else "NOT_REACHED" if stage == "STRATEGY" else "PASS",
                reason if stage in ("RISK", "PORTFOLIO", "EXECUTION") else "not the blocking stage")
            add("EV", "FAIL" if is_ev else "NOT_REACHED" if stage == "STRATEGY" else "PASS",
                reason if is_ev else "not the blocking stage")
    status = c.get("status", "")
    add("PAPER ELIGIBILITY", "PASS" if status in ("PAPER_ELIGIBLE", "TRADED") else "FAIL",
        "PAPER ELIGIBLE (unchanged gates)" if status in ("PAPER_ELIGIBLE", "TRADED") else
        f"not eligible: {c.get('block_reason') or 'no validated strategy covers this setup'}")
    return rows


__all__ = ["CATALYST_FAMILIES", "CATALYST_LABEL", "CAT_FEATURES", "CHAIN_STAGES", "CatalystEngine", "CatalystTriggers",
           "catalyst_summary",
           "SETUP_CLASSES", "catalyst_text", "direction", "evidence_chain", "fundamental_state", "industry_state",
           "setup_class"]
