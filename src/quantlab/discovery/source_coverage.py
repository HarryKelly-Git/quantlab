"""Catalyst data-source coverage: what is actually stored, per source, with its PIT quality.

Reported separately for SEC earnings, other SEC filings, SIC observations, news and fundamentals:
symbol coverage (of the stock universe = stored-bar symbols that are COMMON stock), row counts, date
range, latest availability, PIT status mix, UNKNOWN coverage with reasons, unmapped records and
obvious gaps. Nothing here is estimated: every number is a count over stored rows.
"""
from __future__ import annotations

import json
from typing import Any


import pandas as pd
import pyarrow.parquet as pq

SEC_KINDS = ("earnings_release", "periodic_report", "sec_8k", "foreign_report", "ownership_13d", "registration")


def _read(store, kind: str, columns: list[str], synthetic: bool = False) -> pd.DataFrame:
    frames = []
    for ds in store.dataset_ids(kind, synthetic=synthetic):
        row = store.db.fetchone("SELECT path FROM datasets WHERE dataset_id=?", (ds,))
        path = store.data_dir / row["path"]
        have = set(pq.read_schema(path).names)
        frames.append(pq.read_table(path, columns=[c for c in columns if c in have]).to_pandas())
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _dates(s: pd.Series) -> tuple[str | None, str | None]:
    t = pd.to_datetime(s, utc=True).dropna()
    return (str(t.min()), str(t.max())) if len(t) else (None, None)


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _monthly_gaps(t: pd.Series, windows: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Months inside the covered span (or the given ingestion windows) with no rows or < 25% of the median."""
    t = pd.to_datetime(t, utc=True).dropna()
    if t.empty:
        return []
    m = t.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").value_counts().sort_index()
    if windows:
        want = pd.PeriodIndex(sorted({p for a, b in windows
                                      for p in pd.period_range(pd.Timestamp(a).to_period("M"),
                                                               (pd.Timestamp(b) - pd.Timedelta(days=1)).to_period("M"))}))
    else:
        want = pd.period_range(m.index.min(), m.index.max(), freq="M")
    m = m.reindex(want, fill_value=0)
    med = float(m[m > 0].median()) if (m > 0).any() else 0.0
    return [{"month": str(p), "rows": int(n), "median": med} for p, n in m.items() if n == 0 or n < 0.25 * med]


def source_coverage(ctx, universe: list[str] | None = None, as_of: Any = None, synthetic: bool = False) -> dict[str, Any]:
    """``universe``: the symbols to measure against (default: stored-bar COMMON stocks). ``as_of``:
    the decision cutoff (rows available after it are ignored). Synthetic and real never mix."""
    from quantlab.data.sec_catalysts import catalyst_symbols
    store = ctx.store
    uni = sorted(universe) if universe is not None else catalyst_symbols(store)
    U = set(uni)
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    out: dict[str, Any] = {"universe": {"definition": "stored-bar symbols marked COMMON stock (not ETF/test)",
                                        "symbols": len(uni)}, "as_of": str(now)}

    ev = _read(store, "events", ["symbol", "event_type", "available_at", "pit_status", "payload_json", "provider"], synthetic)
    # decision inputs are cut at the decision time; the registrant snapshot (CIK mapping, filer type)
    # is explanatory metadata for the UNKNOWN breakdown only, stamped with its retrieval time
    ev = ev[(pd.to_datetime(ev["available_at"], utc=True) <= now) | (ev["event_type"] == "sec_registrant")] if len(ev) else ev
    reg = ev[ev["event_type"] == "sec_registrant"].drop_duplicates("symbol", keep="last") if len(ev) else ev
    filer = {r.symbol: (json.loads(r.payload_json) or {}).get("filer", "UNKNOWN") for r in reg.itertuples()}
    mapped = set(filer)
    unmapped = sorted(U - mapped)
    foreign = {s for s, f in filer.items() if f == "FOREIGN"}
    domestic = {s for s, f in filer.items() if f == "DOMESTIC"}

    # ---- SEC earnings ----------------------------------------------------------------------------
    er = ev[ev["event_type"] == "earnings_release"] if len(ev) else ev
    er_syms = set(er["symbol"]) & U
    timing = er["payload_json"].map(lambda p: (json.loads(p) or {}).get("timing", "UNKNOWN")).value_counts().to_dict() \
        if len(er) else {}
    amend = int(er["payload_json"].astype(str).str.contains('"form": "8-K/A"', regex=False).sum()) if len(er) else 0
    recent = er[pd.to_datetime(er["available_at"], utc=True) >= now - pd.Timedelta(days=400)] if len(er) else er
    stale_domestic = sorted((domestic & U) - set(recent["symbol"]))
    lo, hi = _dates(er["available_at"]) if len(er) else (None, None)
    out["sec_earnings"] = {
        "source": "SEC EDGAR 8-K / 8-K/A item 2.02; available_at = acceptanceDateTime (UTC)",
        "symbols_covered": len(er_syms), "coverage_pct": _pct(len(er_syms), len(uni)),
        "events": int(len(er)), "amendments_8k_a": amend, "first": lo, "latest_available_at": hi,
        "pit_status": er["pit_status"].value_counts().to_dict() if len(er) else {},
        "timing_of_sec_acceptance": timing,
        "unknown": {"total": len(U - er_syms), "no_cik_mapping": len(set(unmapped)),
                    "foreign_filer_no_8k_items": len((foreign & U) - er_syms),
                    "domestic_no_2_02_in_window": len((domestic & U) - er_syms),
                    "other": len(U - er_syms - set(unmapped) - foreign - domestic)},
        "unmapped_records": {"symbols_without_cik": len(unmapped), "sample": unmapped[:15]},
        "gaps": {"domestic_filers_without_2_02_in_last_400_days": len(stale_domestic), "sample": stale_domestic[:15]},
        "usable": bool(len(er_syms)),
    }
    # ---- other SEC filings -----------------------------------------------------------------------
    oth = ev[ev["event_type"].isin([k for k in SEC_KINDS if k != "earnings_release"])] if len(ev) else ev
    lo, hi = _dates(oth["available_at"]) if len(oth) else (None, None)
    m8 = oth[oth["event_type"] == "sec_8k"] if len(oth) else oth
    cats: dict[str, int] = {}
    for p in m8["payload_json"] if len(m8) else []:
        for c in (json.loads(p) or {}).get("categories", []):
            cats[c] = cats.get(c, 0) + 1
    out["sec_other_filings"] = {
        "source": "SEC EDGAR submissions (10-Q/10-K, material 8-K items, 20-F/40-F/6-K, SC 13D, S-1/S-3)",
        "symbols_covered": len(set(oth["symbol"]) & U) if len(oth) else 0,
        "coverage_pct": _pct(len(set(oth["symbol"]) & U) if len(oth) else 0, len(uni)),
        "rows": int(len(oth)), "by_type": oth["event_type"].value_counts().to_dict() if len(oth) else {},
        "material_8k_categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "first": lo, "latest_available_at": hi,
        "pit_status": oth["pit_status"].value_counts().to_dict() if len(oth) else {},
        "unknown": {"total": len(U - (set(oth["symbol"]) if len(oth) else set())), "no_cik_mapping": len(unmapped)},
        "usable": bool(len(oth)),
    }
    # ---- SIC observations --------------------------------------------------------------------------
    sic = ev[ev["event_type"] == "sic_observation"] if len(ev) else ev
    codes = sic.assign(code=sic["payload_json"].map(lambda p: (json.loads(p) or {}).get("sic"))) if len(sic) else sic
    changed = int((codes.groupby("symbol")["code"].nunique() > 1).sum()) if len(codes) else 0
    cur_mismatch = 0
    for r in reg.itertuples() if len(reg) else []:
        pl = json.loads(r.payload_json) or {}
        last = codes[codes["symbol"] == r.symbol]["code"].iloc[-1] if len(codes) and (codes["symbol"] == r.symbol).any() else None
        if last is not None and pl.get("sic_current") and str(pl["sic_current"]).zfill(4) != str(last).zfill(4):
            cur_mismatch += 1
    lo, hi = _dates(sic["available_at"]) if len(sic) else (None, None)
    s_syms = set(sic["symbol"]) & U if len(sic) else set()
    out["sic_observations"] = {
        "source": "SIC from each periodic report's SGML header, as of its acceptance (first/last filing, binary search on change)",
        "symbols_covered": len(s_syms), "coverage_pct": _pct(len(s_syms), len(uni)), "observations": int(len(sic)),
        "symbols_with_sic_change_in_window": changed,
        "latest_header_differs_from_current_snapshot": cur_mismatch,
        "first": lo, "latest_available_at": hi,
        "pit_status": sic["pit_status"].value_counts().to_dict() if len(sic) else {},
        "unknown": {"total": len(U - s_syms), "no_cik_mapping": len(unmapped),
                    "no_periodic_report_or_blank_sic": len((mapped & U) - s_syms)},
        "usable": bool(len(s_syms)),
    }
    # ---- news ------------------------------------------------------------------------------------
    nw = _read(store, "news", ["news_id", "symbol", "created_at", "updated_at", "available_at", "pit_status", "provider"],
               synthetic)
    nw = nw[pd.to_datetime(nw["available_at"], utc=True) <= now] if len(nw) else nw
    nw = nw.drop_duplicates(["news_id", "symbol"]) if len(nw) else nw
    windows = []
    for r in store.db.fetchall("SELECT params_json FROM datasets WHERE kind='news' AND is_synthetic=?", (int(synthetic),)):
        pj = json.loads(r["params_json"] or "{}")
        if pj.get("what") == "market_news":
            windows.append((pj["start"][:10], pj["end"][:10]))
    n_syms = set(nw["symbol"]) & U if len(nw) else set()
    tags = nw.groupby("news_id")["symbol"].transform("size") if len(nw) else pd.Series(dtype=float)
    cs = nw[(tags <= 2).to_numpy()] if len(nw) else nw
    rev = (pd.to_datetime(nw["updated_at"], utc=True) - pd.to_datetime(nw["created_at"], utc=True)) if len(nw) else pd.Series(dtype="timedelta64[ns]")
    lo, hi = _dates(nw["available_at"]) if len(nw) else (None, None)
    not_uni = sorted(set(nw["symbol"]) - U) if len(nw) else []
    out["news"] = {
        "source": "Alpaca news (Benzinga), market-wide; available_at = created_at; window selected by updated_at",
        "symbols_covered": len(n_syms), "coverage_pct": _pct(len(n_syms), len(uni)),
        "symbols_with_company_specific_news": len(set(cs["symbol"]) & U) if len(cs) else 0,
        "articles": int(nw["news_id"].nunique()) if len(nw) else 0, "rows": int(len(nw)),
        "ingested_windows": sorted(windows), "first": lo, "latest_available_at": hi,
        "pit_status": nw["pit_status"].value_counts().to_dict() if len(nw) else {},
        "revised_after_60s_pct": round(100 * float((rev > pd.Timedelta(seconds=60)).mean()), 2) if len(nw) else None,
        "unknown": {"total": len(U - n_syms), "reason": "no article tagged with the symbol in the ingested windows "
                                                          "(absence of news is not evidence of no news)"},
        "unmapped_records": {"tags_not_in_stock_universe": len(not_uni),
                             "note": "tags kept as-is (ETFs, crypto, OTC, foreign lines); never dropped or remapped",
                             "sample": not_uni[:15]},
        "gaps": {"months_empty_or_under_25pct_of_median": _monthly_gaps(nw["available_at"], windows) if len(nw) else []},
        "usable": bool(len(n_syms)),
    }
    # ---- fundamentals ----------------------------------------------------------------------------
    fu = _read(store, "fundamentals", ["symbol", "concept", "fiscal_period", "available_at", "pit_status", "provider"],
               synthetic)
    fu = fu[pd.to_datetime(fu["available_at"], utc=True) <= now] if len(fu) else fu
    f_syms = set(fu["symbol"]) & U if len(fu) else set()
    lo, hi = _dates(fu["available_at"]) if len(fu) else (None, None)
    per_concept = {c: len(set(g["symbol"]) & U) for c, g in fu.groupby("concept")} if len(fu) else {}
    out["fundamentals"] = {
        "source": "SEC XBRL companyfacts; available_at = acceptanceDateTime of the reporting filing (accession join); "
                  "'frame' never used",
        "symbols_covered": len(f_syms), "coverage_pct": _pct(len(f_syms), len(uni)), "facts": int(len(fu)),
        "symbols_per_concept": dict(sorted(per_concept.items())), "first": lo, "latest_available_at": hi,
        "pit_status": fu["pit_status"].value_counts().to_dict() if len(fu) else {},
        "unknown": {"total": len(U - f_syms), "no_cik_mapping": len(unmapped),
                    "mapped_without_xbrl_facts": len((mapped & U) - f_syms)},
        "usable": bool(len(f_syms)),
    }
    return out


def coverage_summary(cov: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact rows for the dashboard: label, coverage %, usable, PIT, latest availability."""
    rows = []
    for key, label in (("sec_earnings", "Earnings"), ("news", "News"), ("sic_observations", "Industry"),
                       ("fundamentals", "Fundamentals"), ("sec_other_filings", "SEC events")):
        c = cov.get(key) or {}
        pit = c.get("pit_status") or {}
        rows.append({"key": key, "label": label, "coverage_pct": c.get("coverage_pct"),
                     "symbols": c.get("symbols_covered"), "universe": (cov.get("universe") or {}).get("symbols"),
                     "first": (c.get("first") or "")[:10] or None, "latest": (c.get("latest_available_at") or "")[:16] or None,
                     "pit": ", ".join(f"{k} {v}" for k, v in pit.items()) or "none",
                     "usable": bool(c.get("usable")), "unknown": (c.get("unknown") or {}).get("total"),
                     "source": c.get("source")})
    return rows


__all__ = ["source_coverage", "coverage_summary"]

