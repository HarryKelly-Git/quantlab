"""Event study, step 2 (docs/EVENT-STUDY-PREREG.md + amendment 1): classify events with Benzinga news, then
run the pre-registered tests E1, E2a, E2a-v2, E2b, E2b-v2 and E3 (ATM calls), plus descriptive tables.
PAPER research, 2016-2024 only.

Writes research/alpha/results/events/event_study.json and var/alpha/cache/events_classified.parquet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha import events_study as es  # noqa: E402
from quantlab.alpha import opt_exec as ox  # noqa: E402
from quantlab.alpha import registry  # noqa: E402
from quantlab.alpha.experiments.options_vol import driscoll_kraay_se  # noqa: E402
from quantlab.alpha.options_store import options_dir  # noqa: E402
from quantlab.alpha.store import store_dir  # noqa: E402

OUT = ROOT / "research" / "alpha" / "results" / "events"
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2021-12-31"), "OOS": ("2022-01-01", "2024-12-31")}
OSPLITS = {"TRAIN": ("2019-02-01", "2021-12-31"), "VAL": ("2022-01-01", "2022-12-31"), "OOS": ("2023-01-01", "2024-12-31")}
SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
T_SIG = 2.5


def split_of(d: pd.Series, splits=SPLITS) -> np.ndarray:
    d = pd.to_datetime(d)
    out = np.full(len(d), "NONE", dtype=object)
    for k, (a, b) in splits.items():
        out[((d >= a) & (d <= b)).to_numpy()] = k
    return out


def summ(x: pd.Series, dates: pd.Series) -> dict:
    m, t, nd = es.cluster_t(x, dates)
    z = x.dropna()
    return {"n": int(len(z)), "n_dates": nd, "mean": m, "t": t, "median": float(z.median()) if len(z) else None,
            "hit": float((z > 0).mean()) if len(z) else None}


def diff_test(x: pd.Series, flag: pd.Series, dates: pd.Series) -> dict:
    """Mean(x | flag) - mean(x | not flag) with date-clustered SE (regression on a dummy)."""
    z = pd.DataFrame({"x": x.to_numpy(), "f": flag.astype(float).to_numpy(), "d": dates.astype(str).to_numpy()}).dropna()
    if z["f"].nunique() < 2 or len(z) < 30:
        return {"n": int(len(z))}
    X = np.column_stack([np.ones(len(z)), z["f"].to_numpy()])
    b = np.linalg.lstsq(X, z["x"].to_numpy(), rcond=None)[0]
    se = driscoll_kraay_se(X, z["x"].to_numpy() - X @ b, z["d"].to_numpy(), 0)
    return {"n": int(len(z)), "diff": float(b[1]), "t": float(b[1] / se[1])}


def verdict(by_split: dict, key: str = "mean") -> str:
    tr, va, oo = by_split.get("TRAIN", {}), by_split.get("VAL", {}), by_split.get("OOS", {})
    def ok(s, t=T_SIG):
        return (s.get(key) or -1) > 0 and (s.get("t") or 0) >= t
    years = by_split.get("OOS_years", {})
    yrs_ok = sum(1 for v in years.values() if (v.get(key) or -1) > 0) >= 2 and (oo.get("t") or 0) >= 2
    if ok(tr) and ok(va) and (ok(oo) or ((oo.get(key) or -1) > 0 and yrs_ok)):
        return "SURVIVES"
    if (tr.get(key) or -1) > 0 and (va.get(key) or -1) > 0 and (oo.get(key) or -1) > 0:
        return "PROMISING (forward paper-shadow only)"
    return "FAILS"


def by_splits(df: pd.DataFrame, col: str, date_col: str = "date") -> dict:
    out = {}
    for sp in SPLITS:
        g = df[df["split"] == sp]
        out[sp] = summ(g[col], g[date_col])
    oos = df[df["split"] == "OOS"]
    out["OOS_years"] = {str(y): summ(g[col], g[date_col]) for y, g in oos.groupby(pd.to_datetime(oos[date_col]).dt.year)}
    return out


def calendar() -> pd.DatetimeIndex:
    b = pd.read_parquet(store_dir() / "bars_daily.parquet", columns=["symbol", "date"], filters=[("symbol", "==", "SPY")])
    return pd.DatetimeIndex(sorted(pd.to_datetime(b["date"]).unique()))


def etf_returns() -> pd.DataFrame:
    cols = pd.read_parquet(store_dir() / "bars_daily.parquet", columns=None, filters=[("symbol", "in", list(SECTOR_ETFS))])
    px = "adj_close" if "adj_close" in cols.columns else "close"
    w = cols.pivot(index="date", columns="symbol", values=px).sort_index()
    w.index = pd.to_datetime(w.index)
    return w / w.shift(1) - 1


def call_leg(ev: pd.DataFrame) -> pd.DataFrame:
    """E3: the ~30-DTE ATM call at the first snapshot whose session is AFTER the event day; held to expiry."""
    feat = pd.read_parquet(options_dir() / "features.parquet",
                           columns=["date", "session", "act_symbol", "expiration", "spot", "k_atm", "dte", "exp_close",
                                    "split_or_special_div", "straddle_ask", "straddle_bid", "straddle_mid"])
    feat = feat[(feat["dte"] >= 21) & (feat["dte"] <= 45) & ~feat["split_or_special_div"].astype(bool)].copy()
    feat["session"] = pd.to_datetime(feat["session"])
    feat["dd"] = (feat["dte"] - 30).abs()
    feat = feat.loc[feat.groupby(["date", "act_symbol"])["dd"].idxmin()]
    feat = feat.sort_values("session")
    ev = ev.sort_values("date").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    m = pd.merge_asof(ev[["date", "ticker", "entity", "mdv20"]].rename(columns={"ticker": "act_symbol"}),
                      feat.rename(columns={"session": "snap_session", "date": "snap_date"}),
                      left_on="date", right_on="snap_session", by="act_symbol", direction="forward",
                      allow_exact_matches=False)
    m = m.dropna(subset=["snap_session", "k_atm", "exp_close"])
    m["lag_days"] = (m["snap_session"] - m["date"]).dt.days
    need = m[["snap_date", "act_symbol", "expiration", "k_atm"]]
    months = sorted(set(pd.to_datetime(need["snap_date"]).dt.strftime("%Y-%m")))
    quotes = []
    for mo in months:
        f = options_dir() / f"chain_{mo}.parquet"
        if not f.exists():
            continue
        ch = pd.read_parquet(f, columns=["date", "act_symbol", "expiration", "cp", "strike", "bid", "ask"])
        ch = ch[ch["cp"] == "C"]
        n = need[pd.to_datetime(need["snap_date"]).dt.strftime("%Y-%m") == mo]
        q = n.merge(ch, left_on=["snap_date", "act_symbol", "expiration", "k_atm"],
                    right_on=["date", "act_symbol", "expiration", "strike"], how="inner")
        quotes.append(q[["snap_date", "act_symbol", "expiration", "k_atm", "bid", "ask"]])
    q = pd.concat(quotes, ignore_index=True).drop_duplicates(["snap_date", "act_symbol", "expiration", "k_atm"])
    m = m.merge(q, on=["snap_date", "act_symbol", "expiration", "k_atm"], how="inner")
    m = m[(m["bid"] > 0) & (m["ask"] > m["bid"])]
    k, st = m["k_atm"].to_numpy(), m["exp_close"].to_numpy()
    lv = ox.long_structure_returns(m["bid"].to_numpy(), m["ask"].to_numpy(), 1, st, np.maximum(st - k, 0.0),
                                   (st > k).astype(int), m["mdv20"].to_numpy())
    for lvl, r in lv.items():
        m[f"ret_{lvl}"] = r["ret"]
    m["call_spread_pct"] = (m["ask"] - m["bid"]) / ((m["ask"] + m["bid"]) / 2)
    return m


def main() -> None:
    t0 = time.time()
    cache = store_dir() / "cache"
    ev = pd.read_parquet(cache / "events.parquet")
    lag = pd.read_parquet(cache / "events_laggards.parquet")
    rows = []
    for line in (store_dir() / "external" / "events_news.jsonl").read_text().splitlines():
        rec = json.loads(line)
        rows += rec["articles"]
    news = pd.DataFrame(rows).drop_duplicates("id")
    dates = calendar()
    assert dates.max() < pd.Timestamp("2025-01-01")
    ev = es.classify(ev, news, dates)
    etf = etf_returns()
    ev["sector_etf_ret_t"] = [etf.at[pd.Timestamp(d), s] if s in etf.columns and pd.Timestamp(d) in etf.index else np.nan
                              for d, s in zip(ev["date"], ev["sector"])]
    ev["theme"] = ev["sector_etf_ret_t"] >= 0.02
    ev["split"] = split_of(ev["date"])
    for h in es.HORIZONS:
        ev[f"net_{h}"] = ev[f"excess_{h}"] - ev["cost_rt"]
    lag["split"] = split_of(lag["date"])
    for h in (1, 5, 20):
        lag[f"net_{h}"] = lag[f"excess_{h}"] - lag["cost_rt"]
    theme_days = set(zip(pd.to_datetime(ev.loc[ev["theme"] & (ev["side"] == "up"), "date"]),
                         ev.loc[ev["theme"] & (ev["side"] == "up"), "sector"]))
    lag["theme_day"] = [(pd.Timestamp(d), s) in theme_days for d, s in zip(lag["date"], lag["sector"])]
    ev.to_parquet(cache / "events_classified.parquet", index=False)
    up = ev[ev["side"] == "up"]
    res: dict = {"prereg": "docs/EVENT-STUDY-PREREG.md (+ amendment 1)", "git": registry.git_commit(),
                 "news": {"articles": int(len(news)), "events_with_any_article": int((ev["n_articles"] > 0).sum()),
                          "events": int(len(ev))},
                 "counts": {"up": int(len(up)), "down": int((ev["side"] == "down").sum()),
                            "up_news": int(up["news"].sum()), "up_no_news": int((~up["news"]).sum()),
                            "up_categories": up["category"].value_counts().to_dict(),
                            "up_sector_wide": int(up["sector_wide"].sum()), "up_theme_v2": int(up["theme"].sum()),
                            "laggards": int(len(lag)), "laggards_theme_v2": int(lag["theme_day"].sum())}}
    # E1
    news_non_ma = up[up["news"] & (up["category"] != "ma_target")]
    no_news = up[~up["news"]]
    e1 = {"news_non_ma_net20": by_splits(news_non_ma, "net_20"), "no_news_net20": by_splits(no_news, "net_20"),
          "diff_news_minus_no_news_net20": {sp: diff_test(g["net_20"], g["news"], g["date"]) for sp, g in
                                            up[up["category"] != "ma_target"].groupby("split")}}
    e1["verdict"] = verdict(e1["news_non_ma_net20"])
    e1["horizon_profile_news_non_ma"] = {f"net_{h}": by_splits(news_non_ma, f"net_{h}") for h in es.HORIZONS}
    e1["horizon_profile_no_news"] = {f"net_{h}": by_splits(no_news, f"net_{h}") for h in es.HORIZONS}
    res["E1"] = e1
    # E2a (registered) and E2a-v2 (theme)
    res["E2a"] = {"sector_wide_net20": by_splits(up[up["sector_wide"]], "net_20"),
                  "idiosyncratic_net20": by_splits(up[~up["sector_wide"]], "net_20"),
                  "diff": {sp: diff_test(g["net_20"], g["sector_wide"], g["date"]) for sp, g in up.groupby("split")}}
    res["E2a"]["verdict_diff"] = verdict({sp: {"mean": v.get("diff"), "t": v.get("t")} for sp, v in res["E2a"]["diff"].items()})
    res["E2a_v2_theme"] = {"theme_net20": by_splits(up[up["theme"]], "net_20"),
                           "non_theme_net20": by_splits(up[~up["theme"]], "net_20"),
                           "diff": {sp: diff_test(g["net_20"], g["theme"], g["date"]) for sp, g in up.groupby("split")}}
    res["E2a_v2_theme"]["verdict_diff"] = verdict({sp: {"mean": v.get("diff"), "t": v.get("t")} for sp, v in res["E2a_v2_theme"]["diff"].items()})
    res["E2a_v2_theme"]["theme_net20_verdict"] = verdict(res["E2a_v2_theme"]["theme_net20"])
    # E2b (registered) and E2b-v2
    res["E2b"] = {"laggards_net5": by_splits(lag, "net_5")}
    res["E2b"]["verdict"] = verdict(res["E2b"]["laggards_net5"])
    lt = lag[lag["theme_day"]]
    res["E2b_v2_theme"] = {"laggards_net5": by_splits(lt, "net_5"), "laggards_net1": by_splits(lt, "net_1"),
                           "laggards_net20": by_splits(lt, "net_20")}
    res["E2b_v2_theme"]["verdict"] = verdict(res["E2b_v2_theme"]["laggards_net5"])
    print(f"E1 {e1['verdict']}; E2a {res['E2a']['verdict_diff']}; E2a-v2 {res['E2a_v2_theme']['verdict_diff']}; "
          f"E2b {res['E2b']['verdict']}; E2b-v2 {res['E2b_v2_theme']['verdict']} ({time.time() - t0:.0f}s)", flush=True)
    # descriptive
    res["descriptive"] = {
        "up_by_category_net20": {c: by_splits(g, "net_20") for c, g in up.groupby("category")},
        "up_new_52w_high_net20": {str(k): by_splits(g, "net_20") for k, g in up.groupby("new_52w_high")},
        "down_news_net20": by_splits(ev[(ev["side"] == "down") & ev["news"]], "net_20"),
        "down_no_news_net20": by_splits(ev[(ev["side"] == "down") & ~ev["news"]], "net_20"),
        "theme_and_news_net20": by_splits(up[up["theme"] & up["news"] & (up["category"] != "ma_target")], "net_20"),
    }
    # E3: ATM calls after NEWS non-M&A UP events (options era)
    cl = call_leg(news_non_ma[pd.to_datetime(news_non_ma["date"]) >= pd.Timestamp("2019-02-01")])
    cl["osplit"] = split_of(cl["date"], OSPLITS)
    e3 = {"n_matched": int(len(cl)), "snapshot_lag_days_median": float(cl["lag_days"].median()) if len(cl) else None,
          "call_spread_pct_median": float(cl["call_spread_pct"].median()) if len(cl) else None}
    for sp in OSPLITS:
        g = cl[cl["osplit"] == sp]
        levels = {lvl: {"ret": g[f"ret_{lvl}"].to_numpy()} for lvl in ox.LEVELS}
        e3[sp] = ox.summarize_levels(levels, pd.to_datetime(g["snap_session"]).to_numpy()) if len(g) else {}
        e3[f"{sp}_class"] = ox.classify(e3[sp]) if len(g) else None
    e3["verdict"] = "SURVIVES" if e3.get("OOS_class") == "ROBUST" else "FAILS"
    res["E3_calls"] = e3
    cl.to_parquet(cache / "events_calls.parquet", index=False)
    print(f"E3 {e3['verdict']} (OOS class {e3.get('OOS_class')}); {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "event_study.json").write_text(json.dumps(registry._clean(res), indent=1, default=str))
    registry.append_run(hypothesis_id="EV_news_jumps", family="event_study", split="OOS",
                        spec={"prereg": "EVENT-STUDY-PREREG + amendment 1", "up": es.UP, "vol_mult": es.VOL_MULT,
                              "cooldown": es.COOLDOWN},
                        metrics={"E1": e1["verdict"], "E2a": res["E2a"]["verdict_diff"], "E2a_v2": res["E2a_v2_theme"]["verdict_diff"],
                                 "E2b": res["E2b"]["verdict"], "E2b_v2": res["E2b_v2_theme"]["verdict"], "E3": e3["verdict"]},
                        conclusion="see research/alpha/results/events/event_study.json", seed=7)


if __name__ == "__main__":
    main()
