"""Private scoreboard page: research/alpha/scoreboard.json + the PAPER account's live record, as one HTML file.

Read-only. It only sends GET requests, to the paper trading host and the market-data host, and never places,
changes or cancels an order. The output holds account data, so it goes to the git-ignored
var/alpha/dashboard/ (this repository is public). Publish or open that file privately.

Usage: .venv/bin/python scripts/research/alpha/scoreboard_page.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from quantlab.alpha.store import DATA_URL, TRADING_URL, _auth  # noqa: E402

OUT = ROOT / "var" / "alpha" / "dashboard"
TEMPLATE = Path(__file__).with_name("scoreboard_page.html")
ET = "America/New_York"
MANUAL_NOTE = ("A manual ORCL call test (bought 2026-09-22 for $101, sold 09-25 for $14) and a 1-share SPY order test "
               "were run by hand, not by the bot.")


def _session() -> requests.Session:
    assert TRADING_URL == "https://paper-api.alpaca.markets"
    s = requests.Session()
    s.headers.update(_auth())
    return s


def account_record(s: requests.Session) -> dict:
    def get(path: str, **params):
        r = s.get(f"{TRADING_URL}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    acct = get("/v2/account")
    out = {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
           "account": {k: acct.get(k) for k in ("equity", "last_equity", "cash", "long_market_value", "created_at")},
           "positions": get("/v2/positions"),
           "history": get("/v2/account/portfolio/history", period="3M", timeframe="1D")}
    orders, after = [], "2026-09-01T00:00:00Z"
    while True:
        o = get("/v2/orders", status="all", limit=500, after=after, direction="asc")
        orders += o
        if len(o) < 500:
            break
        after = o[-1]["submitted_at"] or o[-1]["created_at"]
    out["orders"] = orders
    return out


def daily_bars(s: requests.Session, sym: str, start: str, adjustment: str = "raw") -> list[dict]:
    """SIP daily bars from ``start`` up to 20 minutes ago (the free plan refuses the most recent 15 minutes)."""
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    j = s.get(f"{DATA_URL}/v2/stocks/bars", params={"symbols": sym, "timeframe": "1Day", "start": f"{start}T00:00:00Z",
              "end": end, "feed": "sip", "adjustment": adjustment, "limit": 10000}, timeout=60).json()
    return (j.get("bars") or {}).get(sym, [])


def live_summary(s: requests.Session, r: dict) -> dict:
    o = pd.DataFrame(r["orders"])
    bot = o[o["client_order_id"].str.startswith("ql-bot")].copy()
    bot["filled_qty"] = bot["filled_qty"].astype(float)
    filled = bot[(bot["filled_qty"] > 0) & (bot["side"] == "buy")].copy()
    filled["day"] = pd.to_datetime(filled["filled_at"]).dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    fills = []
    for (day, sym), g in filled.groupby(["day", "symbol"]):
        b = [x for x in daily_bars(s, sym, day) if x["t"][:10] == day]
        if not b:
            continue                                   # no official bar yet: UNKNOWN, left out
        q = g["filled_qty"].sum()
        px = float((g["filled_qty"] * g["filled_avg_price"].astype(float)).sum() / q)
        fills.append({"day": day, "symbol": sym, "tif": "/".join(g["time_in_force"]), "filled": q,
                      "wanted": float(g["qty"].astype(float).max()), "fill_px": round(px, 4), "open": b[0]["o"],
                      "fill_vs_open_bps": round((px / b[0]["o"] - 1) * 1e4, 1)})
    h = r["history"]
    eq = pd.Series(h["equity"], index=pd.to_datetime(h["timestamp"], unit="s", utc=True).tz_convert(ET).date)
    eq = eq[eq > 0]
    fetched = pd.Timestamp(r["fetched_at"]).tz_convert(ET)
    # the account's value now belongs to today once the session has opened, otherwise to the last session's close
    t = fetched.tz_localize(None)
    if t.weekday() < 5 and (t.hour, t.minute) >= (9, 30):
        now_day = t.date()
    elif t.weekday() < 5:
        now_day = (t.normalize() - pd.offsets.BDay(1)).date()
    else:
        now_day = pd.offsets.BDay().rollback(t.normalize()).date()
    eq.loc[now_day] = float(r["account"]["equity"])
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    spy_bars = daily_bars(s, "SPY", str(eq.index[0]), adjustment="all")
    spy = {b["t"][:10]: b["c"] for b in spy_bars}
    spy_open = {b["t"][:10]: b["o"] for b in spy_bars}
    first_day = filled.sort_values("day").groupby("symbol")["day"].first().to_dict()
    pos = []
    for p in r["positions"]:
        d0 = first_day.get(p["symbol"])
        ret = spy[max(spy)] / spy_open[d0] - 1 if d0 in spy_open else None
        cost = float(p["avg_entry_price"]) * float(p["qty"])
        pos.append({"symbol": p["symbol"], "qty": float(p["qty"]), "entry": float(p["avg_entry_price"]),
                    "price": float(p["current_price"]), "value": float(p["market_value"]), "pl": float(p["unrealized_pl"]),
                    "pl_pct": float(p["unrealized_plpc"]), "entry_day": d0, "spy_ret": ret,
                    "spy_pl": cost * ret if ret is not None else None})
    opg = bot[bot["time_in_force"] == "opg"]
    day_orders = set(zip(bot.loc[bot["time_in_force"] == "day", "symbol"], bot.loc[bot["time_in_force"] == "day", "created_at"].str[:10]))
    lost = [f"{x.symbol} {x.created_at[:10]}" for x in opg.itertuples()
            if x.filled_qty == 0 and (x.symbol, x.created_at[:10]) not in day_orders]
    partial = [f"{x.symbol} {x.filled_qty:g} of {float(x.qty):g} ({x.created_at[5:10]})" for x in opg.itertuples()
               if 0 < x.filled_qty < float(x.qty)]
    acct = r["account"]
    return {"fetched_at_et": fetched.strftime("%Y-%m-%d %H:%M ET"), "start_capital": 100000.0,
            "equity": float(acct["equity"]), "cash": float(acct["cash"]), "invested": float(acct["long_market_value"]),
            "account_created": acct["created_at"][:10], "first_bot_order": bot["created_at"].min()[:10],
            "equity_curve": [{"d": str(k), "v": round(float(v), 2)} for k, v in eq.items()],
            "spy_curve": [{"d": k, "v": v} for k, v in sorted(spy.items())],
            "positions": sorted(pos, key=lambda p: -p["pl"]),
            "bot_pl": round(sum(p["pl"] for p in pos), 2), "bot_cost": round(sum(p["entry"] * p["qty"] for p in pos), 2),
            "spy_matched_pl": round(sum(p["spy_pl"] for p in pos if p["spy_pl"] is not None), 2),
            "manual_tests": MANUAL_NOTE,
            "execution": {"opg_orders": int(len(opg)), "partial": partial, "opg_partial": int(((opg["filled_qty"] > 0) & (opg["status"] != "filled")).sum()),
                          "lost_entries": lost, "fills": fills, "fill_vs_open_n": len(fills),
                          "fill_vs_open_bps_mean": round(sum(f["fill_vs_open_bps"] for f in fills) / len(fills), 1) if fills else None},
            "trading_days": int(len(pd.bdate_range(acct["created_at"][:10], str(eq.index[-1]))))}


def main() -> None:
    s = _session()
    rec = account_record(s)
    data = {"scoreboard": json.loads((ROOT / "research" / "alpha" / "scoreboard.json").read_text()), "live": live_summary(s, rec)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bot_record.json").write_text(json.dumps(rec, indent=1))
    page = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(data).replace("</", "<\\/"))
    (OUT / "quantlab-scoreboard.html").write_text(page)
    print(f"wrote {(OUT / 'quantlab-scoreboard.html').relative_to(ROOT)}: equity {data['live']['equity']:.2f}, "
          f"{len(data['live']['positions'])} positions, {data['scoreboard']['counts']['ideas']} ideas")


if __name__ == "__main__":
    main()
