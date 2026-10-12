"""Score recent open-market insider purchases (OpenInsider screen, parsed JSON) with price data from Alpaca.
Rules are fixed here BEFORE looking at any forward return; this is a screen, not a validated signal."""
import json, re, sys, time
from collections import defaultdict
import pandas as pd, requests
sys.path.insert(0, "/home/user/quantlab/src")
from quantlab.alpha.store import DATA_URL, _auth

S = sys.argv[1]
ASOF = pd.Timestamp("2026-10-07")
SINCE = ASOF - pd.Timedelta(days=90)
NONOP = re.compile(r"\b(fund|etf|trust inc|income|credit|bdc|acquisition corp|capital corp|closed|portfolio|lp\b|l\.p\.)", re.I)
TOP = re.compile(r"\b(ceo|cfo|pres|coo|cob|chair|chief executive|chief financial|founder)\b", re.I)

def num(x):
    x = (x or "").replace("$", "").replace(",", "").replace("+", "").replace("%", "").strip()
    try: return float(x)
    except ValueError: return None

rows = [r for r in json.load(open(f"{S}/oi_screen.json"))["rows"] if r["Trade Type"].startswith("P")]
for f in ("cluster", "topweek", "topmonth", "uber"):
    rows += [r for r in json.load(open(f"{S}/oi_{f}.json"))["rows"] if r.get("Trade Type", "").startswith("P") and r.get("Insider Name")]
rows = [r for r in rows if SINCE <= pd.Timestamp(r["Trade Date"]) <= ASOF]
seen, uniq = set(), []
for r in rows:
    k = (r["Ticker"], r.get("Insider Name"), r["Trade Date"], r["Qty"])
    if k not in seen:
        seen.add(k); uniq.append(r)
by = defaultdict(list)
for r in uniq:
    if r.get("Company Name") and NONOP.search(r["Company Name"]):
        continue
    by[r["Ticker"].strip(". ")].append(r)

s = requests.Session(); s.headers.update(_auth())
end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
def bars(syms):
    out = {}
    for k in range(0, len(syms), 50):
        tok = None
        while True:
            p = {"symbols": ",".join(syms[k:k+50]), "timeframe": "1Day", "start": "2025-07-01T00:00:00Z", "end": end,
                 "feed": "sip", "adjustment": "all", "limit": 10000}
            if tok: p["page_token"] = tok
            j = s.get(f"{DATA_URL}/v2/stocks/bars", params=p, timeout=60).json()
            for sym, b in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend(b)
            tok = j.get("next_page_token")
            if not tok: break
        time.sleep(0.4)
    return out
B = bars(sorted(by))
res = []
for t, rs in by.items():
    b = B.get(t)
    if not b or len(b) < 40:
        continue
    df = pd.DataFrame(b); df["d"] = pd.to_datetime(df.t.str[:10]); df = df.set_index("d").sort_index()
    names = {r["Insider Name"] for r in rs}
    top = {r["Insider Name"] for r in rs if TOP.search(r.get("Title", ""))}
    only10 = all(re.fullmatch(r"\s*10%\s*", r.get("Title", "")) for r in rs)
    val = sum(num(r["Value"]) or 0 for r in rs)
    first = min(pd.Timestamp(r["Trade Date"]) for r in rs); last = max(pd.Timestamp(r["Trade Date"]) for r in rs)
    qty = sum(num(r["Qty"]) or 0 for r in rs)
    vwap = val / qty if qty else None
    pre = df.c.loc[:first]
    hi52 = pre.iloc[-252:].max() if len(pre) else None
    dd_at_buy = (pre.iloc[-1] / hi52 - 1) if len(pre) and hi52 else None
    adv = float((df.c * df.v).iloc[-60:].median())
    age_days = (df.index[-1] - df.index[0]).days
    px = float(df.c.iloc[-1])
    round_px = all(abs((num(r["Price"]) or 0) - round(num(r["Price"]) or 0)) < 1e-9 for r in rs)
    score = 0
    score += 2 if any(re.search(r"\b(ceo|cfo|chief executive|chief financial)\b", r.get("Title", ""), re.I) for r in rs) else (1 if top else 0)
    score += 3 if len(names) >= 3 else (2 if len(names) == 2 else 0)
    score += 2 if val >= 5e6 else (1 if val >= 1e6 else 0)
    score += 1 if max((num(r["ΔOwn"]) or 0) for r in rs) >= 10 else 0
    score += 1 if dd_at_buy is not None and dd_at_buy <= -0.20 else 0
    flags = []
    if only10: flags.append("10%-holder only"); score -= 3
    if adv < 5e6: flags.append("illiquid"); score -= 2
    if px < 5: flags.append("under $5")
    if age_days < 200 and b[0]["t"][:10] > "2025-07-15": flags.append("recent listing")
    if round_px and len(names) >= 2: flags.append("round-price (offering?)")
    res.append({"ticker": t, "company": rs[0].get("Company Name", ""), "score": score, "insiders": len(names),
                "top_buyers": sorted(top)[:4], "titles": sorted({r.get("Title", "") for r in rs})[:4], "value": round(val),
                "first": str(first.date()), "last": str(last.date()), "vwap": round(vwap, 2) if vwap else None, "price": px,
                "since_first_buy": round(px / pre.iloc[-1] - 1, 4) if len(pre) else None,
                "dd_from_52w_high_at_buy": round(dd_at_buy, 3) if dd_at_buy is not None else None,
                "adv_m": round(adv / 1e6, 1), "flags": flags})
res.sort(key=lambda x: (-x["score"], -x["value"]))
json.dump(res, open(f"{S}/insider_scored.json", "w"), indent=1)
print(len(res), "tickers scored (with Alpaca data)")
for x in res[:40]:
    print(f"{x['score']:2} {x['ticker']:6} {x['company'][:26]:26} ins={x['insiders']} ${x['value']/1e6:6.2f}M {x['first']}..{x['last']} paid~{x['vwap']} now {x['price']:.2f} "
          f"({x['since_first_buy']:+.1%}) dd@buy {x['dd_from_52w_high_at_buy']} adv ${x['adv_m']}M top={x['top_buyers'][:2]} {x['titles'][:2]} {x['flags']}")
