"""Plain-English research scoreboard: research/alpha/scoreboard.json. PAPER research.

One row per idea tested, with a verdict a non-specialist can read, the one number that decided it, and the
file that holds the evidence. Numbers for the newest studies are read from their result files; the older ones
are quoted from the final reports (docs/ALPHA-DISCOVERY-REPORT-2026-10.md, docs/ALPHA-SPRINT-FINAL.md,
docs/WHAT-THE-BOT-NEEDS.md), which are the reviewed source for those results. Nothing here is a buy verdict.

The live paper account is NOT read here (this repository is public); the private dashboard adds it.

Usage: .venv/bin/python scripts/research/alpha/build_scoreboard.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "research" / "alpha" / "results"
OUT = ROOT / "research" / "alpha" / "scoreboard.json"

# verdict codes shown as pills: FAILED (lost money or no better than doing nothing), NO_IMPROVEMENT (a change to
# something that already exists that did not help), WATCH (positive but not proven; record-only), USEFUL (true and
# usable, but not a money-maker by itself), BLOCKED (waiting on data), RUNNING
AREAS = {
    "bot": "The paper bot's own strategies",
    "stocks": "Stock-picking patterns",
    "news": "News and big moves",
    "options": "Options",
    "vol": "Volatility forecasting",
    "market": "Whole-market timing (ETF money)",
}


def _j(rel: str):
    p = RES / rel
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float | None, d: int = 1) -> str:
    return "UNKNOWN" if x is None else f"{x * 100:+.{d}f}%"


def idea(id_, area, name, question, verdict, key, meaning, source):
    return {"id": id_, "area": area, "name": name, "question": question, "verdict": verdict, "key_number": key,
            "meaning": meaning, "source": source}


def market_rows() -> tuple[list[dict], dict]:
    a = _j("new_areas/A_market_overlays.json")
    rows, chart = [], {}
    if a:
        bh = a["rules"]["BH"]["splits"]
        names = {"VM1": "Cut exposure when the market is jumpy (no leverage)",
                 "VM15": "Same, but allowed up to 1.5x when calm (borrowed)",
                 "TR10": "Trend rule: hold the market only above its 10-month average",
                 "VM1+TR10": "Both rules together"}
        for k, nm in names.items():
            s = a["rules"][k]["splits"]
            o, v = s["OOS"], s["VAL"]
            key = (f"2013-24: {o['cagr'] * 100:.1f}%/yr vs {bh['OOS']['cagr'] * 100:.1f}% buy-and-hold; "
                   f"worst fall {o['max_drawdown'] * 100:.0f}% vs {bh['OOS']['max_drawdown'] * 100:.0f}%")
            if k == "TR10":
                meaning = (f"It earns less in normal years but cut the 2000-12 fall from {bh['VAL']['max_drawdown'] * 100:.0f}% "
                           f"to {v['max_drawdown'] * 100:.0f}%. Insurance against a long bear market that costs about "
                           f"{(bh['OOS']['cagr'] - o['cagr']) * 100:.1f} points a year when markets rise.")
            else:
                meaning = "Lower returns than simply holding the market, with no better risk-adjusted return out of sample."
            rows.append(idea(f"A-{k}", "market", nm, "Does a simple rule beat holding the whole US market (1963-2024)?",
                             "NO_IMPROVEMENT", key, meaning, "research/alpha/results/new_areas/A_market_overlays.json"))
        ec = a["equity_curves_monthly"]
        chart = {k: ec[k] for k in ("BH", "TR10", "VM1")}
        chart["_table"] = {k: {sp: {m: round(v[m], 4) for m in ("cagr", "sharpe", "max_drawdown", "avg_exposure")}
                               for sp, v in a["rules"][k]["splits"].items()} for k in a["rules"]}
    c = _j("new_areas/C_spy_overnight.json")
    if c:
        on, bh = c["rules"]["ON"]["splits"], c["rules"]["BH"]["splits"]
        rows.append(idea("C-ON", "market", "Hold SPY only overnight (close to next open)",
                         "Most of the market's gain is said to come overnight. Can you capture it?", "FAILED",
                         f"2022-24: {on['OOS']['cagr'] * 100:.1f}%/yr after costs vs {bh['OOS']['cagr'] * 100:.1f}% holding",
                         "Trading in and out every day costs more than the overnight effect is worth.",
                         "research/alpha/results/new_areas/C_spy_overnight.json"))
        idr = c["rules"]["ID"]["splits"]
        rows.append(idea("C-ID", "market", "Hold SPY only during the trading day", "Is the daytime session better?",
                         "FAILED", f"2022-24: {idr['OOS']['cagr'] * 100:.1f}%/yr after costs",
                         "Loses money after costs.", "research/alpha/results/new_areas/C_spy_overnight.json"))
    return rows, chart


def bot_rows() -> tuple[list[dict], dict]:
    rows = [
        idea("P3-book", "bot", "The bot's six replayable strategies, as configured",
             "Do the bot's strategies make money on 2016-24 data that includes companies that later disappeared?",
             "FAILED", "2022-24 Sharpe +0.01 (about zero) with takeovers booked correctly; -0.77 under the bot's -30% rule",
             "No edge. The bot is a working machine without a money-making rule yet.",
             "research/alpha/results/sprint/P7_leaderboard.json"),
        idea("P3-each", "bot", "Each strategy on its own",
             "Is one of the six carrying the others?", "FAILED",
             "2022-24 Sharpe: relative_strength +0.07, momentum_trend -0.09, sector_rotation -0.35, mean_reversion -0.51, "
             "breakout -0.55, extreme_reversal -0.75",
             "None has an edge on its own.", "docs/ALPHA-SPRINT-FINAL.md section 2"),
        idea("P3-sizing", "bot", "Size positions by forecast volatility",
             "Would betting less on jumpy stocks fix the book?", "NO_IMPROVEMENT",
             "Best rule 2022-24 Sharpe -0.01 vs +0.01 for current sizing (6 forecasts tried)",
             "Current sizing is as good as any alternative tested.", "research/alpha/results/sprint/P7_leaderboard.json"),
        idea("P1-D", "bot", "Loosen the score thresholds", "Is the bot too picky?", "NO_IMPROVEMENT",
             "2022-24 Sharpe -0.74 vs -0.77; 2020-21 -0.61 vs +0.06", "Taking more trades makes it worse.",
             "docs/ALPHA-SPRINT-FINAL.md section 2"),
        idea("P8", "bot", "Switch strategies off in bad market regimes", "Can a regime filter avoid the losing periods?",
             "NO_IMPROVEMENT", "1 of 90 regime cells qualified; 2022-24 Sharpe -0.07 vs +0.01 without it",
             "Regime labels that looked good earlier did not hold up later (54% sign agreement, a coin flip).",
             "research/alpha/results/sprint/P8_regimes.json"),
    ]
    e4 = _j("events/E4_nochase.json")
    if e4:
        c = e4["conventions"]["corrected_delisting"]
        d = {k: v["diff"] for k, v in c["sharpe_diff"].items()}
        rows.append(idea("E4", "bot", "Stop buying stocks the day after a big jump",
                         "22% of the bot's entries follow a jump day. Are those its worst trades?", "NO_IMPROVEMENT",
                         f"Sharpe change 2016-19 {d['TRAIN']:+.2f}, 2020-21 {d['VAL']:+.2f}, 2022-24 {d['OOS']:+.2f}",
                         "Helped in early years, hurt in recent years. Nothing changes in the bot.",
                         "research/alpha/results/events/E4_nochase.json"))
    f = _j("new_areas/F_lowvol_screen.json")
    chart = {}
    if f:
        c = f["conventions"]["corrected_delisting"]
        d = c["sharpe_diff"]
        b, lv = c["base"]["OOS"], c["lowvol"]["OOS"]
        st, sh = c["base_trades_by_forecast_state"], c["base_high_vol_share_by_strategy"]
        hv_share = st["high"]["count"] / sum(v["count"] for v in st.values())
        label = {"TRAIN": "2016-19", "VAL": "2020-21", "OOS": "2022-24"}
        worse = " and ".join(label[k] for k in label if d[k]["diff"] <= 0) or "no period"
        rows.append(idea("F", "bot", "Skip the most volatile 20% of the bot's candidates",
                         "High predicted volatility goes with slightly worse returns. Does dropping those candidates help the bot?",
                         "USEFUL" if c["verdict"] == "IMPROVEMENT" else "NO_IMPROVEMENT",
                         f"Sharpe change 2016-19 {d['TRAIN']['diff']:+.2f}, 2020-21 {d['VAL']['diff']:+.2f}, 2022-24 "
                         f"{d['OOS']['diff']:+.2f} (90% range {d['OOS']['ci90'][0]:+.2f} to {d['OOS']['ci90'][1]:+.2f}); "
                         f"2022-24 return {pct(lv.get('cagr'))}/yr vs {pct(b.get('cagr'))}",
                         ("Passed the pre-registered bar: a candidate change for the bot (needs Harry's approval)."
                          if c["verdict"] == "IMPROVEMENT" else
                          f"{hv_share * 100:.0f}% of the bot's trades are in the most volatile fifth of stocks "
                          f"(momentum_trend {sh.get('momentum_trend', 0) * 100:.0f}%). Those trades averaged "
                          f"{pct(st['high']['mean'])} against {pct(st['low']['mean'])} for the rest, but removing them "
                          f"changed the book's Sharpe the wrong way in {worse}. Nothing changes in the bot."),
                         "research/alpha/results/new_areas/F_lowvol_screen.json"))
        chart = {"base": c["equity_monthly"]["base"], "lowvol": c["equity_monthly"]["lowvol"]}
    else:
        rows.append(idea("F", "bot", "Skip the most volatile 20% of the bot's candidates",
                         "High predicted volatility goes with slightly worse returns. Does dropping those candidates help the bot?",
                         "RUNNING", "Replay running", "Result pending.", "docs/NEW-AREAS-PREREG.md section F"))
    rows.append(idea("P1", "bot", "Which trades did the bot's live filters reject, and were they right?",
                     "Are the bot's real-time safety and quality filters costing money?", "BLOCKED",
                     "Needs about 20 trading days of the bot's own database (around 2026-10-22)",
                     "The cheapest evidence left. The analysis code is ready and tested.", "docs/BOT-DB-IMPORT.md"))
    return rows, chart


def static_rows() -> list[dict]:
    s = "docs/ALPHA-DISCOVERY-REPORT-2026-10.md"
    return [
        idea("H01", "stocks", "Buy recent losers, sell recent winners (short-term reversal)", "Do 1-10 day moves reverse?",
             "FAILED", "2022-24 Sharpe -0.55 (160 variants)", "A textbook effect that no longer pays after costs.", s),
        idea("H03", "stocks", "Momentum (buy 12-month winners)", "Do winners keep winning?", "FAILED",
             "2022-24 Sharpe +0.59 but t 1.01; failed 2020-21", "Not reliable enough to trade.", s),
        idea("H04", "stocks", "Residual momentum", "Momentum after removing market and sector moves?", "FAILED",
             "2022-24 Sharpe +0.32 (t 0.65)", "Too weak.", s),
        idea("H16", "stocks", "Avoid 'lottery' stocks (biggest one-day jumps)", "Do the most explosive stocks lag?",
             "FAILED", "2022-24 Sharpe +0.13 (t 0.25)", "Too weak after costs.", s),
        idea("H17", "stocks", "Low idiosyncratic volatility", "Do calm stocks beat jumpy ones?", "FAILED",
             "2022-24 Sharpe +0.27 (t 0.53)", "Too weak.", s),
        idea("H23", "stocks", "Analyst estimate revisions", "Follow analysts raising forecasts?", "FAILED",
             "2022-24 Sharpe +0.21 (t 0.35)", "Too weak.", s),
        idea("H07", "stocks", "Fade opening gaps", "Do big opening gaps reverse during the day?", "FAILED",
             "-59 basis points a day after costs", "Real before costs, but spreads eat it.", s),
        idea("H20", "stocks", "Buy before earnings announcements", "Do stocks drift up into earnings?", "FAILED",
             "-0.12% per event 2022-24", "No.", s),
        idea("H21", "stocks", "Buy after an earnings beat", "Do beats keep drifting up?", "FAILED",
             "+0.29% per event but t -0.6; -5.4% without the top 5% of events", "Driven by a few outliers.", s),
        idea("H09", "stocks", "Calendar effects on SPY (13 of them: Fed days, month end, jobs Friday...)",
             "Are there reliable good days to hold the market?", "FAILED",
             "None survives multiple-testing correction (best q 0.82)", "Effects are smaller than the cost of trading.", s),
        idea("H08", "stocks", "SPY/QQQ intraday momentum", "Does the first half-hour predict the last?", "FAILED",
             "|t| < 1.3 in every period", "No.", s),
        idea("H15", "stocks", "Pairs trading", "Do similar stocks that drift apart converge?", "FAILED",
             "Returns of +/-1-2% a year, |t| < 1.2", "No.", s),
        idea("H36", "stocks", "Combine many weak strategies", "Do weak signals add up?", "FAILED",
             "The streams are correlated and do not combine into an edge", "No.", s),
        idea("H12", "stocks", "Machine learning on price data", "Can a model predict direction?", "USEFUL",
             "Size of the move is predictable; direction is not (AUC 0.47-0.52)",
             "The key lesson: the lab can forecast how much a stock will move, not which way.", s),
        idea("H02", "stocks", "Survivorship bias check", "How much do tests on today's stocks flatter results?", "USEFUL",
             "+1.2 to +5.5 points a year of fake return for long-only books",
             "Why every test here includes companies that later disappeared.", s),
        idea("E1", "news", "Buy the morning after a news-driven jump", "Do stocks like Vistra keep rising after big news?",
             "FAILED", "20-day return vs market: -0.7%, -1.3%, -0.7% in each period (10,573 jumps)",
             "After big jumps the average stock lags the market.", "docs/WHAT-THE-BOT-NEEDS.md"),
        idea("E2", "news", "Buy jumps on sector 'theme days'", "When a whole sector re-rates, does it keep going?",
             "FAILED", "-5.0%, +0.5%, -1.0% (3,586 cases)", "No.", "docs/WHAT-THE-BOT-NEEDS.md"),
        idea("E2b", "news", "Buy the sector laggards on theme days", "Do the stocks left behind catch up?", "FAILED",
             "-0.5% over 5 days in 2022-24 (39,973 cases)", "Laggards do not catch up.", "docs/WHAT-THE-BOT-NEEDS.md"),
        idea("IN", "news", "React within 15 minutes of a headline", "Can the bot trade the news the same day?", "FAILED",
             "48,773 headlines: the move is done within ~15 minutes; leftover drift is below costs",
             "A retail-speed bot is too late. Do not build a news-reaction feature.", "docs/WHAT-THE-BOT-NEEDS.md"),
        idea("E3", "options", "Buy calls after a news jump", "Is leverage the way to play big news?", "FAILED",
             "-2.2%, -23.2%, -8.8% per trade at the ask", "Calls are expensive right after a jump.",
             "docs/WHAT-THE-BOT-NEEDS.md"),
        idea("O1", "options", "Buy straddles when QuantLab expects a bigger move than the option price implies",
             "Can the volatility forecast pick cheap options?", "WATCH",
             "+8.2% per trade at the ask 2023-24, but 2023 +25.6% and 2024 -11.7%; range includes zero",
             "The only positive options rule. One good year carries it. Record-only shadow tracking, no orders.",
             "docs/ALPHA-SPRINT-FINAL.md section 4"),
        idea("O2", "options", "Buy cheap 25-delta strangles", "Same idea with cheaper options?", "FAILED",
             "+6.4% mean but the median trade loses 98%", "A lottery ticket, not an edge.",
             "docs/ALPHA-SPRINT-FINAL.md section 4"),
        idea("O-all", "options", "Buy every liquid straddle", "Are options generally cheap?", "FAILED",
             "-2.6% per trade at the ask (+0.5% at mid)", "The spread eats the edge.", "docs/ALPHA-SPRINT-FINAL.md"),
        idea("O-sell", "options", "Sell straddles (collect the premium)", "Are options generally expensive?", "FAILED",
             "-6.2% per trade at the bid", "Spreads are bigger than the premium you collect.",
             "docs/ALPHA-SPRINT-FINAL.md"),
        idea("H26", "options", "Pre-earnings straddles", "Do options under-price earnings moves?", "FAILED",
             "-28% to -35% per trade at the ask", "A published effect that does not survive real spreads.", s),
        idea("H27-30", "options", "Option sorts: IV vs realised, option momentum, IV rank, term structure, skew",
             "Does any option ranking make money?", "FAILED", "None profitable after crossing the spread", "No.", s),
        idea("H35", "options", "Sell SPY straddles", "Is the index variance premium tradeable?", "FAILED",
             "About zero in 2019-24", "No.", s),
        idea("P2", "vol", "QuantLab's volatility forecaster", "Can the lab forecast how much a stock will move?",
             "USEFUL", "Best of 7 forecasters at 4 of 5 horizons (20-day QLIKE 0.46 vs 0.49 HAR, 0.56 GARCH)",
             "A real, tested skill. It has not yet turned into money on its own.",
             "research/alpha/results/sprint/P2_vol_forecast.json"),
        idea("P4", "vol", "QuantLab forecast vs option prices", "Does it know more than the options market?",
             "FAILED", "Implied volatility is about twice as accurate (QLIKE 0.16 vs 0.32)",
             "Option prices already contain almost everything the forecast knows.",
             "research/alpha/results/sprint/P4_P6_options.json"),
    ]


FIXES = [
    {"title": "Fake price crashes from spin-offs and share distributions", "where": "Research data",
     "status": "FIXED", "detail": "Alpaca's adjusted prices showed fake falls such as NVS -82% (Alcon spin-off, real move "
     "-12%) and UHAL -90% (holders were actually up about 8%). 94 bad stock-days flagged and repaired; 2 tickers reused by "
     "a different company cut off.", "source": "src/quantlab/alpha/ca_fixes.py"},
    {"title": "Every delisting booked as a -30% loss", "where": "Bot backtester (master)",
     "status": "PROPOSED", "detail": "All 13 delisted trades in the bot's replay were cash takeovers at or above the last "
     "price, yet the rule booked each as -30%. This flips the bot's backtest from about zero to clearly negative and "
     "skews its own outcome statistics. Fix: takeovers with an Alpaca merger record exit at the last close, others keep "
     "-30%. Alpaca has records for 9 of those 13 (all 2022-24; almost none before 2020), so it fixes most takeovers "
     "from now on. 985 tests pass.",
     "source": "https://github.com/HarryKelly-Git/quantlab/pull/3 (draft, not merged)"},
    {"title": "Spin-off days booked as real losses in the bot's live data", "where": "Bot price data (master)",
     "status": "PROPOSED", "detail": "The bot builds returns from raw prices, so a spin-off day looks like a crash. That can "
     "trigger a stop or a false 'buy the dip' signal. Fix: use the spin-off records the data vendor does provide.",
     "source": "https://github.com/HarryKelly-Git/quantlab/pull/3 (draft, not merged)"},
    {"title": "Options matched to the wrong company", "where": "Research data", "status": "FIXED",
     "detail": "Old option chains were joined to whichever company holds the ticker today (e.g. old Caesars vs Eldorado). "
     "One options result showed +31%; corrected, it is +4%.", "source": "docs/ALPHA-DISCOVERY-REPORT-2026-10.md audit"},
    {"title": "Companies that later delisted silently dropped", "where": "Research data", "status": "FIXED",
     "detail": "14% of earnings events and 8% of option rows were missing, all from companies that later disappeared.",
     "source": "docs/ALPHA-DISCOVERY-REPORT-2026-10.md audit"},
    {"title": "Overstated statistics", "where": "Research method", "status": "FIXED",
     "detail": "Overlapping returns had inflated t-statistics about 1.7x; a placebo test compared earnings with earnings.",
     "source": "docs/ALPHA-DISCOVERY-REPORT-2026-10.md audit"},
    {"title": "Replay dividend bias and a winner-picking bug", "where": "Research code", "status": "FIXED",
     "detail": "Caught before any result was read: one-sided dividend rounding biased returns up; the leaderboard picked a "
     "'winner' even when every option lost money.", "source": "docs/ALPHA-SPRINT-FINAL.md deviations"},
]

NEXT = [
    {"rank": 1, "action": "Send the bot's database export after about 20 trading days (around 2026-10-22)",
     "who": "Harry (5 minutes)", "why": "The only evidence on the bot's real filters. Free. Analysis runs the moment it lands.",
     "how": "docs/BOT-DB-IMPORT.md (upload in chat; never commit it: the repo is public)"},
    {"rank": 2, "action": "Review the two bot data fixes (delisting and spin-offs)",
     "who": "Harry", "why": "Makes the bot's own backtests and outcome statistics honest. Not merged without you.",
     "how": "Draft PR #3: https://github.com/HarryKelly-Git/quantlab/pull/3"},
    {"rank": 3, "action": "Keep O1 as a record-only shadow", "who": "Lab",
     "why": "The only positive options rule; needs forward evidence before any money or orders.",
     "how": "Score it on excess over buying every straddle on the same dates"},
    {"rank": 4, "action": "Do not add strategies, data feeds or news features to the bot yet", "who": "Both",
     "why": "About 60 ideas tested, none passes. More machinery without an edge only adds cost.", "how": ""},
]


def findings() -> list[dict]:
    """The few facts that matter most, each with its evidence file."""
    out = [{"text": "The bot's six replayable strategies earn about zero on honest 2016-24 data (2022-24 Sharpe +0.01).",
            "source": "research/alpha/results/sprint/P7_leaderboard.json"}]
    tl = _j("new_areas/bot_tail_losses.json")
    if tl:
        b = tl["big_losers"]
        sd = tl["stop_distance_median_by_strategy"]
        out.append({"text": f"Its losses are concentrated: {b['n']} of {tl['trades']:,} replayed trades ({b['share'] * 100:.0f}%) lost "
                            f"more than 25% each. Together they lost ${-b['pnl']:,.0f}, while all trades together made ${tl['total_pnl']:,.0f}. "
                            f"The worst were speculative names such as {', '.join(w['symbol'].split('@')[0] for w in b['worst'][:4])} "
                            f"(de-SPACs, meme, crypto and small biotech stocks). The bot's stops on its momentum strategies sit "
                            f"{sd.get('momentum_trend', 0) * 100:.0f}-{sd.get('relative_strength', 0) * 100:.0f}% below entry (median), "
                            f"and {tl['stops_at_or_below_zero']} were at or below zero. Screening out volatile stocks did not fix it (Area F).",
                    "source": "research/alpha/results/new_areas/bot_tail_losses.json"})
    out += [{"text": "QuantLab's real, tested skill is forecasting how much a stock will move, not which way. Option prices "
                     "already know most of it.", "source": "research/alpha/results/sprint/P2_vol_forecast.json"},
            {"text": "After big news jumps the average stock lags the market, and prices finish reacting to a headline within "
                     "about 15 minutes. Chasing news is not an edge for this bot.", "source": "docs/WHAT-THE-BOT-NEEDS.md"}]
    return out


def main() -> None:
    m_rows, m_chart = market_rows()
    b_rows, f_chart = bot_rows()
    ideas = b_rows + static_rows() + m_rows
    counts: dict[str, int] = {}
    for r in ideas:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    ledger = (ROOT / "research" / "alpha" / "ledger.jsonl").read_text().splitlines()
    sb = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "headline": "No money-making rule has been proven yet. The paper bot runs, but its strategies earn about zero "
                    "on honest 2016-24 data.",
        "paper_only": True,
        "holdout": "2025+ data is sealed and untouched (0 of 1 uses).",
        "counts": {"ideas": len(ideas), "ledger_runs": len([x for x in ledger if x.strip()]), **counts},
        "findings": findings(),
        "areas": AREAS, "ideas": ideas, "fixes": FIXES, "next_steps": NEXT,
        "charts": {"market": m_chart, "lowvol": f_chart},
    }
    OUT.write_text(json.dumps(sb, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ideas)} ideas, verdicts {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
