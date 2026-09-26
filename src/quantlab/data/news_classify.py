"""Deterministic headline classification for news items (no LLM, no inference beyond the text).

Rules are ordered keyword patterns over the headline as stored. They label what an article is ABOUT
(its event category); they never say whether the news is good or bad for the stock, and a label is
not a trade signal. Anything the rules do not recognise is ``other`` (not material).

Two categories are deliberately NOT company events:
  * ``analyst`` -- rating/price-target changes (opinions about the stock, not company news);
  * ``market_commentary`` -- movers lists, "why is X trading higher", options activity, previews.

Company-specific = the article is tagged with at most ``COMPANY_SPECIFIC_MAX_TAGS`` symbols. Tag
counts must come from a market-wide ingestion (every tag kept); a symbol-filtered ingestion
undercounts tags, which is why ``n_tags`` is computed over all stored rows of an article.

Guidance direction is read only from explicit verbs (raises/lowers/reaffirms ... guidance/outlook);
otherwise it is UNKNOWN. Headlines may be revised after publication (Alpaca returns the latest
version), so these labels are PIT_CONSERVATIVE, like the news rows themselves.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COMPANY_SPECIFIC_MAX_TAGS = 2
MATERIAL = ("earnings", "guidance", "m&a", "financing", "contract", "management", "regulatory", "legal",
            "product", "capital_return")
NOT_EVENTS = ("analyst", "market_commentary", "other")

# (category, pattern) -- FIRST MATCH WINS, so specific/opinion patterns come first
RULES: tuple[tuple[str, str], ...] = (
    ("analyst", r"\b(?:maintains|reiterates|upgrades?|downgrades?|initiates coverage|price target|pt (?:to|raised|lowered)|"
                r"analyst|overweight|underweight|outperform|underperform|equal[- ]weight)\b"),
    ("market_commentary", r"(?:shares are trading|stocks? moving|movers|mid-?day|pre-?market|after-?hours|\bwhy is\b|"
                          r"what'?s going on|options activity|unusual options|short interest|earnings scheduled|"
                          r"preview|ahead of (?:its |the )?(?:q\d |quarterly )?(?:earnings|results|report)|looms|"
                          r"what to expect|to report (?:q\d|earnings|results)|earnings (?:date|call) set|week ahead|"
                          r"top stocks|stocks to watch|trading (?:higher|lower)|new 52-week|\bspy\b|dow jones|"
                          r"nasdaq composite|s&p 500|\bshould you\b|\bis it time\b|\bhere'?s (?:why|how|what)\b|\binsights\b)"),
    ("guidance", r"(?:\bguidance\b|\b(?:raises|lowers|cuts|reaffirms|affirms|maintains|boosts|lifts|withdraws|issues|"
                 r"updates|sees)\b.{0,40}\b(?:outlook|forecast)\b|\bsees (?:fy\s?\d{0,4}|q\d|full[- ]year|20\d\d)\b)"),
    ("earnings", r"(?:\b(?:q[1-4]|fy\s?\d{2,4}|fiscal (?:q\d|20\d\d|year))\b.{0,25}\b(?:eps|earnings|results|sales|revenue)\b|"
                 r"\beps\b.{0,40}\b(?:beat|miss|est|estimate|consensus)|\breports? .{0,30}\b(?:results|earnings)\b|"
                 r"\bfinancial results\b|\bquarterly (?:results|earnings)\b)"),
    ("m&a", r"\b(?:to acquire|acquires|acquisition|merger|merge with|to be acquired|buyout|takeover|tender offer|"
            r"agrees to buy|go[- ]private|divest\w*|spin[- ]?off)\b"),
    ("financing", r"\b(?:public offering|private placement|prices? (?:\$|its |upsized |offering)|offering of|notes due|"
                  r"convertible|at-the-market|shelf registration|credit (?:facility|agreement)|term loan|"
                  r"secondary offering|files to sell|stock sale|share sale|to raise \$|raises \$)"),
    ("capital_return", r"\b(?:dividend|buyback|repurchase|special distribution)\b"),
    ("regulatory", r"\b(?:fda|approval|approves|clearance|cleared|doj|ftc|sec (?:charges|probe|investigation)|"
                   r"investigation|probe|recall|phase (?:1|2|3|i|ii|iii)\b|clinical (?:trial|hold)|breakthrough therapy|"
                   r"orphan drug|pdufa|warning letter|delist\w*|compliance notice|deficiency notice)"),
    ("legal", r"\b(?:lawsuit|sues|sued|settle(?:s|ment)|class action|verdict|litigation|jury|patent infringement|"
              r"injunction|court (?:rules|ruling|orders|dismisses))\b"),
    ("management", r"(?:\b(?:appoints?|appointed|names|named|hires|resigns?|resignation|steps? down|to step down|"
                   r"depart\w*|retire\w*|succeed\w*|ousted|fired)\b.{0,40}\b(?:ceo|cfo|coo|chief \w+ officer|"
                   r"president|chair\w*|director)\b|\b(?:ceo|cfo|coo|chief \w+ officer|chair\w*)\b.{0,40}\b(?:resigns?|"
                   r"steps? down|to step down|depart\w*|retire\w*|to leave|ousted|appointed|named|succeed\w*)\b)"),
    ("contract", r"\b(?:contract|awarded|agreement with|partnership|partners with|collaboration|supply (?:deal|agreement)|"
                 r"order (?:from|for)|licens(?:e|ing) (?:deal|agreement)|selected by)\b"),
    ("product", r"\b(?:launches|launch of|unveils|introduces|rolls out|new product|debuts)\b"),
)
_COMPILED = tuple((c, re.compile(p, re.IGNORECASE)) for c, p in RULES)
_UP = re.compile(r"\b(raises|raised|boosts|lifts|increases|ups|above)\b", re.IGNORECASE)
_DOWN = re.compile(r"\b(lowers|lowered|cuts|cut|reduces|slashes|below|withdraws|suspends)\b", re.IGNORECASE)
_FLAT = re.compile(r"\b(reaffirms|affirms|maintains|reiterates|confirms)\b", re.IGNORECASE)


def classify_headline(headline: str) -> str:
    h = headline or ""
    for cat, rx in _COMPILED:
        if rx.search(h):
            return cat
    return "other"


def guidance_direction(headline: str) -> str:
    """UP / DOWN / REAFFIRM from explicit verbs only; UNKNOWN otherwise."""
    h = headline or ""
    if _DOWN.search(h):
        return "DOWN"
    if _UP.search(h):
        return "UP"
    if _FLAT.search(h):
        return "REAFFIRM"
    return "UNKNOWN"


def classify_frame(news: pd.DataFrame) -> pd.DataFrame:
    """news rows + ``category``, ``guidance_dir``, ``n_tags``, ``company_specific``, ``material``.

    Vectorised: each rule is applied once to the distinct headlines (first match wins)."""
    if news.empty:
        return news.assign(category=pd.Series(dtype=object), guidance_dir=pd.Series(dtype=object),
                           n_tags=pd.Series(dtype=float), company_specific=pd.Series(dtype=bool),
                           material=pd.Series(dtype=bool))
    heads = pd.Series(pd.unique(news["headline"].astype(str)))
    cat = pd.Series("other", index=heads.index, dtype=object)
    todo = np.ones(len(heads), dtype=bool)
    for c, rx in _COMPILED:
        hit = todo & heads.str.contains(rx, na=False).to_numpy()
        cat[hit] = c
        todo &= ~hit
    gdir = pd.Series("UNKNOWN", index=heads.index, dtype=object)
    g = (cat == "guidance").to_numpy()
    if g.any():
        gdir[g] = heads[g].map(guidance_direction)
    lut_c = dict(zip(heads, cat))
    lut_g = dict(zip(heads, gdir))
    h = news["headline"].astype(str)
    n_tags = news["n_tags"] if "n_tags" in news.columns else news.groupby("news_id")["symbol"].transform("size")
    out = news.assign(category=h.map(lut_c), guidance_dir=h.map(lut_g), n_tags=n_tags.astype(float))
    out["company_specific"] = out["n_tags"] <= COMPANY_SPECIFIC_MAX_TAGS
    out["material"] = out["company_specific"] & out["category"].isin(MATERIAL)
    return out


__all__ = ["COMPANY_SPECIFIC_MAX_TAGS", "MATERIAL", "NOT_EVENTS", "RULES", "classify_frame", "classify_headline",
           "guidance_direction"]
