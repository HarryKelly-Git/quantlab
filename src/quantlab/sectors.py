"""Symbol -> sector ETF mapping (the sector context used by relative-strength features, sector
rotation, portfolio sector limits and reports).

PIT STATUS: ASSUMED_STATIC. Every input here (the provider's sector/industry labels, the SEC
submissions ``sic``) is a CURRENT snapshot. A company that changed industry is mapped to its
current sector for all of history. Anything built on this map must carry ASSUMED_STATIC.

Resolution order for each symbol (first hit wins, per reference row, newest row first):
  1. ``reference.industry`` is itself a configured sector-ETF ticker (the synthetic generator
     and any provider that already resolved the ETF do this);
  2. ``reference.sector`` is a sector NAME -> ETF, using the configured names
     (``benchmarks.sectors``: ticker -> name) plus a few common vendor spellings;
  3. ``reference.sic`` -> ETF via :data:`SIC_SECTOR_RANGES`;
  4. otherwise ``None`` (unknown sector: sector features are NaN, never guessed).

SIC APPROXIMATION. SIC (1987) and GICS (the basis of the Select Sector SPDRs) are different
taxonomies. :data:`SIC_SECTOR_RANGES` approximates the 11 GICS sectors from 4-digit SIC ranges.
Known imperfections: SIC 7370-7379 goes to XLK although GICS puts interactive media / internet
platforms (e.g. SIC 7370 filers such as search and social-media companies) in XLC; cruise lines
(4400s) go to XLI instead of XLY; conglomerates are mapped by their primary SIC only; SIC
6770 (blank checks) lands in XLF. A Fama-French SIC-industry file could replace this table later;
the table is plain data so it can be audited line by line.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from quantlab.core.types import PitStatus
from quantlab.data.panel import DataBundle

SECTOR_MAP_PIT_STATUS = PitStatus.ASSUMED_STATIC

# (low, high, etf) inclusive 4-digit SIC ranges. FIRST MATCH WINS: specific exceptions come
# before the broad ranges they carve out of.
SIC_SECTOR_RANGES: tuple[tuple[int, int, str], ...] = (
    # --- exceptions (must precede the broad ranges below) ----------------------------------
    (800, 899, "XLB"),      # forestry
    (1520, 1549, "XLY"),    # residential building contractors / homebuilders
    (2450, 2452, "XLY"),    # wood buildings & mobile homes
    (2710, 2741, "XLC"),    # newspapers, periodicals, books, misc publishing (GICS media)
    (2830, 2836, "XLV"),    # pharmaceuticals, biologicals, diagnostics
    (2840, 2844, "XLP"),    # soap, detergents, cosmetics, personal care
    (3080, 3089, "XLB"),    # plastics products
    (3410, 3412, "XLB"),    # metal cans & containers
    (3533, 3533, "XLE"),    # oil & gas field machinery
    (3570, 3579, "XLK"),    # computer & office equipment
    (3630, 3639, "XLY"),    # household appliances
    (3650, 3652, "XLY"),    # household audio & video equipment
    (3660, 3679, "XLK"),    # communications equipment, electronic components, semiconductors
    (3695, 3695, "XLK"),    # magnetic & optical recording media
    (3711, 3711, "XLY"),    # motor vehicles & passenger car bodies
    (3714, 3714, "XLY"),    # motor vehicle parts & accessories
    (3716, 3716, "XLY"),    # motor homes
    (3751, 3751, "XLY"),    # motorcycles, bicycles & parts
    (3792, 3792, "XLY"),    # travel trailers & campers
    (3812, 3812, "XLI"),    # search, detection, navigation, guidance (defense electronics)
    (3826, 3826, "XLV"),    # laboratory analytical instruments (life-science tools)
    (3820, 3829, "XLK"),    # measuring & controlling instruments
    (3841, 3851, "XLV"),    # surgical/medical/dental instruments, x-ray, electromedical, ophthalmic
    (3861, 3861, "XLK"),    # photographic equipment
    (3873, 3873, "XLY"),    # watches & clocks
    (3910, 3914, "XLY"),    # jewelry & silverware
    (3940, 3949, "XLY"),    # toys & sporting goods
    (4610, 4619, "XLE"),    # pipelines (crude / refined)
    (4724, 4725, "XLY"),    # travel agencies & tour operators
    (4922, 4923, "XLE"),    # natural gas transmission (midstream)
    (4950, 4959, "XLI"),    # sanitary services / waste management
    (5045, 5045, "XLK"),    # computers & peripherals wholesale
    (5047, 5047, "XLV"),    # medical & hospital equipment wholesale
    (5065, 5065, "XLK"),    # electronic parts wholesale
    (5122, 5122, "XLV"),    # drugs & proprietaries wholesale
    (5140, 5149, "XLP"),    # groceries wholesale
    (5171, 5172, "XLE"),    # petroleum products wholesale
    (5180, 5182, "XLP"),    # beer, wine & distilled beverages wholesale
    (5331, 5331, "XLP"),    # variety stores (GICS moved general-merchandise staples retail to XLP in 2023)
    (5399, 5399, "XLP"),    # misc general merchandise (warehouse clubs)
    (5411, 5499, "XLP"),    # grocery & food stores
    (5912, 5912, "XLP"),    # drug stores
    (6324, 6324, "XLV"),    # hospital & medical service plans (managed care)
    (6500, 6553, "XLRE"),   # real estate operators, lessors, agents, developers
    (6798, 6798, "XLRE"),   # real estate investment trusts
    (7311, 7319, "XLC"),    # advertising
    (7370, 7379, "XLK"),    # computer programming, software, data processing (see module note)
    (7812, 7833, "XLC"),    # motion pictures: production, distribution, theaters
    (7841, 7841, "XLC"),    # video tape rental
    (8731, 8731, "XLV"),    # commercial physical & biological research (biotech R&D)
    # --- broad ranges -------------------------------------------------------------------------
    (100, 999, "XLP"),      # agriculture, agricultural services, fishing
    (1000, 1099, "XLB"),    # metal mining
    (1200, 1299, "XLE"),    # coal mining
    (1300, 1399, "XLE"),    # oil & gas extraction and services
    (1400, 1499, "XLB"),    # non-metallic minerals mining
    (1500, 1799, "XLI"),    # construction
    (2000, 2199, "XLP"),    # food, beverages, tobacco
    (2200, 2399, "XLY"),    # textiles & apparel
    (2400, 2499, "XLB"),    # lumber & wood products
    (2500, 2599, "XLY"),    # furniture & fixtures
    (2600, 2699, "XLB"),    # paper & allied products
    (2700, 2799, "XLI"),    # commercial printing
    (2800, 2899, "XLB"),    # chemicals
    (2900, 2999, "XLE"),    # petroleum refining
    (3000, 3079, "XLY"),    # tires, rubber & plastic footwear
    (3090, 3099, "XLB"),    # misc plastics
    (3100, 3199, "XLY"),    # leather goods & footwear
    (3200, 3299, "XLB"),    # stone, clay, glass, concrete
    (3300, 3399, "XLB"),    # primary metals
    (3400, 3499, "XLI"),    # fabricated metal products
    (3500, 3599, "XLI"),    # industrial machinery
    (3600, 3699, "XLI"),    # electrical equipment
    (3700, 3799, "XLI"),    # aerospace, ships, rail equipment, trucks
    (3800, 3899, "XLI"),    # other instruments
    (3900, 3999, "XLI"),    # misc manufacturing
    (4000, 4799, "XLI"),    # railroads, trucking, shipping, airlines, logistics
    (4800, 4899, "XLC"),    # telecommunications & broadcasting
    (4900, 4999, "XLU"),    # electric, gas, water utilities
    (5000, 5199, "XLI"),    # wholesale distribution
    (5200, 5999, "XLY"),    # retail
    (6000, 6799, "XLF"),    # banks, brokers, insurance, holding companies
    (7000, 7099, "XLY"),    # hotels & lodging
    (7200, 7299, "XLY"),    # personal services
    (7300, 7399, "XLI"),    # business services
    (7500, 7599, "XLY"),    # auto repair, rental, parking
    (7600, 7699, "XLI"),    # misc repair services
    (7900, 7999, "XLY"),    # amusement, recreation, casinos
    (8000, 8099, "XLV"),    # health services
    (8100, 8199, "XLI"),    # legal services
    (8200, 8299, "XLY"),    # educational services
    (8300, 8399, "XLV"),    # social services
    (8700, 8799, "XLI"),    # engineering, accounting, research, management services
)

# Vendor spellings of sector names seen in common data sources (lower-case).
SECTOR_NAME_ALIASES: dict[str, str] = {
    "materials": "XLB", "basic materials": "XLB",
    "communication services": "XLC", "communications": "XLC", "telecommunication services": "XLC",
    "telecommunications": "XLC",
    "energy": "XLE",
    "financials": "XLF", "financial": "XLF", "financial services": "XLF", "finance": "XLF",
    "industrials": "XLI", "industrial": "XLI",
    "information technology": "XLK", "technology": "XLK",
    "consumer staples": "XLP", "consumer defensive": "XLP",
    "real estate": "XLRE",
    "utilities": "XLU",
    "health care": "XLV", "healthcare": "XLV",
    "consumer discretionary": "XLY", "consumer cyclical": "XLY",
}


def _clean_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def parse_sic(v: Any) -> int | None:
    """SIC codes arrive as '3571', 3571, 3571.0 or missing. Returns None when unusable."""
    s = _clean_str(v)
    if s is None:
        return None
    try:
        code = int(float(s))
    except ValueError:
        return None
    return code if 100 <= code <= 9999 else None


def sic_to_sector_etf(sic: Any, allowed: set[str] | None = None) -> str | None:
    """Approximate GICS sector ETF for a SIC code (see module docstring for the caveats)."""
    code = parse_sic(sic)
    if code is None:
        return None
    for lo, hi, etf in SIC_SECTOR_RANGES:
        if lo <= code <= hi:
            return etf if (allowed is None or etf in allowed) else None
    return None


def configured_sectors(bundle: DataBundle | None = None, config: Any = None) -> dict[str, str]:
    """Configured sector ETFs (ticker -> sector name): config ``benchmarks.sectors`` if a config
    is given, else the bundle's benchmarks."""
    if config is not None:
        sectors = config.get("benchmarks.sectors", {}) or {}
    elif bundle is not None:
        sectors = bundle.sector_etfs
    else:
        sectors = {}
    return {str(k).upper(): str(v) for k, v in sectors.items()}


def sector_name_to_etf(name: Any, sectors: dict[str, str]) -> str | None:
    s = _clean_str(name)
    if s is None:
        return None
    key = s.lower()
    by_name = {v.strip().lower(): k for k, v in sectors.items()}
    if key in by_name:
        return by_name[key]
    etf = SECTOR_NAME_ALIASES.get(key)
    return etf if etf in sectors else None


def _resolve_row(row: pd.Series, sectors: dict[str, str]) -> str | None:
    industry = _clean_str(row.get("industry"))
    if industry is not None and industry.upper() in sectors:
        return industry.upper()
    etf = sector_name_to_etf(row.get("sector"), sectors)
    if etf is not None:
        return etf
    return sic_to_sector_etf(row.get("sic"), allowed=set(sectors))


def sector_map(bundle: DataBundle, config: Any = None) -> dict[str, str | None]:
    """symbol -> sector ETF ticker (or None when unknown) for every panel/reference symbol.

    ASSUMED_STATIC (current metadata). When a symbol has several reference rows (several
    sources), the most recently retrieved row that yields a mapping wins.
    """
    sectors = configured_sectors(bundle, config)
    symbols = set(map(str, bundle.panel.symbols))
    ref = bundle.reference
    out: dict[str, str | None] = {s: None for s in symbols}
    if not sectors or ref is None or ref.empty:
        return out
    ref = ref.copy()
    ref["symbol"] = ref["symbol"].astype(str).str.upper()
    if "retrieved_at" in ref.columns:
        ref = ref.sort_values("retrieved_at", ascending=False, kind="mergesort")
    for sym, grp in ref.groupby("symbol", sort=False):
        etf = None
        for _, row in grp.iterrows():
            etf = _resolve_row(row, sectors)
            if etf is not None:
                break
        out[str(sym)] = etf
    return out
