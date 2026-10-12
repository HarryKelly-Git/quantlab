"""Options execution engine (sprint P5; docs/ALPHA-SPRINT-PREREG.md section 5). PAPER research.

Every option result is priced at four fill levels. For a buy, with h = (ask - bid) / 2 of the whole
structure (the sum of the legs' half-spreads):

| Level | Buy fill |
|---|---|
| MID | mid |
| CONSERVATIVE | mid + 0.5h |
| PESSIMISTIC | ask |
| WORST_REASONABLE | ask + 0.25h |

A sale mirrors around mid. Alpaca PAPER only fills marketable limits (buy >= ask), so PESSIMISTIC is the
realistic paper level. MID is never evidence of an edge.

**Fees:** $0.03 per contract per fill (ASSUMED regulatory and clearing fees; Alpaca charges no options
commission). Fees are expressed per share of underlying (/100).

**Holding:** to expiry, settled at intrinsic value on the expiry close. An ITM long leg is exercised and
the delivered stock sold. That pays the equity one-way cost on the FULL stock notional, which is the
primary and conservative treatment. The pre-registered wording charged the cost on the intrinsic value
only; that version is reported as ``settle_cost_on_intrinsic``.

**Not in this data, reported UNKNOWN:** option volume, open interest, quote freshness. The DoltHub
chains are end-of-day snapshots without timestamps.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

LEVELS = {"MID": 0.0, "CONSERVATIVE": 0.5, "PESSIMISTIC": 1.0, "WORST_REASONABLE": 1.25}
FEE_PER_SHARE = 0.03 / 100.0
EQ_TIERS = ((100e6, 2.0), (20e6, 5.0), (5e6, 10.0), (0.0, 25.0))     # master's half-spread tiers (bps)
EQ_SLIPPAGE_BPS = 5.0
UNKNOWN_FIELDS = ("option_volume", "open_interest", "quote_freshness")


def equity_cost_frac(mdv20: np.ndarray) -> np.ndarray:
    m = np.asarray(mdv20, dtype=float)
    hs = np.select([m >= t for t, _ in EQ_TIERS], [b for _, b in EQ_TIERS], default=EQ_TIERS[-1][1])
    return (hs + EQ_SLIPPAGE_BPS) / 1e4


def fill(bid: np.ndarray, ask: np.ndarray, level: str, side: str = "buy") -> np.ndarray:
    mid, h = (bid + ask) / 2.0, (ask - bid) / 2.0
    lam = LEVELS[level]
    return mid + lam * h if side == "buy" else np.maximum(mid - lam * h, 0.0)


def long_structure_returns(bid: np.ndarray, ask: np.ndarray, n_legs: int, spot_t: np.ndarray, payoff: np.ndarray,
                           itm_legs: np.ndarray, mdv20: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Net return per $ paid of a LONG structure held to expiry, at every fill level. ``payoff`` = intrinsic
    per share at the expiry close; ``itm_legs`` = number of legs that finish in the money."""
    c = equity_cost_frac(mdv20)
    out = {}
    for lv in LEVELS:
        paid = fill(bid, ask, lv, "buy") + n_legs * FEE_PER_SHARE
        net_full = payoff - itm_legs * c * spot_t
        net_intr = payoff - c * payoff
        out[lv] = {"ret": net_full / paid - 1.0, "ret_settle_cost_on_intrinsic": net_intr / paid - 1.0, "paid": paid}
    return out


def short_structure_returns(bid: np.ndarray, ask: np.ndarray, n_legs: int, spot_t: np.ndarray, payoff: np.ndarray,
                            itm_legs: np.ndarray, mdv20: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Net P&L per $ of premium received of a SHORT structure held to expiry (assignment -> the stock is
    bought back at the equity cost). Unbounded loss; reported per credit, not per margin."""
    c = equity_cost_frac(mdv20)
    out = {}
    for lv in LEVELS:
        got = fill(bid, ask, lv, "sell") - n_legs * FEE_PER_SHARE
        pnl = got - payoff - itm_legs * c * spot_t
        out[lv] = {"ret": np.where(got > 0, pnl / np.where(got > 0, got, np.nan), np.nan), "received": got}
    return out


def date_cluster_ci(x: np.ndarray, dates: np.ndarray, n_boot: int = 2000, seed: int = 7) -> list[float] | None:
    d = pd.DataFrame({"x": x, "d": dates}).dropna()
    g = d.groupby("d")["x"].agg(["sum", "count"])
    if len(g) < 5:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    m = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarize_levels(res: dict[str, dict[str, np.ndarray]], dates: np.ndarray) -> dict[str, Any]:
    out = {}
    for lv, r in res.items():
        x = np.asarray(r["ret"], dtype=float)
        ok = np.isfinite(x)
        out[lv] = {"n": int(ok.sum()), "n_dates": int(pd.Series(dates[ok]).nunique()),
                   "mean": float(np.mean(x[ok])) if ok.any() else None,
                   "median": float(np.median(x[ok])) if ok.any() else None,
                   "hit_rate": float(np.mean(x[ok] > 0)) if ok.any() else None,
                   "ci95_by_date": date_cluster_ci(x[ok], dates[ok]) if ok.sum() > 10 else None}
        if "ret_settle_cost_on_intrinsic" in r:
            y = np.asarray(r["ret_settle_cost_on_intrinsic"], dtype=float)
            out[lv]["mean_settle_cost_on_intrinsic"] = float(np.nanmean(y)) if np.isfinite(y).any() else None
    return out


def classify(levels: dict[str, Any]) -> str:
    """ROBUST: mean > 0 at PESSIMISTIC with the date-clustered CI above 0. EXECUTION-SENSITIVE: mean > 0 at
    MID or CONSERVATIVE only. NONVIABLE: mean <= 0 at MID."""
    pe, mid, co = levels.get("PESSIMISTIC", {}), levels.get("MID", {}), levels.get("CONSERVATIVE", {})
    if pe.get("mean") is not None and pe["mean"] > 0 and pe.get("ci95_by_date") and pe["ci95_by_date"][0] > 0:
        return "ROBUST"
    if (mid.get("mean") or -1) > 0 or (co.get("mean") or -1) > 0:
        return "EXECUTION-SENSITIVE"
    return "NONVIABLE"


# ------------------------------------------------------------------------------------------------
# model expectations under |move| = sigma sqrt(n/252) |Z|, Z unit-variance Student-t
# ------------------------------------------------------------------------------------------------
_Q = (np.arange(400) + 0.5) / 400


def _z(nu: int) -> np.ndarray:
    return sps.t.ppf(_Q, nu) * math.sqrt((nu - 2.0) / nu)


def expected_payoff(spot: np.ndarray, k_call: np.ndarray, k_put: np.ndarray, sigma: np.ndarray, n: np.ndarray,
                    nu: int) -> np.ndarray:
    """E[max(S_T - Kc, 0) + max(Kp - S_T, 0)], S_T = spot * exp(sigma sqrt(n/252) Z) (zero drift)."""
    z = _z(nu)[None, :]
    s = (sigma * np.sqrt(np.asarray(n, dtype=float) / 252.0))[:, None]
    st = np.asarray(spot)[:, None] * np.exp(s * z)
    return (np.maximum(st - np.asarray(k_call)[:, None], 0) + np.maximum(np.asarray(k_put)[:, None] - st, 0)).mean(axis=1)


def prob_profit_long(spot, k_call, k_put, sigma, n, nu, paid) -> np.ndarray:
    z = _z(nu)[None, :]
    s = (sigma * np.sqrt(np.asarray(n, dtype=float) / 252.0))[:, None]
    st = np.asarray(spot)[:, None] * np.exp(s * z)
    pay = np.maximum(st - np.asarray(k_call)[:, None], 0) + np.maximum(np.asarray(k_put)[:, None] - st, 0)
    return (pay > np.asarray(paid)[:, None]).mean(axis=1)


# ------------------------------------------------------------------------------------------------
# 25-delta strangle quotes from the chains (for O2)
# ------------------------------------------------------------------------------------------------
def strangle_quotes(chain: pd.DataFrame, target: float = 0.25) -> pd.DataFrame:
    """Per (date, act_symbol, expiration): the call with delta nearest +target and the put nearest -target
    (bid > 0, ask > bid, 0.01 < IV < 5)."""
    c = chain[(chain["bid"] > 0) & (chain["ask"] > chain["bid"]) & (chain["vol"] > 0.01) & (chain["vol"] < 5)
              & chain["delta"].notna()].copy()
    key = ["date", "act_symbol", "expiration"]
    out = []
    for cp, tgt in (("C", target), ("P", -target)):
        x = c[c["cp"] == cp].copy()
        x["dd"] = (x["delta"] - tgt).abs()
        x = x.loc[x.groupby(key)["dd"].idxmin(), key + ["strike", "bid", "ask", "delta", "vol"]]
        out.append(x.rename(columns={k: f"{k}_{cp.lower()}25" for k in ("strike", "bid", "ask", "delta", "vol")}))
    return out[0].merge(out[1], on=key, how="inner")
