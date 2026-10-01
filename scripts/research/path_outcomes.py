"""Path-dependent outcomes for every replay observation (REAL data, PIT, holdout untouched).

Input : feat_obs.parquet (date, symbol, features; 2021-03..2024-11, research universe).
Output: path_obs.parquet with, per observation, entry at the NEXT session's open and for up to 20
sessions:
  * day the +T% target is first TOUCHED (intraday high), T in 5/8/10/15/20%
  * day the stop (entry * (1 - 2 x atr14_pct)) is first touched, and day -T% is first touched
  * stop_first_T: the stop was touched before (or on the same day as) the target -- same-day order
    is unknowable from daily bars, so it is scored against us
  * mae_before_T: lowest low from entry up to the target day (or the whole window if never hit)
  * dist_52w_high at the decision, SPY regime at the decision (above 200d MA, 63d return)
Bars on/after the holdout start are never read.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\harry\Downloads\Businesspilot\quantlab")
sys.path.insert(0, str(REPO))
from quantlab.config import load_config                    # noqa: E402
from quantlab.context import AppContext                    # noqa: E402

SPW = Path(r"C:/Users/harry/AppData/Local/Temp/claude/C--Users-harry-Downloads-Businesspilot/a815f6ae-90eb-4d6b-8043-36bbc3440be6/scratchpad/research")
TARGETS = (5, 8, 10, 15, 20)
H = 20
t0 = time.time()
ctx = AppContext.create(load_config(SPW / "cfg.yaml", root=REPO), init_logging=False)
b = ctx.store.load_bundle(ctx.config.section("benchmarks"), snapshot=ctx.store.snapshot(synthetic=False), synthetic=False)
holdout = pd.Timestamp(ctx.config.get("validation.holdout.start", "2025-01-01"))
p = b.panel
dates = p.dates
last = int(dates.searchsorted(holdout, side="left")) - 1             # last readable index
pos = {d: i for i, d in enumerate(dates)}
col = {s: j for j, s in enumerate(p.symbols)}
ao, ah, al, ac = (p.aopen.to_numpy(float), p.ahigh.to_numpy(float), p.alow.to_numpy(float), p.aclose.to_numpy(float))
print(f"bundle loaded {ao.shape} in {time.time() - t0:.0f}s", flush=True)

o = pd.read_parquet(SPW / "feat_obs.parquet")
i = o["date"].map(pos).to_numpy()
c = o["symbol"].map(col).to_numpy()
e = i + 1
entry = np.where(e <= last, ao[np.minimum(e, len(dates) - 1), c], np.nan)
stop = entry * (1 - 2 * o["atr14_pct"].to_numpy(float))
n = len(o)
first_hit = {t: np.full(n, np.nan) for t in TARGETS}
first_dn = {t: np.full(n, np.nan) for t in TARGETS}
first_stop = np.full(n, np.nan)
run_low = np.full(n, np.inf)
mae_before = {t: np.full(n, np.nan) for t in TARGETS}
for k in range(H):
    idx = e + k
    ok = (idx <= last) & np.isfinite(entry)
    hi = np.where(ok, ah[np.minimum(idx, len(dates) - 1), c], np.nan)
    lo = np.where(ok, al[np.minimum(idx, len(dates) - 1), c], np.nan)
    run_low = np.where(np.isfinite(lo), np.minimum(run_low, lo), run_low)
    st = np.isnan(first_stop) & np.isfinite(lo) & (lo <= stop)
    first_stop[st] = k + 1
    for t in TARGETS:
        hit = np.isnan(first_hit[t]) & np.isfinite(hi) & (hi >= entry * (1 + t / 100))
        first_hit[t][hit] = k + 1
        mae_before[t][hit] = run_low[hit] / entry[hit] - 1
        dn = np.isnan(first_dn[t]) & np.isfinite(lo) & (lo <= entry * (1 - t / 100))
        first_dn[t][dn] = k + 1
full_window_low = np.where(np.isfinite(run_low), run_low / entry - 1, np.nan)
out = o.copy()
out["entry"] = entry
out["stop_pct"] = 2 * o["atr14_pct"].to_numpy(float)
out["days_to_stop"] = first_stop
out["mae_20"] = full_window_low
for t in TARGETS:
    out[f"days_to_{t}"] = first_hit[t]
    out[f"days_to_dn{t}"] = first_dn[t]
    out[f"mae_before_{t}"] = np.where(np.isfinite(first_hit[t]), mae_before[t], full_window_low)
    # stop touched no later than the target (same day counts against us), within 20 sessions
    out[f"stop_first_{t}"] = np.isfinite(first_stop) & (~np.isfinite(first_hit[t]) | (first_stop <= first_hit[t]))
# distance from the 52-week high and SPY regime at the decision session (rows <= i only)
hi252 = pd.DataFrame(ah).rolling(252, min_periods=200).max().to_numpy()
out["dist_52w_high"] = ac[i, c] / hi252[i, c] - 1
spy = col.get(b.market_symbol)
spy_c = ac[:, spy]
ma200 = pd.Series(spy_c).rolling(200, min_periods=150).mean().to_numpy()
out["spy_above_200"] = (spy_c[i] > ma200[i]).astype(float)
out["spy_ret_63"] = spy_c[i] / spy_c[np.maximum(i - 63, 0)] - 1
out.to_parquet(SPW / "path_obs.parquet", index=False)
print(f"DONE {len(out)} obs in {time.time() - t0:.0f}s; P(+10% in 20) = {np.isfinite(first_hit[10]).mean():.3f}", flush=True)
ctx.close()
