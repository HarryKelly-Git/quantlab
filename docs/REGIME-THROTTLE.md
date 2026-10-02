# Regime throttle (PAPER_EXPLORATION)

A **risk throttle, not a ban**. When the market is below its long-term trend, the exploration
layer opens fewer new paper positions per session. It never stops exploration, never touches the
selection score, sizing, stops, hold arms, or the pinned / industry checks, and it changes nothing in
STRICT mode beyond the number of SHADOW selections. Paper only.

## Rule

```yaml
exploration:
  regime_throttle:
    enabled: true
    metric: market_trend_200      # SPY close vs its 200-session average, e.g. 0.062 = 6.2% above
    below: 0.0                    # throttle when the metric is strictly below this
    max_new_per_session: 2        # cap while throttled; effective cap = min(exploration.max_new_per_session, this)
```

* **Source.** `regime_snapshots` row whose `as_of_date` is the decision session D (`metrics_json`
  holds `market_trend_200`). The daily pipeline writes it in the `research` step, which runs before
  `discover` and `explore` (`pipeline/daily.py` STEPS), from the bundle truncated to D. Its inputs are
  trailing market features (`tests/regime/test_regime.py` checks they are point-in-time).
* **No look-ahead.** Only D's own snapshot is read. A later snapshot is never read, and an earlier one
  is never carried forward. `tests/exploration/test_regime_watchlist.py` covers both cases plus a
  truncation-invariance test of the cap.
* **Missing = UNKNOWN.** With no snapshot for D, or an unknown metric (fewer than 200 sessions of SPY),
  nothing is throttled, and the decisions record the state as `UNKNOWN`.
* **Recorded on every decision.** Each decision's `pre_trade_json["regime"]` holds `as_of_date`,
  `label`, `market_trend_200`, `throttled`, `effective_max_new`, `state` (THROTTLED, NOT_THROTTLED,
  UNKNOWN or DISABLED), and `displaced_by_throttle`.
* **The counterfactual is kept.** Eligible candidates beyond the throttled cap, up to the normal budget,
  are `WATCHED_NOT_TRADED` with a reason naming the throttle. They do not use up the ordinary
  watched-not-traded allowance, so the regular comparison group is the same as in an unthrottled session.
* **Forward test.** `learning_report()` (CLI `quantlab explore learn`) splits matured outcomes by
  throttle state (`by_throttle`) and by regime label (`by_regime`). Throttle-displaced candidates are a
  separate arm.

## Evidence (pre-registered, point-in-time, real data 2021-03 to 2024-11)

| SPY vs 200-day average | net at 5 sessions | 10 sessions | 20 sessions |
|---|---|---|---|
| above | +19 bps | +39 bps | +46 bps |
| below | -2 bps | -8 bps | -24 bps |

These are the bot's picks in the **2021-23 half**. The **2023-24 half cannot confirm or refute it**:
SPY was above its 200-day average on 98% of those sessions, so the "below" bucket is close to empty.
The split is therefore supported by one half-sample only. It is consistent with the trend-filter
literature (time-series momentum and moving-average market timing), but it is **not validated**.

## Status

**Unvalidated hypothesis, on by default as a risk throttle.** The case for it is asymmetric. Throttling
in a downtrend costs at most 3 exploratory entries a session, and the displaced candidates are still
scored, so the forward data shows what the throttle cost or saved. Review it with `by_throttle` once the
"below" bucket has a meaningful number of matured outcomes. Do not tune `below` or
`max_new_per_session` to those results: they are fixed in advance, like every other exploration limit.

## Switching it off

Set `exploration.regime_throttle.enabled: false` (for example in `config/local.yaml`). Decisions then
record `state: DISABLED` and the full `max_new_per_session` applies. The regime label is still
recorded, so the by-regime split keeps accumulating.
