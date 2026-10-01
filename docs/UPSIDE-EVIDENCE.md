# What do big winners have in common?

Study run 2026-10-01 on real data. Scripts: `scripts/research/` (feat_replay → path_outcomes →
upside_analysis → build_upside_table). The protocol was fixed in the script docstring before any
result was seen.

## Data and protocol

- PIT replay of the live discovery scan, 2021-03-12 → 2024-11-27, every 5th session: 188 sessions,
  **334,901** research-universe observations with ADV ≥ $5M. The 2025+ holdout was never read.
- Entry at the next session's open. Each candidate's path is followed day by day for 20 sessions:
  the first touch of +5/8/10/15/20% (intraday high), of −T%, and of the 2×ATR stop. When the target
  and stop fall on the same day the order is unknowable from daily bars, so it is scored as a loss.
- **Clean win (T, h)**: +T% touched within h sessions before the stop. Pairs (5%, 5), (8%, 10),
  (10%, 20) — one per live hold arm.
- **Vol-neutral lift**: each outcome minus the mean for the same date and ATR quintile. Volatility
  mechanically raises every big-move probability, so a characteristic only counts if it adds
  something beyond volatility.
- Discovery half A (< 2023-01-24) / confirmation half B. Confirmed = |t| ≥ 3 on A, same sign with
  |t| ≥ 2 on B. Six combinations were pre-specified.

## Big moves are common, and almost symmetric

| target | within | touched | before stop | −T% first | median sessions | median dip first |
|---|---|---|---|---|---|---|
| +5% | 5 | 27.3% | 27.0% | 25.1% | 3 | −1.3% |
| +8% | 10 | 24.0% | 23.2% | 22.2% | 5 | −1.8% |
| +10% | 20 | 30.0% | 27.3% | 26.7% | 10 | −2.4% |
| +15% | 20 | 15.8% | 14.5% | 14.0% | 11 | −2.6% |

A liquid US stock is about as likely to fall 10% first as to rise 10% first.

## Volatility decides the odds, not the payoff

P(+10% within 20 before the stop) by ATR quintile: **11.6% → 18.9% → 25.8% → 34.0% → 46.4%**.
Net return over the same 20 sessions: +0.5%, +0.5%, +0.5%, +0.3%, **−0.7%**. The most volatile
fifth makes the most big winners and loses money on average. The odds are also higher with SPY below
its 200-day average (35.7% vs 24.9%) — volatile markets, not better ones.

## Four characteristics pass the bar — and all four are volatility, not direction

| characteristic | lift in clean wins | lift in −T% first | net (stop + time exit) |
|---|---|---|---|
| time since last earnings (longer) | +3.6 / +7.2 pp (5%/5d) | +4.1 / +4.6 pp | +5 / +29 bps, inconsistent |
| range contraction (more compressed) | +2.9 / +3.1 pp | +3.1 / +5.0 pp | +13 / +12 bps, t < 2 |
| distance below 52-week high (further) | +3.6 / +2.9 pp | +2.3 / +3.1 pp | not significant |
| one-day volume spike | +3.1 / +3.2 pp | +1.6 / +2.5 pp | +3 / +4 bps, t < 1 |

(A / B halves; the sign of compression and distance from the high is "more compressed" and "further
below" for readability.)

Each one raises the odds of a big move **down** about as much as **up**: an earnings date coming
into the window, a compressed range about to expand, a beaten-down stock, a volume spike. They
predict the **size** of the next move, not its direction, and none improves net return consistently.
None of the six pre-specified combinations was confirmed. Post-earnings strength (EAR z ≥ 1.5)
was negative in half B (t −3.4), consistent with the earlier catalyst study. The descriptive
discovery score showed no lift.

## Which hold suits which opportunity type

Stop + time exit as traded live, net per trade (bps), halves A / B:

| type | 5 sessions | 10 sessions | 20 sessions |
|---|---|---|---|
| all liquid | −28 / +3 | −39 / +29 | −51 / +52 |
| momentum | −60 / −3 | −94 / +12 | −127 / +20 |
| relative strength | −55 / +3 | −88 / +19 | −127 / +36 |
| volume | −7 / −9 | −20 / +18 | −75 / +47 |
| breakout | −70 / −6 | −83 / +24 | −74 / +21 |
| mean reversion | −66 / +15 | −63 / +77 | −67 / +127 |

No type is positive in both halves at any hold. The hold that works depends on the market: longer
holds earn more of the drift in a rising market (B) and lose more in a falling one (A). The live
5/10/20 experiment measures this forward; mean reversion at 20 sessions is the arm to watch.

## What changed in QuantLab

- `exploration/upside.py` + `exploration/data/upside_table.json`: 45 cells (5 ATR buckets × 3
  range-contraction buckets × 3 holds). Each cell holds P(touch +5/8/10/15%), P(before stop),
  P(−T% first), stop rate, median sessions to target, median dip before target, and net-return and
  MFE quantiles. Every exploratory decision — traded, watched and skipped — records its profile for
  its own hold.
- `quantlab explore learn`: missed big winners (not traded, touched +10%), failed picks (traded,
  stop breached), outcomes by selection × hold arm, and the recorded profile vs what happened.
- Stock selection is **unchanged**. Ranking for "big upside" would rank for volatility, which this
  study shows adds big losers as fast as big winners and lowers net return. The profile is used to
  describe and to calibrate, and it gives the options layer an empirical move distribution — the
  place where predicting move *size* can pay, if options underprice it. That is a forward test,
  because no historical option quotes exist (docs/OPTIONS.md).
