"""Immutable research splits (Part 5 of the alpha brief). Changing a date here invalidates every
result that used it; the test-suite pins these values.

EQUITY (Alpaca SIP daily, survivorship-free, 2016-01 .. 2024-12 in the store):
  TRAIN       2016-01-01 .. 2019-12-31   choose specs here (12-month lookbacks are live from 2017)
  VALIDATION  2020-01-01 .. 2021-12-31   confirm the chosen spec (sign + rough size), no re-tuning
  OOS         2022-01-01 .. 2024-12-31   ONE evaluation per locked spec (registry enforces)
  HOLDOUT     2025-01-01 ..              NOT in the store at all; needs a separate download + unlock

OPTIONS (DoltHub chains: weekly 2019, Mon/Wed/Fri 2020-2024):
  TRAIN       2019-02-01 .. 2021-12-31
  VALIDATION  2022-01-01 .. 2022-12-31
  OOS         2023-01-01 .. 2024-12-31
  HOLDOUT     2025-01-01 ..              (daily snapshots and a truly point-in-time earnings calendar)

Honesty note: QuantLab's earlier daily price/volume research used 2021-03..2024-11, so for price and
volume ideas the equity OOS window is not virgin territory for the researchers. The clean tests left
are the 2025+ holdout (already read twice for stock strategies) and forward paper trading.
"""
from __future__ import annotations

import pandas as pd

SPLITS: dict[str, dict[str, tuple[str, str | None]]] = {
    "equity": {"TRAIN": ("2016-01-01", "2019-12-31"), "VALIDATION": ("2020-01-01", "2021-12-31"),
               "OOS": ("2022-01-01", "2024-12-31"), "HOLDOUT": ("2025-01-01", None)},
    "options": {"TRAIN": ("2019-02-01", "2021-12-31"), "VALIDATION": ("2022-01-01", "2022-12-31"),
                "OOS": ("2023-01-01", "2024-12-31"), "HOLDOUT": ("2025-01-01", None)},
}
DEVELOPMENT = ("TRAIN", "VALIDATION")


class HoldoutAccessError(RuntimeError):
    pass


def window(dataset: str, split: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    a, b = SPLITS[dataset][split]
    return pd.Timestamp(a), (pd.Timestamp(b) if b else None)


def slice_split(x: pd.Series | pd.DataFrame, dataset: str, split: str):
    if split == "HOLDOUT":
        raise HoldoutAccessError("the holdout is not available to research code; see docs/ALPHA-DISCOVERY-PLAN.md")
    a, b = window(dataset, split)
    return x.loc[a:b]


def label_series(index: pd.DatetimeIndex, dataset: str) -> pd.Series:
    lab = pd.Series("PRE", index=index, dtype=object)
    for s in ("TRAIN", "VALIDATION", "OOS"):
        a, b = window(dataset, s)
        lab[(index >= a) & (index <= b)] = s
    lab[index >= window(dataset, "HOLDOUT")[0]] = "HOLDOUT"
    return lab
