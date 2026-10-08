"""Ken French data-library CSV blocks (monthly portfolio files). PAPER research.

A file holds several blocks, for example "Average Value Weight Returns -- Monthly" and then equal-weighted and
annual blocks. Each block is a title line, a header line starting with a comma, and rows ``YYYYMM, v1, v2, ...``
in percent. Missing values are coded -99.99 or -999 and become NaN (UNKNOWN), never zero.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = (-99.99, -999.0)


def parse_block(text: str, *needles: str) -> pd.DataFrame:
    """The first block whose title line contains every one of ``needles`` (case-insensitive), as decimals indexed
    by month-end dates. With no needles, the file's first block (e.g. the monthly factors). Raises if no such
    block or no rows."""
    lines = text.splitlines()
    keys = [n.lower() for n in needles]
    try:
        t = next(i for i, ln in enumerate(lines) if all(k in ln.lower() for k in keys)
                 and not ln.strip().startswith(",")) if keys else -1
    except StopIteration:
        raise ValueError(f"no block titled like {needles!r}") from None
    h = next(i for i in range(t + 1, len(lines)) if lines[i].strip().startswith(","))
    cols = [c.strip() for c in lines[h].split(",")[1:]]
    rows, idx = [], []
    for ln in lines[h + 1:]:
        p = [x.strip() for x in ln.split(",")]
        if not p[0].isdigit() or len(p[0]) != 6:
            break
        idx.append(pd.Timestamp(year=int(p[0][:4]), month=int(p[0][4:]), day=1) + pd.offsets.MonthEnd(0))
        rows.append([float(x) if x not in ("", None) else np.nan for x in p[1:len(cols) + 1]])
    if not rows:
        raise ValueError(f"block {needles!r} has no rows")
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=cols)
    df = df.mask(df.isin(MISSING) | (df <= -99.0))
    return df / 100.0


def read_zip_block(path: Path, *needles: str) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_block(text, *needles)
