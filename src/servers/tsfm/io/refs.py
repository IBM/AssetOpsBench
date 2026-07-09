"""io_refs.py — file-pointer data I/O (IoT data is passed by reference, not inline).

IoT sensor data and every step output are **file pointers** (path / file:// / s3:// URI to a
CSV/Parquet), exactly like HuggingGPT chains tasks via resource files (`<resource>-task_id`).
The MCP tools carry small JSON (a `data_ref`), never the array; this layer loads a ref into the
sktime-native container (pd.Series univariate / pd.DataFrame multivariate) and writes step
outputs back to new file pointers so downstream steps can consume them.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import List, Optional

import numpy as np
import pandas as pd

WORKDIR = os.environ.get("TSFM_WORKDIR", "/tmp/tsfm_work")


def _path(ref: str) -> str:
    return ref[7:] if ref.startswith("file://") else ref


def _ensure_workdir():
    os.makedirs(WORKDIR, exist_ok=True)
    return WORKDIR


def load_series(data_ref: str, *, value_col: Optional[str] = None,
                time_col: Optional[str] = None, channels: Optional[List[str]] = None):
    """Resolve a file pointer to an sktime container. CSV/Parquet → pd.Series (univariate) or
    pd.DataFrame (multivariate, channel-subset aware). Index becomes the time axis."""
    p = _path(data_ref)
    df = pd.read_parquet(p) if p.endswith((".parquet", ".pq")) else pd.read_csv(p)
    # auto-detect the time column if not given (timestamp/time/date/datetime)
    if time_col is None:
        for c in df.columns:
            if c.lower() in ("timestamp", "time", "date", "datetime"):
                time_col = c
                break
    if time_col and time_col in df.columns:
        df = df.drop(columns=[time_col])
    # numeric value columns only; drop any column that isn't numeric (coerces all-NaN away)
    num = df.apply(pd.to_numeric, errors="coerce").astype("float64")  # no nullable dtypes / pd.NA
    num = num.loc[:, num.notna().any(axis=0)]
    cols = channels or list(num.columns)
    out = num[cols].reset_index(drop=True)               # RangeIndex keeps sktime happy offline
    return out[cols[0]] if len(cols) == 1 else out


def write_series(obj, *, name: str = "result") -> str:
    """Write a forecast/series/frame to a file pointer and return the ref."""
    _ensure_workdir()
    p = os.path.join(WORKDIR, f"{name}_{uuid.uuid4().hex[:8]}.csv")
    (obj.to_frame() if isinstance(obj, pd.Series) else pd.DataFrame(obj)).to_csv(p, index=False)
    return f"file://{p}"


def write_json(obj: dict, *, name: str = "result") -> str:
    _ensure_workdir()
    p = os.path.join(WORKDIR, f"{name}_{uuid.uuid4().hex[:8]}.json")
    json.dump(obj, open(p, "w"), indent=2, default=str)
    return f"file://{p}"


def materialize_iot(series, *, asset_id: str = "asset", value_cols=None, freq: str = "H") -> str:
    """Test helper: write an in-memory series/array to an IoT-style CSV file pointer."""
    _ensure_workdir()
    arr = np.asarray(series, float)
    if arr.ndim == 1:
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(arr), freq=freq),
                           "value": arr})
    else:
        cols = value_cols or [f"ch{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        df.insert(0, "timestamp", pd.date_range("2020-01-01", periods=len(arr), freq=freq))
    p = os.path.join(WORKDIR, f"iot_{asset_id}.csv")
    df.to_csv(p, index=False)
    return f"file://{p}"
