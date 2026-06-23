"""Ported dependency-free module: reasoning.dataquality (pandas)."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ..reasoning import dataquality as dq


def test_dataquality_segmentation_pure_pandas():
    # 15-min regular series with a gap in the middle → two continuous segments
    n = 40
    ts = list(pd.date_range("2020-01-01", periods=n // 2, freq="15min"))
    ts += list(pd.date_range("2020-02-01", periods=n // 2, freq="15min"))  # big jump
    df = pd.DataFrame({"Timestamp": ts, "v": np.sin(np.arange(n) / 3.0)})
    seg = dq._dq_timeseries_segmentation(df, timestamp_tag="Timestamp")
    assert isinstance(seg, pd.DataFrame) and "segment_id" in seg.columns
    assert seg["segment_id"].nunique() >= 2


def test_dataquality_nan_removal():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, np.nan, 3.0, 4.0]})
    out = dq._efficient_nan_removal(df)
    assert not out["df_filter"].isna().any().any() and out["cost_total"] >= 0
