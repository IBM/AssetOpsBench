"""Tasks + data/evidence tools: list_tasks, profile_series, characterize_series, data_quality.

These take FILE POINTERS (dataset_path) and return typed results; the server gives evidence, the
agent makes the decisions.
"""

import asyncio
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ..io import refs
from ..main import mcp

N = 240


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _iot(n=N, channels=1):
    """An IoT-style CSV file pointer (timestamp + channel columns)."""
    t = np.arange(n)
    if channels == 1:
        return refs.materialize_iot(np.sin(t / 6.0) + 0.01 * t, asset_id="ev_uni")
    X = np.column_stack([np.sin(t / 6.0), np.cos(t / 4.0) + 0.01 * t, np.sin(t / 3.0)])
    return refs.materialize_iot(X, asset_id="ev_mv")


def test_evidence_surface_present():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"list_tasks", "profile_series", "characterize_series", "data_quality"} <= names


def test_list_tasks():
    r = call("list_tasks", {})
    ids = {t["task_id"] for t in r["tasks"]}
    assert {"tsfm_forecasting", "tsfm_anomaly_detection"} <= ids


def test_profile_series_gives_facts():
    r = call("profile_series", {"dataset_path": _iot(), "timestamp_column": "timestamp"})
    assert r["n_observations"] == N and r["n_channels"] == 1
    assert "dominant_period" in r and "non_stationary" in r
    # evidence only - no decisions
    assert not any(k.startswith("recommend") or k.startswith("chosen") for k in r)
    assert "error" in call("profile_series", {"dataset_path": ""})


def test_profile_series_multivariate():
    r = call("profile_series", {"dataset_path": _iot(channels=3),
                                "timestamp_column": "timestamp"})
    assert r["n_channels"] == 3 and len(r["channels"]) == 3


def test_characterize_series():
    r = call("characterize_series", {"dataset_path": _iot(), "timestamp_column": "timestamp"})
    assert r["evidence_file"].startswith("file://")
    assert "error" in call("characterize_series", {"dataset_path": ""})


def test_data_quality_cleans_and_reports():
    n = 120
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="15min"),
                       "value": np.sin(np.arange(n) / 5.0)})
    df.loc[10:14, "value"] = np.nan
    refs._ensure_workdir()
    p = os.path.join(refs.WORKDIR, "ev_dq.csv")
    df.to_csv(p, index=False)

    r = call("data_quality", {"dataset_path": p, "timestamp_column": "timestamp"})
    assert r["status"] == "success"
    assert r["cleaned_file"].startswith("file://")
    assert r["rows_out"] < r["rows_in"]              # NaN rows removed
    assert "value" in r["nan_per_column"]
    assert "error" in call("data_quality", {"dataset_path": ""})


def test_data_quality_output_feeds_profile():
    """The cleaned file pointer is consumable by the next tool (the chaining contract)."""
    n = 120
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="15min"),
                       "value": np.sin(np.arange(n) / 5.0)})
    df.loc[5:9, "value"] = np.nan
    refs._ensure_workdir()
    p = os.path.join(refs.WORKDIR, "ev_chain.csv")
    df.to_csv(p, index=False)

    cleaned = call("data_quality", {"dataset_path": p, "timestamp_column": "timestamp"})["cleaned_file"]
    prof = call("profile_series", {"dataset_path": cleaned, "timestamp_column": "timestamp"})
    assert prof["n_observations"] == n - 5
