"""Anomaly via run_recipe (task dispatch) + the data-quality tool — through the MCP boundary."""

import asyncio, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ..main import mcp
from ..io import refs


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _spiky_iot(n=300):
    y = np.sin(np.arange(n) / 6.0) + 0.05 * np.random.RandomState(0).randn(n)
    y[120] += 8.0; y[240] -= 8.0                     # two injected anomalies
    return refs.materialize_iot(y, asset_id="adq")


# ---- anomaly runs through run_recipe (task dispatch), not a separate tool ----
def test_run_recipe_anomaly_sublof_flags_spikes():
    ref = _spiky_iot()
    r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"task": "tsfm_anomaly_detection",
                                       "estimator": {"model_id": "sublof"}}})
    assert r["status"] == "success" and r["results_file"].startswith("file://")
    assert r["n_anomalies"] >= 1 and r["n_observations"] == 300
    assert r["training_regime"] == "fit_on_series"          # SubLOF is classical
    rec = json.load(open(refs._path(r["results_file"])))
    assert len(rec["anomaly_label"]) == 300 and sum(rec["anomaly_label"]) == r["n_anomalies"]


def test_anomaly_validation_and_tspulse_gated():
    ref = _spiky_iot()
    assert "error" in call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                                          "target_columns": ["value"],
                                          "recipe": {"task": "tsfm_anomaly_detection"}})  # no detector
    # TSPulse zero-shot resolves to the sktime detector but needs tsfm_public → typed error, not crash
    r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"task": "tsfm_anomaly_detection",
                                       "estimator": {"model_id": "tspulse_ad"}}})
    assert "error" in r


# ---- data quality tool ----
def test_data_quality_cleans_and_summarizes():
    n = 100
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq="15min"),
                       "value": np.sin(np.arange(n) / 5.0)})
    df.loc[10:14, "value"] = np.nan                  # a gap of NaNs
    refs._ensure_workdir()
    p = os.path.join(refs.WORKDIR, "dq_in.csv"); df.to_csv(p, index=False)
    r = call("data_quality", {"dataset_path": p, "timestamp_column": "timestamp"})
    assert r["status"] == "success" and r["cleaned_file"].startswith("file://")
    assert r["rows_out"] < r["rows_in"]              # NaN rows removed
    assert "error" in call("data_quality", {"dataset_path": ""})
