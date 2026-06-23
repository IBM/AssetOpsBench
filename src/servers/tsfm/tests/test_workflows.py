"""The two target workflows, end-to-end through the MCP boundary (file pointer in → result out):
  1. forecasting        — run_recipe with a forecaster card
  2. prediction-based AD with conformal — run_recipe(task=anomaly, method=conformal)
Both run on classical sktime models (no tsfm_public); swap the card for TTM/TSPulse in prod."""

import asyncio, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
from ..main import mcp
from ..io import refs


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


# ---- Workflow 1: forecasting ----
def test_forecasting_workflow():
    ref = refs.materialize_iot(20 + 5 * np.sin(np.arange(240) / 12.0) + 0.02 * np.arange(240),
                               asset_id="wf_fc")
    r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"estimator": {"model_id": "naive_persistence"},
                                       "fh": [1, 2, 3, 4, 5]}})
    assert r["status"] == "success" and r["results_file"].startswith("file://")
    assert r["metric"] and isinstance(r["backtest_score"], (int, float))
    rec = json.load(open(refs._path(r["results_file"])))
    assert len(rec["forecast_head"]) == 5


# ---- Workflow 2: prediction-based AD with conformal ----
def test_conformal_anomaly_workflow():
    y = 5 * np.sin(np.arange(200) / 8.0) + 0.2 * np.random.RandomState(0).randn(200)
    y[196] += 25.0                                    # injected anomaly in the recent window
    ref = refs.materialize_iot(y, asset_id="wf_cad")
    r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"task": "tsfm_anomaly_detection", "method": "conformal",
                                       "estimator": {"model_id": "naive_persistence"},
                                       "conformal": {"coverage": 0.9}, "fh": list(range(1, 11))}})
    assert r["status"] == "success" and r["n_anomalies"] >= 1
    rec = json.load(open(refs._path(r["results_file"])))
    assert rec["anomaly_indices"] and 196 in rec["anomaly_indices"]   # the spike is flagged


def test_dq_then_conformal_chain():
    """The scenario shape: data_quality (clean) → feed cleaned file into conformal AD."""
    y = 5 * np.sin(np.arange(160) / 8.0)
    y[150] += 20.0
    import pandas as pd, os
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=160, freq="15min"), "value": y})
    df.loc[20:22, "value"] = np.nan
    refs._ensure_workdir(); p = os.path.join(refs.WORKDIR, "wf_dq.csv"); df.to_csv(p, index=False)
    dq = call("data_quality", {"dataset_path": p, "timestamp_column": "timestamp"})
    assert dq["status"] == "success" and dq["rows_out"] < dq["rows_in"]
    r = call("run_recipe", {"dataset_path": dq["cleaned_file"], "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"task": "tsfm_anomaly_detection", "method": "conformal",
                                       "estimator": {"model_id": "naive_persistence"},
                                       "conformal": {"coverage": 0.9}, "fh": list(range(1, 13))}})
    assert r["status"] == "success" and r["n_anomalies"] >= 1
