"""Top-layer hardening: every tool exercised through the real MCP boundary (mcp.call_tool),
covering success + validation/error paths. No tsfm_public/torch required."""

import asyncio, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ..main import mcp
from ..io import refs

NAIVE = {"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                       "params": {"strategy": "last"}}, "fh": [1, 2, 3]}


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _iot_ref(n=120):
    return refs.materialize_iot(np.sin(np.arange(n) / 4.0) + 0.01 * np.arange(n), asset_id="surf")


def _panel_path():
    rng = np.random.RandomState(0); n, T = 40, 24
    X = np.zeros((n, T)); y = np.array([0, 1] * (n // 2))
    for i in range(n):
        X[i] = np.sin(np.arange(T) * (0.2 if y[i] == 0 else 0.8)) + 0.1 * rng.randn(T)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(T)]); df["label"] = y
    refs._ensure_workdir(); p = os.path.join(refs.WORKDIR, "surf_panel.csv"); df.to_csv(p, index=False)
    return p


# ---- discovery ----
def test_list_tasks():
    r = call("list_tasks", {})
    assert "tasks" in r and len(r["tasks"]) == 8


def test_discover_components_ok_and_error():
    assert "components" in call("discover_components", {"task": "tsfm_forecasting"})
    assert "error" in call("discover_components", {"task": "   "})


def test_find_models_ok_and_error():
    assert "models" in call("find_models", {"task_id": "tsfm_forecasting"})
    assert "error" in call("find_models", {"task_id": ""})


def test_describe_candidates_error():
    assert "error" in call("describe_candidates", {"task_id": ""})


def test_find_features_ok():
    assert "features" in call("find_features", {})


def test_get_component_ok_and_error():
    assert "error" in call("get_component", {"component_id": "  "})
    assert "error" in call("get_component", {"component_id": "does_not_exist"})
    r = call("get_component", {"component_id": "ttm_96_28"})       # seed model
    assert r.get("component_id") == "ttm_96_28"


# ---- evidence / learn (file pointers) ----
def test_profile_series_ok_and_error():
    r = call("profile_series", {"dataset_path": _iot_ref()})
    assert r["n_observations"] == 120
    assert "error" in call("profile_series", {"dataset_path": ""})


def test_select_features_ok_and_error():
    r = call("select_features", {"dataset_path": _iot_ref(300), "target_column": "value",
                                 "reference_feature": "kurtosis"})
    assert r["detail_file"].startswith("file://")
    assert "error" in call("select_features", {"dataset_path": ""})


# ---- compose + run ----
def test_run_recipe_ok_and_validation():
    r = call("run_recipe", {"dataset_path": _iot_ref(), "timestamp_column": "timestamp",
                            "target_columns": ["value"], "recipe": NAIVE})
    assert r["status"] == "success" and r["results_file"].startswith("file://")
    assert "error" in call("run_recipe", {"dataset_path": _iot_ref(), "timestamp_column": "timestamp",
                                          "target_columns": ["value"], "recipe": {}})  # no estimator


def test_run_tabular_recipe_ok_and_validation():
    recipe = {"task": "tsfm_classification",
              "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                            "params": {"n_estimators": 40, "random_state": 0}}}
    r = call("run_tabular_recipe", {"dataset_path": _panel_path(), "recipe": recipe, "label_column": "label"})
    assert r["status"] == "success" and r["task"] == "tsfm_classification"
    assert "error" in call("run_tabular_recipe", {"dataset_path": _panel_path(), "recipe": {}})


def test_run_plan_and_evaluate_validation():
    assert "error" in call("run_plan", {"plan_spec": {}})
    assert "error" in call("evaluate", {"recipe": NAIVE, "configs": []})       # empty configs
    assert "error" in call("evaluate", {"recipe": {}, "configs": [{"x": 1}]})  # bad recipe


# ---- write-back ----
def test_register_model_ok_and_error():
    assert "error" in call("register_model", {"model": {}})              # empty
    assert "error" in call("register_model", {"model": {"model_id": "x"}})  # missing required fields
    r = call("register_model", {"model": {
        "model_id": "surface_test_model",
        "sktime_class": "sktime.forecasting.naive.NaiveForecaster",
        "description": "surface test model", "task_ids": ["tsfm_forecasting"]}})
    assert r.get("status") == "registered" and r.get("id") == "surface_test_model"


def test_register_feature_error():
    assert "error" in call("register_feature", {"feature": {}})


# ---- results / runs ----
def test_results_and_runs():
    assert "error" in call("get_result", {"task_type": "tsfm_forecasting", "result_id": "nope"})
    assert "results" in call("list_results", {"task_type": "tsfm_forecasting"})
    assert "error" in call("get_run", {"run_id": "nope"})
    runs = call("list_runs", {})
    assert "runs" in runs and "plans" in runs

def _mv_path(n=120):
    """A 2-channel CSV (timestamp + a + b) for multivariate extraction."""
    refs._ensure_workdir()
    t = pd.date_range("2020-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        {"timestamp": t, "a": np.sin(np.arange(n) / 4.0), "b": np.cos(np.arange(n) / 6.0)}
    )
    p = os.path.join(refs.WORKDIR, "surf_mv.csv")
    df.to_csv(p, index=False)
    return p


# ---- extract_features ----
def test_extract_features_whole_series():
    r = call("extract_features", {"dataset_path": _iot_ref(120), "timestamp_column": "timestamp",
                                  "target_columns": ["value"], "extractors": ["mean", "std"]})
    assert r["n_windows"] == 1                       # whole series -> one row
    assert r["columns"] == ["mean", "std"]           # single channel -> bare names
    assert len(r["features"]) == 1 and len(r["features"][0]) == 2


def test_extract_features_windowed():
    r = call("extract_features", {"dataset_path": _iot_ref(120), "timestamp_column": "timestamp",
                                  "target_columns": ["value"], "extractors": ["mean"], "window": 30})
    assert r["n_windows"] == 4                        # 120 / 30 non-overlapping tiles
    assert len(r["features"]) == 4


def test_extract_features_multivariate():
    r = call("extract_features", {"dataset_path": _mv_path(120), "timestamp_column": "timestamp",
                                  "target_columns": ["a", "b"], "extractors": ["mean"]})
    assert set(r["columns"]) == {"a.mean", "b.mean"}  # per-channel prefixed columns
    assert r["n_windows"] == 1


def test_extract_features_validation():
    assert "error" in call("extract_features",
                           {"dataset_path": _iot_ref(), "extractors": []})            # none picked
    assert "error" in call("extract_features",
                           {"dataset_path": _iot_ref(), "extractors": ["not_a_real_extractor"]})