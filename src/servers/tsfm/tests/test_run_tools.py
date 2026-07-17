"""Compose-and-run tools + the run/result ledger.

These actually FIT models. The suite deliberately uses classical sktime estimators (NaiveForecaster,
RandomForest) so it is a real fit but fast, offline, and free of torch/HuggingFace downloads. The
foundation-model paths (TTM/Chronos) are exercised by the smoke scripts against real CouchDB.
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

NAIVE = "sktime.forecasting.naive.NaiveForecaster"


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _series(n=200, asset="rt"):
    return refs.materialize_iot(np.sin(np.arange(n) / 6.0) + 0.01 * np.arange(n), asset_id=asset)


def _panel():
    rng = np.random.RandomState(0)
    n, T = 40, 24
    X = np.zeros((n, T)); y = np.array([0, 1] * (n // 2))
    for i in range(n):
        X[i] = np.sin(np.arange(T) * (0.2 if y[i] == 0 else 0.8)) + 0.1 * rng.randn(T)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(T)]); df["label"] = y
    refs._ensure_workdir()
    p = os.path.join(refs.WORKDIR, "rt_panel.csv"); df.to_csv(p, index=False)
    return p


def test_run_surface_present():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"run_recipe", "run_tabular_recipe", "run_plan", "evaluate",
            "get_result", "list_results", "get_run", "list_runs"} <= names


# ---- run_recipe: a real fit + backtest ----
def test_run_recipe_fits_and_backtests():
    r = call("run_recipe", {"dataset_path": _series(), "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"estimator": {"sktime_class": NAIVE,
                                                     "params": {"strategy": "drift"}},
                                       "fh": [1, 2, 3, 4, 5]}})
    assert r["status"] == "success"
    assert r["run_id"].startswith("run:")
    assert r["metric"] and isinstance(r["backtest_score"], (int, float))
    assert r["results_file"].startswith("file://")


def test_run_recipe_validates_the_recipe():
    ref = _series()
    assert "non-empty object" in call("run_recipe", {"dataset_path": ref,
        "timestamp_column": "timestamp", "target_columns": ["value"], "recipe": {}})["error"]
    assert "estimator" in call("run_recipe", {"dataset_path": ref,
        "timestamp_column": "timestamp", "target_columns": ["value"],
        "recipe": {"fh": [1]}})["error"]
    assert "error" in call("run_recipe", {"dataset_path": "", "timestamp_column": "timestamp",
        "target_columns": ["value"], "recipe": {"estimator": {"sktime_class": NAIVE}}})


def test_run_recipe_is_recorded_in_the_ledger():
    r = call("run_recipe", {"dataset_path": _series(asset="ledger"),
                            "timestamp_column": "timestamp", "target_columns": ["value"],
                            "recipe": {"estimator": {"sktime_class": NAIVE}, "fh": [1, 2]}})
    run_id = r["run_id"]
    assert any(x["run_id"] == run_id for x in call("list_runs", {})["runs"])
    assert call("get_run", {"run_id": run_id})["run_id"] == run_id
    assert "error" in call("get_run", {"run_id": "run:nope"})


# ---- run_tabular_recipe ----
def test_run_tabular_recipe_classifies():
    r = call("run_tabular_recipe", {"dataset_path": _panel(), "label_column": "label",
             "recipe": {"task": "tsfm_classification",
                        "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                                      "params": {"n_estimators": 10}}}})
    assert r["status"] == "success" and r["task"] == "tsfm_classification"
    assert r["metric"] == "accuracy" and r["cv_score"] > 0.5
    assert r["n_features"] > 0                       # FLOps features were extracted


def test_run_tabular_recipe_validates():
    assert "error" in call("run_tabular_recipe", {"dataset_path": "", "recipe": {}})
    assert "error" in call("run_tabular_recipe", {"dataset_path": _panel(),
        "label_column": "not_a_column",
        "recipe": {"estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier"}}})


# ---- run_plan: a recipe DAG ----
def test_run_plan_executes_a_step():
    ref = _series(asset="plan")
    r = call("run_plan", {"asset_id": "chiller6", "plan_spec": {"steps": [
        {"id": "s1", "task": "forecast", "args": {"data_ref": ref},
         "recipe": {"estimator": {"sktime_class": NAIVE, "params": {"strategy": "drift"}},
                    "fh": [1, 2, 3]}}]}})
    assert r["status"] == "success" and r["plan_id"].startswith("plan:")
    assert r["outputs"]["s1"]["ref"].startswith("file://")


def test_run_plan_runs_a_two_step_dag():
    ref = _series(asset="dag")
    r = call("run_plan", {"asset_id": "chiller6", "plan_spec": {"steps": [
        {"id": "s1", "task": "forecast", "args": {"data_ref": ref},
         "recipe": {"estimator": {"sktime_class": NAIVE, "params": {"strategy": "last"}}, "fh": [1, 2]}},
        {"id": "s2", "task": "forecast", "dep": ["s1"], "args": {"data_ref": ref},
         "recipe": {"estimator": {"sktime_class": NAIVE, "params": {"strategy": "mean"}}, "fh": [1, 2]}}]}})
    assert r["status"] == "success"
    assert set(r["outputs"]) == {"s1", "s2"}         # both steps ran, in dependency order


# ---- evaluate ----
def test_evaluate_scores_a_recipe():
    r = call("evaluate", {"recipe": {"estimator": {"sktime_class": NAIVE,
                                                   "params": {"strategy": "drift"}}},
                          "configs": [{"name": "c1", "y": list(np.sin(np.arange(120) / 6.0)),
                                       "fh": [1, 2, 3], "sp": 24}]})
    assert r["status"] == "success"
    assert r["results_file"].startswith("file://")
    assert r["per_config"] and "agg" in r


def test_evaluate_validates():
    assert "error" in call("evaluate", {"recipe": {}, "configs": [{"name": "c", "y": [1, 2]}]})
    assert "must not be empty" in call("evaluate", {
        "recipe": {"estimator": {"sktime_class": NAIVE}}, "configs": []})["error"]


# ---- results ledger ----
def test_list_results_reads_the_ledger():
    r = call("list_results", {"task_type": "tsfm_forecasting"})
    assert "results" in r
    assert "error" in call("get_result", {"task_type": "tsfm_forecasting", "result_id": "nope"})
