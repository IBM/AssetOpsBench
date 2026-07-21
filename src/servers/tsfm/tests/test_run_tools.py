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


def test_mcp_guidance_advertises_anomaly_run_recipe_path():
    instructions = mcp.instructions
    assert "anomaly detection" in instructions
    assert "find_models(task_id=\"tsfm_anomaly_detection\")" in instructions
    assert "search_models" in instructions
    assert "run_recipe" in instructions
    assert "tsfm_anomaly_detection" in instructions
    assert "\"estimator\": {\"model_id\": \"<selected_model_id>\"}" in instructions
    assert "flagged outputs" in instructions

    descriptions = {t.name: (t.description or "") for t in asyncio.run(mcp.list_tools())}
    search_models_doc = descriptions["search_models"]
    assert "model-catalog discovery tool" in search_models_doc
    assert "task_ids" in search_models_doc

    find_models_doc = descriptions["find_models"]
    assert "find_models(task_id=\"tsfm_anomaly_detection\")" in find_models_doc

    run_recipe_doc = descriptions["run_recipe"]
    assert "time-series anomaly detection" in run_recipe_doc
    assert "model-catalog tools" in run_recipe_doc
    assert "tsfm_anomaly_detection" in run_recipe_doc
    assert "\"estimator\": {\"model_id\": \"<selected>\"}" in run_recipe_doc
    assert "final anomaly segment/JSON answer" in run_recipe_doc


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


# ---- recipe_template: the contract must be TRUE, so every example is executed ----
def test_recipe_template_returns_the_contract():
    t = call("recipe_template", {})
    assert t["task_choices"] and t["estimator_spec"] and t["optional_blocks"]
    assert t["rules"] and t["examples"]


def test_recipe_template_documents_the_finetune_persistence_pair():
    """save_to is what feeds register_finetuned. An agent reads only recipe_template, so if the
    template omits save_to the persistence step is undiscoverable even though run_recipe supports
    it. finetune and save_to must both be advertised."""
    blocks = call("recipe_template", {})["optional_blocks"]
    assert any(b.startswith("finetune") for b in blocks)
    assert any(b.startswith("save_to") for b in blocks)


def test_every_forecast_example_actually_runs():
    """A template that lies is worse than none: run each example as-is."""
    t = call("recipe_template", {})
    ref = _series(n=240, asset="tmpl")
    for name in ["forecast_inline_estimator", "anomaly_detection"]:
        r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                                "target_columns": ["value"], "recipe": t["examples"][name]})
        assert "error" not in r, f"{name} -> {r.get('error')}"
        assert r["status"] == "success"


def test_catalog_model_example_has_the_right_shape():
    """The catalog example points at ttm_96_28, a real foundation card that needs transformers -
    so run its SHAPE against a classical card instead of pulling torch into the test suite."""
    t = call("recipe_template", {})
    ex = t["examples"]["forecast_with_a_catalog_model"]
    assert set(ex) == {"estimator", "fh"} and set(ex["estimator"]) == {"model_id"}

    call("register_model", {"model": {
        "model_id": "tmpl_classical", "description": "classical stand-in for the shape check",
        "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE,
        "params": {"strategy": "drift"}}})
    recipe = {**ex, "estimator": {"model_id": "tmpl_classical"}}
    r = call("run_recipe", {"dataset_path": _series(asset="tmpl_cat"),
                            "timestamp_column": "timestamp", "target_columns": ["value"],
                            "recipe": recipe})
    assert "error" not in r, r.get("error")
    assert r["status"] == "success"


def test_unknown_model_id_in_a_recipe_errors():
    r = call("run_recipe", {"dataset_path": _series(asset="ghost"),
                            "timestamp_column": "timestamp", "target_columns": ["value"],
                            "recipe": {"estimator": {"model_id": "no_such_card"}, "fh": [1]}})
    assert "error" in r and "not in catalog" in r["error"]


def test_ensemble_with_conformal_example_runs():
    """Regression: the conformal block used to crash on serialisation - sktime returns MultiIndex
    columns, so prediction_interval was keyed by tuples, which json.dump cannot write."""
    t = call("recipe_template", {})
    ref = _series(n=60, asset="tmpl_cf")           # short: conformal refits repeatedly
    r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": t["examples"]["forecast_ensemble_with_intervals"]})
    assert "error" not in r, r.get("error")
    assert r["status"] == "success"


def test_tabular_example_actually_runs():
    t = call("recipe_template", {})
    r = call("run_tabular_recipe", {"dataset_path": _panel(), "label_column": "label",
                                    "recipe": t["examples"]["tabular_classification"]})
    assert "error" not in r, r.get("error")
    assert r["status"] == "success" and r["task"] == "tsfm_classification"


def test_conformal_intervals_are_json_serialisable():
    """The whole result is written to a JSON file pointer, so every value must survive json.dump."""
    ref = _series(n=60, asset="cf_json")
    r = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                            "target_columns": ["value"],
                            "recipe": {"estimator": {"sktime_class": NAIVE,
                                                     "params": {"strategy": "drift"}},
                                       "fh": [1, 2], "conformal": {"coverage": 0.9}}})
    assert "error" not in r, r.get("error")
    payload = json.load(open(refs._path(r["results_file"])))
    pi = payload.get("prediction_interval") or {}
    assert all(isinstance(k, str) for k in pi), f"non-string keys reach json: {list(pi)[:3]}"


# ---- the result index: run_recipe must make its result findable via list_results/get_result ----
def test_run_recipe_indexes_its_result_for_get_result():
    """Every execution returns a results_file AND registers the result in its typed collection, so
    an agent can find it later by task_type without knowing the run_id. Before this was wired,
    list_results returned nothing because no run tool called results.write_result."""
    ref = _series(n=240, asset="idx_fc")
    call("register_model", {"model": {
        "model_id": "idx_nn", "description": "seasonal naive for the result-index test",
        "task_ids": ["tsfm_forecasting"], "provenance": "trained",
        "sktime_class": "sktime.forecasting.naive.NaiveForecaster",
        "params": {"strategy": "last", "sp": 24}}})
    run = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                              "target_columns": ["value"], "asset_id": "idx_fc",
                              "recipe": {"estimator": {"model_id": "idx_nn"}, "fh": [1, 2, 3]}})
    assert run["status"] == "success" and run["results_file"]

    listed = call("list_results", {"task_type": "tsfm_forecasting", "asset_id": "idx_fc"})["results"]
    assert listed, "run_recipe did not index its result"
    got = call("get_result", {"task_type": "tsfm_forecasting", "result_id": listed[0]["result_id"]})
    assert got["results_file"] == run["results_file"]      # points at the same payload
    assert got["summary"]["horizon"] == 3


def test_anomaly_run_indexes_its_result():
    ref = _series(n=300, asset="idx_ad")
    call("register_model", {"model": {
        "model_id": "idx_sublof", "description": "sublof for the anomaly result-index test",
        "task_ids": ["tsfm_anomaly_detection"], "provenance": "trained",
        "sktime_class": "sktime.detection.lof.SubLOF",
        "params": {"window_size": 24, "n_neighbors": 5, "novelty": True}}})
    run = call("run_recipe", {"dataset_path": ref, "timestamp_column": "timestamp",
                              "target_columns": ["value"], "asset_id": "idx_ad",
                              "recipe": {"task": "tsfm_anomaly_detection",
                                         "estimator": {"model_id": "idx_sublof"}}})
    assert run["status"] == "success"
    listed = call("list_results", {"task_type": "tsfm_anomaly_detection",
                                   "asset_id": "idx_ad"})["results"]
    assert listed
    got = call("get_result", {"task_type": "tsfm_anomaly_detection",
                              "result_id": listed[0]["result_id"]})
    assert got["results_file"] == run["results_file"]
    assert "anomaly_count" in got["summary"]
