"""The redesigned tools take FILE POINTERS (dataset_path) and return typed Pydantic results
with a results_file pointer — matching the legacy TSFM tools and the IoT data model."""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ..io import refs
from ..core.results_models import (
    ProfileResult, FeatureSelectionResult, RecipeResult, TabularResult, ErrorResult,
)
from ..main import (
    profile_series, select_features, run_recipe, run_tabular_recipe, list_tasks,
)

NAIVE = {"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                       "params": {"strategy": "last"}}, "fh": [1, 2, 3]}


def test_profile_series_takes_file_pointer():
    ref = refs.materialize_iot(np.sin(np.arange(200) / 3.0) + 0.01 * np.arange(200), asset_id="p")
    r = profile_series(ref)                       # dataset_path, not inline data
    assert isinstance(r, ProfileResult)
    assert r.n_observations == 200 and r.dominant_period and r.source == ref


def test_profile_series_validates_input():
    assert isinstance(profile_series("   "), ErrorResult)


def test_run_recipe_file_pointer_in_results_file_out():
    ref = refs.materialize_iot(np.sin(np.arange(120) / 4.0) + 0.01 * np.arange(120), asset_id="r")
    r = run_recipe(ref, "timestamp", ["value"], NAIVE)
    assert isinstance(r, RecipeResult) and r.status == "success"
    assert r.results_file.startswith("file://") and isinstance(r.backtest_score, float)
    # the results_file pointer actually resolves to the full run record
    rec = __import__("json").load(open(refs._path(r.results_file)))
    assert rec["run_id"] == r.run_id and "forecast_head" in rec


def test_run_recipe_validates_inputs():
    assert isinstance(run_recipe("", "timestamp", ["value"], NAIVE), ErrorResult)
    ref = refs.materialize_iot(np.arange(50.0), asset_id="r2")
    assert isinstance(run_recipe(ref, "timestamp", [], NAIVE), ErrorResult)


def test_select_features_file_pointer():
    ref = refs.materialize_iot(np.sin(np.arange(300) / 4.0) + 0.01 * np.arange(300), asset_id="s")
    r = select_features(ref, target_column="value", reference_feature="kurtosis")
    assert isinstance(r, FeatureSelectionResult)
    assert r.detail_file.startswith("file://") and isinstance(r.selected, list)


def test_run_tabular_recipe_file_pointer():
    rng = np.random.RandomState(0)
    n, T = 40, 24
    X = np.zeros((n, T)); y = np.array([0, 1] * (n // 2))
    for i in range(n):
        f = 0.2 if y[i] == 0 else 0.8
        X[i] = np.sin(np.arange(T) * f) + 0.1 * rng.randn(T)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(T)]); df["label"] = y
    refs._ensure_workdir()
    path = os.path.join(refs.WORKDIR, "panel.csv"); df.to_csv(path, index=False)
    recipe = {"task": "tsfm_classification",
              "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                            "params": {"n_estimators": 40, "random_state": 0}}}
    r = run_tabular_recipe(path, recipe, label_column="label")
    assert isinstance(r, TabularResult) and r.status == "success" and r.task == "tsfm_classification"
    assert r.results_file.startswith("file://")
