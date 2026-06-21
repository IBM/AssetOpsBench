"""Series→tabular run path: FeatureUnion(extractors) → estimator for reg/clf/clustering."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tsfm.engine import composition as C


def _two_class_panel(n=40, T=60, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, T)); y = np.array([0, 1] * (n // 2))
    for i in range(n):
        f = 0.2 if y[i] == 0 else 0.8
        X[i] = np.sin(np.arange(T) * f) + 0.1 * rng.randn(T)
    return X, y


def test_classification_default_library():
    X, y = _two_class_panel()
    r = C.run_tabular_recipe(
        None, X,
        {"task": "tsfm_classification",
         "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                       "params": {"n_estimators": 60, "random_state": 0}}},
        y=y)
    assert r["metric"] == "accuracy" and r["cv_score"] > 0.8
    assert r["n_features"] == 15 and r["training_regime"] == "fit_on_series"


def test_regression_with_explicit_extractors():
    rng = np.random.RandomState(1)
    n, T = 60, 50; X = np.zeros((n, T)); y = np.zeros(n)
    for i in range(n):
        slope = rng.uniform(-0.05, 0.05); X[i] = slope * np.arange(T) + 0.1 * rng.randn(T)
        y[i] = slope                                            # target = the trend slope
    r = C.run_tabular_recipe(
        None, X,
        {"task": "tsfm_regression",
         "transforms": [{"extractors": ["slope", "mean", "std", "energy"]}],
         "estimator": {"sktime_class": "sklearn.ensemble.RandomForestRegressor",
                       "params": {"n_estimators": 80, "random_state": 0}}},
        y=y)
    assert r["metric"] == "r2" and r["cv_score"] > 0.5 and r["n_features"] == 4


def test_clustering_silhouette():
    X, _ = _two_class_panel()
    r = C.run_tabular_recipe(
        None, X,
        {"task": "tsfm_clustering",
         "estimator": {"sktime_class": "sklearn.cluster.KMeans",
                       "params": {"n_clusters": 2, "n_init": 10, "random_state": 0}}})
    assert r["metric"] == "silhouette" and isinstance(r["cv_score"], float)


def test_flops_select_columns():
    """flops_select extracts the full library then keeps the most relevant columns vs y."""
    X, y = _two_class_panel()
    r = C.run_tabular_recipe(
        None, X,
        {"task": "tsfm_classification",
         "transforms": [{"flops_select": {"top_k": 5}}],
         "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                       "params": {"n_estimators": 60, "random_state": 0}}},
        y=y)
    assert r["n_features"] == 5 and r["cv_score"] > 0.8


def test_sktime_panel_transformer():
    """An sktime panel transformer (SummaryTransformer, dependency-free) composes too."""
    X, y = _two_class_panel()
    r = C.run_tabular_recipe(
        None, X,
        {"task": "tsfm_classification",
         "transforms": [{"sktime_class": "sktime.transformations.series.summarize.SummaryTransformer"}],
         "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                       "params": {"n_estimators": 60, "random_state": 0}}},
        y=y)
    assert r["n_features"] >= 5 and isinstance(r["cv_score"], float)
