"""Untrusted recipe targets must never import or construct arbitrary callables."""

import pytest

from servers.tsfm.substrate import resolver


@pytest.mark.parametrize(
    "path",
    [
        "subprocess.Popen",
        "os.system",
        "builtins.eval",
        "importlib.import_module",
        "sktime_evil.models.Estimator",
        "sklearn_evil.models.Estimator",
        "pyod_evil.models.Detector",
        "sktime.utils._testing.Estimator",
        "sktime..Estimator",
        "sktime.base.__dict__",
        "sktime",
        42,
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_untrusted_targets_are_rejected_before_import(path, nested, monkeypatch):
    imports = []

    def forbidden_import(module):
        imports.append(module)
        raise AssertionError("untrusted target reached import")

    monkeypatch.setattr(resolver.importlib, "import_module", forbidden_import)
    with pytest.raises((ValueError, TypeError)):
        if nested:
            resolver._build({"wrapped": [{"_target_": path, "params": {}}]})
        else:
            resolver.resolve({"sktime_class": path, "params": {}})
    assert imports == []


@pytest.mark.parametrize(
    "path",
    [
        "sktime.base.BaseObject",
        "sklearn.base.clone",
        "sklearn.base.np",
    ],
)
def test_supported_namespace_does_not_allow_functions_or_modules(path):
    with pytest.raises(TypeError, match="estimator class"):
        resolver.resolve({"sktime_class": path})


def test_naive_forecaster_still_resolves():
    from sktime.forecasting.naive import NaiveForecaster

    model = resolver.resolve(
        {
            "sktime_class": "sktime.forecasting.naive.NaiveForecaster",
            "params": {"strategy": "drift"},
        }
    )
    assert isinstance(model, NaiveForecaster)
    assert model.strategy == "drift"


def test_non_estimator_class_is_rejected_without_construction(monkeypatch):
    import sklearn.ensemble

    constructed = []

    class NotAnEstimator:
        def __init__(self):
            constructed.append(True)

    monkeypatch.setattr(sklearn.ensemble, "RandomForestRegressor", NotAnEstimator)
    with pytest.raises(TypeError, match="estimator class"):
        resolver.resolve({"sktime_class": "sklearn.ensemble.RandomForestRegressor"})
    assert constructed == []


def test_nested_sklearn_estimator_still_resolves():
    from sklearn.ensemble import RandomForestRegressor
    from sktime.forecasting.compose import RecursiveTabularRegressionForecaster

    model = resolver.resolve(
        {
            "sktime_class": "sktime.forecasting.compose.RecursiveTabularRegressionForecaster",
            "params": {
                "estimator": {
                    "_target_": "sklearn.ensemble.RandomForestRegressor",
                    "params": {"n_estimators": 3},
                },
                "window_length": 2,
            },
        }
    )
    assert isinstance(model, RecursiveTabularRegressionForecaster)
    assert isinstance(model.estimator, RandomForestRegressor)
    assert model.estimator.n_estimators == 3


def test_nested_pyod_detector_still_resolves():
    from pyod.models.iforest import IForest

    model = resolver._build(
        {"_target_": "pyod.models.iforest.IForest", "params": {"n_estimators": 3}}
    )
    assert isinstance(model, IForest)
    assert model.n_estimators == 3
