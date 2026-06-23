"""Zero-shot regime detection + the no-retrain fast path in run_recipe."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from ..substrate import resolver as R
from ..engine import composition as C

CHRONOS = {"model_id": "chronos", "sktime_class": "sktime.forecasting.chronos.ChronosForecaster"}
NAIVE = {"model_id": "naive", "sktime_class": "sktime.forecasting.naive.NaiveForecaster"}


def test_foundation_is_zero_shot_by_default():
    assert R.training_regime(CHRONOS) == "zero_shot"
    assert R.is_foundation(CHRONOS)


def test_finetune_params_flip_regime():
    ft = {**CHRONOS, "params": {"num_train_epochs": 3}}
    assert R.training_regime(ft) == "fine_tune"


def test_classical_is_fit_on_series():
    assert R.training_regime(NAIVE) == "fit_on_series"
    assert not R.is_foundation(NAIVE)


def test_explicit_regime_wins():
    assert R.training_regime({**NAIVE, "training_regime": "zero_shot"}) == "zero_shot"


def test_recipe_regime_zero_shot_only_if_all_members():
    mixed = {"ensemble": {"members": [CHRONOS, NAIVE], "combine": "mean"}}
    assert C._recipe_regime(mixed, None) == "fit_on_series"
    allzs = {"ensemble": {"members": [CHRONOS, {**NAIVE, "training_regime": "zero_shot"}]}}
    assert C._recipe_regime(allzs, None) == "zero_shot"


def test_zero_shot_fast_path_skips_retrain_loop():
    """Force a zero-shot regime on a cheap forecaster: fast path runs ONE holdout (folds==1),
    reports trained=False — proving no expanding-window retraining happened."""
    y = list(np.sin(np.arange(80) / 4.0) + 0.01 * np.arange(80))
    zs = {"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                        "params": {"strategy": "last"}, "training_regime": "zero_shot"},
          "fh": [1, 2, 3, 4, 5], "eval": {"metrics": ["smape"]}}
    r = C.run_recipe(None, y, zs)
    assert r["training_regime"] == "zero_shot" and r["trained"] is False
    assert isinstance(r["backtest_score"], float) and len(r["forecast_head"]) == 5


def test_fit_on_series_uses_expanding_backtest():
    y = list(np.sin(np.arange(80) / 4.0) + 0.01 * np.arange(80))
    fos = {"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                         "params": {"strategy": "last"}},
           "fh": [1, 2, 3], "eval": {"metrics": ["smape"]}}
    r = C.run_recipe(None, y, fos)
    assert r["training_regime"] == "fit_on_series" and r["trained"] is True
