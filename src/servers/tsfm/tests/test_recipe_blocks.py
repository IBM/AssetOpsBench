"""Recipe-block params (finetune / anomaly): schema + validation + audit in run_recipe."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from ..reasoning import param_space as PS
from ..engine import composition as C


def test_block_schema_exposes_hints():
    sch = PS.block_schema("finetune")
    assert "lr" in sch["params"] and "epochs" in sch["params"]
    assert "ad_model_type" in PS.block_schema("anomaly")["params"]


def test_validate_finetune_good_and_bad():
    ok = PS.validate_block("finetune", {"lr": 0.001, "epochs": 4, "backbone_frozen": True})
    assert ok["ok"] and ok["param_audit"]["lr"]["in_range"]
    bad = PS.validate_block("finetune", {"epochs": 9999, "scheduler": "magic", "bogus": 1})
    assert not bad["ok"]
    j = " ".join(bad["issues"])
    assert "out of range" in j and "not in choices" in j and "unknown" in j


def test_validate_anomaly_good_and_bad():
    ok = PS.validate_block("anomaly", {"false_alarm": 0.05, "ad_model_type": "timeseries_conformal_adaptive"})
    assert ok["ok"]
    bad = PS.validate_block("anomaly", {"false_alarm": 0.9, "threshold_function": "nope"})
    assert not bad["ok"]


def test_run_recipe_records_block_audit():
    y = list(np.sin(np.arange(80) / 4.0) + 0.01 * np.arange(80))
    recipe = {"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                            "params": {"strategy": "last"}},
              "fh": [1, 2, 3],
              "finetune": {"lr": 0.001, "epochs": 4},
              "anomaly": {"false_alarm": 0.05}}
    r = C.run_recipe(None, y, recipe)
    assert r["block_audit"]["finetune"]["ok"] and r["block_audit"]["anomaly"]["ok"]


def test_discover_components_exposes_recipe_blocks():
    out = C.discover_components(None, task="tsfm_forecasting")
    assert "finetune" in out["recipe_blocks"] and "anomaly" in out["recipe_blocks"]
    assert "lr" in out["recipe_blocks"]["finetune"]
