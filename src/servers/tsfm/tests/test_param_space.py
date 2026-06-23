"""Per-model parameter schema + reasoning hints + validation."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..reasoning import param_space as PS

NAIVE = {"model_id": "naive", "sktime_class": "sktime.forecasting.naive.NaiveForecaster"}


def test_introspects_real_sktime_params():
    sch = PS.param_schema(NAIVE)
    assert set(sch["params"]) >= {"strategy", "sp", "window_length"}
    # reasoning hints are attached for known params
    assert "suggest" in sch["params"]["sp"]["hint"]
    assert sch["params"]["strategy"]["hint"]["choices"] == ["last", "mean", "drift"]


def test_validate_good_params():
    v = PS.validate_params(NAIVE, {"strategy": "drift", "sp": 24})
    assert v["ok"] and v["param_audit"]["sp"]["in_range"]


def test_validate_rejects_bad_params():
    v = PS.validate_params(NAIVE, {"strategy": "wizard", "sp": 99999, "bogus": 1})
    assert not v["ok"]
    joined = " ".join(v["issues"])
    assert "not in choices" in joined and "out of range" in joined and "unknown param" in joined


def test_works_across_scitypes():
    for cls in ["sktime.detection.lof.SubLOF",
                "sktime.clustering.k_means.TimeSeriesKMeans"]:
        sch = PS.param_schema({"model_id": "x", "sktime_class": cls})
        assert sch["params"]  # non-empty introspected schema
