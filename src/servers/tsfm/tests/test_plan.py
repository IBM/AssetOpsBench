"""File-pointer data I/O + recipe DAG (HuggingGPT task-list) + candidate selection."""

import os, sys, warnings, tempfile
warnings.filterwarnings("ignore")
os.environ.setdefault("TSFM_WORKDIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from ..core.store import MemoryStore
from ..io import refs as io_refs
from ..engine import plan as P
from ..stores import model_store as ms

NF = "sktime.forecasting.naive.NaiveForecaster"
_Y = np.sin(np.arange(180) / 4.0) + 0.03 * np.arange(180)


def test_iot_data_is_a_file_pointer():
    ref = io_refs.materialize_iot(_Y, asset_id="t")
    assert ref.startswith("file://")
    s = io_refs.load_series(ref)                 # auto-drops the timestamp column
    assert len(s) == len(_Y) and float(np.asarray(s).std()) > 0


def test_resource_reference_resolution():
    out = {"s1": {"ref": "file:///tmp/x.csv"}}
    assert P._resolve_arg("@s1", out) == "file:///tmp/x.csv"
    assert P._resolve_arg("file:///lit.csv", out) == "file:///lit.csv"


def test_recipe_dag_forecast_then_evaluate():
    s = MemoryStore()
    ref = io_refs.materialize_iot(_Y, asset_id="chiller_6")
    plan = {"steps": [
        {"id": "f1", "task": "forecast", "dep": [], "args": {"data_ref": ref},
         "recipe": {"estimator": {"sktime_class": NF, "params": {"strategy": "drift"}}, "fh": [1, 2, 3, 4]}},
        {"id": "e1", "task": "evaluate", "dep": ["f1"],
         "args": {"data_ref": ref, "fh": [1, 2, 3, 4], "sp": 1, "by": "norm_mase",
                  "recipes": {"drift": {"estimator": {"sktime_class": NF, "params": {"strategy": "drift"}}}}}},
    ]}
    res = P.run_plan(s, plan, asset_id="chiller_6")
    assert res["steps"] == ["f1", "e1"]
    assert res["outputs"]["f1"]["ref"].startswith("file://")     # output is a file pointer
    assert res["outputs"]["e1"]["ref"].startswith("file://")
    assert s.find("tsfm_plans")                                  # DAG run persisted (state)


def test_describe_candidates_hugginggpt_style():
    s = MemoryStore()
    ms.register_model(s, {"model_id": "drift", "sktime_class": NF, "params": {"strategy": "drift"},
                          "task_ids": ["tsfm_forecasting"], "description": "drift baseline", "downloads": 100})
    ms.register_model(s, {"model_id": "ttm", "sktime_class": "sktime.forecasting.ttm.TinyTimeMixerForecaster",
                          "params": {}, "task_ids": ["tsfm_forecasting"], "description": "TTM foundation",
                          "downloads": 90000})
    cands = ms.describe_candidates(s, "tsfm_forecasting", top_k=5)
    assert cands[0]["model_id"] == "ttm"        # ranked by downloads (HuggingGPT)
    assert all("description" in c for c in cands)
