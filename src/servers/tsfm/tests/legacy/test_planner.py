"""Planner tests — the parameter reasoning that shows tools are complex."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tsfm.bootstrap import fresh_store
from tsfm.legacy import planner


def test_forecast_plan_lookback_drives_model_context():
    s = fresh_store()
    p = planner.plan_forecasting(s, "chiller_6", question="forecast the next 48 hours", domain="energy")
    par = p["parameters"]
    assert par["forecast_horizon"] == 48                      # parsed from question
    assert par["context_length"] >= par["lookback"]           # interdependency enforced
    assert par["model_id"] is not None
    assert p["complexity"]["n_decisions"] >= 5
    # every decision carries rationale + risk (reasoning, not defaults)
    assert all(r["rationale"] and r["risk_if_wrong"] for r in p["reasoning"])


def test_anomaly_pipeline_depends_on_data_shape():
    s = fresh_store()
    multi = planner.plan_anomaly(s, "chiller_6")               # 3 channels
    two = planner.plan_anomaly(s, "motor_01")                  # 2 channels
    assert multi["parameters"]["model_id"] == "akits_relationshipad"
    assert two["parameters"]["model_id"] != "akits_relationshipad"


def test_thresholding_reasons_about_stationarity():
    s = fresh_store()
    a = planner.plan_anomaly(s, "chiller_6")
    # injected trend → non-stationary → dynamic thresholding + train_test mode
    assert a["stationarity"]["non_stationary"]
    assert a["parameters"]["thresholding"] == "dynamic"
    assert a["parameters"]["mode"] == "train_test"
