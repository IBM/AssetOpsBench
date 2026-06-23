"""Standardized task registry (the 8 TS-AI tasks + their contracts)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..core import tasks as ts


def test_eight_tasks_standardized():
    ids = set(ts.TASKS)
    assert {"tsfm_forecasting", "tsfm_regression", "tsfm_classification", "tsfm_anomaly_detection",
            "tsfm_imputation", "tsfm_evaluation", "tsfm_similarity_search", "tsfm_clustering"} <= ids
    # supervised vs unsupervised split is explicit
    assert ts.get_task("tsfm_forecasting").supervised
    assert not ts.get_task("tsfm_anomaly_detection").supervised
    # output verbs differ; protocols are leakage-safe (no plain shuffled cv)
    assert ts.get_task("tsfm_clustering").output_verb == "assign"
    assert ts.get_task("tsfm_classification").leakage_split == "stratified_blocked"


def test_request_validation():
    assert not ts.validate_request("tsfm_forecasting", {"series": [1, 2]})["ok"]      # no horizon
    assert ts.validate_request("tsfm_forecasting", {"series": [1, 2], "horizon": 8})["ok"]
    assert ts.validate_request("tsfm_imputation", {"series": [1], "mask": [0]})["requires_inverse_transforms"]
