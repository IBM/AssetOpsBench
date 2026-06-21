"""Standardized task registry + T-Daub selection."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tsfm.core import tasks as ts
from tsfm.reasoning import selection as selector


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


def test_tdaub_finds_best_under_budget():
    rng = np.random.RandomState(1); L = 2000
    truth = {f"p{i:02d}": 0.10 + 0.012 * i for i in range(24)}     # p00 best
    def score_fn(p, n):
        return truth[p] * (1.0 + 0.8 * np.exp(-n / 300)) + rng.normal(0, 0.003)
    res = selector.tdaub_select(list(truth), L, score_fn, top_k=4)
    assert res["winner"] == "p00"                                  # correct selection
    assert res["budget_fraction"] < 1.0                           # cheaper than train-all


def test_label_free_selection_for_unsupervised():
    em = {"deepad": 0.81, "windowad": 0.55, "relationshipad": 0.78}
    r = selector.label_free_rank(list(em), lambda p: em[p])        # higher EM/AL is better
    assert r["winner"] == "deepad"


def test_select_dispatch_by_supervision():
    # supervised → tdaub; unsupervised → label-free, via one entry point
    rng = np.random.RandomState(0); L = 800
    truth = {f"q{i}": 0.1 + 0.05 * i for i in range(6)}
    sup = selector.select(True, list(truth), L=L,
                          score_fn=lambda p, n: truth[p] * (1 + np.exp(-n / 150)), top_k=2)
    assert sup["winner"] == "q0"
    uns = selector.select(False, ["a", "b"], signal_fn=lambda p: {"a": 0.9, "b": 0.3}[p])
    assert uns["winner"] == "a"
