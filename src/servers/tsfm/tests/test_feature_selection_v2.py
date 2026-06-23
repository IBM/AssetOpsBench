"""FLOps multi-config selection: |corr| + F-test + mutual-info + model-importance, mean-rank."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from ..reasoning import feature_selection as F


def _signal(n=400):
    t = np.arange(n)
    return np.sin(t / 6.0) + 0.3 * np.sin(t / 23.0) + 0.02 * t + 0.05 * np.random.RandomState(0).randn(n)


def test_multiconfig_runs_all_scorers():
    r = F.select_features(_signal(), reference_feature="kurtosis")  # weak reference
    assert set(r["scorers"]) == {"corr", "f_test", "mutual_info", "model"}
    assert set(r["per_scorer"]) == {"corr", "f_test", "mutual_info", "model"}
    # aggregate scores are normalized in [0,1]; ranking is ordered best→worst
    assert max(r["scores"].values()) <= 1.0 + 1e-9
    assert r["selected"] and r["ranking"][0][1] >= r["ranking"][-1][1]


def test_reference_excluded_and_selection_nonempty():
    r = F.select_features(_signal(), reference_feature="kurtosis", cd_margin=0.05)
    assert "kurtosis" not in r["selected"] and len(r["selected"]) >= 1


def test_fast_path_corr_only_backward_compatible():
    r = F.select_features(_signal(), scorers=["corr"])
    assert r["scorers"] == ["corr"] and r["selected"]


def test_aggregation_is_robust_vs_single_scorer():
    """A feature top-ranked by the aggregate should rank well under >1 scorer, not just one."""
    r = F.select_features(_signal(), reference_feature="mean")
    top = r["ranking"][0][0]
    k = max(5, len(r["scores"]) // 5)               # top-quintile (scales with library size)
    good = sum(1 for s in r["scorers"]
               if top in sorted(r["per_scorer"][s], key=r["per_scorer"][s].get, reverse=True)[:k])
    assert good >= 2  # the aggregate top-ranked feature ranks well under >=2 scorers
