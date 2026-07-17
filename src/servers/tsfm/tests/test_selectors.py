"""The selectors we build must be valid CouchDB Mango, not just MemoryStore-friendly.

Context: model_store used to emit {"task_ids": {"$elemMatch": "tsfm_forecasting"}} - a bare scalar
operand. CouchDB requires a condition object and answers 400 bad_request, so every task-filtered
tool (list_models / find_models / describe_candidates / list_domains) was broken in production
while the whole suite stayed green, because MemoryStore accepted the scalar.

These tests pin both halves of the fix: we emit spec-valid Mango, and MemoryStore no longer blesses
what CouchDB rejects.
"""

import warnings

warnings.filterwarnings("ignore")

import pytest

from ..core.store import MemoryStore, _match
from ..stores import model_store


class _SpySt(MemoryStore):
    """Records the selector handed to find(), so we can assert on the wire format."""

    def __init__(self):
        super().__init__()
        self.selectors = []

    def find(self, collection, selector=None, limit=1000):
        self.selectors.append(selector)
        return super().find(collection, selector, limit)


def test_task_filter_emits_a_condition_object_not_a_scalar():
    s = _SpySt()
    model_store.list_models(s, task_id="tsfm_forecasting")
    sel = s.selectors[-1]
    assert sel["task_ids"] == {"$elemMatch": {"$eq": "tsfm_forecasting"}}, (
        "task_ids must use a condition object; a bare scalar is a 400 on CouchDB"
    )


def test_usage_mode_filter_emits_a_condition_object():
    s = _SpySt()
    model_store.list_models(s, usage_mode="zero_shot")
    assert s.selectors[-1]["usage_modes"] == {"$elemMatch": {"$eq": "zero_shot"}}


def test_memorystore_rejects_a_scalar_elemmatch_like_couchdb():
    """The leniency that hid the bug must stay gone."""
    with pytest.raises(ValueError, match="selector object"):
        _match({"task_ids": ["tsfm_forecasting"]}, {"task_ids": {"$elemMatch": "tsfm_forecasting"}})


def test_condition_object_elemmatch_still_matches():
    doc = {"task_ids": ["tsfm_forecasting", "tsfm_imputation"]}
    assert _match(doc, {"task_ids": {"$elemMatch": {"$eq": "tsfm_forecasting"}}})
    assert not _match(doc, {"task_ids": {"$elemMatch": {"$eq": "tsfm_clustering"}}})


def test_task_filter_actually_filters():
    s = MemoryStore()
    model_store.register_model(s, {"model_id": "fc", "description": "forecaster card",
                                   "task_ids": ["tsfm_forecasting"],
                                   "sktime_class": "sktime.forecasting.naive.NaiveForecaster"})
    model_store.register_model(s, {"model_id": "cl", "description": "clusterer card",
                                   "task_ids": ["tsfm_clustering"],
                                   "sktime_class": "sktime.clustering.k_means.TimeSeriesKMeans"})
    got = [m["model_id"] for m in model_store.list_models(s, task_id="tsfm_forecasting")]
    assert got == ["fc"]
