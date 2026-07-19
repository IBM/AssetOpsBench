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

NAIVE = "sktime.forecasting.naive.NaiveForecaster"
# a finetune base must accept params.model_path; NaiveForecaster does not
FAKE_TTM = "servers.tsfm.tests.fake_hf_forecaster.FakeTTMForecaster"


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


def test_model_store_reads_collection_name_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_CATALOG_DBNAME", "custom_model_catalog")
    s = MemoryStore()
    card = {
        "model_id": "env_model",
        "description": "model in custom collection",
        "task_ids": ["tsfm_forecasting"],
        "sktime_class": NAIVE,
    }

    model_store.register_model(s, card)

    assert model_store.collection_name() == "custom_model_catalog"
    assert model_store.get_model(s, "env_model")["model_id"] == "env_model"
    assert s.get(model_store.COLLECTION, "model:env_model") is None
    assert s.get("custom_model_catalog", "model:env_model")["model_id"] == "env_model"


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


# ---- the fixes: a card must never silently point at the wrong model ----
def test_finetuned_rejects_an_unknown_base_instead_of_guessing():
    """Regression: register_finetuned used to do `get_model(...) or {}` and then default
    sktime_class to TinyTimeMixerForecaster - a typo'd base produced a card that loads the wrong
    architecture, and it "succeeded"."""
    s = MemoryStore()
    with pytest.raises(ValueError, match="not in the catalog"):
        model_store.register_finetuned(
            s, model_id="ft", checkpoint_path="/ckpt", base_model_id="ghost",
            context_length=96, prediction_length=28, description="finetune of a ghost")
    assert model_store.get_model(s, "ft") is None


def test_finetuned_rejects_a_base_with_no_sktime_class():
    s = MemoryStore()
    model_store.register_model(s, {"model_id": "stub", "description": "toolkit-only card",
                                   "task_ids": ["tsfm_forecasting"],
                                   "model_checkpoint": "anomalykits://iforest"})
    with pytest.raises(ValueError, match="no sktime_class"):
        model_store.register_finetuned(
            s, model_id="ft2", checkpoint_path="/ckpt", base_model_id="stub",
            context_length=96, prediction_length=28, description="finetune of a stub")


def test_finetuned_still_inherits_from_a_real_base():
    s = MemoryStore()
    model_store.register_model(s, {"model_id": "base", "description": "checkpoint-backed base",
                                   "task_ids": ["tsfm_forecasting"], "sktime_class": FAKE_TTM,
                                   "params": {"model_path": "fake-hub/ttm", "sp": 24}})
    card = model_store.register_finetuned(
        s, model_id="ft3", checkpoint_path="/ckpt/x", base_model_id="base",
        context_length=96, prediction_length=28, description="finetune of the base")
    assert card["sktime_class"] == FAKE_TTM                   # inherited, not invented
    assert card["params"]["model_path"] == "/ckpt/x"
    assert card["training_regime"] == "zero_shot"             # serving loads, does not train
    assert card["params"]["sp"] == 24            # other base params inherited too


def test_register_model_rejects_a_duplicate_id():
    s = MemoryStore()
    card = {"model_id": "dup", "description": "first version",
            "task_ids": ["tsfm_forecasting"], "sktime_class": FAKE_TTM}
    model_store.register_model(s, card)
    with pytest.raises(ValueError, match="already exists"):
        model_store.register_model(s, {**card, "description": "second version"})
    assert model_store.get_model(s, "dup")["description"] == "first version"


# ---- resolvable: computed AND persisted, with sktime_class counting as a pointer ----
def test_sktime_class_alone_makes_a_card_resolvable():
    """Regression: sktime_class was missing from the validator's refs, so the operative loader
    did not count and such a card was flagged unresolvable."""
    from ..core.schemas import ModelCard
    c = ModelCard(model_id="m", description="sktime pointer only",
                  task_ids=["tsfm_forecasting"], sktime_class=NAIVE, params={"strategy": "drift"})
    assert c.resolvable is True


def test_resolvable_survives_into_the_stored_card():
    """Regression: the flag was set with object.__setattr__ on an undeclared field, so
    model_dump() dropped it and the stored card never carried it."""
    from ..core.schemas import ModelCard
    doc = ModelCard(model_id="m", description="hf pointer",
                    task_ids=["tsfm_forecasting"], hf_repo="amazon/chronos-t5-small").to_doc()
    assert doc["resolvable"] is True


def test_a_card_with_no_pointer_is_flagged_unresolvable():
    from ..core.schemas import ModelCard
    doc = ModelCard(model_id="stub", description="catalog-only stub",
                    task_ids=["tsfm_forecasting"]).to_doc()
    assert doc["resolvable"] is False
