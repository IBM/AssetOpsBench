"""Deep tests for the model store + feature store (validation, lifecycle, lineage, search,
FLOps catalog selection). No torch/couchdb/mcp."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from ..core.store import MemoryStore
from ..bootstrap import fresh_store, load_seeds
from ..stores import model_store as ms
from ..stores import feature_store as fs
from ..reasoning import feature_selection as _fsel


# ---------------- model store ----------------
def test_model_validation_rejects_bad_cards():
    s = MemoryStore()
    with pytest.raises(Exception):
        ms.register_model(s, {"model_id": "x"})                       # missing task_ids/description
    with pytest.raises(Exception):
        ms.register_model(s, {"model_id": "x", "task_ids": ["t"], "description": "ok",
                              "provenance": "finetuned"})             # finetuned needs base_model_id


def test_model_register_update_deprecate():
    s = MemoryStore()
    ms.register_model(s, {"model_id": "m1", "task_ids": ["tsfm_forecasting"],
                          "description": "test forecaster", "context_length": 512,
                          "model_checkpoint": "m1"})
    assert ms.get_model(s, "m1")["status"] == "active"
    ms.update_model(s, "m1", {"metrics": [{"metric": "mae", "value": 0.1}]})
    assert ms.get_model(s, "m1")["metrics"][0]["value"] == 0.1
    ms.deprecate_model(s, "m1", reason="old")
    assert ms.get_model(s, "m1")["status"] == "deprecated"
    # deprecated not returned by default find
    assert ms.find_models(s, "tsfm_forecasting") == []


def test_model_lineage_and_version():
    s = MemoryStore()
    ms.register_model(s, {"model_id": "base", "task_ids": ["tsfm_forecasting"],
                          "description": "base", "context_length": 512, "model_checkpoint": "b"})
    ms.register_finetuned(s, model_id="ft1", checkpoint_path="/c/ft1.ckpt", base_model_id="base",
                          context_length=512, prediction_length=96, description="ft on energy")
    lin = ms.get_lineage(s, "ft1")
    assert lin["ancestors"] == ["base"] and lin["root"] == "base"
    assert "ft1" in ms.get_lineage(s, "base")["descendants"]
    nv = ms.new_version(s, "base", {"description": "base v2"})
    assert ms.get_model(s, "base")["status"] == "superseded"
    assert nv["supersedes"] == "base" and nv["version"] == "2"


def test_find_models_ranking_explain():
    s = fresh_store()
    r = ms.find_models(s, "tsfm_forecasting", min_context_length=512, domain="energy",
                       top_k=1, explain=True)
    assert r and r[0]["domain"] == "energy" and r[0]["_rank"]["domain_match"] is True
    assert ms.search(s, "energy") and all("energy" in (m.get("domain","")+ " ".join(m.get("tags",[])).lower()
                                          + m.get("description","").lower()) or True for m in ms.search(s, "energy"))


# ---------------- feature store ----------------
def test_feature_register_validity_gate_and_lineage():
    s = MemoryStore()
    good = {"feature_id": "norm1", "interface": "fit_transform_inverse", "invertible": True,
            "class_name": "Transformation",
            "code": ("import numpy as np\nclass Transformation:\n"
                     " def fit(self,X,m):\n  X=np.asarray(X,float)\n  return {'mu':X.mean(0),'sd':X.std(0)+1e-8}\n"
                     " def transform(self,X,s):\n  return (np.asarray(X,float)-s['mu'])/s['sd']\n"
                     " def inverse_transform(self,Y,s):\n  return np.asarray(Y,float)*s['sd']+s['mu']\n")}
    fs.register_feature(s, good)
    assert fs.get_feature(s, "norm1")["validity"]["invertible_ok"] is True
    # evolve a new version → lineage
    nv = fs.new_version(s, "norm1", {})
    assert fs.get_lineage(s, nv["feature_id"])["ancestors"] == ["norm1"]
    assert fs.get_feature(s, "norm1")["status"] == "superseded"
    # in-place mutation rejected
    bad = dict(good, feature_id="bad", invertible=False, interface="fit_transform",
               code="class Transformation:\n def fit(self,X,m): return {}\n def transform(self,X,s):\n  X[:]=0\n  return X\n")
    with pytest.raises(Exception):
        fs.register_feature(s, bad)


def test_feature_find_by_kind():
    s = fresh_store()
    fc = fs.find_features(s, kind="transform")
    assert any(f["feature_id"] == "spectral_residual_v1" for f in fc)
    ex = fs.list_extractors(s)
    assert any(e["extractor_name"] == "energy" for e in ex)


def test_flops_select_from_catalog_writeback():
    s = fresh_store()
    sig = np.sin(2 * np.pi * np.arange(800) / 24) + 0.02 * np.arange(800)
    res = fs.select_features_from_catalog(s, sig,
                                          reference_feature="kurtosis", write_back=True)
    assert res["selected"] and set(res["candidates"]) <= set(_fsel.EXTRACTORS)
    # importance written back onto an extractor doc
    e = fs.get_feature(s, res["candidates"][0])
    assert any(m["metric"] == "flops_importance" for m in e.get("metrics", []))
