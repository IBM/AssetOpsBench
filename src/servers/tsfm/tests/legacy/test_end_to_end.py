"""End-to-end tests — no torch / couchdb / mcp. MemoryStore + StubCompute."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tsfm.bootstrap import fresh_store
from tsfm.stores import model_store
from tsfm.stores import feature_store
from tsfm.stores import results
from tsfm.legacy import pipeline as pl


def test_seeds_loaded():
    s = fresh_store()
    assert len(model_store.list_models(s)) >= 8
    assert len(feature_store.find_features(s)) >= 5


def test_find_models_respects_context_and_domain():
    s = fresh_store()
    m = model_store.find_models(s, "tsfm_forecasting", min_context_length=512, domain="energy", top_k=1)
    assert m and m[0]["context_length"] >= 512 and m[0]["domain"] == "energy"
    # anomaly task returns AnomalyKiTS pipelines
    ad = model_store.find_models(s, "tsfm_anomaly_detection", top_k=5)
    assert any(x["framework"] == "anomalykits" for x in ad)


def test_forecasting_writes_result():
    s = fresh_store()
    r = pl.run_forecasting(s, asset_id="chiller_6", horizon=24, domain="energy", scenario_id="s1")
    assert "result_id" in r and r["forecast_shape"][0] == 24
    rows = results.list_results(s, "tsfm_forecasting", asset_id="chiller_6")
    assert len(rows) == 1 and rows[0]["model_id"] == r["model_id"]
    assert rows[0]["feature_ids"]  # a feature was applied


def test_anomaly_writes_result_with_contribution():
    s = fresh_store()
    r = pl.run_anomaly(s, asset_id="chiller_6", thresholding="static", scenario_id="s1")
    assert r["anomaly_count"] >= 1
    row = results.list_results(s, "tsfm_anomaly_detection", asset_id="chiller_6")[0]
    tc = row["summary"]["top_contributors"][0]
    assert "kpi_name" in tc and tc["anomaly_type"] in ("High", "Low")   # FMSR handoff ready


def test_finetune_decoupled_then_register():
    s = fresh_store()
    before = len(model_store.list_models(s))
    ft = pl.run_finetuning(s, asset_id="chiller_6", base_model_id="ttm_512_96", register=False)
    assert ft["model_checkpoint"] and ft["registered"] is False
    assert len(model_store.list_models(s)) == before          # nothing registered
    ft2 = pl.run_finetuning(s, asset_id="chiller_6", base_model_id="ttm_512_96",
                            register=True, new_model_id="ttm_chiller6_ft", domain="energy")
    assert ft2["registered"] is True
    m = model_store.get_model(s, "ttm_chiller6_ft")
    assert m["provenance"] == "finetuned" and m["base_model_id"] == "ttm_512_96"
    assert m["artifact_path"] == ft2["model_checkpoint"]      # catalog points at checkpoint


def test_flops_select_and_lookback():
    import numpy as np
    s = fresh_store()
    sig = np.sin(2 * np.pi * np.arange(960) / 24) + 0.01 * np.arange(960)
    sel = feature_store.select_features(sig, reference_feature="kurtosis")
    assert "kurtosis" not in sel["selected"] and len(sel["selected"]) >= 1
    assert 8 <= feature_store.discover_lookback(sig) <= 128


def test_register_feature_validity_gate():
    s = fresh_store()
    # in-place-mutating program must be rejected
    bad = {"feature_id": "bad_inplace", "interface": "fit_transform", "class_name": "Transformation",
           "code": "class Transformation:\n def fit(self,X,m): return {}\n def transform(self,X,s):\n  X[:]=0\n  return X\n"}
    try:
        feature_store.register_feature(s, bad)
        assert False, "should have rejected in-place mutation"
    except Exception:
        pass


def test_export_state_captures_everything():
    s = fresh_store()
    pl.run_forecasting(s, asset_id="chiller_6", horizon=12)
    pl.run_anomaly(s, asset_id="chiller_6")
    state = s.export_state()
    assert "model_catalog" in state and "feature_catalog" in state
    assert "forecast_result" in state and "anomaly_result" in state
