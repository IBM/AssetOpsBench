"""Catalog lifecycle tools (pull + update + version + retire) via the real MCP boundary."""

import asyncio, json, warnings
warnings.filterwarnings("ignore")

from ..main import mcp


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


# ---- model store ----
def test_list_models():
    assert len(call("list_models", {})["models"]) >= 30                  # full active catalog
    fc = call("list_models", {"task_id": "tsfm_forecasting"})["models"]
    assert fc and all("tsfm_forecasting" in m["task_ids"] for m in fc)
    assert "error" in call("list_models", {"task_id": "tsfm_made_up"})    # validated


def test_search_models():
    r = call("search_models", {"text": "chronos"})
    assert r["models"] and all("chronos" in (m.get("model_id", "") + " " + m.get("model_family", "")).lower()
                               for m in r["models"])
    assert "error" not in call("search_models", {})          # empty text → all active


def test_model_lineage_update_deprecate():
    assert "error" in call("get_model_lineage", {"model_id": ""})
    lin = call("get_model_lineage", {"model_id": "ttm_96_28"})
    assert "model_id" in lin or "ancestors" in lin or "descendants" in lin
    upd = call("update_model", {"model_id": "ttm_96_28", "fields": {"domain": "energy"}})
    assert upd.get("domain") == "energy"
    dep = call("deprecate_model", {"model_id": "autoarima", "reason": "test"})
    assert dep.get("status") == "deprecated"
    assert "error" in call("update_model", {"model_id": "x", "fields": {}})


def test_new_model_version_and_finetuned():
    nv = call("new_model_version", {"model_id": "naive_persistence",
                                    "fields": {"description": "naive baseline, revised"}})
    assert nv.get("supersedes") == "naive_persistence" and nv.get("model_id", "").startswith("naive_persistence")
    ft = call("register_finetuned", {
        "model_id": "ttm_chiller_ft", "checkpoint_path": "/art/ttm_chiller", "base_model_id": "ttm_96_28",
        "context_length": 96, "prediction_length": 28, "description": "TTM fine-tuned on chiller data"})
    assert ft.get("provenance") == "finetuned" and ft.get("base_model_id") == "ttm_96_28"
    assert "error" in call("register_finetuned", {"model_id": "", "checkpoint_path": "p",
                                                  "base_model_id": "b", "context_length": 1,
                                                  "prediction_length": 1, "description": "x"})


# ---- feature store ----
def test_search_and_list_extractors():
    assert call("search_features", {"text": "normalization"})["features"]
    assert len(call("list_extractors", {})["features"]) >= 100


def test_feature_lineage_update_version_deprecate():
    assert "error" in call("get_feature_lineage", {"feature_id": ""})
    lin = call("get_feature_lineage", {"feature_id": "trend_slope_v1"})
    assert isinstance(lin, dict)
    upd = call("update_feature", {"feature_id": "trend_slope_v1", "fields": {"tags": ["degradation"]}})
    assert "degradation" in (upd.get("tags") or [])
    nv = call("new_feature_version", {"feature_id": "trend_slope_v1",
                                      "fields": {"name": "trend slope v2"}})
    assert nv.get("parent_feature_id") == "trend_slope_v1"
    dep = call("deprecate_feature", {"feature_id": "channel_select_v1"})
    assert dep.get("status") == "deprecated"
