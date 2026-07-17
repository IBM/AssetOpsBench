"""Model-catalog tool surface: template -> register -> resolve -> update/deprecate/version."""

import asyncio
import json
import warnings

warnings.filterwarnings("ignore")

from ..main import mcp

NAIVE = "sktime.forecasting.naive.NaiveForecaster"


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _card(model_id, **over):
    card = {"model_id": model_id, "description": "a test forecaster",
            "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE,
            "params": {"strategy": "drift"}}
    card.update(over)
    return card


def test_model_catalog_surface_present():
    """The 15 model-catalog tools (4-18). Other groups may add to the surface alongside them."""
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"list_models", "search_models", "find_models", "describe_candidates",
            "describe_models", "count_models", "list_domains", "get_model_lineage",
            "register_model", "model_template", "register_finetuned", "update_model",
            "deprecate_model", "new_model_version", "resolve_model"} <= names


# ---- discovery / read (tools 4-11) ----
def test_search_models():
    call("register_model", {"model": _card("srch_ttm", model_family="TinyTimeMixer",
                                           tags=["zero-shot"])})
    r = call("search_models", {"text": "tinytimemixer"})
    assert any(m["model_id"] == "srch_ttm" for m in r["models"])
    assert call("search_models", {"text": "no_such_model_xyz"})["models"] == []


def test_find_models_filters_by_task_and_domain():
    call("register_model", {"model": _card("fm_energy", domain="energy", context_length=512)})
    r = call("find_models", {"task_id": "tsfm_forecasting", "domain": "energy"})
    assert all(m["domain"] == "energy" for m in r["models"])
    assert "error" in call("find_models", {"task_id": "tsfm_made_up"})
    # min_context_length excludes shorter cards
    r = call("find_models", {"task_id": "tsfm_forecasting", "min_context_length": 100000})
    assert r["models"] == []


def test_describe_candidates_shortlist():
    r = call("describe_candidates", {"task_id": "tsfm_forecasting", "top_k": 2})
    assert len(r["candidates"]) <= 2
    assert all("model_id" in c and "description" in c for c in r["candidates"])
    assert "error" in call("describe_candidates", {"task_id": ""})


def test_describe_models_by_ids():
    call("register_model", {"model": _card("dm_a", model_family="naive", domain="energy")})
    r = call("describe_models", {"model_ids": ["dm_a", "ghost"]})
    assert r["models"][0]["model_id"] == "dm_a"
    assert r["models"][0]["family"] == "naive" and r["models"][0]["domain"] == "energy"
    assert r["unknown"] == ["ghost"]
    assert "error" in call("describe_models", {"model_ids": []})


def test_count_models():
    r = call("count_models", {})
    assert r["total"] >= 1 and "tsfm_forecasting" in r["by_task"]


def test_list_domains():
    call("register_model", {"model": _card("ld_x", domain="energy")})
    r = call("list_domains", {})
    assert "energy" in r["domains"]
    assert "error" in call("list_domains", {"task_id": "tsfm_made_up"})


def test_get_model_lineage():
    call("register_model", {"model": _card("lin_base")})
    call("register_finetuned", {"model_id": "lin_ft", "checkpoint_path": "/ckpt/x",
                                "base_model_id": "lin_base", "context_length": 96,
                                "prediction_length": 28, "description": "ft of base"})
    lin = call("get_model_lineage", {"model_id": "lin_ft"})
    assert "lin_base" in lin["ancestors"]
    assert call("get_model_lineage", {"model_id": "lin_base"})["descendants"] == ["lin_ft"]
    assert "error" in call("get_model_lineage", {"model_id": ""})


def test_list_models_and_task_validation():
    assert call("list_models", {})["models"]                      # seeded catalog
    assert "error" in call("list_models", {"task_id": "tsfm_made_up"})


def test_model_template_example_registers():
    t = call("model_template", {})
    assert t["required_fields"] == ["model_id", "description", "task_ids"]
    assert call("register_model", {"model": t["example"]})["status"] == "registered"


def test_register_validates():
    assert call("register_model", {"model": {}})["error"]
    assert "error" in call("register_model", {"model": {"model_id": "x", "description": "ab",
                                                        "task_ids": ["tsfm_forecasting"]}})
    assert "error" in call("register_model", {"model": _card("x", provenance="finetuned")})


def test_register_then_resolve():
    call("register_model", {"model": _card("rt_naive")})
    r = call("resolve_model", {"model_id": "rt_naive"})
    assert r["sktime_class"].endswith("NaiveForecaster")
    assert "error" in call("resolve_model", {"model_id": "nope"})
    assert "error" in call("resolve_model", {"model_id": ""})


def test_update_and_deprecate():
    # NOTE: CardResult declares no fields (extra=allow), so the card comes back FLAT here,
    # unlike register_model which nests it under "card".
    call("register_model", {"model": _card("lc_naive")})
    assert call("update_model", {"model_id": "lc_naive",
                                 "fields": {"domain": "energy"}})["domain"] == "energy"
    assert call("deprecate_model", {"model_id": "lc_naive",
                                    "reason": "superseded"})["status"] == "deprecated"
    assert "error" in call("update_model", {"model_id": "ghost", "fields": {"domain": "x"}})


def test_new_version_links_predecessor():
    call("register_model", {"model": _card("ver_naive")})
    out = call("new_model_version", {"model_id": "ver_naive", "fields": {"context_length": 128}})
    assert out["supersedes"] == "ver_naive"


def test_register_finetuned_inherits_and_links():
    call("register_model", {"model": _card("base_ttm")})
    out = call("register_finetuned", {"model_id": "ft_ttm", "checkpoint_path": "/ckpt/ft",
                                      "base_model_id": "base_ttm", "context_length": 96,
                                      "prediction_length": 28, "description": "fine-tuned ttm"})
    card = out
    assert card["base_model_id"] == "base_ttm" and card["provenance"] == "finetuned"
    assert card["sktime_class"] == NAIVE                 # inherited from the base
    assert card["params"]["model_path"] == "/ckpt/ft"    # points at the checkpoint
