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


def test_surface_is_the_model_catalog():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"list_models", "model_template", "register_model", "register_finetuned",
                     "update_model", "deprecate_model", "new_model_version", "resolve_model"}


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
