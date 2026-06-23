"""The agent can learn the vocabulary: glossary is canonical, surfaced, and errors teach."""

import asyncio, json, warnings
warnings.filterwarnings("ignore")

from ..core import glossary as G
from ..main import mcp

KEY_TERMS = {"task", "component", "candidate", "model", "feature", "recipe",
             "transform", "ensemble", "training_regime", "param_schema", "data_ref"}


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def test_glossary_is_canonical_and_complete():
    g = G.glossary()
    assert KEY_TERMS <= set(g["terms"])
    for term, meta in g["terms"].items():
        assert meta["definition"] and meta.get("tool")     # every term defined + points to a tool
    assert len(g["workflow"]) >= 5 and g["principles"]


def test_discover_components_teaches_vocabulary():
    out = call("discover_components", {"task": "tsfm_forecasting"})["components"]
    assert KEY_TERMS <= set(out["glossary"])
    assert "workflow" in out and "principles" in out
    assert "recipe_blocks" in out and "training_regimes" in out


def test_list_tasks_have_plain_descriptions():
    tasks = call("list_tasks", {})["tasks"]
    assert len(tasks) == 8
    assert all(t.get("description") for t in tasks)         # every task is explained in plain words


def test_unknown_task_error_lists_valid_tasks():
    for tool in ("discover_components", "find_models", "describe_candidates"):
        arg = "task" if tool == "discover_components" else "task_id"
        r = call(tool, {arg: "tsfm_made_up"})
        assert "error" in r and "Valid tasks" in r["error"] and "tsfm_forecasting" in r["error"]


def test_short_glossary_in_server_instructions():
    instr = mcp.instructions or ""
    assert "VOCABULARY" in instr and "WORKFLOW" in instr
    assert "recipe" in instr and "component" in instr
