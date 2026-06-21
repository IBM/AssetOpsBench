"""Smoke test for the MCP tool surface — it builds and exposes the expected tools."""

import os, sys, asyncio, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

mcp_lib = pytest.importorskip("mcp")          # skip cleanly if mcp isn't installed
from tsfm.server import build_server

EXPECTED = {
    "list_tasks", "discover_components", "describe_candidates", "find_models", "find_features",
    "get_component", "profile_series", "select_features", "run_recipe", "run_tabular_recipe",
    "run_plan", "evaluate", "finetune", "register_model", "register_feature",
    "get_result", "list_results", "get_run", "list_runs",
}


def _tool_names():
    mcp = build_server()
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def test_server_builds_and_registers_surface():
    names = _tool_names()
    assert len(names) >= 16
    missing = EXPECTED - names
    assert not missing, f"missing tools: {missing}"


def test_core_recipe_tools_present():
    names = _tool_names()
    for t in ["run_recipe", "run_tabular_recipe", "run_plan", "evaluate", "discover_components"]:
        assert t in names
