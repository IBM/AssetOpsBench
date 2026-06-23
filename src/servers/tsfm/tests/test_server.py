"""Smoke test for the MCP tool surface — main.py builds and exposes both groups."""

import asyncio, warnings
warnings.filterwarnings("ignore")

import pytest

pytest.importorskip("mcp")          # skip cleanly if mcp isn't installed
from ..main import mcp

SURFACE = {
    "list_tasks", "discover_components", "describe_candidates", "find_models", "find_features",
    "get_component", "profile_series", "select_features", "characterize_series",
    "run_recipe", "run_tabular_recipe",
    "run_plan", "evaluate", "data_quality", "register_model", "register_feature",
    "get_result", "list_results", "get_run", "list_runs",
}


def _names():
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_surface_present():
    names = _names()
    missing = SURFACE - names
    assert not missing, f"missing tools: {missing}"


def test_no_legacy_compat_tools():
    names = _names()
    gone = {"run_tsfm_forecasting", "run_tsfm_finetuning", "run_tsad", "run_integrated_tsad",
            "get_ai_tasks", "get_tsfm_models"}
    assert not (gone & names), f"legacy tools should be removed: {gone & names}"


def test_core_recipe_tools_present():
    names = _names()
    for t in ["run_recipe", "run_tabular_recipe", "run_plan", "evaluate", "discover_components"]:
        assert t in names
