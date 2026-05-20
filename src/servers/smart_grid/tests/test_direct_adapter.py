"""Direct-adapter contract tests.

These tests don't touch CSVs (data files may not exist on the AOB-side dev
box). They only exercise the adapter's tool-registration, parameter
introspection, and JSON-schema-ish output, all of which are pure-Python
operations on the imported server modules.
"""

from __future__ import annotations

import pytest

from servers.smart_grid.direct_adapter import (
    ToolSpec,
    get_tool,
    get_tools,
    list_tool_specs_for_llm,
)


def test_get_tools_returns_19_tools():
    tools = get_tools()
    assert len(tools) == 19


def test_get_tools_returns_toolspec_instances():
    for t in get_tools():
        assert isinstance(t, ToolSpec)
        assert t.name and "." in t.name
        assert t.domain in {"iot", "fmsr", "tsfm", "wo"}
        assert t.bare_name
        assert callable(t.fn)


def test_tools_split_by_domain():
    by_domain = {}
    for t in get_tools():
        by_domain.setdefault(t.domain, []).append(t.name)
    assert sorted(by_domain) == ["fmsr", "iot", "tsfm", "wo"]
    assert len(by_domain["iot"]) == 4
    assert len(by_domain["fmsr"]) == 5
    assert len(by_domain["tsfm"]) == 4
    assert len(by_domain["wo"]) == 6


def test_get_tool_by_name():
    t = get_tool("iot.list_assets")
    assert t.domain == "iot"
    assert t.bare_name == "list_assets"


def test_get_tool_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_tool("nonexistent.tool")


def test_parameters_extract_signature_correctly():
    t = get_tool("iot.get_sensor_readings")
    params = t.parameters()
    assert "transformer_id" in params
    assert params["transformer_id"].get("required") is True
    assert "limit" in params
    assert params["limit"].get("default") == 100


def test_list_tool_specs_for_llm_shape():
    specs = list_tool_specs_for_llm()
    assert len(specs) == 19
    for s in specs:
        assert set(s.keys()) >= {"name", "description", "parameters"}


def test_doc_first_paragraph_extracted():
    t = get_tool("iot.list_assets")
    assert t.doc.startswith("List all Smart Grid")
    # First paragraph only — should NOT contain "Args:" subsection.
    assert "Args:" not in t.doc
