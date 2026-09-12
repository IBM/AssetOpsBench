"""The client-side adapter: the only surface an evolved patch may change.

The property that matters most is the first one tested here. An empty adapter
must leave the tool surface byte-identical, because that is the baseline arm of
every comparison. If it drifts, every reported gain is measuring the plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.stirrup_agent.adapter import (
    AdapterError,
    AdapterSpec,
    adapt_tools,
    unknown_tool_names,
)


@dataclass
class FakeTool:
    """Stands in for a Stirrup Tool; only name/description/schema are used."""

    name: str
    description: str
    schema: dict

    def input_schema(self) -> dict:
        return self.schema


def wo_tool() -> FakeTool:
    return FakeTool(
        name="wo__query_work_orders",
        description="Query work orders.",
        schema={
            "type": "object",
            "properties": {
                "siteid": {"type": "string"},
                "status": {"type": "string"},
                "include_archived": {"type": "boolean"},
                "cursor": {"type": "string"},
            },
            "required": ["siteid"],
        },
    )


def iot_tool() -> FakeTool:
    return FakeTool(name="iot__get_sensor_history", description="History.", schema={})


# -- the baseline guarantee -------------------------------------------------


def test_empty_adapter_changes_nothing() -> None:
    tools = [wo_tool(), iot_tool()]

    adapted = adapt_tools(tools, AdapterSpec.empty())

    assert [a.name for a in adapted] == [t.name for t in tools]
    assert [a.description for a in adapted] == [t.description for t in tools]
    assert adapted[0].schema == tools[0].schema
    assert adapted[0].translate({"siteid": "X"}) == {"siteid": "X"}


def test_empty_spec_reports_itself_empty() -> None:
    assert AdapterSpec.empty().is_empty
    assert not AdapterSpec.from_raw({"hidden": ["a"]}).is_empty


# -- argument modification --------------------------------------------------


def test_rename_presents_one_name_and_sends_another() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"rename": {"site": "siteid"}}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert "site" in tool.schema["properties"]
    assert "siteid" not in tool.schema["properties"]
    assert tool.schema["required"] == ["site"]
    assert tool.translate({"site": "MAIN"}) == {"siteid": "MAIN"}


def test_drop_hides_an_argument_and_never_sends_it() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"drop": ["cursor"]}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert "cursor" not in tool.schema["properties"]
    assert tool.translate({"siteid": "MAIN", "cursor": "abc"}) == {"siteid": "MAIN"}


def test_a_dropped_argument_with_a_default_is_still_sent() -> None:
    """Hiding a parameter while pinning its value is the common repair."""
    spec = AdapterSpec.from_raw(
        {
            "tools": {
                "wo__query_work_orders": {
                    "drop": ["include_archived"],
                    "defaults": {"include_archived": False},
                }
            }
        }
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert "include_archived" not in tool.schema["properties"]
    assert tool.translate({"siteid": "MAIN"}) == {
        "siteid": "MAIN",
        "include_archived": False,
    }


def test_a_default_makes_an_argument_optional() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"defaults": {"siteid": "MAIN"}}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert tool.schema["required"] == []
    assert tool.translate({}) == {"siteid": "MAIN"}


def test_the_agent_can_override_a_default() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"defaults": {"status": "APPR"}}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert tool.translate({"siteid": "M", "status": "WAPPR"})["status"] == "WAPPR"


def test_require_adds_an_argument_the_server_left_optional() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"require": ["status"]}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert tool.schema["required"] == ["siteid", "status"]


def test_strip_empty_removes_nulls_before_the_server_sees_them() -> None:
    """The repair AutoSaddler reports as its largest single gain."""
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"strip_empty": True}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert tool.translate(
        {"siteid": "MAIN", "status": None, "cursor": "", "extra": []}
    ) == {"siteid": "MAIN"}


def test_strip_empty_keeps_false_and_zero() -> None:
    """False is a value, not an absence. Stripping it would change meaning."""
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query_work_orders": {"strip_empty": True}}}
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert tool.translate({"include_archived": False, "n": 0}) == {
        "include_archived": False,
        "n": 0,
    }


# -- steering ---------------------------------------------------------------


def test_description_is_replaced_and_the_hook_is_carried() -> None:
    spec = AdapterSpec.from_raw(
        {
            "tools": {
                "wo__query_work_orders": {
                    "description": "Query work orders. Filter by site first.",
                    "hook": "Confirm the site before querying.",
                }
            }
        }
    )

    tool = adapt_tools([wo_tool()], spec)[0]

    assert tool.description.endswith("Filter by site first.")
    assert tool.hook == "Confirm the site before querying."


def test_hidden_tools_disappear_from_the_surface() -> None:
    spec = AdapterSpec.from_raw({"hidden": ["iot__get_sensor_history"]})

    adapted = adapt_tools([wo_tool(), iot_tool()], spec)

    assert [a.name for a in adapted] == ["wo__query_work_orders"]


# -- composites -------------------------------------------------------------


def test_a_composite_is_added_as_a_tool() -> None:
    spec = AdapterSpec.from_raw(
        {
            "composites": [
                {
                    "name": "wo__open_at_site",
                    "description": "Open work orders at a site.",
                    "steps": [
                        {
                            "tool": "wo__query_work_orders",
                            "arguments": {"siteid": "$input.site"},
                        }
                    ],
                }
            ]
        }
    )

    adapted = adapt_tools([wo_tool()], spec)

    assert [a.name for a in adapted][-1] == "wo__open_at_site"
    assert adapted[-1].is_composite
    assert adapted[-1].target is None


def test_a_composite_may_not_shadow_a_server_tool() -> None:
    spec = AdapterSpec.from_raw(
        {
            "composites": [
                {
                    "name": "wo__query_work_orders",
                    "description": "d",
                    "steps": [{"tool": "wo__query_work_orders"}],
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="collides"):
        adapt_tools([wo_tool()], spec)


def test_composite_step_references_resolve() -> None:
    from agent.stirrup_agent.adapter.apply import resolve_step_arguments

    resolved = resolve_step_arguments(
        {"asset": "$0.rows.0.assetnum", "site": "$input.site", "n": 5},
        inputs={"site": "MAIN"},
        results=[{"rows": [{"assetnum": "CHILLER6"}]}],
    )

    assert resolved == {"asset": "CHILLER6", "site": "MAIN", "n": 5}


def test_a_reference_to_a_step_that_has_not_run_raises() -> None:
    from agent.stirrup_agent.adapter.apply import resolve_step_arguments

    with pytest.raises(KeyError, match="has not run"):
        resolve_step_arguments({"x": "$3.y"}, inputs={}, results=[])


# -- the spec is a gate, not a suggestion -----------------------------------


def test_an_unknown_key_is_rejected() -> None:
    with pytest.raises(AdapterError, match="unknown adapter keys"):
        AdapterSpec.from_raw({"tools": {}, "sneaky": 1})


def test_an_unknown_tool_key_is_rejected() -> None:
    with pytest.raises(AdapterError, match="unknown adapter keys"):
        AdapterSpec.from_raw({"tools": {"t": {"exec": "rm -rf /"}}})


def test_renaming_two_arguments_onto_one_is_rejected() -> None:
    with pytest.raises(AdapterError, match="two agent names"):
        AdapterSpec.from_raw({"tools": {"t": {"rename": {"a": "x", "b": "x"}}}})


def test_requiring_a_dropped_argument_is_rejected() -> None:
    """The agent could never satisfy it, so the tool would be unusable."""
    with pytest.raises(AdapterError, match="could never satisfy"):
        AdapterSpec.from_raw({"tools": {"t": {"drop": ["a"], "require": ["a"]}}})


def test_a_composite_without_steps_is_rejected() -> None:
    with pytest.raises(AdapterError, match="no steps"):
        AdapterSpec.from_raw(
            {"composites": [{"name": "c", "description": "d", "steps": []}]}
        )


def test_adapting_a_tool_no_server_serves_is_caught() -> None:
    """A patch against a nonexistent tool is a silent no-op. Reject it."""
    spec = AdapterSpec.from_raw({"tools": {"wo__typo": {"strip_empty": True}}})

    assert unknown_tool_names([wo_tool()], spec) == ["wo__typo"]


def test_a_composite_over_a_real_tool_is_not_flagged_unknown() -> None:
    spec = AdapterSpec.from_raw(
        {
            "composites": [
                {
                    "name": "c",
                    "description": "d",
                    "steps": [{"tool": "wo__query_work_orders"}],
                }
            ]
        }
    )

    assert unknown_tool_names([wo_tool()], spec) == []


# -- loading ----------------------------------------------------------------


def test_no_directory_loads_the_empty_adapter(tmp_path: Path) -> None:
    assert AdapterSpec.load(None).is_empty


def test_a_missing_spec_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="no adapter.json"):
        AdapterSpec.load(tmp_path)


def test_malformed_json_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "adapter.json").write_text("{not json")

    with pytest.raises(AdapterError, match="not valid JSON"):
        AdapterSpec.load(tmp_path)


def test_a_spec_round_trips_from_disk(tmp_path: Path) -> None:
    (tmp_path / "adapter.json").write_text(
        json.dumps({"tools": {"wo__query_work_orders": {"strip_empty": True}}})
    )

    spec = AdapterSpec.load(tmp_path)

    assert spec.for_tool("wo__query_work_orders").strip_empty is True
    assert spec.for_tool("anything_else").strip_empty is False
