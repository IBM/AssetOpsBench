"""Regression coverage for canonical tool names that overlap everyday language."""

import pytest

from scenarios.constraints import validate_negative_scenario, validate_scenario
from scenarios.constraints.tool_references import find_tool_references


TOOLS = {
    "iot": ("sites", "assets", "history", "asset_ids", "measured_sensors"),
    "fmsr": ("get_failure_modes",),
    "wo": ("list_workorders", "generate_work_order"),
}


def scenario(text, characteristic_form="Use iot.history to retrieve past readings."):
    return {"text": text, "category": "Data Query", "characteristic_form": characteristic_form}


@pytest.mark.parametrize("text", [
    "Compare the maintenance history across assets at both sites.",
    "History shows a pressure rise; compare the sites before recommending repairs.",
    "Compare history (from last week) with today's readings.",
    "Inspect `oil_temperature` and explain the readings in history_buffer.",
    "Compare Temperature(C) measurements across the assets.",
])
def test_operator_prose_is_not_mistaken_for_tool_references(text):
    assert validate_scenario("iot", scenario(text), tool_names_by_focus=TOOLS) == []


@pytest.mark.parametrize("reference", [
    "iot.history", "iot.history()", "history()", "`history`", "`iot.history`",
    "the history tool", "the tool named history", "get_failure_modes",
    "get_sites", "get_assets", "get_sensors", "get_history",
    "iot.get_history", "iot.invented_tool()",
])
def test_explicit_references_are_rejected_in_operator_text(reference):
    errors = validate_scenario(
        "iot", scenario(f"Use {reference} to compare the readings."), tool_names_by_focus=TOOLS
    )
    assert any("text must not contain explicit" in error for error in errors)


@pytest.mark.parametrize("reference", [
    "iot.history", "iot.history()", "history()", "`history`", "`iot.history`",
    "the history tool", "the tool named history", "IOT.HISTORY",
])
def test_explicit_current_references_satisfy_primary_focus(reference):
    assert validate_scenario(
        "iot", scenario("Compare last week's readings.", f"Use {reference}."),
        tool_names_by_focus=TOOLS,
    ) == []


def test_bare_common_word_does_not_establish_primary_focus():
    errors = validate_scenario(
        "iot", scenario("Compare past readings.", "Review maintenance history."),
        tool_names_by_focus=TOOLS,
    )
    assert any("at least one concrete iot tool" in error for error in errors)


def test_bare_distinctive_current_identifier_remains_supported():
    assert validate_scenario(
        "iot", scenario("Find the installed units.", "Use asset_ids."),
        tool_names_by_focus=TOOLS,
    ) == []


def test_prose_does_not_supply_a_second_multiagent_focus():
    errors = validate_scenario(
        "multiagent",
        scenario("Assess the fault.", "Use fmsr.get_failure_modes and discuss maintenance history."),
        tool_names_by_focus=TOOLS,
    )
    assert any("at least two distinct focuses" in error for error in errors)


def test_explicit_tools_from_two_namespaces_satisfy_multiagent_check():
    assert validate_scenario(
        "multiagent",
        scenario("Assess the fault.", "Use iot.history and then fmsr.get_failure_modes."),
        tool_names_by_focus=TOOLS,
    ) == []


@pytest.mark.parametrize("reference", ["wo.history", "iot.invented_tool", "get_history"])
def test_invalid_references_do_not_pass_alongside_a_valid_tool(reference):
    errors = validate_scenario(
        "iot", scenario("Compare readings.", f"Use iot.history and {reference}."),
        tool_names_by_focus=TOOLS,
    )
    assert any(f"unknown MCP tool reference '{reference}'" in error for error in errors)


def test_unqualified_collision_cannot_count_as_multiple_namespaces():
    tools = {"iot": ("history",), "wo": ("history",)}
    errors = validate_scenario(
        "multiagent", scenario("Compare readings.", "Use history()."),
        tool_names_by_focus=tools,
    )
    assert any("ambiguous MCP tool reference" in error for error in errors)
    assert any("at least two distinct focuses" in error for error in errors)
    assert validate_scenario(
        "multiagent", scenario("Compare readings.", "Use iot.history and wo.history."),
        tool_names_by_focus=tools,
    ) == []


def test_reference_matching_uses_identifier_boundaries_and_current_registry():
    references = find_tool_references("pre_iot.history iot.history_extra iot.history.field", TOOLS)
    assert all(ref.canonical is None for ref in references)
    assert [ref.spelling for ref in references] == ["iot.history_extra"]
    tools = {"iot": ("new_api",)}
    references = find_tool_references("Use iot.new_api.", tools)
    assert [ref.canonical for ref in references] == ["iot.new_api"]



def test_negative_scenarios_apply_the_same_prose_and_reference_rules():
    row = scenario(
        "Correlate the maintenance history of these assets with tomorrow's weather.",
        "Cannot answer because the external weather data is unavailable.",
    )
    assert validate_negative_scenario("iot", row, tool_names_by_focus=TOOLS) == []
    row["text"] += " Use iot.history."
    errors = validate_negative_scenario("iot", row, tool_names_by_focus=TOOLS)
    assert any("text must not contain explicit" in error for error in errors)
    row["text"] = "Fetch tomorrow's weather."
    row["characteristic_form"] += " Use get_history."
    errors = validate_negative_scenario("iot", row, tool_names_by_focus=TOOLS)
    assert any("unknown MCP tool reference 'get_history'" in error for error in errors)
