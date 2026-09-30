"""Grounding and duplicate checks at the scenario validation boundary."""

from scenarios.constraints import (
    validate_negative_scenario,
    validate_negative_scenario_batch,
    validate_scenario,
)
from scenarios.models import AssetInstance, AssetProfile


TOOLS = {"iot": ("history",)}


def test_open_form_requires_grounding_while_closed_form_allows_inline_data():
    row = {
        "text": "Explain an oil temperature of 110 C compared with a 90 C operating limit.",
        "category": "Data Query", "characteristic_form": "Use iot.history for comparison.",
    }
    profile = AssetProfile(
        asset_name="Transformer", description="Transformer with live sensor coverage.",
        asset_instances=[AssetInstance(site_name="MAIN", asset_id="Transformer 1", has_iot=True)],
    )
    assert validate_scenario(
        "iot", row, profile=profile, generation_mode="closed_form", tool_names_by_focus=TOOLS,
    ) == []
    errors = validate_scenario(
        "iot", row, profile=profile, generation_mode="open_form", tool_names_by_focus=TOOLS,
    )
    assert any("must use grounded" in error for error in errors)

    row["text"] = "Explain the recent oil temperature trend for Transformer 1 at MAIN."
    assert validate_scenario(
        "iot", row, profile=profile, generation_mode="open_form", tool_names_by_focus=TOOLS,
    ) == []
    errors = validate_scenario(
        "iot", row, generation_mode="open_form", tool_names_by_focus=TOOLS,
    )
    assert errors == ["open-form validation requires an Asset Profile"]


def test_negative_answerability_is_not_decided_by_keywords():
    row = {
        "text": "Give one oil temperature reading that is both strictly above and strictly below 100 C.",
        "category": "Data Query",
        "characteristic_form": "Cannot answer consistently because the two requirements contradict each other.",
    }
    profile = AssetProfile(
        asset_name="Transformer", description="Transformer with live sensor coverage.",
        asset_instances=[AssetInstance(site_name="MAIN", asset_id="Transformer 1", has_iot=True)],
    )
    assert validate_negative_scenario(
        "iot", row, profile=profile, generation_mode="open_form", tool_names_by_focus=TOOLS,
    ) == []


def test_negative_duplicates_are_rejected_against_prior_rows_and_within_batch():
    def negative(text):
        return {
            "text": text, "category": "Data Query",
            "characteristic_form": "Cannot answer because the external weather data is unavailable.",
        }

    prior = negative("Compare Transformer 1 oil temperature with tomorrow's weather at MAIN.")
    exact = negative(prior["text"].upper().replace(".", "!"))
    near = negative(prior["text"].replace("tomorrow's", "today's"))
    distinct = negative("Estimate how electricity market prices affect the next maintenance budget.")
    repeated = negative(distinct["text"])
    valid, failures = validate_negative_scenario_batch(
        "iot", [exact, near, distinct, repeated],
        accepted_scenarios=[prior], tool_names_by_focus=TOOLS,
    )
    assert valid == [distinct]
    assert [failure.scenario for failure in failures] == [exact, near, repeated]
    assert all(
        any("duplicate or near-duplicate" in reason for reason in failure.reasons)
        for failure in failures
    )
