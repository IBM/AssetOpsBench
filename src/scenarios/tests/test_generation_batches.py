"""Cross-batch duplicate rejection and the negative-generation text contract."""

import json
from types import SimpleNamespace

from scenarios.generator.agent import ScenarioGeneratorAgent
from scenarios.models import AssetInstance, AssetProfile
from scenarios.constraints.policies import format_accepted_scenarios_for_prompt


def test_later_batches_can_see_recent_and_early_accepted_texts():
    rows = [{"text": f"accepted scenario {i}"} for i in range(45)]
    visible = json.loads(format_accepted_scenarios_for_prompt(rows))
    assert visible == [row["text"] for row in rows]
    assert json.loads(format_accepted_scenarios_for_prompt(rows, limit=2)) == visible[:2]


def test_duplicates_from_earlier_batches_are_retried_until_count_is_met(monkeypatch):
    agent = ScenarioGeneratorAgent(backend="codex", batch_size=2)
    prior, first, second, third = [
        {"text": text, "category": "Data Query", "characteristic_form": "Use iot.history."}
        for text in [
            "Summarize missing samples for Transformer 1 at MAIN.",
            "When did oil temperature peak on Transformer 1?",
            "Compare the load before and after yesterday's shutdown at MAIN.",
            "Find sustained pressure drops over the past week at MAIN.",
        ]
    ]
    near_duplicate = {**first, "text": first["text"].replace("peak", "peak today")}
    responses = iter([[first, second], [prior, near_duplicate], [third]])
    monkeypatch.setattr(agent, "_generate_attempt_batch", lambda **kwargs: next(responses))
    # The model fails to repair its duplicates; deterministic validation must still reject them.
    monkeypatch.setattr(agent, "validate_and_repair", lambda scenarios, **kwargs: scenarios)
    profile = AssetProfile(
        asset_name="Transformer", description="test", generation_mode="open_form",
        asset_instances=[AssetInstance(site_name="MAIN", asset_id="Transformer 1", has_iot=True)],
    )
    rows = agent.generate_validated_scenarios(
        "iot", 3, profile, {}, accepted_scenarios=[prior],
        validation_tool_names={"iot": ("history",)},
    )
    assert rows == [first, second, third]


def test_negative_generation_retries_duplicates_from_positives_and_other_focuses(monkeypatch):
    agent = ScenarioGeneratorAgent(backend="codex")
    prior, first, second = [
        {"text": text, "category": "Data Query", "characteristic_form": "Cannot answer with the available data."}
        for text in [
            "Compare the oil temperature trend for Transformer 1 at MAIN.",
            "Compare Transformer 1 oil temperature with local weather forecasts.",
            "Compare vibration levels with equipment procurement invoices.",
        ]
    ]
    near = {**first, "text": first["text"].replace("local weather", "hourly weather")}
    accepted_positive = {**prior, "characteristic_form": "Use iot.history to compare readings."}
    responses = iter([[prior], [first], [near], [second]])
    agent.llm = SimpleNamespace(generate=lambda prompt: json.dumps(next(responses)))
    # A repair pass that returns duplicates unchanged must not let them through.
    monkeypatch.setattr(agent, "validate_and_repair", lambda scenarios, **kwargs: scenarios)
    result = agent.generate_negative_scenarios(
        2, AssetProfile(asset_name="Transformer", description="test"), {},
        accepted_scenarios=[accepted_positive],
        validation_tool_names={"iot": ("history",), "fmsr": ("get_failure_modes",)},
    )
    assert [row.text for row in result] == [first["text"], second["text"]]
    assert [row.type for row in result] == ["iot", "fmsr"]
