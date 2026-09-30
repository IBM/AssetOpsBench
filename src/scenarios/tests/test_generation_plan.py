"""Explicit quotas must control generation and completion end to end."""

from collections import Counter
import json
import sys

import pytest

from scenarios.config import GeneratorConfig
from scenarios.generator import agent as agent_module, cli, service
from scenarios.models import AssetProfile, Scenario, ScenarioGenerationResult


def row(text, characteristic="Use iot.history."):
    return {"text": text, "category": "Data Query", "characteristic_form": characteristic}


@pytest.fixture
def scripted_generation(monkeypatch, tmp_path):
    def install(batches, allocation=None):
        replies = [batch for batch in batches for _ in range(2)]  # Generation, then repair.
        if allocation is not None:
            replies.insert(0, {"allocation": allocation})
        replies = iter(replies)
        prompts = []

        class Model:
            def generate(self, prompt, **kwargs):
                prompts.append(prompt)
                return json.dumps(next(replies))

        async def descriptions(self):
            return {
                "iot": "  - history(): Retrieve historical readings.",
                "fmsr": "  - get_failure_modes(): List known failure modes.",
            }

        async def profile(self, **kwargs):
            return AssetProfile(asset_name="Transformer", description="Transformer", generation_mode="closed_form")

        async def unexpected_allocation(self, *args, **kwargs):
            pytest.fail("An explicit plan must never invoke the model's budget allocator")

        monkeypatch.setattr(agent_module, "create_backend", lambda *args: Model())
        monkeypatch.setattr(agent_module.Executor, "get_server_descriptions", descriptions)
        monkeypatch.setattr(agent_module.ScenarioGeneratorAgent, "build_asset_profile", profile)
        if allocation is None:
            monkeypatch.setattr(agent_module.ScenarioGeneratorAgent, "allocate_budget", unexpected_allocation)
        monkeypatch.setattr(agent_module, "fetch_hf_fewshot", lambda **kwargs: [])
        monkeypatch.setattr(service, "default_scenario_output_path", lambda asset: tmp_path / "runs" / "run" / "scenarios.json")
        return prompts

    return install


def test_json_plan_controls_cli_generation_and_saved_results(monkeypatch, tmp_path, scripted_generation):
    # Input order deliberately differs from execution order; multiagent still follows single focuses.
    plan = {"multiagent": {"positive": 1, "negative": 1}, "fmsr": {"positive": 1}, "iot": {"positive": 2, "negative": 1}}
    iot = [row("Compare oil temperature readings."), row("List gaps in historical load samples.")]
    fmsr = [row("Rank likely winding failure mechanisms.", "Use fmsr.get_failure_modes.")]
    multi = [row("Connect the recorded temperature rise to known fault mechanisms.", "Use iot.history and fmsr.get_failure_modes.")]
    negative_iot = [row("Give one reading both above and below 100 C.", "Cannot answer because the constraints contradict each other.")]
    negative_multi = [row("Reconstruct the exact unrecorded incident and prove its root cause.", "Cannot answer conclusively without observations and diagnostic evidence.")]
    prompts = scripted_generation([iot, fmsr, multi, negative_iot, negative_multi])
    monkeypatch.setattr(sys, "argv", ["generator", "Transformer", "--scenario-plan", json.dumps(plan), "--log"])

    cli.main()

    manifest_path, = (tmp_path / "runs").glob("*/run.json")
    manifest = json.loads(manifest_path.read_text())
    positives = json.loads((manifest_path.parent / "scenarios.json").read_text())
    negatives = json.loads((manifest_path.parent / "negative_scenarios.json").read_text())
    assert [item["text"] for item in positives] == [item["text"] for item in iot + fmsr + multi]
    assert Counter(item["type"] for item in negatives) == {"iot": 1, "multiagent": 1}
    assert [item["text"] for item in negatives] == [negative_iot[0]["text"], negative_multi[0]["text"]]
    assert manifest["status"] == "complete"
    assert (manifest["config"]["num_scenarios"], manifest["config"]["num_negative_scenarios"]) == (4, 2)
    assert manifest["config"]["scenario_plan"] == {**plan, "fmsr": {"positive": 1, "negative": 0}}
    assert GeneratorConfig.model_validate(manifest["config"]).scenario_plan.negative_counts == {"iot": 1, "multiagent": 1}
    multi_negative_prompt = next(prompt for prompt in prompts if "intentionally unanswerable evaluation scenarios with primary focus 'multiagent'" in prompt)
    catalogue = multi_negative_prompt.split("Available Focus Tools:\n", 1)[1].split("\n\nSuggested", 1)[0]
    assert json.loads(catalogue) == {
        "iot": "  - history(): Retrieve historical readings.",
        "fmsr": "  - get_failure_modes(): List known failure modes.",
    }


@pytest.mark.anyio
@pytest.mark.parametrize("focus,kind,characteristic", [
    ("multiagent", "positive", "Use iot.history and fmsr.get_failure_modes."),
    ("fmsr", "negative", "Cannot answer because the requested conclusion contradicts the supplied evidence."),
])
async def test_single_track_plans_do_not_generate_unrequested_scenarios(tmp_path, scripted_generation, focus, kind, characteristic):
    expected = row("Assess the stated fault conclusion against the observations.", characteristic)
    scripted_generation([[expected]])
    run = await service.generate_scenarios(
        GeneratorConfig(asset_name="Transformer", scenario_plan={focus: {kind: 1}}),
        output_dir=tmp_path / "only-requested",
    )
    assert run.complete
    assert [item.text for item in run.result.scenarios] == ([expected["text"]] if kind == "positive" else [])
    assert [item.text for item in run.result.negative_scenarios] == ([expected["text"]] if kind == "negative" else [])


@pytest.mark.parametrize("payload,extra,error", [
    ('{"unknown": {"positive": 1}}', [], "literal_error"),
    ('{"iot": {"negative": -1}}', [], "greater_than_equal"),
    ('{"iot": {"positive": true}}', [], "int_type"),
    ('{"iot": {"positve": 1}}', [], "extra_forbidden"),
    ('{"iot": {"positive": 0}}', [], "at least one"),
    ('{', [], "json_invalid"),
    ('{"iot": {"positive": 1}}', ["--scenario-counts", '{"positive":1}'], "not allowed with argument"),
])
def test_invalid_cli_plans_fail_before_generation(monkeypatch, capsys, payload, extra, error):
    monkeypatch.setattr(sys, "argv", ["generator", "Transformer", "--scenario-plan", payload, *extra])
    monkeypatch.setattr(cli, "generate_scenarios", lambda *args: pytest.fail("Invalid plan reached generation"))
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert error in capsys.readouterr().err


@pytest.mark.parametrize("counts", [
    {"positive": 3, "negative": 2},
    {"positive": 1, "negative": 0},
    {"positive": 0, "negative": 2},
])
def test_json_totals_drive_automatic_generation(monkeypatch, tmp_path, scripted_generation, counts):
    allocation = None
    batches = []
    if counts["positive"]:
        allocation = {"iot": 1}
        iot = [row("Find missing oil temperature measurements.")]
        if counts["positive"] == 3:
            allocation = {"iot": 2, "fmsr": 1}
            iot.append(row("Compare yesterday's electrical load with today's readings."))
        batches.append(iot)
        if "fmsr" in allocation:
            batches.append([row("Rank likely winding failure mechanisms.", "Use fmsr.get_failure_modes.")])
    if counts["negative"]:
        batches.extend([
            [row("Provide one reading both above and below 100 C.", "Cannot answer because the requirements conflict.")],
            [row("Prove the exact fault cause without diagnostic evidence.", "Cannot answer conclusively with the available data.")],
        ])
    scripted_generation(batches, allocation=allocation)
    monkeypatch.setattr(sys, "argv", ["generator", "Transformer", "--scenario-counts", json.dumps(counts)])

    cli.main()

    manifest_path, = (tmp_path / "runs").glob("*/run.json")
    manifest = json.loads(manifest_path.read_text())
    positives = json.loads((manifest_path.parent / "scenarios.json").read_text())
    negative_path = manifest_path.parent / "negative_scenarios.json"
    negatives = json.loads(negative_path.read_text()) if negative_path.exists() else []
    assert Counter(item["type"] for item in positives) == (allocation or {})
    assert Counter(item["type"] for item in negatives) == ({"iot": 1, "fmsr": 1} if counts["negative"] else {})
    assert (len(positives), len(negatives)) == (counts["positive"], counts["negative"])
    assert manifest["status"] == "complete"
    assert manifest["config"]["scenario_counts"] == counts
    restored = GeneratorConfig.model_validate(manifest["config"])
    assert (restored.num_scenarios, restored.num_negative_scenarios) == (counts["positive"], counts["negative"])


@pytest.mark.parametrize("payload,error", [
    ('{"positive":-1}', "greater_than_equal"),
    ('{"positive":2,"negative":true}', "int_type"),
    ('{}', "at least one"),
    ('{"iot":{"positive":2}}', "extra_forbidden"),
])
def test_invalid_totals_fail_before_generation(monkeypatch, capsys, payload, error):
    monkeypatch.setattr(sys, "argv", ["generator", "Transformer", "--scenario-counts", payload])
    monkeypatch.setattr(cli, "generate_scenarios", lambda *args: pytest.fail("Invalid totals reached generation"))
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert error in capsys.readouterr().err


def test_python_api_rejects_conflicting_count_sources():
    with pytest.raises(ValueError, match="either scenario_plan or scenario_counts"):
        GeneratorConfig(
            asset_name="Transformer",
            scenario_plan={"iot": {"positive": 1}},
            scenario_counts={"positive": 1},
        )
    with pytest.raises(ValueError, match="conflicts with scenario_counts"):
        GeneratorConfig(asset_name="Transformer", scenario_counts={"positive": 1}, num_scenarios=2)


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["positive", "negative"])
async def test_matching_total_with_wrong_focus_is_incomplete(monkeypatch, tmp_path, kind):
    class WrongFocusAgent:
        def __init__(self, **kwargs):
            pass

        async def run(self, *args, **kwargs):
            rows = [Scenario(id=f"wrong-{i}", type="iot", **row(f"Question {i}")) for i in range(2)]
            return ScenarioGenerationResult(
                scenarios=rows if kind == "positive" else [],
                negative_scenarios=rows if kind == "negative" else [],
            )

    monkeypatch.setattr(service, "ScenarioGeneratorAgent", WrongFocusAgent)
    run = await service.generate_scenarios(
        GeneratorConfig(asset_name="Transformer", scenario_plan={"iot": {kind: 1}, "fmsr": {kind: 1}}),
        output_dir=tmp_path / "wrong-focus",
    )
    assert not run.complete
    assert json.loads((run.output_path.parent / "run.json").read_text())["status"] == "incomplete"
