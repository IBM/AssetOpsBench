"""Generation results survive partial runs, failures, and repeated output paths."""

import json

import pytest

from scenarios.config import GeneratorConfig
from scenarios.generator import service
from scenarios.models import Scenario, ScenarioGenerationResult


def install_agent(monkeypatch, result=None, error=None):
    class Agent:
        def __init__(self, **kwargs):
            pass

        async def run(self, asset_name, **kwargs):
            if error:
                raise error
            return result

    monkeypatch.setattr(service, "ScenarioGeneratorAgent", Agent)


def sample_result(scenario_id="test-1"):
    row = Scenario(id=scenario_id, type="iot", text="Compare readings.", category="Data Query",
                   characteristic_form="Use iot.history.")
    return ScenarioGenerationResult(scenarios=[row], negative_scenarios=[])


@pytest.mark.anyio
async def test_repeated_runs_preserve_previous_results(monkeypatch, tmp_path):
    install_agent(monkeypatch, sample_result("first"))
    monkeypatch.setattr(service, "default_scenario_output_path", lambda asset: tmp_path / "same-time" / "scenarios.json")
    config = GeneratorConfig(asset_name="Transformer", num_scenarios=1, num_negative_scenarios=0)
    first = await service.generate_scenarios(config)
    original = first.output_path.read_bytes()

    install_agent(monkeypatch, sample_result("second"))
    second = await service.generate_scenarios(config)
    assert json.loads(first.output_path.read_text())[0]["id"] == "first"
    assert json.loads(second.output_path.read_text())[0]["id"] == "second"
    assert first.output_path.read_bytes() == original
    for run in (first, second):
        assert run.complete
        manifest = json.loads((run.output_path.parent / "run.json").read_text())
        assert manifest["status"] == "complete"
        assert (manifest["scenario_count"], manifest["negative_count"]) == (1, 0)

    with pytest.raises(FileExistsError):
        await service.generate_scenarios(config, output_dir=first.output_path.parent)
    assert first.output_path.read_bytes() == original


@pytest.mark.anyio
@pytest.mark.parametrize("positive_count,negative_count", [(2, 0), (1, 1)])
async def test_partial_run_is_saved_and_reported_incomplete(monkeypatch, tmp_path, positive_count, negative_count):
    install_agent(monkeypatch, sample_result())
    config = GeneratorConfig(
        asset_name="Transformer", num_scenarios=positive_count,
        num_negative_scenarios=negative_count,
    )
    run = await service.generate_scenarios(config, output_dir=tmp_path / "partial")
    assert not run.complete
    assert json.loads(run.output_path.read_text()) == [{
        "id": "test-1", "type": "iot", "text": "Compare readings.",
        "category": "Data Query", "characteristic_form": "Use iot.history.",
    }]
    manifest = json.loads((run.output_path.parent / "run.json").read_text())
    assert manifest["status"] == "incomplete"
    assert (manifest["scenario_count"], manifest["negative_count"]) == (1, 0)


@pytest.mark.anyio
async def test_failure_status_does_not_include_provider_error_body(monkeypatch, tmp_path):
    failure = RuntimeError("private provider response")
    install_agent(monkeypatch, error=failure)
    with pytest.raises(RuntimeError) as caught:
        await service.generate_scenarios(GeneratorConfig(asset_name="Transformer"), output_dir=tmp_path / "failed")
    assert caught.value is failure
    manifest = (tmp_path / "failed" / "run.json").read_text()
    assert "private provider response" not in manifest
    assert json.loads(manifest)["status"] == "failed"
    assert json.loads(manifest)["error_type"] == "RuntimeError"
