"""Scenario grounding against main's registry and failure-mode APIs."""

import json

import pytest

from scenarios import grounding


@pytest.mark.parametrize("cached_mapping", [False, True])
def test_grounding_uses_asset_class_and_optional_cached_mapping(monkeypatch, tmp_path, cached_mapping):
    monkeypatch.setattr(grounding, "_FAILURE_MAPPING_DIR", tmp_path)
    monkeypatch.setattr(grounding, "get_asset_coverage", lambda: [{
        "site_name": "NORTH",
        "asset_id": "Transformer 1",
        "asset_class": "TRANSFORMER",
        "sensors": ["hydrogen"],
        "time_range": {"start": "2024-01-01", "end": "2024-01-02", "total_observations": 2},
    }, {
        "site_name": "SOUTH",
        "asset_id": "Chiller 6",
        "asset_class": "CHILLER",
        "sensors": ["supply_temperature"],
        "time_range": {"start": "2024-01-01", "end": "2024-01-02", "total_observations": 2},
    }])
    monkeypatch.setattr(grounding, "get_vibration_asset_coverage", lambda: [{
        "site_name": "NORTH", "asset_id": "Transformer 1", "sensors": ["acceleration"],
        "time_range": {"start": "2024-02-01", "end": "2024-02-03", "total_observations": 24},
    }, {
        # Same asset id at another site must not contaminate this instance.
        "site_name": "SOUTH", "asset_id": "Transformer 1", "sensors": ["wrong_site_sensor"],
        "time_range": {"start": "2020-01-01", "end": "2020-01-02", "total_observations": 999},
    }])
    requested_classes = []

    def get_failure_modes(*, asset_class):
        requested_classes.append(asset_class)
        return {"failure_modes": ["Partial Discharge in Oil"]}

    monkeypatch.setattr(grounding, "get_failure_modes", get_failure_modes)
    mapping = {"Partial Discharge in Oil": ["hydrogen"]}
    if cached_mapping:
        (tmp_path / "transformer.json").write_text(json.dumps({
            "fm2sensor": mapping, "sensor2fm": {"hydrogen": ["Partial Discharge in Oil"]},
        }))

    result = grounding.discover_grounding("transformer", requested_open_form=True)

    assert requested_classes == ["transformer"]
    assert result.open_form_eligible
    assert len(result.asset_instances) == 1
    instance = result.asset_instances[0]
    assert (instance.site_name, instance.asset_id) == ("NORTH", "Transformer 1")
    assert instance.has_iot and instance.has_vibration
    assert (instance.iot_time_range.start, instance.iot_time_range.end,
            instance.iot_time_range.total_observations) == ("2024-01-01", "2024-01-02", 2)
    assert (instance.vibration_time_range.start, instance.vibration_time_range.end,
            instance.vibration_time_range.total_observations) == ("2024-02-01", "2024-02-03", 24)
    assert result.iot_sensors == ["hydrogen"]
    assert result.vibration_sensors == ["acceleration"]
    assert result.failure_modes == ["Partial Discharge in Oil"]
    assert result.failure_sensor_mapping == (mapping if cached_mapping else {})
    assert result.sensor_failure_mapping == (
        {"hydrogen": ["Partial Discharge in Oil"]} if cached_mapping else {}
    )


def test_no_matching_asset_class_is_not_open_form_eligible(monkeypatch):
    monkeypatch.setattr(grounding, "get_asset_coverage", lambda: [{
        "asset_id": "Chiller 6", "asset_class": "CHILLER",
    }])
    monkeypatch.setattr(grounding, "get_vibration_asset_coverage", lambda: [])
    result = grounding.discover_grounding("transformer", requested_open_form=True)
    assert not result.open_form_eligible
    assert result.asset_instances == []
    assert result.iot_sensors == []
    assert result.vibration_sensors == []


def test_closed_form_does_not_query_live_data(monkeypatch):
    def unexpected_query(*args, **kwargs):
        pytest.fail("closed-form generation should not query live grounding sources")

    monkeypatch.setattr(grounding, "get_asset_coverage", unexpected_query)
    monkeypatch.setattr(grounding, "get_vibration_asset_coverage", unexpected_query)
    monkeypatch.setattr(grounding, "get_failure_modes", unexpected_query)
    result = grounding.discover_grounding("transformer")
    assert not result.requested_open_form
    assert not result.open_form_eligible


@pytest.mark.anyio
async def test_open_mode_without_inventory_fails_before_research_or_generation(monkeypatch, tmp_path):
    from scenarios.config import GeneratorConfig
    from scenarios.generator import agent, cli, service

    class NoGeneration:
        async def generate(self, *args, **kwargs):
            pytest.fail("Missing open-mode inventory must stop before any model generation")

    async def descriptions(self):
        return {"iot": "Available tools"}

    def no_research(*args, **kwargs):
        pytest.fail("Missing open-mode inventory must stop before research retrieval")

    monkeypatch.setattr(agent, "create_backend", lambda *args: NoGeneration())
    monkeypatch.setattr(agent.Executor, "get_server_descriptions", descriptions)
    monkeypatch.setattr(grounding, "get_asset_coverage", lambda: [])
    monkeypatch.setattr(grounding, "get_vibration_asset_coverage", lambda: [])
    monkeypatch.setattr(agent, "retrieve_asset_evidence", no_research)
    config = GeneratorConfig.model_validate(vars(cli.build_parser().parse_args([
        "Transformer", "--mode", "open", "--log",
    ])))
    destination = tmp_path / "failed-open-run"
    with pytest.raises(ValueError, match="Open mode requires matching live asset inventory") as caught:
        await service.generate_scenarios(config, output_dir=destination)
    assert "CouchDB" in str(caught.value)
    assert "--mode closed" in str(caught.value)
    manifest = json.loads((destination / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["config"]["mode"] == "open"
    assert not (destination / "scenarios.json").exists()
    discovery, = destination.glob("logs/01_grounding/*.json")
    assert json.loads(discovery.read_text())["open_form_eligible"] is False


def test_legacy_config_mode_conflicts_are_rejected():
    from scenarios.config import GeneratorConfig

    assert GeneratorConfig(asset_name="Transformer", live_data=True).mode == "open"
    assert GeneratorConfig(asset_name="Transformer").mode == "closed"
    with pytest.raises(ValueError, match="conflicts with mode"):
        GeneratorConfig(asset_name="Transformer", mode="closed", live_data=True)


@pytest.mark.anyio
async def test_discovered_iot_names_work_with_validation():
    from scenarios.constraints import validate_scenario
    from servers.iot.main import mcp
    from scenarios.generator.prompt_helpers import _validation_tool_names_by_focus

    tools = await mcp.list_tools()
    descriptions = {"iot": "\n".join(f"  - {tool.name}(): {tool.description}" for tool in tools)}
    names = _validation_tool_names_by_focus(descriptions)
    assert "history" in names["iot"]
    assert "get_history" not in names["iot"]
    assert validate_scenario(
        "iot", {"text": "Compare maintenance history across assets at both sites.",
                "category": "Data Query", "characteristic_form": "Use iot.history."},
        tool_names_by_focus=names,
    ) == []
