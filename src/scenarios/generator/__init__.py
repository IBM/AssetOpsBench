"""Scenario generation: orchestration, validation loop, and CLI (`python -m scenarios.generator`)."""

from .agent import ScenarioGeneratorAgent
from .cli import main
from ..config import GeneratorConfig
from ..planning import ScenarioCounts, ScenarioPlan
from .service import GenerationRun, generate_scenarios
from .prompt_helpers import (
    DEFAULT_GENERATED_SCENARIOS_DIR,
    default_scenario_output_path,
    negative_scenario_output_path,
)

__all__ = [
    "DEFAULT_GENERATED_SCENARIOS_DIR",
    "ScenarioGeneratorAgent",
    "GeneratorConfig",
    "ScenarioCounts",
    "ScenarioPlan",
    "GenerationRun",
    "generate_scenarios",
    "default_scenario_output_path",
    "negative_scenario_output_path",
    "main",
]
