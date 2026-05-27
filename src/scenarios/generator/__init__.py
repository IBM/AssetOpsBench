"""Scenario generation: orchestration, validation loop, and CLI (`python -m scenarios.generator`)."""

from .agent import ScenarioGeneratorAgent
from .cli import main
from .prompt_helpers import (
    DEFAULT_GENERATED_SCENARIOS_DIR,
    default_scenario_output_path,
    negative_scenario_output_path,
)

__all__ = [
    "DEFAULT_GENERATED_SCENARIOS_DIR",
    "ScenarioGeneratorAgent",
    "default_scenario_output_path",
    "negative_scenario_output_path",
    "main",
]
