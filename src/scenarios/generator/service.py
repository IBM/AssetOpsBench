"""Reusable execution service; provider selection and CLI presentation stay separate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import mkdtemp

from ..config import GeneratorConfig
from ..models import ScenarioGenerationResult
from .agent import ScenarioGeneratorAgent
from .prompt_helpers import default_scenario_output_path, negative_scenario_output_path


@dataclass(frozen=True)
class GenerationRun:
    config: GeneratorConfig
    result: ScenarioGenerationResult
    output_path: Path
    negative_output_path: Path | None

    @property
    def complete(self) -> bool:
        if self.config.scenario_plan is not None:
            return (
                Counter(row.type for row in self.result.scenarios) == self.config.scenario_plan.positive_counts
                and Counter(row.type for row in self.result.negative_scenarios) == self.config.scenario_plan.negative_counts
            )
        return (
            len(self.result.scenarios) == self.config.num_scenarios
            and len(self.result.negative_scenarios) == self.config.num_negative_scenarios
        )


async def generate_scenarios(
    config: GeneratorConfig, *, output_dir: str | Path | None = None,
) -> GenerationRun:
    """Generate and save one run with a validated, credential-free configuration.

    Default directories are unique even for simultaneous runs of the same
    asset. A supplied directory must not already exist, preventing overwrites.
    Environment loading is the caller's responsibility.
    """
    if output_dir is None:
        proposed = default_scenario_output_path(config.asset_name).parent
        proposed.parent.mkdir(parents=True, exist_ok=True)
        run_dir = Path(mkdtemp(prefix=proposed.name + "_", dir=proposed.parent))
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "scenarios.json"
    manifest_path = run_dir / "run.json"
    manifest = {"config": config.resolved_dict(), "status": "running"}

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    try:
        agent = ScenarioGeneratorAgent(
            model_id=config.resolved_model, backend=config.backend,
            batch_size=config.batch_size,
            show_workflow=config.show_workflow,
            log_dir=str(run_dir / "logs") if config.log else None,
            retriever=config.retriever, research_file=config.research_file,
        )
        result = await agent.run(
            config.asset_name, num_scenarios=config.num_scenarios,
            live_data=config.live_data,
            num_negative_scenarios=config.num_negative_scenarios,
            scenario_plan=config.scenario_plan,
        )
        output_path.write_text(
            json.dumps([row.to_dict() for row in result.scenarios], indent=2) + "\n",
            encoding="utf-8",
        )
        negative_path = None
        if result.negative_scenarios:
            negative_path = negative_scenario_output_path(output_path)
            negative_path.write_text(
                json.dumps([row.to_dict() for row in result.negative_scenarios], indent=2) + "\n",
                encoding="utf-8",
            )
        run = GenerationRun(config, result, output_path, negative_path)
        manifest.update(
            status="complete" if run.complete else "incomplete",
            scenario_count=len(result.scenarios), negative_count=len(result.negative_scenarios),
        )
        save_manifest()
        return run
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        save_manifest()
        raise
