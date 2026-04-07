"""Batch evaluation runner for the skills benchmark."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm import LiteLLMBackend

from .ablation import SkillGranularity, SkillStructure
from .conditions import EvalCondition, create_runner
from .metrics import bootstrap_ci, compute_normalized_gain, compute_pass_rate
from .scorer import score_result
from .task_categories import categorize_scenario

_log = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Settings for a single ablation condition."""

    label: str
    structure: SkillStructure = SkillStructure.FULL
    granularity: SkillGranularity | None = None


@dataclass
class EvalConfig:
    """Configuration for a full evaluation run."""

    models: list[str] = field(
        default_factory=lambda: ["watsonx/meta-llama/llama-3-3-70b-instruct"]
    )
    conditions: list[EvalCondition] = field(
        default_factory=lambda: list(EvalCondition),
    )
    skills_dir: Path = Path("src/skills/generated")
    runs_per_scenario: int = 5
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95
    random_seed: int = 42
    max_concurrent: int = 5
    checkpoint_every: int = 10
    output_dir: Path = Path("results")
    judge_model_id: int = 16
    ablations: list[AblationConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalConfig":
        """Load config from a YAML file."""
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ev = raw.get("evaluation", raw)
        conditions = [
            EvalCondition(c)
            for c in ev.get("conditions", ["no_skills", "mcp_only", "mcp_skills"])
        ]
        ablations: list[AblationConfig] = []
        for abl in ev.get("ablations", []):
            ablations.append(
                AblationConfig(
                    label=abl["label"],
                    structure=SkillStructure(abl.get("structure", "full")),
                    granularity=(
                        SkillGranularity(abl["granularity"])
                        if abl.get("granularity")
                        else None
                    ),
                )
            )
        return cls(
            models=ev.get("models", ["watsonx/meta-llama/llama-3-3-70b-instruct"]),
            conditions=conditions,
            skills_dir=Path(ev.get("skills_dir", "src/skills/generated")),
            runs_per_scenario=ev.get("runs_per_scenario", 5),
            bootstrap_samples=ev.get("bootstrap_samples", 10_000),
            confidence_level=ev.get("confidence_level", 0.95),
            random_seed=ev.get("random_seed", 42),
            max_concurrent=ev.get("max_concurrent", 5),
            checkpoint_every=ev.get("checkpoint_every", 10),
            output_dir=Path(ev.get("output_dir", "results")),
            judge_model_id=ev.get("judge_model_id", 16),
            ablations=ablations,
        )


@dataclass
class ScenarioResult:
    """Result of evaluating one scenario under one condition."""

    scenario_id: str
    model: str
    condition: str
    run: int
    passed: bool
    method: str
    category: str
    criteria: list[dict] = field(default_factory=list)
    answer: str = ""
    error: str | None = None


async def _eval_single(
    scenario: Any,
    condition: EvalCondition,
    llm_model: str,
    skills_dir: Path,
    run_idx: int,
    judge_model_id: int,
    *,
    skill_structure: SkillStructure = SkillStructure.FULL,
    skill_granularity: SkillGranularity | None = None,
    condition_label: str | None = None,
) -> ScenarioResult:
    """Evaluate one scenario once."""
    llm = LiteLLMBackend(llm_model)
    runner = create_runner(
        condition,
        llm,
        skills_dir if condition == EvalCondition.MCP_SKILLS else None,
        skill_structure=skill_structure,
        skill_granularity=skill_granularity,
    )
    scenario_id = getattr(scenario, "id", "unknown")
    category = categorize_scenario(scenario)
    cond_value = condition_label or condition.value

    try:
        result = await runner.run(getattr(scenario, "text", ""))
        trace = "\n".join(
            f"Step {r.step_number}: {r.task} -> {r.response if r.success else r.error}"
            for r in result.history
        )
        score = score_result(scenario, result.answer, trace, judge_model_id)
        return ScenarioResult(
            scenario_id=scenario_id,
            model=llm_model,
            condition=cond_value,
            run=run_idx,
            passed=score["passed"],
            method=score["method"],
            category=category,
            criteria=score.get("criteria", []),
            answer=result.answer,
        )
    except Exception as exc:
        _log.exception("Error on scenario %s run %d", scenario_id, run_idx)
        return ScenarioResult(
            scenario_id=scenario_id,
            model=llm_model,
            condition=cond_value,
            run=run_idx,
            passed=False,
            method="error",
            category=category,
            error=str(exc),
        )


async def run_evaluation_batch(
    scenarios: list,
    condition: EvalCondition,
    llm_model: str,
    skills_dir: Path,
    runs: int = 5,
    max_concurrent: int = 5,
    judge_model_id: int = 16,
    *,
    skill_structure: SkillStructure = SkillStructure.FULL,
    skill_granularity: SkillGranularity | None = None,
    condition_label: str | None = None,
) -> list[ScenarioResult]:
    """Run all scenarios under one condition with concurrency control."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _guarded(s: Any, r: int) -> ScenarioResult:
        async with sem:
            return await _eval_single(
                s,
                condition,
                llm_model,
                skills_dir,
                r,
                judge_model_id,
                skill_structure=skill_structure,
                skill_granularity=skill_granularity,
                condition_label=condition_label,
            )

    tasks = [_guarded(s, r) for s in scenarios for r in range(1, runs + 1)]
    return list(await asyncio.gather(*tasks))


def _checkpoint(results: list[ScenarioResult], path: Path) -> None:
    """Write intermediate results to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "scenario_id": r.scenario_id,
            "model": r.model,
            "condition": r.condition,
            "run": r.run,
            "passed": r.passed,
            "method": r.method,
            "category": r.category,
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _log.info("Checkpoint: %d results -> %s", len(data), path)


async def run_full_evaluation(
    scenarios: list,
    config: EvalConfig,
) -> list[ScenarioResult]:
    """Run the complete evaluation matrix: models × conditions × scenarios × runs.

    When ``config.ablations`` is non-empty, each ablation is run as an
    additional MCP_SKILLS condition with the specified structure/granularity
    stripping.

    Returns all :class:`ScenarioResult` records.
    """
    all_results: list[ScenarioResult] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for model in config.models:
        for cond in config.conditions:
            _log.info(
                "Evaluating model=%s condition=%s (%d scenarios × %d runs)",
                model,
                cond.value,
                len(scenarios),
                config.runs_per_scenario,
            )
            batch = await run_evaluation_batch(
                scenarios,
                cond,
                model,
                config.skills_dir,
                runs=config.runs_per_scenario,
                max_concurrent=config.max_concurrent,
                judge_model_id=config.judge_model_id,
            )
            all_results.extend(batch)
            ckpt = (
                config.output_dir
                / f"checkpoint_{model.replace('/', '_')}_{cond.value}.json"
            )
            _checkpoint(batch, ckpt)

        # Ablation runs (always use MCP_SKILLS as base condition).
        for abl in config.ablations:
            _log.info(
                "Ablation model=%s label=%s (%d scenarios × %d runs)",
                model,
                abl.label,
                len(scenarios),
                config.runs_per_scenario,
            )
            batch = await run_evaluation_batch(
                scenarios,
                EvalCondition.MCP_SKILLS,
                model,
                config.skills_dir,
                runs=config.runs_per_scenario,
                max_concurrent=config.max_concurrent,
                judge_model_id=config.judge_model_id,
                skill_structure=abl.structure,
                skill_granularity=abl.granularity,
                condition_label=abl.label,
            )
            all_results.extend(batch)
            safe = abl.label.replace("/", "_").replace(" ", "_")
            ckpt = (
                config.output_dir / f"checkpoint_{model.replace('/', '_')}_{safe}.json"
            )
            _checkpoint(batch, ckpt)

    return all_results


def aggregate_results(
    results: list[ScenarioResult],
    config: EvalConfig,
) -> dict[str, Any]:
    """Compute aggregate metrics from raw results.

    Returns a nested dict keyed by model -> condition -> metrics.
    """
    from collections import defaultdict

    grouped: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    by_cat: dict[str, dict[str, dict[str, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for r in results:
        grouped[r.model][r.condition].append(r.passed)
        by_cat[r.model][r.condition][r.category].append(r.passed)

    output: dict[str, Any] = {}
    for model, conds in grouped.items():
        model_out: dict[str, Any] = {}
        for cond, vals in conds.items():
            pr = compute_pass_rate(vals)
            lo, hi = bootstrap_ci(
                [float(v) for v in vals],
                n_bootstrap=config.bootstrap_samples,
                ci=config.confidence_level,
                seed=config.random_seed,
            )
            model_out[cond] = {
                "pass_rate": pr,
                "ci_lower": lo,
                "ci_upper": hi,
                "n": len(vals),
            }

        # Normalized gain (MCP_SKILLS vs MCP_ONLY)
        mcp_only = model_out.get(EvalCondition.MCP_ONLY.value, {}).get("pass_rate", 0.0)
        mcp_skills = model_out.get(EvalCondition.MCP_SKILLS.value, {}).get(
            "pass_rate", 0.0
        )
        model_out["normalized_gain"] = compute_normalized_gain(mcp_skills, mcp_only)

        # Per-category breakdown
        cats: dict[str, dict[str, Any]] = {}
        for cond, cat_vals in by_cat[model].items():
            for cat, vals in cat_vals.items():
                cats.setdefault(cat, {})[cond] = {
                    "pass_rate": compute_pass_rate(vals),
                    "n": len(vals),
                }
        model_out["per_category"] = cats
        output[model] = model_out

    return output
