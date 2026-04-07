"""GEPA-based reflective skill refinement.

Uses DSPy's optimisation to iteratively improve auto-generated skills based on
execution trace feedback from failed scenarios.

Requires the ``dspy`` package (install via ``uv add --group gepa dspy``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

try:
    import dspy

    _HAS_DSPY = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_DSPY = False


def _ensure_dspy() -> None:
    if not _HAS_DSPY:
        raise ImportError(
            "dspy is required for GEPA refinement. "
            "Install it with: uv add --group gepa dspy"
        )


class SkillRefiner:
    """Refine a single skill using LLM reflection on failed execution traces.

    This wraps DSPy's ``ChainOfThought`` to propose targeted mutations to a
    SKILL.md file (trigger clarity, tool parameter specificity, decision
    logic).
    """

    def __init__(self, skill_text: str, domain: str) -> None:
        _ensure_dspy()
        self._skill_text = skill_text
        self._domain = domain
        self._refine = dspy.ChainOfThought(
            "execution_trace, skill_text -> refined_skill"
        )

    def refine(self, execution_trace: str) -> str:
        """Return a refined version of the skill given a failed trace."""
        prediction = self._refine(
            execution_trace=execution_trace,
            skill_text=self._skill_text,
        )
        return prediction.refined_skill


def _score_skill_quality(refined_skill: str, execution_trace: str) -> float:
    """Heuristic quality metric for a refined skill.

    Checks for presence of key structural elements that indicate a
    well-formed, actionable skill.
    """
    score = 0.0
    markers = [
        "## When to Use",
        "## Prerequisites",
        "## Procedure",
        "## Decision Logic",
        "## Expected Outputs",
        "call ",
    ]
    for marker in markers:
        if marker in refined_skill:
            score += 1.0
    # Normalise to [0, 1]
    return score / len(markers)


def _score_skill_pass_rate(
    refined_skill: str,
    eval_results: list[dict],
) -> float:
    """Task-pass-rate metric for a refined skill.

    *eval_results* is a list of dicts with at least a ``"passed"`` bool key,
    typically collected by running the evaluation batch with the refined skill
    injected.  Returns the fraction of scenarios that passed.
    """
    if not eval_results:
        return 0.0
    return sum(1 for r in eval_results if r.get("passed")) / len(eval_results)


async def run_gepa_refinement(
    cluster_name: str,
    seed_skills: list[Path],
    failed_traces: list[dict[str, Any]],
    llm_model: str = "claude-sonnet-4-6",
    max_metric_calls: int = 300,
    output_dir: Path = Path("src/skills/gepa-refined"),
    eval_results_by_scenario: dict[str, bool] | None = None,
) -> list[Path]:
    """Refine skills for a task cluster using GEPA-style optimisation.

    Args:
        cluster_name: Task cluster name (e.g. ``"diagnostics"``).
        seed_skills: Paths to the auto-generated SKILL.md files to refine.
        failed_traces: List of dicts with ``scenario_id``, ``trace``, and
                       ``expected_result`` from failed evaluations.
        llm_model: Model string for the DSPy LM.
        max_metric_calls: Budget per skill (approx. 300 per cluster).
        output_dir: Where to write refined skills.
        eval_results_by_scenario: Optional mapping of scenario_id → passed.
            When provided the optimiser uses a combined metric (0.7 × task
            pass rate + 0.3 × structural quality) instead of structural
            quality alone.

    Returns:
        Paths to the refined SKILL.md files.
    """
    _ensure_dspy()

    lm = dspy.LM(llm_model)
    dspy.configure(lm=lm)

    def skill_quality_metric(example: Any, prediction: Any, trace: Any = None) -> float:
        refined = prediction.refined_skill
        structural = _score_skill_quality(refined, example.execution_trace)
        if eval_results_by_scenario:
            # Build per-scenario results list for pass-rate scoring.
            results = [
                {"scenario_id": sid, "passed": passed}
                for sid, passed in eval_results_by_scenario.items()
            ]
            pass_rate = _score_skill_pass_rate(refined, results)
            return 0.7 * pass_rate + 0.3 * structural
        return structural

    refined_paths: list[Path] = []
    cluster_dir = output_dir / cluster_name
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for skill_path in seed_skills:
        skill_text = skill_path.read_text(encoding="utf-8")
        refiner = SkillRefiner(skill_text, domain=cluster_name)

        # Build training examples from failed traces
        examples = []
        for ft in failed_traces:
            examples.append(
                dspy.Example(
                    execution_trace=ft["trace"],
                    skill_text=skill_text,
                ).with_inputs("execution_trace", "skill_text")
            )

        if not examples:
            _log.info("No failed traces for %s — skipping GEPA", skill_path.name)
            refined_paths.append(skill_path)
            continue

        # Run optimisation
        optimizer = dspy.MIPROv2(
            metric=skill_quality_metric,
            num_candidates=min(max_metric_calls, len(examples) * 10),
            num_threads=1,
        )

        try:
            optimized = optimizer.compile(
                refiner._refine,
                trainset=examples[:max_metric_calls],
            )
            # Use the optimised module to refine the skill one final time
            best_trace = failed_traces[0]["trace"]
            result = optimized(
                execution_trace=best_trace,
                skill_text=skill_text,
            )
            refined_text = result.refined_skill
        except Exception:
            _log.exception(
                "GEPA optimisation failed for %s — using original", skill_path.name
            )
            refined_text = skill_text

        out_path = cluster_dir / skill_path.name
        out_path.write_text(refined_text, encoding="utf-8")
        _log.info("Refined skill written to %s", out_path)
        refined_paths.append(out_path)

    return refined_paths
