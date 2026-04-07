"""Evaluation conditions for the skills benchmark."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from llm import LLMBackend
from agent.plan_execute.runner import PlanExecuteRunner

from .ablation import SkillGranularity, SkillStructure
from .skill_loader import load_skills_as_text


class EvalCondition(Enum):
    """Evaluation conditions tested in the skills benchmark.

    * ``NO_SKILLS``  – vanilla agent, no MCP access.
    * ``MCP_ONLY``   – agent with MCP tools, no skills.
    * ``MCP_SKILLS`` – agent with MCP tools **and** auto-generated skills.
    """

    NO_SKILLS = "no_skills"
    MCP_ONLY = "mcp_only"
    MCP_SKILLS = "mcp_skills"


def create_runner(
    condition: EvalCondition,
    llm: LLMBackend,
    skills_dir: Path | None = None,
    *,
    skill_structure: SkillStructure = SkillStructure.FULL,
    skill_granularity: SkillGranularity | None = None,
) -> PlanExecuteRunner:
    """Instantiate the appropriate runner for *condition*.

    Args:
        condition: Which evaluation condition to set up.
        llm: LLM backend for planning and execution.
        skills_dir: Directory containing ``*.md`` skill files
                    (only used when *condition* is ``MCP_SKILLS``).
        skill_structure: Which building blocks to keep in each skill (ablation).
        skill_granularity: Filter skills by complexity level (ablation).

    Returns:
        A configured :class:`PlanExecuteRunner` (or subclass).
    """
    # Lazy import to avoid circular dependency at module level.
    from .skill_aware_runner import SkillAwarePlanExecuteRunner

    if condition == EvalCondition.NO_SKILLS:
        # No MCP access at all — empty server paths.
        return PlanExecuteRunner(llm=llm, server_paths={})

    if condition == EvalCondition.MCP_ONLY:
        # All MCP servers, no skills in prompt.
        return PlanExecuteRunner(llm=llm)

    # MCP_SKILLS — load skills (with ablation) and inject into planner prompt.
    skills_text = ""
    if skills_dir is not None:
        skills_text = load_skills_as_text(
            skills_dir,
            structure=skill_structure,
            granularity=skill_granularity,
        )
    return SkillAwarePlanExecuteRunner(llm=llm, skills_text=skills_text)
