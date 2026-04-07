"""Skill-aware plan-execute runner.

Extends the base :class:`PlanExecuteRunner` and :class:`Planner` to inject
auto-generated skills into the planning prompt **without** modifying the
original classes.
"""

from __future__ import annotations

from pathlib import Path

from llm import LLMBackend
from agent.plan_execute.planner import Planner, parse_plan
from agent.plan_execute.runner import PlanExecuteRunner

_SKILL_AUGMENTED_PLAN_PROMPT = """\
You are a planning assistant for industrial asset operations and maintenance.

Decompose the question below into a sequence of subtasks. For each subtask,
assign a server and select the exact tool to call. Do NOT include tool arguments —
they will be resolved at execution time from the task description and prior results.

Available servers and tools:
{servers}

{skills_section}

Output format — one block per step, exactly:

#Task1: <task description>
#Server1: <exact server name>
#Tool1: <exact tool name, or "none" if no tool call is needed>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Server2: <exact server name>
#Tool2: <exact tool name>
#Dependency2: #S1
#ExpectedOutput2: <what this step should produce>

Rules:
- Server and tool names must exactly match those listed above.
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if none.
- When a loaded skill matches the question, follow its procedure and decision \
logic to select the right tools and ordering.
- Keep tasks specific and actionable.

Question: {question}

Plan:
"""


class SkillAwarePlanner(Planner):
    """Planner that includes loaded skills in the planning prompt."""

    def __init__(self, llm: LLMBackend, skills_text: str = "") -> None:
        super().__init__(llm)
        self._skills_text = skills_text

    def generate_plan(self, question, server_descriptions):
        servers_text = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in server_descriptions.items()
        )
        if self._skills_text:
            skills_section = (
                "## Loaded Skills (follow these procedures when applicable)\n\n"
                + self._skills_text
            )
        else:
            skills_section = ""

        prompt = _SKILL_AUGMENTED_PLAN_PROMPT.format(
            servers=servers_text,
            skills_section=skills_section,
            question=question,
        )
        raw = self._llm.generate(prompt)
        return parse_plan(raw)


class SkillAwarePlanExecuteRunner(PlanExecuteRunner):
    """PlanExecuteRunner that injects skills into the planning phase.

    This subclass replaces the default planner with a
    :class:`SkillAwarePlanner` that appends loaded skills to the planning
    prompt.  Everything else (execution, summarisation) is unchanged.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        skills_text: str = "",
    ) -> None:
        super().__init__(llm, server_paths)
        # Override the planner set by the parent __init__.
        self._planner = SkillAwarePlanner(llm, skills_text)
