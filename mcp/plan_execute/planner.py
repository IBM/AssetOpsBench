"""LLM-based plan generation for the plan-execute orchestrator.

Each plan step now includes the specific tool to call and its arguments,
so the executor needs no additional LLM calls — it calls the tool directly.
"""

from __future__ import annotations

import json
import re

from llm import LLMBackend
from .models import Plan, PlanStep

_PLAN_PROMPT = """\
You are a planning assistant for industrial asset operations and maintenance.

Decompose the question below into a sequence of subtasks. For each subtask,
assign an agent and select the exact tool to call with its arguments.

Available agents and tools:
{agents}

For argument values that can only be known from a prior step's result,
use the placeholder {{step_N}} (e.g., {{step_1}}) as the value.

Output format — one block per step, exactly:

#Task1: <task description>
#Agent1: <exact agent name>
#Tool1: <exact tool name, or "none" if no tool call is needed>
#Args1: <JSON object of tool arguments, e.g. {{"site_name": "MAIN"}}>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Agent2: <exact agent name>
#Tool2: <exact tool name>
#Args2: {{"site_name": "MAIN", "asset_id": "{{step_1}}"}}
#Dependency2: #S1
#ExpectedOutput2: <what this step should produce>

Rules:
- Agent and tool names must exactly match those listed above.
- #Args must be a valid JSON object on a single line.
- Use {{step_N}} as a placeholder when an argument depends on step N's result.
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if none.
- Keep tasks specific and actionable.

Question: {question}

Plan:
"""

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_AGENT_RE = re.compile(r"#Agent(\d+):\s*(.+)")
_TOOL_RE = re.compile(r"#Tool(\d+):\s*(.+)")
_ARGS_RE = re.compile(r"#Args(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


def parse_plan(raw: str) -> Plan:
    """Parse an LLM-generated plan string into a Plan object."""
    tasks = {int(m.group(1)): m.group(2).strip() for m in _TASK_RE.finditer(raw)}
    agents = {int(m.group(1)): m.group(2).strip() for m in _AGENT_RE.finditer(raw)}
    tools = {int(m.group(1)): m.group(2).strip() for m in _TOOL_RE.finditer(raw)}
    deps_raw = {int(m.group(1)): m.group(2).strip() for m in _DEP_RE.finditer(raw)}
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_RE.finditer(raw)}

    args: dict[int, dict] = {}
    for m in _ARGS_RE.finditer(raw):
        n = int(m.group(1))
        try:
            args[n] = json.loads(m.group(2).strip())
        except json.JSONDecodeError:
            args[n] = {}

    steps = [
        PlanStep(
            step_number=n,
            task=tasks[n],
            agent=agents.get(n, ""),
            tool=tools.get(n, ""),
            tool_args=args.get(n, {}),
            dependencies=(
                []
                if deps_raw.get(n, "None").strip().lower() == "none"
                else [int(x) for x in _DEP_NUM_RE.findall(deps_raw.get(n, ""))]
            ),
            expected_output=outputs.get(n, ""),
        )
        for n in sorted(tasks)
    ]
    return Plan(steps=steps, raw=raw)


class Planner:
    """Decomposes a question into a structured execution plan using an LLM."""

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def generate_plan(
        self,
        question: str,
        agent_descriptions: dict[str, str],
    ) -> Plan:
        """Generate a plan for a question given available agents and their tools.

        Args:
            question: The user question to answer.
            agent_descriptions: Mapping of agent_name -> formatted tool signatures.

        Returns:
            A Plan where each PlanStep includes the tool to call and its arguments.
        """
        agents_text = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in agent_descriptions.items()
        )
        prompt = _PLAN_PROMPT.format(agents=agents_text, question=question)
        raw = self._llm.generate(prompt)
        return parse_plan(raw)
