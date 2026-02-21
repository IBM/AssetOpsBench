"""Entry-point runner for the plan-execute workflow using MCP servers.

Replaces AgentHive's combination of PlanningWorkflow + SequentialWorkflow with
an MCP-native implementation:

  AgentHive                       plan_execute
  ────────────────────────────    ─────────────────────────────
  PlanningWorkflow.generate_steps → Planner.generate_plan
  SequentialWorkflow.run          → Executor.execute_plan
  ReactAgent.execute_task         → _list_tools + _call_tool (MCP stdio)
"""

from __future__ import annotations

import logging
from pathlib import Path

from llm import LLMBackend

_log = logging.getLogger(__name__)

from .executor import Executor
from .models import OrchestratorResult
from .planner import Planner

_SUMMARIZE_PROMPT = """\
You are summarizing the results of a multi-step task execution for an \
industrial asset operations system.

Original question: {question}

Step-by-step execution results:
{results}

Provide a concise, direct answer to the original question based on the results
above. Do not repeat the individual steps — just give the final answer.
"""


class PlanExecuteRunner:
    """Entry-point for plan-and-execute workflows using MCP servers as tool providers.

    Usage::

        from plan_execute import PlanExecuteRunner
        from llm import LiteLLMBackend

        runner = PlanExecuteRunner(llm=LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct"))
        result = await runner.run("What are the assets at site MAIN?")
        print(result.answer)

    Args:
        llm: LLM backend used for planning, tool selection, and summarisation.
        server_paths: Override MCP server script paths.  Keys must match the
                      agent names the planner will assign steps to.  Defaults
                      to IoTAgent and Utilities servers in mcp/servers/.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path] | None = None,
    ) -> None:
        self._llm = llm
        self._planner = Planner(llm)
        self._executor = Executor(llm, server_paths)

    async def run(self, question: str) -> OrchestratorResult:
        """Run the full plan-execute loop for a question.

        Steps:
          1. Discover available agents from registered MCP servers.
          2. Use the LLM to decompose the question into an execution plan.
          3. Execute each plan step by routing tool calls to MCP servers.
          4. Summarise the step results into a final answer.

        Args:
            question: The user question to answer.

        Returns:
            OrchestratorResult with the final answer, the generated plan, and
            the per-step execution history.
        """
        # 1. Discover
        _log.info("Discovering agent capabilities...")
        agent_descriptions = await self._executor.get_agent_descriptions()

        # 2. Plan
        _log.info("Planning...")
        plan = self._planner.generate_plan(question, agent_descriptions)
        _log.info("Plan has %d step(s).", len(plan.steps))

        # 3. Execute
        history = await self._executor.execute_plan(plan, question)

        # 4. Summarise
        _log.info("Summarising...")
        results_text = "\n\n".join(
            f"Step {r.step_number} — {r.task} (agent: {r.agent}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in history
        )
        answer = self._llm.generate(
            _SUMMARIZE_PROMPT.format(question=question, results=results_text)
        )

        return OrchestratorResult(
            question=question,
            answer=answer,
            plan=plan,
            history=history,
        )
