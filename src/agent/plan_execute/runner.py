"""Plan-and-execute agent runner using MCP servers as tool providers.

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
import time
from pathlib import Path

from llm import LLMBackend, LLMResult
from observability import agent_run_span, persist_trajectory

from .executor import Executor
from .models import OrchestratorResult
from .planner import Planner
from ..runner import AgentRunner


class _TokenMeter(LLMBackend):
    """Wraps an :class:`LLMBackend` and sums token usage across calls.

    ``Planner`` / ``Executor`` call ``generate()`` and only need a string;
    this wrapper transparently pulls usage from the inner backend's
    ``generate_with_usage()`` and accumulates it in-place.  Totals are
    reset at the start of each :meth:`PlanExecuteRunner.run` call so
    per-run span attributes reflect that run alone.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        result = self._inner.generate_with_usage(prompt, temperature)
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        return result.text

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0
    ) -> LLMResult:
        result = self._inner.generate_with_usage(prompt, temperature)
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        return result

    @property
    def model_id(self) -> str:
        return self._inner.model_id

_log = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are summarizing the results of a multi-step task execution for an \
industrial asset operations system.

Original question: {question}

Step-by-step execution results:
{results}

Provide a concise, direct answer to the original question based on the results
above. Do not repeat the individual steps — just give the final answer.
"""


class PlanExecuteRunner(AgentRunner):
    """Entry-point for plan-and-execute workflows using MCP servers as tool providers.

    Usage::

        from agent import PlanExecuteRunner
        from llm import LiteLLMBackend

        runner = PlanExecuteRunner(llm=LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct"))
        result = await runner.run("What are the assets at site MAIN?")
        print(result.answer)

    Args:
        llm: LLM backend used for planning, tool selection, and summarisation.
        server_paths: Override MCP server specs.  Keys must match the server
                      names the planner will assign steps to.  Values are
                      either a uv entry-point name (str) or a Path to a
                      script file.  Defaults to all five registered servers.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        self._meter = _TokenMeter(llm)
        self._planner = Planner(self._meter)
        self._executor = Executor(self._meter, server_paths)

    async def run(self, question: str) -> OrchestratorResult:
        """Run the full plan-execute loop for a question.

        Steps:
          1. Discover available servers from registered MCP servers.
          2. Use the LLM to decompose the question into an execution plan.
          3. Execute each plan step by routing tool calls to MCP servers.
          4. Summarise the step results into a final answer.

        Args:
            question: The user question to answer.

        Returns:
            OrchestratorResult with the final answer, the generated plan, and
            the per-step execution trajectory.
        """
        with agent_run_span(
            "plan-execute", model=self._llm.model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            self._meter.reset()

            # 1. Discover
            _log.info("Discovering server capabilities...")
            server_descriptions = await self._executor.get_server_descriptions()

            # 2. Plan
            _log.info("Planning...")
            planning_started = time.perf_counter()
            plan = self._planner.generate_plan(question, server_descriptions)
            planning_ms = (time.perf_counter() - planning_started) * 1000
            _log.info("Plan has %d step(s).", len(plan.steps))

            # 3. Execute
            trajectory = await self._executor.execute_plan(plan, question)

            # 4. Summarise
            _log.info("Summarising...")
            results_text = "\n\n".join(
                f"Step {r.step_number} — {r.task} (server: {r.server}):\n"
                + (r.response if r.success else f"ERROR: {r.error}")
                for r in trajectory
            )
            summarization_started = time.perf_counter()
            answer = self._meter.generate(
                _SUMMARIZE_PROMPT.format(question=question, results=results_text)
            )
            summarization_ms = (time.perf_counter() - summarization_started) * 1000
            duration_ms = (time.perf_counter() - run_started) * 1000

            result = OrchestratorResult(
                question=question,
                answer=answer,
                plan=plan,
                trajectory=trajectory,
            )
            span.set_attribute("agent.plan.steps", len(plan.steps))
            span.set_attribute("agent.answer.length", len(answer or ""))
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.planning_time_ms", planning_ms)
            span.set_attribute("agent.summarization_time_ms", summarization_ms)
            span.set_attribute("gen_ai.usage.input_tokens", self._meter.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", self._meter.output_tokens)
            # plan-execute's "LLM time" is the time spent on direct LLM calls
            # controlled by the runner (planning + summarisation).  Per-step
            # arg-resolution LLM calls are included in each StepResult's
            # duration_ms instead.
            span.set_attribute("agent.llm_time_ms", planning_ms + summarization_ms)
            persist_trajectory(
                runner_name="plan-execute",
                model=self._llm.model_id,
                question=question,
                answer=answer or "",
                trajectory=trajectory,
            )
            return result
