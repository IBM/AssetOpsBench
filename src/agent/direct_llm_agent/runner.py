"""Direct LLM baseline runner.

This runner does not use MCP tools, planning, code execution, or retrieval.
It sends the benchmark question directly to the selected LLM and returns the
model's answer. This is useful as a model-only baseline.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from llm import LLMBackend
from observability import agent_run_span, persist_trajectory

from ..models import AgentResult, Trajectory, TurnRecord
from ..runner import AgentRunner


_DIRECT_LLM_SYSTEM_PROMPT = """\
You are a direct LLM baseline for AssetOpsBench.

Read the user task carefully and answer in the requested format.
You do not have access to tools, databases, workorders, sensors, or external files.

If the task asks for information you cannot access, do not refuse only because
data is missing. Use the task wording, general industrial-maintenance knowledge,
and a reasonable guess if needed.

Return only the final answer requested by the user. Do not include reasoning,
tool-use claims, markdown, or extra explanation unless explicitly requested.
"""


class DirectLLMAgentRunner(AgentRunner):
    """A simple model-only runner with no MCP tool calls."""

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        super().__init__(llm, server_paths)

    async def run(self, question: str) -> AgentResult:
        """Run one direct LLM call and return an AgentResult."""
        with agent_run_span(
            "direct-llm-agent",
            model=self._llm.model_id,
            question=question,
        ) as span:
            started = datetime.now(timezone.utc).isoformat()
            run_started = time.perf_counter()

            prompt = f"""{_DIRECT_LLM_SYSTEM_PROMPT}

User task:
{question}

Final answer:"""

            call_started = time.perf_counter()
            result = self._llm.generate_with_usage(prompt, temperature=0.0)
            duration_ms = (time.perf_counter() - call_started) * 1000
            total_duration_ms = (time.perf_counter() - run_started) * 1000

            answer = result.text.strip()

            trajectory = Trajectory(
                started_at=started,
                turns=[
                    TurnRecord(
                        index=0,
                        text=answer,
                        tool_calls=[],
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        duration_ms=duration_ms,
                    )
                ],
            )

            span.set_attribute("agent.answer.length", len(answer))
            span.set_attribute("agent.duration_ms", total_duration_ms)
            span.set_attribute("agent.llm_time_ms", duration_ms)
            span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
            span.set_attribute("agent.tool_calls", 0)

            persist_trajectory(
                runner_name="direct-llm-agent",
                model=self._llm.model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )

            return AgentResult(
                question=question,
                answer=answer,
                trajectory=trajectory,
            )