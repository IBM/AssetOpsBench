"""AgentRunner backed by Artificial Analysis' Stirrup framework.

Stirrup is an in-process Python library (not a separate binary), so this
runner is structurally the same as :class:`~agent.deep_agent.DeepAgentRunner`:
build a client, attach the AssetOpsBench MCP servers as a tool provider, run
the agent loop, and map the returned message history onto the shared
:class:`~agent.models.Trajectory`.

Model routing:
  * ``litellm_proxy/<model>``  -> Stirrup ``ChatCompletionsClient`` pointed at
    the LiteLLM proxy (OpenAI-compatible), matching the other runners.
  * ``<provider>/<model>``     -> Stirrup ``LiteLLMClient``, which reaches
    Anthropic, watsonx, Bedrock, etc. natively through LiteLLM.  This means
    ``watsonx/...`` models work directly here, without the proxy detour Goose
    needed.

Tracks (the code switch):
  * ``code_enabled=False`` -> tools are *only* the MCP servers; directly
    comparable to claude-agent / openai-agent / deep-agent.
  * ``code_enabled=True``  -> a sandboxed code-execution tool (Docker by
    default) is added, so the agent may solve a scenario by writing code.
    Report on its own leaderboard track; the bypass metric records whether it
    did so instead of calling the domain tools.

Stirrup's web/default tools are deliberately NOT attached: the environment
under test is the MCP servers (plus, on the code track, code execution), so
adding web search would contaminate the benchmark.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from pathlib import Path

from observability import agent_run_span, persist_trajectory

from .._litellm import LITELLM_PREFIX, resolve_model
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, Trajectory
from ..runner import AgentRunner
from .trajectory import build_trajectory, classify_tool, final_answer

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# A code-track image needs the scientific stack the WO/vibration analyses use.
_DEFAULT_CODE_IMAGE = os.environ.get("STIRRUP_CODE_IMAGE", "python:3.12-slim")


class StirrupAgentRunner(AgentRunner):
    """Run a question through a Stirrup agent against the MCP servers.

    Args:
        llm: Unused; accepted for :class:`AgentRunner` interface parity.
        server_paths: MCP server specs (defaults to all registered servers).
        model: ``litellm_proxy/<provider>/<model>`` or native ``<provider>/<model>``.
        code_enabled: Add a sandboxed code-execution tool (the code track).
        code_backend: ``"docker"`` (sandboxed, default), ``"local"``, or ``"e2b"``.
        max_turns: Stirrup agent loop bound.
        max_tokens: Context window hint passed to the client.
    """

    def __init__(
        self,
        llm=None,
        server_paths=None,
        model: str = _DEFAULT_MODEL,
        code_enabled: bool = True,
        code_backend: str = "docker",
        max_turns: int = 30,
        max_tokens: int = 16_384,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._code_enabled = code_enabled
        self._code_backend = code_backend
        self._max_turns = max_turns
        self._max_tokens = max_tokens

    # -- client / tools ----------------------------------------------------

    def _build_client(self):
        """Build a Stirrup LLM client for the configured model id."""
        if self._model_id.startswith(LITELLM_PREFIX):
            base_url = os.environ.get("LITELLM_BASE_URL")
            api_key = os.environ.get("LITELLM_API_KEY")
            if not base_url or not api_key:
                raise ValueError(
                    "LITELLM_BASE_URL and LITELLM_API_KEY must be set when "
                    f"using the {LITELLM_PREFIX!r} model prefix"
                )
            from stirrup.clients.chat_completions_client import ChatCompletionsClient

            return ChatCompletionsClient(
                model=resolve_model(self._model_id),
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                max_tokens=self._max_tokens,
            )
        from stirrup.clients.litellm_client import LiteLLMClient

        return LiteLLMClient(model=self._model_id, max_tokens=self._max_tokens)

    def _build_mcp_provider(self):
        """Build a Stirrup ``MCPToolProvider`` for the AssetOpsBench servers.

        Each server is a stdio process launched exactly as the other runners
        launch it: ``uv run --directory <repo> <entry-point>``.
        """
        from stirrup.tools.mcp import MCPConfig, MCPToolProvider

        servers: dict[str, dict] = {}
        for name, spec in self._server_paths.items():
            cmd_arg = str(spec)
            servers[name] = {
                "command": "uv",
                "args": ["run", "--directory", str(_REPO_ROOT), cmd_arg],
                "cwd": str(_REPO_ROOT),
            }
        config = MCPConfig.model_validate({"mcpServers": servers})
        return MCPToolProvider(config=config)

    def _build_code_provider(self):
        """Build the sandboxed code-execution provider for the code track."""
        if self._code_backend == "local":
            from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

            return LocalCodeExecToolProvider()
        if self._code_backend == "e2b":
            from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider

            return E2BCodeExecToolProvider()
        from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

        return DockerCodeExecToolProvider.from_image(_DEFAULT_CODE_IMAGE)

    def _build_tools(self) -> list:
        tools: list = []
        if self._code_enabled:
            tools.append(self._build_code_provider())
        tools.append(self._build_mcp_provider())
        return tools

    # -- run ---------------------------------------------------------------

    async def run(self, question: str) -> AgentResult:
        from stirrup import Agent

        with agent_run_span(
            "stirrup-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()

            agent = Agent(
                client=self._build_client(),
                name="assetops",
                system_prompt=AGENT_SYSTEM_PROMPT,
                tools=self._build_tools(),
                max_turns=self._max_turns,
            )

            _log.info(
                "StirrupAgentRunner: starting (model=%s, code=%s, backend=%s)",
                self._model_id, self._code_enabled, self._code_backend,
            )

            async with agent.session() as session:
                finish_params, history, _metadata = await session.run(question)

            trajectory = build_trajectory(history)
            trajectory.started_at = started_at
            answer = final_answer(history, finish_params)

            self._annotate_span(span, trajectory, answer, run_started)
            persist_trajectory(
                runner_name="stirrup-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)

    def _annotate_span(self, span, trajectory: Trajectory, answer: str, started: float) -> None:
        domain_servers = set(self._server_paths)
        counts = {"domain": 0, "code": 0, "other": 0}
        for tc in trajectory.all_tool_calls:
            counts[classify_tool(tc.name, domain_servers)] += 1
        total_tools = sum(counts.values())
        bypass = self._code_enabled and counts["code"] > 0 and counts["domain"] == 0

        span.set_attribute("agent.answer.length", len(answer))
        span.set_attribute("gen_ai.usage.input_tokens", trajectory.total_input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", trajectory.total_output_tokens)
        span.set_attribute("agent.turns", len(trajectory.turns))
        span.set_attribute("agent.tool_calls", total_tools)
        span.set_attribute("agent.duration_ms", (time.perf_counter() - started) * 1000)
        span.set_attribute("agent.code_track", self._code_enabled)
        span.set_attribute("agent.domain_tool_calls", counts["domain"])
        span.set_attribute("agent.code_tool_calls", counts["code"])
        span.set_attribute("agent.tool_bypass", bypass)

        _log.info(
            "StirrupAgentRunner: done (turns=%d, domain=%d, code=%d, bypass=%s)",
            len(trajectory.turns), counts["domain"], counts["code"], bypass,
        )