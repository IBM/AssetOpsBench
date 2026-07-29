"""AgentRunner backed by Artificial Analysis' Stirrup framework.

Stirrup is an in-process Python library (not a separate binary), so this
runner is structurally the same as :class:`~agent.deep_agent.DeepAgentRunner`:
build a client, attach the AssetOpsBench MCP servers as a tool provider, run
the agent loop, and map the returned message history onto the shared
:class:`~agent.models.Trajectory`.

Model routing:
  * OpenAI-compatible router prefixes -> Stirrup ``OpenResponsesClient`` first,
    with a sticky fallback to ``ChatCompletionsClient`` when the endpoint does
    not support the Responses interface.
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
import shutil
import time
from pathlib import Path
from typing import Any

from observability import agent_run_span, persist_trajectory

from llm.routers import resolve_model, resolve_router_creds
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, Trajectory
from ..runner import AgentRunner
from .trajectory import build_trajectory, classify_tool, final_answer

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# A code-track image needs the scientific stack the WO/vibration analyses use.
_DEFAULT_CODE_IMAGE = os.environ.get("STIRRUP_CODE_IMAGE", "python:3.12-slim")
_RESPONSES_FALLBACK_STATUS_CODES = frozenset(
    {400, 404, 405, 415, 422, 500, 501, 502, 503, 504}
)


def _responses_error_status_code(exc: Exception) -> int | None:
    """Return an HTTP status, including errors wrapped by Tenacity retries."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    last_attempt = getattr(exc, "last_attempt", None)
    if last_attempt is None:
        return None
    nested = last_attempt.exception()
    nested_status = getattr(nested, "status_code", None)
    return nested_status if isinstance(nested_status, int) else None


class _ResponsesThenChatClient:
    """Prefer Open Responses, then stick to Chat Completions if unavailable."""

    def __init__(self, responses_client: Any, chat_client: Any) -> None:
        self._responses_client = responses_client
        self._chat_client = chat_client
        self._use_chat_completions = False

    @property
    def max_tokens(self) -> int:
        return self._responses_client.max_tokens

    @property
    def model_slug(self) -> str:
        return self._responses_client.model_slug

    async def generate(
        self, messages: list[Any], tools: dict[str, Any]
    ) -> Any:
        if self._use_chat_completions:
            return await self._chat_client.generate(messages, tools)

        try:
            return await self._responses_client.generate(messages, tools)
        except Exception as exc:
            status_code = _responses_error_status_code(exc)
            if status_code not in _RESPONSES_FALLBACK_STATUS_CODES:
                raise
            self._use_chat_completions = True
            _log.warning(
                "Responses API unavailable for model %s (HTTP %s); "
                "using Chat Completions for the remainder of the run",
                self.model_slug,
                status_code,
            )
            return await self._chat_client.generate(messages, tools)


def _copy_workspace_contents(source: Path, destination: Path) -> None:
    """Copy code-exec workspace contents out of Stirrup's temp child dir."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _preserving_provider_class(provider_cls):
    class _PreservingCodeExecToolProvider(provider_cls):
        def __init__(self, *args, preserve_dir: Path, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._assetops_preserve_dir = preserve_dir

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            temp_dir = self.temp_dir
            if temp_dir is not None and temp_dir.exists():
                try:
                    fix_ownership = getattr(self, "_fix_file_ownership", None)
                    if fix_ownership is not None:
                        await fix_ownership()
                    _copy_workspace_contents(temp_dir, self._assetops_preserve_dir)
                    _log.info(
                        "Preserved Stirrup code workspace %s -> %s",
                        temp_dir,
                        self._assetops_preserve_dir,
                    )
                except Exception:
                    _log.warning(
                        "Failed to preserve Stirrup code workspace %s",
                        temp_dir,
                        exc_info=True,
                    )
            return await super().__aexit__(exc_type, exc_val, exc_tb)

    return _PreservingCodeExecToolProvider


class StirrupAgentRunner(AgentRunner):
    """Run a question through a Stirrup agent against the MCP servers.

    Args:
        llm: Unused; accepted for :class:`AgentRunner` interface parity.
        server_paths: MCP server specs (defaults to all registered servers).
        model: ``litellm_proxy/<provider>/<model>`` or native ``<provider>/<model>``.
        code_enabled: Add a sandboxed code-execution tool (the code track).
        code_backend: ``"docker"`` (sandboxed, default) or ``"local"``.
        workspace_dir: Optional host base directory for Docker/local code execution.
        preserve_workspace: Copy final code-execution files back into ``workspace_dir``.
        max_turns: Stirrup agent loop bound.
        temperature: Optional sampling temperature passed to the Stirrup client.
        reasoning_effort: Optional reasoning effort passed to the Stirrup client.
    """

    def __init__(
        self,
        llm=None,
        server_paths=None,
        model: str = _DEFAULT_MODEL,
        code_enabled: bool = True,
        code_backend: str = "docker",
        workspace_dir: Path | str | None = None,
        preserve_workspace: bool = False,
        max_turns: int = 30,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        if code_backend not in {"docker", "local"}:
            raise ValueError("code_backend must be 'docker' or 'local'")
        self._model_id = model
        self._code_enabled = code_enabled
        self._code_backend = code_backend
        self._workspace_dir = (
            Path(workspace_dir).expanduser().resolve()
            if workspace_dir is not None
            else None
        )
        if preserve_workspace and self._workspace_dir is None:
            raise ValueError("workspace_dir is required when preserve_workspace is enabled")
        if preserve_workspace and code_backend not in {"docker", "local"}:
            raise ValueError(
                "preserve_workspace is only supported with docker or local code backends"
            )
        self._preserve_workspace = preserve_workspace
        self._max_turns = max_turns
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    # -- client / tools ----------------------------------------------------

    def _build_client(self):
        """Build a Stirrup LLM client for the configured model id."""
        client_kwargs = (
            {"temperature": self._temperature}
            if self._temperature is not None
            else None
        )
        creds = resolve_router_creds(self._model_id)
        if creds is not None:
            from stirrup.clients.chat_completions_client import ChatCompletionsClient
            from stirrup.clients.open_responses_client import OpenResponsesClient

            common_kwargs = {
                "model": resolve_model(self._model_id),
                "base_url": creds.base_url.rstrip("/"),
                "api_key": creds.api_key,
                "reasoning_effort": self._reasoning_effort,
                "kwargs": client_kwargs,
            }
            return _ResponsesThenChatClient(
                responses_client=OpenResponsesClient(**common_kwargs),
                chat_client=ChatCompletionsClient(**common_kwargs),
            )
        from stirrup.clients.litellm_client import LiteLLMClient

        return LiteLLMClient(
            model=self._model_id,
            reasoning_effort=self._reasoning_effort,
            kwargs=client_kwargs,
        )

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

            provider_cls = LocalCodeExecToolProvider
            kwargs = {"temp_base_dir": self._workspace_dir}
            if self._preserve_workspace:
                provider_cls = _preserving_provider_class(provider_cls)
                kwargs["preserve_dir"] = self._workspace_dir
            return provider_cls(**kwargs)
        from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

        if self._preserve_workspace:
            provider_cls = _preserving_provider_class(DockerCodeExecToolProvider)
            return provider_cls(
                _DEFAULT_CODE_IMAGE,
                is_dockerfile=False,
                temp_base_dir=self._workspace_dir,
                preserve_dir=self._workspace_dir,
            )
        return DockerCodeExecToolProvider.from_image(
            _DEFAULT_CODE_IMAGE,
            temp_base_dir=self._workspace_dir,
        )

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
                "StirrupAgentRunner: starting (model=%s, code=%s, backend=%s, workspace=%s, preserve=%s)",
                self._model_id,
                self._code_enabled,
                self._code_backend,
                self._workspace_dir,
                self._preserve_workspace,
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

    def _annotate_span(
        self, span, trajectory: Trajectory, answer: str, started: float
    ) -> None:
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
            len(trajectory.turns),
            counts["domain"],
            counts["code"],
            bypass,
        )
