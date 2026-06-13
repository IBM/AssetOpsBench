"""Shared base for CLI coding-agent runners (Codex, Claude Code, Gemini, ...).

This is AssetOpsBench's port of the per-agent adapter pattern from
``rdi-berkeley/agents-last-exam`` (``ale_run/agents/<agent>/{config,deployer}.py``).
ALE wires each CLI agent to its own computer-use MCP bridges and grades on files
produced in a provisioned VM.  AssetOpsBench instead keeps the existing
:class:`AgentRunner` contract — answer a question by calling the six domain MCP
servers (iot / utilities / fmsr / tsfm / wo / vibration) — and grades the
trajectory.  So this base lifts ALE's three-method lifecycle
(install/launch/parse) but points every agent at *your* MCP servers and folds
its event stream into *your* :class:`Trajectory`.

A concrete agent subclass supplies only the three things that differ between
CLIs (exactly the ALE ``config.py`` / ``deployer.py`` split):

1. ``_write_config(home, base_url)``   — write the agent's MCP + provider config
   (Codex: ``config.toml``; Claude Code: ``.mcp.json``; Gemini:
   ``.gemini/settings.json``) and return any extra process env.
2. ``_build_command(home, system_prompt, question)`` — the headless launch argv.
3. ``_handle_event(event, trajectory)`` — fold one parsed JSON event into the
   trajectory, returning the agent's answer text when present.

Everything else — env validation, subprocess streaming, timeout, observability
span, ``persist_trajectory`` — lives here once.

Requirements: the agent CLI and ``uv`` on PATH, plus ``LITELLM_BASE_URL`` /
``LITELLM_API_KEY`` in the environment (same contract as the SDK runners).
"""

from __future__ import annotations

import abc
import asyncio
import datetime as _dt
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from observability import agent_run_span, persist_trajectory

from .._litellm import LITELLM_PREFIX, resolve_model
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

# src/agent/cli_agent/base.py -> repo root is four parents up (matches the
# other runners' _REPO_ROOT computation).
_REPO_ROOT = Path(__file__).parent.parent.parent.parent


class McpServerSpec:
    """One AssetOpsBench MCP server, launched the same way every runner does:
    ``uv run <entry-point>`` with ``cwd`` at the repo root."""

    __slots__ = ("name", "command", "args", "cwd")

    def __init__(self, name: str, spec: Path | str) -> None:
        self.name = name
        cmd_arg = str(spec) if isinstance(spec, Path) else spec
        self.command = "uv"
        self.args = ["run", cmd_arg]
        self.cwd = str(_REPO_ROOT)


class CliCodingAgentRunner(AgentRunner, abc.ABC):
    """Base AgentRunner that drives a headless CLI coding agent as a subprocess.

    Args mirror the SDK runners so the benchmark harness treats every runner
    identically.

    Args:
        llm: Unused — accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs (defaults to all registered servers).
        model: LiteLLM-prefixed or native model string. The prefix is stripped
            before being handed to the CLI; the proxy is reached via
            ``LITELLM_BASE_URL`` / ``LITELLM_API_KEY``.
        timeout_s: Hard wall-clock cap on a single run.
    """

    #: Short identifier used in spans / persisted trajectories (e.g. "codex").
    agent_name: str = "cli-agent"
    #: Default model string for this agent; subclasses override.
    default_model: str = "litellm_proxy/azure/gpt-5.4"

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model or self.default_model
        self._timeout_s = timeout_s

    # ------------------------------------------------------------------ #
    # Subclass hooks (the only per-agent code).
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _write_config(self, home: Path, base_url: str) -> dict[str, str]:
        """Write the agent's MCP + provider config under *home*.

        Use :meth:`_mcp_servers` for the server list and
        :attr:`_resolved_model` for the model id.  Return a dict of extra env
        vars the launch needs (e.g. ``{"CODEX_HOME": str(home)}``)."""

    @abc.abstractmethod
    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        """Return the headless launch argv for this agent."""

    @abc.abstractmethod
    def _handle_event(self, event: dict, trajectory: Trajectory) -> str | None:
        """Fold one parsed JSON event into *trajectory*.

        Return the agent's answer text when the event carries it (the last
        non-empty return wins), else ``None``."""

    def _cwd(self, home: Path) -> str:
        """Working directory for the agent subprocess.

        Defaults to the repo root so ``uv run <server>`` resolves.  Agents that
        discover their config from the cwd (e.g. Gemini's ``.gemini/``) override
        this to *home* and pin each MCP server's ``cwd`` to the repo root in
        their own config instead."""
        return str(_REPO_ROOT)

    # ------------------------------------------------------------------ #
    # Helpers shared by every adapter.
    # ------------------------------------------------------------------ #
    @property
    def _resolved_model(self) -> str:
        """Model id with the ``litellm_proxy/`` prefix stripped."""
        return resolve_model(self._model_id)

    def _mcp_servers(self) -> list[McpServerSpec]:
        return [McpServerSpec(name, spec) for name, spec in self._server_paths.items()]

    @staticmethod
    def _ensure_turn(trajectory: Trajectory) -> TurnRecord:
        if not trajectory.turns:
            trajectory.turns.append(TurnRecord(index=0, text=""))
        return trajectory.turns[-1]

    @staticmethod
    def _add_tool_call(
        trajectory: Trajectory,
        *,
        name: str,
        input: dict,
        id: str = "",
        output: object = None,
    ) -> None:
        CliCodingAgentRunner._ensure_turn(trajectory).tool_calls.append(
            ToolCall(name=name, input=input, id=id, output=output)
        )

    @staticmethod
    def _add_message(trajectory: Trajectory, text: str) -> None:
        trajectory.turns.append(TurnRecord(index=len(trajectory.turns), text=text))

    # ------------------------------------------------------------------ #
    # The lifecycle (install is implicit: CLI assumed on PATH).
    # ------------------------------------------------------------------ #
    async def run(self, question: str) -> AgentResult:
        base_url = os.environ.get("LITELLM_BASE_URL", "")
        api_key = os.environ.get("LITELLM_API_KEY", "")
        if self._model_id.startswith(LITELLM_PREFIX) and (not base_url or not api_key):
            raise ValueError(
                "LITELLM_BASE_URL and LITELLM_API_KEY must be set "
                f"when using {LITELLM_PREFIX!r} model prefix"
            )

        with agent_run_span(
            self.agent_name, model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()
            trajectory = Trajectory()
            trajectory.started_at = started_at
            answer = ""

            with tempfile.TemporaryDirectory(prefix=f"{self.agent_name}-home-") as home_s:
                home = Path(home_s)
                extra_env = self._write_config(home, base_url)

                env = dict(os.environ)
                env.update(extra_env)

                args = self._build_command(home, AGENT_SYSTEM_PROMPT, question)
                _log.info(
                    "%s: starting query (model=%s, servers=%d)",
                    self.agent_name,
                    self._resolved_model,
                    len(self._server_paths),
                )

                proc = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=self._cwd(home),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                async def _consume() -> str:
                    final = ""
                    assert proc.stdout is not None
                    async for raw in proc.stdout:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # plain log line; agents interleave these
                        if not isinstance(event, dict):
                            continue
                        got = self._handle_event(event, trajectory)
                        if got:
                            final = got
                    return final

                try:
                    answer = await asyncio.wait_for(_consume(), timeout=self._timeout_s)
                    await proc.wait()
                except asyncio.TimeoutError:
                    proc.kill()
                    _log.warning("%s: timed out after %ss", self.agent_name, self._timeout_s)

                if proc.returncode not in (0, None) and not answer:
                    stderr = (
                        (await proc.stderr.read()).decode("utf-8", "replace")
                        if proc.stderr
                        else ""
                    )
                    _log.error(
                        "%s: exit %s, stderr: %s",
                        self.agent_name,
                        proc.returncode,
                        stderr[-2000:],
                    )

            _log.info(
                "%s: done (turns=%d, input_tokens=%d, output_tokens=%d)",
                self.agent_name,
                len(trajectory.turns),
                trajectory.total_input_tokens,
                trajectory.total_output_tokens,
            )

            span.set_attribute("agent.answer.length", len(answer))
            span.set_attribute("gen_ai.usage.input_tokens", trajectory.total_input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", trajectory.total_output_tokens)
            span.set_attribute("agent.turns", len(trajectory.turns))
            span.set_attribute("agent.tool_calls", len(trajectory.all_tool_calls))
            span.set_attribute(
                "agent.duration_ms", (time.perf_counter() - run_started) * 1000
            )
            persist_trajectory(
                runner_name=self.agent_name,
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
