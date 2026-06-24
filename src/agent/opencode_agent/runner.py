"""AgentRunner implementation backed by the OpenCode CLI.

OpenCode is configured at runtime with the AssetOpsBench MCP servers and run
through ``opencode run --format json``.  This keeps it usable from the same
CLI/evaluator flow as the SDK-backed agents without requiring a checked-in
OpenCode project config.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.routers import resolve_model, resolve_router_creds
from observability import agent_run_span, persist_trajectory

from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = "opencode/gpt-5.1-codex"
_DEFAULT_AGENT_NAME = "assetops"

_OPENCODE_SYSTEM_PROMPT = (
    AGENT_SYSTEM_PROMPT
    + """

Use the configured AssetOpsBench MCP tools for operational data. Do not ask
the user follow-up questions during benchmark runs; make reasonable
assumptions and answer with the evidence you found. Do not edit files, run
shell commands, or browse the web unless those capabilities have been enabled
for this run.
"""
)


@dataclass
class OpenCodeTrajectory(Trajectory):
    """Trajectory plus raw OpenCode JSON events for debugging parser drift."""

    raw_events: list[dict[str, Any]] = field(default_factory=list)
    stderr: str = ""


def _build_mcp_config(
    server_paths: dict[str, Path | str],
    *,
    cwd: Path = _REPO_ROOT,
    timeout_ms: int = 30000,
) -> dict[str, dict[str, Any]]:
    """Convert AssetOpsBench MCP server specs into OpenCode local MCP config."""
    mcp: dict[str, dict[str, Any]] = {}
    for name, spec in server_paths.items():
        cmd_arg = str(spec) if isinstance(spec, Path) else spec
        mcp[name] = {
            "type": "local",
            "command": ["uv", "run", cmd_arg],
            "cwd": str(cwd),
            "enabled": True,
            "timeout": timeout_ms,
        }
    return mcp


def _build_permissions(
    server_names: list[str],
    *,
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_web: bool = False,
) -> dict[str, Any]:
    """Build non-interactive permissions for benchmark-safe OpenCode runs."""
    permission: dict[str, Any] = {
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "lsp": "allow",
        "edit": "allow" if allow_edit else "deny",
        "bash": "allow" if allow_bash else "deny",
        "task": "deny",
        "skill": "deny",
        "question": "deny",
        "webfetch": "allow" if allow_web else "deny",
        "websearch": "allow" if allow_web else "deny",
        "doom_loop": "deny",
    }
    for name in server_names:
        permission[f"{name}_*"] = "allow"
    return permission


def _resolve_opencode_model_and_provider(
    model_id: str,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Translate AssetOpsBench router model IDs into OpenCode config.

    OpenCode wants ``provider/model``.  For AssetOpsBench router prefixes such
    as ``litellm_proxy/`` and ``tokenrouter/``, declare a custom
    OpenAI-compatible provider and register the requested model explicitly.
    """
    creds = resolve_router_creds(model_id, strict=True)
    if creds is None:
        return model_id, {}, {}

    provider_id = creds.prefix.rstrip("/").replace("_", "-")
    provider_name = "TokenRouter" if provider_id == "tokenrouter" else "LiteLLM Proxy"
    model_name = resolve_model(model_id)
    opencode_model = f"{provider_id}/{model_name}"
    provider = {
        provider_id: {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider_name,
            "options": {
                "baseURL": creds.base_url,
                "apiKey": "{env:ASSETOPSBENCH_OPENCODE_API_KEY}",
            },
            "models": {
                model_name: {
                    "name": model_name,
                }
            },
        }
    }
    env = {
        "ASSETOPSBENCH_OPENCODE_API_KEY": creds.api_key,
    }
    return opencode_model, provider, env


def _build_opencode_config(
    *,
    model: str,
    agent_name: str,
    max_steps: int,
    server_paths: dict[str, Path | str],
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_web: bool = False,
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Return (OpenCode config, env overrides, resolved OpenCode model)."""
    opencode_model, provider, env = _resolve_opencode_model_and_provider(model)
    permission = _build_permissions(
        list(server_paths),
        allow_bash=allow_bash,
        allow_edit=allow_edit,
        allow_web=allow_web,
    )
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": opencode_model,
        "autoupdate": False,
        "mcp": _build_mcp_config(server_paths),
        "agent": {
            agent_name: {
                "description": "AssetOpsBench MCP benchmark agent",
                "mode": "primary",
                "model": opencode_model,
                "prompt": _OPENCODE_SYSTEM_PROMPT,
                "permission": permission,
                "steps": max_steps,
                "temperature": 0.1,
            }
        },
    }
    if provider:
        config["provider"] = provider
    return config, env, opencode_model


def _json_events(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse OpenCode's JSON-lines event stream, preserving non-JSON lines."""
    stripped = stdout.strip()
    if not stripped:
        return [], []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)], []
    if isinstance(parsed, dict):
        return [parsed], []

    events: list[dict[str, Any]] = []
    plain_lines: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            plain_lines.append(line)
            continue
        if isinstance(item, dict):
            events.append(item)
        else:
            plain_lines.append(line)
    return events, plain_lines


def _candidate_part(event: dict[str, Any]) -> dict[str, Any] | None:
    """Find the message/tool part inside common OpenCode event shapes."""
    for key in ("part", "messagePart"):
        value = event.get(key)
        if isinstance(value, dict):
            return value

    for container_key in ("properties", "data", "payload"):
        container = event.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("part", "messagePart"):
            value = container.get(key)
            if isinstance(value, dict):
                return value
    return None


def _coerce_tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if value is None:
        return {}
    return {"value": value}


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _usage_from_events(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Extract a conservative max token usage from possibly cumulative events."""
    input_tokens = 0
    output_tokens = 0
    for event in events:
        for item in _walk_dicts(event):
            in_value = (
                item.get("input_tokens")
                or item.get("inputTokens")
                or item.get("prompt_tokens")
                or item.get("promptTokens")
            )
            out_value = (
                item.get("output_tokens")
                or item.get("outputTokens")
                or item.get("completion_tokens")
                or item.get("completionTokens")
            )
            if isinstance(in_value, int):
                input_tokens = max(input_tokens, in_value)
            if isinstance(out_value, int):
                output_tokens = max(output_tokens, out_value)
    return input_tokens, output_tokens


def _build_trajectory_from_events(
    events: list[dict[str, Any]],
    plain_lines: list[str],
    *,
    duration_ms: float | None = None,
    stderr: str = "",
) -> tuple[str, OpenCodeTrajectory]:
    """Convert OpenCode events into the shared SDK-style trajectory shape."""
    text_parts: OrderedDict[str, str] = OrderedDict()
    tool_calls: OrderedDict[str, ToolCall] = OrderedDict()

    for index, event in enumerate(events):
        part = _candidate_part(event)
        if part is None:
            part = event

        part_type = str(part.get("type") or part.get("kind") or "")
        part_id = str(
            part.get("id")
            or part.get("partID")
            or part.get("messageID")
            or f"event_{index}"
        )

        text_value = part.get("text") or part.get("content")
        if isinstance(text_value, str) and (
            not part_type or "text" in part_type or part.get("role") == "assistant"
        ):
            text_parts[part_id] = text_value

        tool_name = (
            part.get("tool")
            or part.get("toolName")
            or part.get("name")
            or part.get("function")
        )
        if tool_name and (
            "tool" in part_type
            or any(key in part for key in ("input", "arguments", "args", "params"))
        ):
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            raw_input = (
                part.get("input")
                or part.get("arguments")
                or part.get("args")
                or part.get("params")
                or state.get("input")
            )
            output = (
                part.get("output")
                or part.get("result")
                or state.get("output")
                or state.get("result")
            )
            tool_calls[part_id] = ToolCall(
                name=str(tool_name),
                input=_coerce_tool_input(raw_input),
                id=part_id,
                output=output,
            )

    answer = "".join(text_parts.values()).strip()
    if not answer and plain_lines:
        answer = "\n".join(plain_lines).strip()

    input_tokens, output_tokens = _usage_from_events(events)
    trajectory = OpenCodeTrajectory(raw_events=events, stderr=stderr)
    if answer or tool_calls or input_tokens or output_tokens:
        trajectory.turns.append(
            TurnRecord(
                index=0,
                text=answer,
                tool_calls=list(tool_calls.values()),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
            )
        )
    return answer, trajectory


class OpenCodeAgentRunner(AgentRunner):
    """Agent runner that delegates the agentic loop to ``opencode run``."""

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str = _DEFAULT_MODEL,
        max_steps: int = 30,
        agent_name: str = _DEFAULT_AGENT_NAME,
        opencode_bin: str = "opencode",
        attach: str | None = None,
        timeout_s: float | None = 900,
        allow_bash: bool = False,
        allow_edit: bool = False,
        allow_web: bool = False,
        dangerously_skip_permissions: bool = True,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._max_steps = max_steps
        self._agent_name = agent_name
        self._opencode_bin = opencode_bin
        self._attach = attach
        self._timeout_s = timeout_s
        self._dangerously_skip_permissions = dangerously_skip_permissions
        self._config, self._env_overrides, self._opencode_model = (
            _build_opencode_config(
                model=model,
                agent_name=agent_name,
                max_steps=max_steps,
                server_paths=self._server_paths,
                allow_bash=allow_bash,
                allow_edit=allow_edit,
                allow_web=allow_web,
            )
        )

    async def run(self, question: str) -> AgentResult:
        """Run OpenCode for *question* and return a benchmark result."""
        with agent_run_span(
            "opencode-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()

            cmd = [
                self._opencode_bin,
                "run",
                "--pure",
                "--format",
                "json",
                "--model",
                self._opencode_model,
                "--agent",
                self._agent_name,
                "--dir",
                str(_REPO_ROOT),
                "--title",
                "AssetOpsBench",
            ]
            if self._attach:
                cmd.extend(["--attach", self._attach])
            if self._dangerously_skip_permissions:
                cmd.append("--dangerously-skip-permissions")
            cmd.append(question)

            env = os.environ.copy()
            env.update(self._env_overrides)
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(self._config)
            env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "true")
            env.setdefault("NO_COLOR", "1")

            _log.info(
                "OpenCodeAgentRunner: starting query (model=%s, opencode_model=%s)",
                self._model_id,
                self._opencode_model,
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(_REPO_ROOT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout_s
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                raise TimeoutError(
                    f"opencode run timed out after {self._timeout_s} seconds"
                ) from None

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(
                    "opencode run failed with exit code "
                    f"{proc.returncode}\nSTDERR:\n{stderr[-4000:]}\nSTDOUT:\n{stdout[-4000:]}"
                )

            duration_ms = (time.perf_counter() - run_started) * 1000
            events, plain_lines = _json_events(stdout)
            answer, trajectory = _build_trajectory_from_events(
                events,
                plain_lines,
                duration_ms=duration_ms,
                stderr=stderr,
            )
            trajectory.started_at = started_at

            span.set_attribute("agent.answer.length", len(answer))
            span.set_attribute(
                "gen_ai.usage.input_tokens", trajectory.total_input_tokens
            )
            span.set_attribute(
                "gen_ai.usage.output_tokens", trajectory.total_output_tokens
            )
            span.set_attribute("agent.turns", len(trajectory.turns))
            span.set_attribute("agent.tool_calls", len(trajectory.all_tool_calls))
            span.set_attribute("agent.duration_ms", duration_ms)
            persist_trajectory(
                runner_name="opencode-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
