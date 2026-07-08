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
import re
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
shell commands, browse the web, or inspect local files unless those
capabilities have been enabled for this run.

When file or bash access is enabled, use the current working directory as the
run workspace. Write any scripts, temporary files, intermediate data, and final
artifacts there. Do not read or write files outside the current workspace.
Do not inspect parent directories, repository folders, reports, traces,
groundtruth files, previous agent outputs, or hidden evaluation artifacts.
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
    allow_files: bool = False,
) -> dict[str, Any]:
    """Build non-interactive permissions for benchmark-safe OpenCode runs."""
    permission: dict[str, Any] = {
        "read": "allow" if allow_files else "deny",
        "glob": "allow" if allow_files else "deny",
        "grep": "allow" if allow_files else "deny",
        "lsp": "allow" if allow_files else "deny",
        "edit": "allow" if allow_edit else "deny",
        "bash": "allow" if allow_bash else "deny",
        "task": "deny",
        "skill": "deny",
        "question": "deny",
        "webfetch": "allow" if allow_web else "deny",
        "websearch": "allow" if allow_web else "deny",
        "external_directory": "deny",
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
    as ``litellm_proxy/`` and ``tokenrouter/``, declare a custom provider and
    register the requested model explicitly. TokenRouter Claude models need the
    Anthropic protocol so OpenCode preserves native Anthropic message handling.
    """
    creds = resolve_router_creds(model_id, strict=True)
    if creds is None:
        return model_id, {}, {}

    provider_id = creds.prefix.rstrip("/").replace("_", "-")
    provider_name = "TokenRouter" if provider_id == "tokenrouter" else "LiteLLM Proxy"
    model_name = resolve_model(model_id)
    opencode_model = f"{provider_id}/{model_name}"
    provider_npm = (
        "@ai-sdk/anthropic"
        if provider_id == "tokenrouter" and model_name.startswith("anthropic/")
        else "@ai-sdk/openai-compatible"
    )
    provider = {
        provider_id: {
            "npm": provider_npm,
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
    allow_files: bool = False,
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Return (OpenCode config, env overrides, resolved OpenCode model)."""
    opencode_model, provider, env = _resolve_opencode_model_and_provider(model)
    permission = _build_permissions(
        list(server_paths),
        allow_bash=allow_bash,
        allow_edit=allow_edit,
        allow_web=allow_web,
        allow_files=allow_files,
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


def _resolve_run_dir(
    *,
    workspace_dir: Path | str | None = None,
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_files: bool = False,
) -> Path:
    """Return OpenCode's working directory for this run.

    The safe default is the repo root with local file/bash/edit tools denied.
    Any filesystem/code capability must opt into a dedicated workspace folder.
    This keeps the default comparable to tools-only MCP agents while allowing a
    separate CLI/code-capable track.
    """
    workspace_requested = allow_bash or allow_edit or allow_files
    if workspace_requested and workspace_dir is None:
        raise ValueError(
            "--workspace-dir is required when enabling files, edits, or bash"
        )
    if workspace_dir is None:
        return _REPO_ROOT

    run_dir = Path(workspace_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def _token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _opencode_usage_tokens(tokens: dict[str, Any]) -> tuple[int, int]:
    """Return input/output totals from OpenCode's step-finish token schema."""
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    input_tokens = (
        _token_count(tokens.get("input"))
        + _token_count(cache.get("read"))
        + _token_count(cache.get("write"))
    )
    output_tokens = _token_count(tokens.get("output")) + _token_count(
        tokens.get("reasoning")
    )
    return input_tokens, output_tokens


def _usage_from_events(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Extract token usage from OpenCode events.

    OpenCode emits per-step usage as ``tokens.input`` / ``tokens.output`` plus
    cache and reasoning buckets. Older/test fixtures may use SDK-style usage
    names, which are treated as possibly cumulative and deduplicated by max.
    """
    input_tokens = 0
    output_tokens = 0
    sdk_input_tokens = 0
    sdk_output_tokens = 0
    for event in events:
        for item in _walk_dicts(event):
            tokens = item.get("tokens")
            if isinstance(tokens, dict):
                in_value, out_value = _opencode_usage_tokens(tokens)
                input_tokens += in_value
                output_tokens += out_value
                continue

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
            sdk_input_tokens = max(sdk_input_tokens, _token_count(in_value))
            sdk_output_tokens = max(sdk_output_tokens, _token_count(out_value))
    return input_tokens or sdk_input_tokens, output_tokens or sdk_output_tokens


_REASONING_HINTS = ("reasoning", "thinking", "thought")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _visible_text(text: str) -> str:
    """Remove optional model thinking markup from user-facing text."""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text.lower() and "<think>" not in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def _is_answer_text(part_type: str, part: dict[str, Any]) -> bool:
    """Whether a part contributes to the user-facing answer."""
    if any(hint in part_type for hint in _REASONING_HINTS):
        return False
    if "text" in part_type:
        return True
    if not part_type and part.get("role") == "assistant":
        return True
    return False


def _merge_text(existing: str, new: str) -> str:
    """Combine repeated emissions for one text part id.

    OpenCode streams can look like snapshots (new value contains the previous
    value) or deltas (new value is only the next chunk). Handle both shapes.
    """
    if not existing:
        return new
    if new.startswith(existing):
        return new
    if existing.startswith(new):
        return existing
    return existing + new


def _is_step_finish(
    event: dict[str, Any], part: dict[str, Any], part_type: str
) -> bool:
    """True for OpenCode step-finish boundaries."""
    event_type = str(event.get("type") or "").lower()
    return ("step" in part_type and "finish" in part_type) or (
        "step" in event_type and "finish" in event_type
    )


def _final_answer(
    text_by_part: OrderedDict[str, str], msg_by_part: dict[str, str]
) -> str:
    """Return text from the final assistant message only."""
    if not text_by_part:
        return ""
    last_msg = None
    for part_id in text_by_part:
        last_msg = msg_by_part.get(part_id, part_id)
    answer = "".join(
        text
        for part_id, text in text_by_part.items()
        if msg_by_part.get(part_id, part_id) == last_msg
    )
    return _visible_text(answer)


def _build_trajectory_from_events(
    events: list[dict[str, Any]],
    plain_lines: list[str],
    *,
    duration_ms: float | None = None,
    stderr: str = "",
) -> tuple[str, OpenCodeTrajectory]:
    """Convert OpenCode events into the shared SDK-style trajectory shape.

    Text/tool parts are merged by part id, turns are split on OpenCode
    step-finish boundaries, and the canonical answer is scoped to the final
    assistant message instead of concatenating all intermediate narration.
    """
    text_by_part: OrderedDict[str, str] = OrderedDict()
    msg_by_part: dict[str, str] = {}
    tool_calls: OrderedDict[str, ToolCall] = OrderedDict()

    steps: list[dict[str, Any]] = []
    pending_text: list[str] = []
    pending_tools: list[str] = []

    def _close_step(input_tokens: int, output_tokens: int) -> None:
        if not (pending_text or pending_tools or input_tokens or output_tokens):
            return
        steps.append(
            {
                "text_ids": list(pending_text),
                "tool_ids": list(pending_tools),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        pending_text.clear()
        pending_tools.clear()

    for index, event in enumerate(events):
        part = _candidate_part(event)
        if part is None:
            part = event

        part_type = str(part.get("type") or part.get("kind") or "").lower()
        part_id = str(
            part.get("id")
            or part.get("partID")
            or part.get("messageID")
            or f"event_{index}"
        )

        if _is_step_finish(event, part, part_type):
            step_in, step_out = _usage_from_events([event])
            _close_step(step_in, step_out)
            continue

        text_value = part.get("text") or part.get("content")
        if isinstance(text_value, str) and _is_answer_text(part_type, part):
            text_by_part[part_id] = _merge_text(
                text_by_part.get(part_id, ""), text_value
            )
            msg_by_part[part_id] = str(
                part.get("messageID") or part.get("messageId") or part_id
            )
            if part_id not in pending_text:
                pending_text.append(part_id)

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
            if part_id not in pending_tools:
                pending_tools.append(part_id)

    _close_step(0, 0)

    answer = _final_answer(text_by_part, msg_by_part)
    if not answer and plain_lines:
        answer = _visible_text("\n".join(plain_lines))

    total_input, total_output = _usage_from_events(events)
    trajectory = OpenCodeTrajectory(raw_events=events, stderr=stderr)

    if steps:
        for index, step in enumerate(steps):
            turn_text = _visible_text(
                "".join(text_by_part.get(tid, "") for tid in step["text_ids"])
            )
            turn_tools = [
                tool_calls[tid] for tid in step["tool_ids"] if tid in tool_calls
            ]
            trajectory.turns.append(
                TurnRecord(
                    index=index,
                    text=turn_text,
                    tool_calls=turn_tools,
                    input_tokens=step["input_tokens"],
                    output_tokens=step["output_tokens"],
                )
            )
        if sum(
            step["input_tokens"] + step["output_tokens"] for step in steps
        ) == 0 and (total_input or total_output):
            trajectory.turns[-1].input_tokens = total_input
            trajectory.turns[-1].output_tokens = total_output
        trajectory.turns[-1].duration_ms = duration_ms
    elif answer or tool_calls or total_input or total_output:
        trajectory.turns.append(
            TurnRecord(
                index=0,
                text=answer,
                tool_calls=list(tool_calls.values()),
                input_tokens=total_input,
                output_tokens=total_output,
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
        thinking: bool = False,
        variant: str | None = None,
        allow_bash: bool = False,
        allow_edit: bool = False,
        allow_web: bool = False,
        allow_files: bool = False,
        workspace_dir: Path | str | None = None,
        dangerously_skip_permissions: bool = True,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._max_steps = max_steps
        self._agent_name = agent_name
        self._opencode_bin = opencode_bin
        self._attach = attach
        self._timeout_s = timeout_s
        self._thinking = thinking
        self._variant = variant
        self._dangerously_skip_permissions = dangerously_skip_permissions
        self._run_dir = _resolve_run_dir(
            workspace_dir=workspace_dir,
            allow_bash=allow_bash,
            allow_edit=allow_edit,
            allow_files=allow_files,
        )
        self._config, self._env_overrides, self._opencode_model = (
            _build_opencode_config(
                model=model,
                agent_name=agent_name,
                max_steps=max_steps,
                server_paths=self._server_paths,
                allow_bash=allow_bash,
                allow_edit=allow_edit,
                allow_web=allow_web,
                allow_files=allow_files,
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
                str(self._run_dir),
                "--title",
                "AssetOpsBench",
            ]
            if self._attach:
                cmd.extend(["--attach", self._attach])
            if self._variant:
                cmd.extend(["--variant", self._variant])
            if self._thinking:
                cmd.append("--thinking")
            if self._dangerously_skip_permissions:
                cmd.append("--dangerously-skip-permissions")
            cmd.append(question)

            env = os.environ.copy()
            # The OpenCode subprocess should not expose host-side evaluation
            # output paths to file/bash tools. The Python wrapper persists the
            # trajectory after OpenCode exits, using the parent process env.
            env.pop("AGENT_TRAJECTORY_DIR", None)
            env.pop("SCENARIOS_DATA_DIR", None)
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
                cwd=str(self._run_dir),
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
