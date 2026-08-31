"""AgentRunner implementation backed by OpenClaw CLI.

OpenClaw is configured at runtime with AssetOpsBench MCP servers and run via
``openclaw agent --local --json``.  Router model IDs such as
``tokenrouter/MiniMax-M3`` are mapped onto OpenClaw's OpenAI provider with a
custom OpenAI-compatible base URL.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import shutil
import tempfile
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
_DEFAULT_MODEL = "tokenrouter/MiniMax-M3"
_DEFAULT_AGENT_NAME = "main"

_OPENCLAW_SYSTEM_PROMPT = (
    AGENT_SYSTEM_PROMPT
    + """

Use the configured AssetOpsBench MCP tools for operational data. Do not ask
the user follow-up questions during benchmark runs; make reasonable
assumptions and answer with the evidence you found. Do not edit files, run
shell commands, browse the web, or inspect local files unless those
capabilities have been enabled for this run.

When file or shell access is enabled, use the current working directory as the
run workspace. Write any scripts, temporary files, intermediate data, and final
artifacts there. Do not read or write files outside the current workspace.
"""
)

_BASE_DISABLED_TOOLS = (
    "web_search",
    "web_fetch",
    "image_generate",
    "video_generate",
    "music_generate",
    "memory_search",
    "sessions_yield",
    "sessions_list",
    "sessions_history",
    "sessions_spawn",
    "sessions_send",
    "cron",
)
_FILE_TOOLS = (
    "read",
    "write",
    "edit",
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
)
_SHELL_TOOLS = (
    "exec",
    "shell",
    "bash",
    "terminal",
)


@dataclass
class OpenClawCliTrajectory(Trajectory):
    """Trajectory plus raw OpenClaw events and captured stderr."""

    raw_events: list[dict[str, Any]] = field(default_factory=list)
    stderr: str = ""


def _build_mcp_config(
    server_paths: dict[str, Path | str],
    *,
    cwd: Path = _REPO_ROOT,
    timeout_s: int = 30,
) -> dict[str, dict[str, Any]]:
    """Convert AssetOpsBench MCP server specs into OpenClaw MCP config."""
    mcp: dict[str, dict[str, Any]] = {}
    for name, spec in server_paths.items():
        cmd_arg = str(spec) if isinstance(spec, Path) else spec
        mcp[name] = {
            "command": "uv",
            "args": ["run", cmd_arg],
            "cwd": str(cwd),
            "enabled": True,
            "timeout": timeout_s,
        }
    return mcp


def _disabled_tools(
    *,
    allow_files: bool = False,
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_web: bool = False,
) -> list[str]:
    """Return built-in OpenClaw tools to hide for the requested mode."""
    disabled = list(_BASE_DISABLED_TOOLS)
    if allow_web:
        disabled = [tool for tool in disabled if tool not in {"web_search", "web_fetch"}]
    if not allow_files:
        disabled.extend(_FILE_TOOLS)
    elif not allow_edit:
        disabled.extend(("write", "edit", "write_file"))
    if not allow_bash:
        disabled.extend(_SHELL_TOOLS)
    return sorted(dict.fromkeys(disabled))


def _resolve_openclaw_model_and_auth(
    model_id: str,
) -> tuple[str, str, str, str | None]:
    """Return ``(model_ref, provider, api_key, base_url)`` for OpenClaw.

    TokenRouter and LiteLLM router IDs are OpenAI-compatible gateways. OpenClaw
    can use them by routing through the ``openai`` provider with a provider
    ``baseUrl`` override and the router API key.
    """
    creds = resolve_router_creds(model_id, strict=True)
    if creds is not None:
        if creds.prefix not in ("tokenrouter/", "litellm_proxy/"):
            raise ValueError(f"Unsupported router prefix for OpenClaw: {creds.prefix}")
        model_name = resolve_model(model_id)
        return f"openai/{model_name}", "openai", creds.api_key, creds.base_url

    if model_id.startswith("openai/"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for openai/* models")
        return model_id, "openai", api_key, None

    if model_id.startswith("anthropic/"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set for anthropic/* models")
        return model_id, "anthropic", api_key, None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Unprefixed OpenClaw models use the OpenAI provider; set OPENAI_API_KEY "
            "or use tokenrouter/<model>."
        )
    return f"openai/{model_id}", "openai", api_key, None


def _bare_model_id(model_ref: str, provider: str) -> str:
    head, sep, tail = model_ref.partition("/")
    return tail if sep and head == provider else model_ref


def _build_openclaw_config(
    *,
    model_ref: str,
    provider: str,
    base_url: str | None,
    server_paths: dict[str, Path | str],
    timeout_s: float | None,
    allow_files: bool = False,
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_web: bool = False,
) -> dict[str, Any]:
    """Build ``openclaw.json`` for an isolated OpenClaw home."""
    disabled = _disabled_tools(
        allow_files=allow_files,
        allow_bash=allow_bash,
        allow_edit=allow_edit,
        allow_web=allow_web,
    )
    config: dict[str, Any] = {
        "agents": {
            "defaults": {
                "model": {"primary": model_ref},
                "timeoutSeconds": int(timeout_s or 900),
                "models": {model_ref: {}},
            }
        },
        "plugins": {
            "allow": [provider, "memory-core"],
            "deny": [],
        },
        "tools": {
            "deny": disabled,
            "exec": {
                "host": "gateway",
                "security": "full",
                "ask": "off",
            },
        },
        "gateway": {
            "mode": "local",
            "bind": "loopback",
        },
        "mcp": {
            "servers": _build_mcp_config(server_paths),
        },
    }
    if base_url:
        bare = _bare_model_id(model_ref, provider)
        config["models"] = {
            "providers": {
                provider: {
                    "baseUrl": base_url,
                    "models": [{"id": bare, "name": bare}],
                }
            }
        }
    return config


def _resolve_run_dir(
    *,
    workspace_dir: Path | str | None = None,
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_files: bool = False,
) -> Path:
    """Return OpenClaw's working directory for this run."""
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


def _stage_openclaw_home(
    *,
    home_root: Path,
    agent_name: str,
    config: dict[str, Any],
    provider: str,
    api_key: str,
) -> Path:
    """Create an isolated HOME containing OpenClaw config files."""
    oc_home = home_root / ".openclaw"
    oc_home.mkdir(parents=True, exist_ok=True)
    (oc_home / "openclaw.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (oc_home / ".env").write_text("OPENCLAW_RAW_STREAM=0\n", encoding="utf-8")

    agent_dir = oc_home / "agents" / agent_name / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    auth_profiles = {
        "version": 1,
        "profiles": {
            f"{provider}:default": {
                "provider": provider,
                "type": "api_key",
                "key": api_key,
            }
        },
        "lastGood": {
            provider: f"{provider}:default",
        },
    }
    (agent_dir / "auth-profiles.json").write_text(
        json.dumps(auth_profiles, indent=2), encoding="utf-8"
    )

    approvals = {
        "version": 1,
        "defaults": {
            "security": "full",
            "ask": "off",
            "askFallback": "full",
        },
        "socket": {},
        "agents": {},
    }
    (oc_home / "exec-approvals.json").write_text(
        json.dumps(approvals, indent=2), encoding="utf-8"
    )

    workspace = oc_home / "workspace"
    state_dir = workspace / ".openclaw"
    state_dir.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.UTC).isoformat()
    (state_dir / "workspace-state.json").write_text(
        json.dumps(
            {"version": 1, "bootstrapSeededAt": now, "setupCompletedAt": now},
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in (
        workspace / "MEMORY.md",
        workspace / "memory" / f"{_dt.datetime.now().date()}.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    bootstrap = workspace / "BOOTSTRAP.md"
    if bootstrap.exists():
        bootstrap.unlink()
    return home_root


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the OpenClaw ``--json`` envelope from stdout or stderr."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _answer_from_envelope(envelope: dict[str, Any]) -> str:
    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    answer = meta.get("finalAssistantVisibleText")
    if isinstance(answer, str):
        return answer.strip()

    payloads = envelope.get("payloads")
    if isinstance(payloads, list):
        parts = [
            payload.get("text", "")
            for payload in payloads
            if isinstance(payload, dict) and isinstance(payload.get("text"), str)
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _usage_from_envelope(envelope: dict[str, Any]) -> tuple[int, int]:
    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    agent_meta = (
        meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    )
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    try:
        return int(usage.get("input") or 0), int(usage.get("output") or 0)
    except (TypeError, ValueError):
        return 0, 0


def _session_id_from_envelope(envelope: dict[str, Any]) -> str | None:
    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    agent_meta = (
        meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    )
    session_id = agent_meta.get("sessionId")
    return session_id if isinstance(session_id, str) and session_id else None


def _tool_input(value: Any) -> dict[str, Any]:
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


def _parse_transcript(path: Path) -> tuple[list[dict[str, Any]], list[ToolCall]]:
    """Parse OpenClaw session JSONL enough to recover tool calls."""
    if not path.exists():
        return [], []

    events: list[dict[str, Any]] = []
    calls: OrderedDict[str, ToolCall] = OrderedDict()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)

        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content") or event.get("content") or []
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            content = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type in {"toolCall", "tool_call", "tool_use"}:
                call_id = str(
                    block.get("id")
                    or block.get("call_id")
                    or block.get("toolCallId")
                    or f"tool_{len(calls)}"
                )
                name = (
                    block.get("name")
                    or block.get("tool")
                    or block.get("toolName")
                    or block.get("function")
                    or ""
                )
                raw_input = (
                    block.get("input")
                    or block.get("args")
                    or block.get("arguments")
                    or block.get("params")
                )
                calls[call_id] = ToolCall(
                    id=call_id,
                    name=str(name),
                    input=_tool_input(raw_input),
                )
            elif block_type in {"tool_result", "toolResult"}:
                call_id = str(
                    block.get("call_id")
                    or block.get("toolCallId")
                    or block.get("id")
                    or ""
                )
                output = block.get("output", block.get("result", block.get("content")))
                if call_id in calls:
                    calls[call_id].output = output

        if event.get("type") == "tool_result":
            call_id = str(event.get("call_id") or event.get("toolCallId") or "")
            output = event.get("output", event.get("result"))
            if call_id in calls:
                calls[call_id].output = output

    return events, list(calls.values())


class OpenClawCliAgentRunner(AgentRunner):
    """Agent runner that delegates the agentic loop to ``openclaw agent``."""

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        agent_name: str = _DEFAULT_AGENT_NAME,
        openclaw_bin: str = "openclaw",
        timeout_s: float | None = 900,
        thinking: str = "off",
        allow_bash: bool = False,
        allow_edit: bool = False,
        allow_web: bool = False,
        allow_files: bool = False,
        workspace_dir: Path | str | None = None,
        home_dir: Path | str | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._agent_name = agent_name
        self._openclaw_bin = openclaw_bin
        self._timeout_s = timeout_s
        self._thinking = thinking
        self._run_dir = _resolve_run_dir(
            workspace_dir=workspace_dir,
            allow_bash=allow_bash,
            allow_edit=allow_edit,
            allow_files=allow_files,
        )
        (
            self._openclaw_model,
            self._provider,
            self._api_key,
            self._base_url,
        ) = _resolve_openclaw_model_and_auth(model)
        self._config = _build_openclaw_config(
            model_ref=self._openclaw_model,
            provider=self._provider,
            base_url=self._base_url,
            server_paths=self._server_paths,
            timeout_s=timeout_s,
            allow_files=allow_files,
            allow_bash=allow_bash,
            allow_edit=allow_edit,
            allow_web=allow_web,
        )
        if home_dir is None:
            if workspace_dir is None:
                root = Path(tempfile.mkdtemp(prefix="assetopsbench-openclaw-"))
            else:
                root = self._run_dir / ".openclaw_home"
        else:
            root = Path(home_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._home_root = _stage_openclaw_home(
            home_root=root,
            agent_name=agent_name,
            config=self._config,
            provider=self._provider,
            api_key=self._api_key,
        )

    def _build_prompt(self, question: str) -> str:
        return f"{_OPENCLAW_SYSTEM_PROMPT}\n\nQuestion:\n{question}\n"

    async def run(self, question: str) -> AgentResult:
        """Run OpenClaw CLI for *question* and return a benchmark result."""
        with agent_run_span(
            "openclaw-cli-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()
            prompt = self._build_prompt(question)

            cmd = [
                self._openclaw_bin,
                "agent",
                "--local",
                "--agent",
                self._agent_name,
                "--message",
                prompt,
                "--json",
                "--timeout",
                str(int(self._timeout_s or 900)),
                "--thinking",
                self._thinking,
                "--model",
                self._openclaw_model,
            ]

            env = os.environ.copy()
            env["HOME"] = str(self._home_root)
            if self._provider == "openai":
                env["OPENAI_API_KEY"] = self._api_key
            elif self._provider == "anthropic":
                env["ANTHROPIC_API_KEY"] = self._api_key
            env.setdefault("NO_COLOR", "1")
            env.setdefault("OPENCLAW_RAW_STREAM", "0")

            _log.info(
                "OpenClawCliAgentRunner: starting query "
                "(model=%s, openclaw_model=%s, provider=%s)",
                self._model_id,
                self._openclaw_model,
                self._provider,
            )
            if shutil.which(self._openclaw_bin) is None:
                raise RuntimeError(
                    f"OpenClaw CLI executable not found: {self._openclaw_bin!r}. "
                    "Install OpenClaw first, make sure it is on PATH, or pass "
                    "--openclaw-bin /path/to/openclaw."
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
                    f"openclaw CLI run timed out after {self._timeout_s} seconds"
                ) from None

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            envelope = _extract_json_object(stdout) or _extract_json_object(stderr)
            if proc.returncode != 0:
                raise RuntimeError(
                    "openclaw CLI run failed with exit code "
                    f"{proc.returncode}\nSTDERR:\n{stderr[-4000:]}\nSTDOUT:\n{stdout[-4000:]}"
                )
            if envelope is None:
                raise RuntimeError(
                    "openclaw CLI did not return a JSON envelope\n"
                    f"STDERR:\n{stderr[-4000:]}\nSTDOUT:\n{stdout[-4000:]}"
                )

            duration_ms = (time.perf_counter() - run_started) * 1000
            answer = _answer_from_envelope(envelope)
            input_tokens, output_tokens = _usage_from_envelope(envelope)

            raw_events: list[dict[str, Any]] = [envelope]
            tool_calls: list[ToolCall] = []
            session_id = _session_id_from_envelope(envelope)
            if session_id:
                transcript = (
                    self._home_root
                    / ".openclaw"
                    / "agents"
                    / self._agent_name
                    / "sessions"
                    / f"{session_id}.jsonl"
                )
                transcript_events, tool_calls = _parse_transcript(transcript)
                raw_events.extend(transcript_events)

            trajectory = OpenClawCliTrajectory(
                raw_events=raw_events,
                stderr=stderr,
                started_at=started_at,
            )
            trajectory.turns.append(
                TurnRecord(
                    index=0,
                    text=answer,
                    tool_calls=tool_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                )
            )

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
                runner_name="openclaw-cli-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
