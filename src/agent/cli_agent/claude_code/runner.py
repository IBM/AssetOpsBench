"""Claude Code CLI adapter.

Ported from ``ale_run/agents/claude_code`` and kept close to that deployer's
launch shape: ``claude -p - --output-format stream-json --verbose
--mcp-config <f> --model <m> --dangerously-skip-permissions`` with the prompt
(system prompt folded in) fed via **stdin**.  The stream-json events are folded
into AssetOpsBench's :class:`Trajectory`.

LiteLLM routing: Claude Code talks to an Anthropic-compatible endpoint, so we
point it at the proxy with ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN``.
(The proxy must expose the Anthropic Messages API — e.g. LiteLLM's ``/anthropic``
route — for the chosen model.)

stream-json events:
    {"type":"assistant","message":{"content":[{type:text|tool_use,...}], usage}}
    {"type":"user","message":{"content":[{type:tool_result,...}]}}
    {"type":"result","subtype":"success","result":"<final>","usage":{...}}
"""

from __future__ import annotations

import json
from pathlib import Path

from ...models import Trajectory
from .._providers import ResolvedProvider
from ..base import CliCodingAgentRunner


class ClaudeCodeRunner(CliCodingAgentRunner):
    agent_name = "claude-code"
    default_model = "litellm_proxy/aws/claude-opus-4-6"
    # Claude Code speaks the Anthropic Messages API, so it needs an
    # Anthropic-compatible endpoint (LiteLLM /anthropic, a suitably-configured
    # TokenRouter, or direct Anthropic). OpenRouter is OpenAI-format only.
    supported_providers = frozenset({"litellm", "tokenrouter", "direct"})

    def __init__(self, *args, claude_bin: str = "claude", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._claude_bin = claude_bin
        self._tool_use_ids: dict[str, object] = {}

    def _write_config(self, home: Path, provider: ResolvedProvider) -> dict[str, str]:
        servers = {
            s.name: {"command": s.command, "args": s.args}
            for s in self._mcp_servers()
        }
        (home / "mcp_config.json").write_text(json.dumps({"mcpServers": servers}, indent=2))
        self._tool_use_ids.clear()  # reset per run (runner may be reused)

        env = {"CLAUDE_CONFIG_DIR": str(home)}
        if provider.base_url:
            # Route Claude Code's Anthropic client at the proxy; the proxy key
            # is the bearer token (ALE's openrouter dance). Clear
            # ANTHROPIC_API_KEY so the AUTH_TOKEN path is used.
            env.update(
                {
                    "ANTHROPIC_BASE_URL": provider.base_url.rstrip("/"),
                    "ANTHROPIC_AUTH_TOKEN": provider.api_key,
                    "ANTHROPIC_API_KEY": "",
                }
            )
        # direct: leave the user's native ANTHROPIC_API_KEY in place.
        return env

    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        # Prompt comes from stdin via `-p -` (see _stdin_text).
        return [
            self._claude_bin,
            "-p",
            "-",
            "--output-format",
            "stream-json",
            "--verbose",
            "--mcp-config",
            str(home / "mcp_config.json"),
            "--model",
            self._resolved_model,
            "--dangerously-skip-permissions",
        ]

    def _stdin_text(self, system_prompt: str, question: str) -> str:
        return f"{system_prompt}\n\n---\n\nUser question: {question}"

    def _handle_event(self, event: dict, trajectory: Trajectory) -> str | None:
        etype = event.get("type", "")
        answer: str | None = None

        if etype == "assistant":
            msg = event.get("message", {}) or {}
            text_parts: list[str] = []
            for block in msg.get("content", []) or []:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tc_id = block.get("id", "") or ""
                    self._add_tool_call(
                        trajectory,
                        name=block.get("name", ""),
                        input=block.get("input", {}) or {},
                        id=tc_id,
                    )
                    if tc_id and trajectory.turns:
                        self._tool_use_ids[tc_id] = trajectory.turns[-1].tool_calls[-1]
            if text_parts:
                self._add_message(trajectory, "".join(text_parts))
            usage = msg.get("usage", {}) or {}
            if usage and trajectory.turns:
                trajectory.turns[-1].input_tokens = int(usage.get("input_tokens", 0) or 0)
                trajectory.turns[-1].output_tokens = int(usage.get("output_tokens", 0) or 0)

        elif etype == "user":
            msg = event.get("message", {}) or {}
            for block in msg.get("content", []) or []:
                if block.get("type") == "tool_result":
                    call = self._tool_use_ids.get(block.get("tool_use_id", ""))
                    if call is not None:
                        call.output = block.get("content")

        elif etype == "result":
            result_text = event.get("result", "") or ""
            if result_text.strip():
                answer = result_text
            usage = event.get("usage", {}) or {}
            if usage and trajectory.turns:
                trajectory.turns[-1].input_tokens = (
                    trajectory.turns[-1].input_tokens or int(usage.get("input_tokens", 0) or 0)
                )
                trajectory.turns[-1].output_tokens = (
                    trajectory.turns[-1].output_tokens or int(usage.get("output_tokens", 0) or 0)
                )

        return answer
