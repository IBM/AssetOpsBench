"""Claude Code CLI adapter.

Ported from ``ale_run/agents/claude_code`` but pointed at AssetOpsBench's MCP
servers and folding the stream-json output into AssetOpsBench's
:class:`Trajectory`.

Config: MCP servers go in a ``.mcp.json`` passed via ``--mcp-config``.  The
servers run as stdio children inheriting the process ``cwd`` (the repo root),
so ``uv run <entry-point>`` resolves exactly like every other runner.

LiteLLM routing: Claude Code talks to an Anthropic-compatible endpoint, so we
point it at the proxy with ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN``.
(Your LiteLLM proxy must expose the Anthropic Messages API — e.g. the
``/anthropic`` passthrough — for the chosen model.)

Output: ``--output-format stream-json --verbose`` emits JSONL events:
    {"type":"assistant","message":{"content":[{type:text|tool_use,...}], usage}}
    {"type":"user","message":{"content":[{type:tool_result,...}]}}
    {"type":"result","subtype":"success","result":"<final>","usage":{...}}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..models import Trajectory
from .base import CliCodingAgentRunner


class ClaudeCodeRunner(CliCodingAgentRunner):
    agent_name = "claude-code"
    default_model = "litellm_proxy/aws/claude-opus-4-6"

    def __init__(self, *args, claude_bin: str = "claude", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._claude_bin = claude_bin
        self._tool_use_ids: dict[str, object] = {}

    def _write_config(self, home: Path, base_url: str) -> dict[str, str]:
        servers = {
            s.name: {"command": s.command, "args": s.args}
            for s in self._mcp_servers()
        }
        (home / ".mcp.json").write_text(json.dumps({"mcpServers": servers}, indent=2))
        # Route Claude Code's Anthropic client at the LiteLLM proxy. The proxy
        # key is the bearer token; clear ANTHROPIC_API_KEY so the AUTH_TOKEN
        # path is used.
        return {
            "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": os.environ.get("LITELLM_API_KEY", ""),
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CONFIG_DIR": str(home),
        }

    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        return [
            self._claude_bin,
            "-p",
            question,
            "--append-system-prompt",
            system_prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--mcp-config",
            str(home / ".mcp.json"),
            "--model",
            self._resolved_model,
        ]

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
            # Attach tool_result outputs back onto the matching tool call.
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
                # Final usage often summarises the whole run; record on last turn.
                trajectory.turns[-1].input_tokens = (
                    trajectory.turns[-1].input_tokens or int(usage.get("input_tokens", 0) or 0)
                )
                trajectory.turns[-1].output_tokens = (
                    trajectory.turns[-1].output_tokens or int(usage.get("output_tokens", 0) or 0)
                )

        return answer
