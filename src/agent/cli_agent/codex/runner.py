"""Codex CLI adapter.

Ported from ``ale_run/agents/codex`` and kept close to that deployer's launch
shape: ``codex exec --model <m> --json`` with the prompt fed via **stdin** and
headless ``--dangerously-bypass-approvals-and-sandbox``.  The NDJSON stream is
folded into AssetOpsBench's :class:`Trajectory`.

NDJSON event shapes (``codex exec --json``):
    thread.started | turn.started | turn.completed (usage) | item.started |
    item.completed | error
Item types: agent_message, reasoning, command_execution, mcp_tool_call,
file_change, web_search, error.  As in ALE, only ``item.completed`` is emitted
(``item.started`` is ignored) so tool calls are not double-counted.

Config is written to ``$CODEX_HOME/config.toml`` (throwaway home, so the
developer's real ``~/.codex`` is untouched). The model is passed on the CLI;
the proxy is declared as a custom ``[model_providers.<name>]`` block.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...models import Trajectory
from .._providers import ResolvedProvider
from ..base import CliCodingAgentRunner


class CodexCliRunner(CliCodingAgentRunner):
    agent_name = "codex"
    default_model = "litellm_proxy/azure/gpt-5.4"
    # Codex speaks the OpenAI wire format, so every OpenAI-compatible proxy
    # works (LiteLLM, OpenRouter, TokenRouter) plus direct OpenAI auth.
    supported_providers = frozenset({"litellm", "openrouter", "tokenrouter", "direct"})

    def __init__(self, *args, codex_bin: str = "codex", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._codex_bin = codex_bin

    def _write_config(self, home: Path, provider: ResolvedProvider) -> dict[str, str]:
        # Top-level keys must precede any [table]. Model is passed on the CLI
        # (ALE parity), so config carries only reasoning effort + provider.
        lines = ['model_reasoning_effort = "high"']
        if provider.base_url:
            lines.append(f'model_provider = "{provider.name}"')
        lines.append("")
        if provider.base_url:
            lines += [
                f"[model_providers.{provider.name}]",
                f'name = "{provider.name}"',
                f'base_url = "{provider.base_url.rstrip("/")}"',
                f'env_key = "{provider.api_key_env}"',
                "",
            ]
        # MCP servers inherit the agent's cwd (repo root), so `uv run <server>`
        # resolves without a per-server cwd key.
        for s in self._mcp_servers():
            arg_list = ", ".join(json.dumps(a) for a in s.args)
            lines += [
                f"[mcp_servers.{s.name}]",
                'command = "uv"',
                f"args = [{arg_list}]",
                "startup_timeout_sec = 60",
                "",
            ]
        (home / "config.toml").write_text("\n".join(lines))
        return {"CODEX_HOME": str(home)}

    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        # Prompt comes from stdin (see _stdin_text); no positional prompt.
        return [
            self._codex_bin,
            "exec",
            "--model",
            self._resolved_model,
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

    def _stdin_text(self, system_prompt: str, question: str) -> str:
        return f"{system_prompt}\n\n---\n\nUser question: {question}"

    def _handle_event(self, event: dict, trajectory: Trajectory) -> str | None:
        etype = event.get("type", "")
        answer: str | None = None

        # Only item.completed carries final data; item.started is ignored so
        # tool calls aren't counted twice (ALE parity).
        if etype == "item.completed":
            item = event.get("item", {}) or {}
            itype = item.get("type", "")
            item_id = item.get("id", "") or ""

            if itype == "agent_message":
                text = item.get("text", "") or ""
                self._add_message(trajectory, text)
                if text.strip():
                    answer = text

            elif itype == "mcp_tool_call":
                self._add_tool_call(
                    trajectory,
                    name=item.get("tool", "") or "",
                    input=item.get("arguments", {}) or {},
                    id=item_id,
                    output=item.get("result", item.get("error")),
                )

            elif itype == "command_execution":
                self._add_tool_call(
                    trajectory,
                    name="shell",
                    input={"command": item.get("command", "")},
                    id=item_id,
                    output=item.get("aggregated_output", ""),
                )

            elif itype == "web_search":
                self._add_tool_call(
                    trajectory,
                    name="web_search",
                    input={"query": item.get("query", "")},
                    id=item_id,
                )

        elif etype in ("turn.completed", "thread.completed"):
            usage = event.get("usage", {}) or {}
            if usage and trajectory.turns:
                trajectory.turns[-1].input_tokens = int(usage.get("input_tokens", 0) or 0)
                trajectory.turns[-1].output_tokens = int(usage.get("output_tokens", 0) or 0)

        return answer
