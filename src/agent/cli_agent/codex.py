"""Codex CLI adapter.

Ported from ``ale_run/agents/codex`` but pointed at AssetOpsBench's MCP servers
and folding the NDJSON stream into AssetOpsBench's :class:`Trajectory`.

Reference (ALE codex README), NDJSON event shapes from ``codex exec --json``:
    thread.started | turn.started | turn.completed (usage) | item.started |
    item.completed | error
Item types: agent_message, reasoning, command_execution, mcp_tool_call,
file_change, web_search, error.

Config is written to ``$CODEX_HOME/config.toml``; we use a throwaway CODEX_HOME
so the benchmark never touches the developer's real ``~/.codex``.  The LiteLLM
proxy is declared as a custom ``[model_providers.litellm]`` block, mirroring how
ALE wires its ``openrouter`` provider.

Pin ``codex_version`` for reproducibility (ALE pins 0.114.0); install with
``npm install -g @openai/codex@<version>``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Trajectory
from .base import CliCodingAgentRunner, _REPO_ROOT


class CodexCliRunner(CliCodingAgentRunner):
    agent_name = "codex"
    default_model = "litellm_proxy/azure/gpt-5.4"

    def __init__(self, *args, codex_bin: str = "codex", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._codex_bin = codex_bin

    def _write_config(self, home: Path, base_url: str) -> dict[str, str]:
        lines = [
            f'model = "{self._resolved_model}"',
            'model_provider = "litellm"',
            'approval_policy = "never"',
            'model_reasoning_effort = "high"',
            "",
            "[model_providers.litellm]",
            'name = "LiteLLM Proxy"',
            f'base_url = "{base_url.rstrip("/")}"',
            'env_key = "LITELLM_API_KEY"',
            "",
        ]
        for s in self._mcp_servers():
            arg_list = ", ".join(json.dumps(a) for a in s.args)
            lines += [
                f"[mcp_servers.{s.name}]",
                'command = "uv"',
                f"args = [{arg_list}]",
                f"cwd = {json.dumps(s.cwd)}",
                "startup_timeout_sec = 60",
                "",
            ]
        (home / "config.toml").write_text("\n".join(lines))
        return {"CODEX_HOME": str(home)}

    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        prompt = f"{system_prompt}\n\n---\n\nUser question: {question}"
        return [
            self._codex_bin,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(_REPO_ROOT),
            prompt,
        ]

    def _handle_event(self, event: dict, trajectory: Trajectory) -> str | None:
        etype = event.get("type", "")
        item = event.get("item") or (event if etype.startswith("item") else None)
        answer: str | None = None

        if isinstance(item, dict):
            itype = item.get("type", "") or item.get("item_type", "")
            if itype == "mcp_tool_call" or "tool" in itype:
                raw_args = item.get("arguments", item.get("input", {})) or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        raw_args = {"raw": raw_args}
                name = item.get("tool") or item.get("name") or ""
                server = item.get("server", "")
                self._add_tool_call(
                    trajectory,
                    name=f"{server}.{name}" if server else name,
                    input=raw_args if isinstance(raw_args, dict) else {"raw": raw_args},
                    id=item.get("id", "") or item.get("call_id", "") or "",
                    output=item.get("result", item.get("output")),
                )
            elif itype == "agent_message":
                text = item.get("text", "") or item.get("content", "") or ""
                self._add_message(trajectory, text)
                if text.strip():
                    answer = text

        if etype in ("turn.completed", "thread.completed"):
            usage = event.get("usage", {}) or {}
            if usage and trajectory.turns:
                trajectory.turns[-1].input_tokens = int(usage.get("input_tokens", 0) or 0)
                trajectory.turns[-1].output_tokens = int(usage.get("output_tokens", 0) or 0)

        return answer
