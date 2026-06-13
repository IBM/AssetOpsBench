"""Gemini CLI adapter.

Ported from ``ale_run/agents/gemini_cli`` but pointed at AssetOpsBench's MCP
servers and folding Gemini's JSON output into AssetOpsBench's :class:`Trajectory`.

Config: Gemini discovers ``mcpServers`` from a project ``.gemini/settings.json``
in its working directory.  To keep the repo clean we run Gemini from a throwaway
home dir (overriding :meth:`_cwd`) and write the settings there, while pinning
each MCP server's ``cwd`` to the repo root so ``uv run <entry-point>`` resolves.

LiteLLM routing — IMPORTANT: Gemini CLI does not speak the OpenAI wire format,
so it can't use the same provider block as Codex.  It can be pointed at a
Gemini-API-compatible endpoint via ``GOOGLE_GEMINI_BASE_URL``; LiteLLM exposes a
``/gemini`` passthrough that fits.  Verify both the env-var name and the
passthrough path against your LiteLLM + Gemini CLI versions — this is the most
version-sensitive part of this adapter.

Output: ``--output-format json`` prints a final object (``{"response": ...,
"stats": {...}}``); newer builds also support a streamed event form with a
``type`` field.  Both are handled best-effort below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..models import Trajectory
from .base import CliCodingAgentRunner


class GeminiCliRunner(CliCodingAgentRunner):
    agent_name = "gemini-cli"
    default_model = "gemini-2.5-pro"

    def __init__(self, *args, gemini_bin: str = "gemini", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gemini_bin = gemini_bin

    def _cwd(self, home: Path) -> str:
        # Run from the config home so Gemini discovers .gemini/settings.json
        # there instead of in the repo.
        return str(home)

    def _write_config(self, home: Path, base_url: str) -> dict[str, str]:
        gem_dir = home / ".gemini"
        gem_dir.mkdir(parents=True, exist_ok=True)
        servers = {
            s.name: {"command": s.command, "args": s.args, "cwd": s.cwd, "timeout": 60000}
            for s in self._mcp_servers()
        }
        (gem_dir / "settings.json").write_text(
            json.dumps({"mcpServers": servers}, indent=2)
        )
        return {
            # Point Gemini CLI at the LiteLLM Gemini-compatible passthrough.
            "GOOGLE_GEMINI_BASE_URL": f"{base_url.rstrip('/')}/gemini",
            "GEMINI_API_KEY": os.environ.get("LITELLM_API_KEY", ""),
        }

    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        prompt = f"{system_prompt}\n\n---\n\nUser question: {question}"
        return [
            self._gemini_bin,
            "--prompt",
            prompt,
            "--model",
            self._resolved_model,
            "--yolo",  # auto-approve tool calls (headless)
            "--output-format",
            "json",
        ]

    def _handle_event(self, event: dict, trajectory: Trajectory) -> str | None:
        # Streamed event form (newer builds): dispatch on "type".
        etype = event.get("type", "")
        if etype:
            if etype in ("tool_call", "tool_use", "function_call"):
                self._add_tool_call(
                    trajectory,
                    name=event.get("name", "") or event.get("tool", ""),
                    input=event.get("args", event.get("input", {})) or {},
                    id=event.get("id", "") or "",
                    output=event.get("result", event.get("output")),
                )
                return None
            if etype in ("content", "assistant", "message"):
                text = event.get("text", "") or event.get("content", "") or ""
                if text:
                    self._add_message(trajectory, text)
                return text or None

        # Final single-object form: {"response": "...", "stats": {...}}.
        if "response" in event:
            text = event.get("response", "") or ""
            if text:
                self._add_message(trajectory, text)
            stats = event.get("stats", {}) or {}
            usage = stats.get("usage", stats) if isinstance(stats, dict) else {}
            if trajectory.turns:
                trajectory.turns[-1].input_tokens = int(
                    usage.get("input_tokens", usage.get("promptTokenCount", 0)) or 0
                )
                trajectory.turns[-1].output_tokens = int(
                    usage.get("output_tokens", usage.get("candidatesTokenCount", 0)) or 0
                )
            return text or None

        return None
