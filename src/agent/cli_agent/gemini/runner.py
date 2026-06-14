"""Gemini CLI adapter.

Ported from ``ale_run/agents/gemini_cli`` and kept close to that deployer's
launch shape: ``gemini -p - --model <m> --output-format stream-json
--approval-mode yolo`` with the prompt fed via **stdin**.  The stream-json
NDJSON events are folded into AssetOpsBench's :class:`Trajectory`.

Config: Gemini discovers ``mcpServers`` from a project ``.gemini/settings.json``
in its working directory, so we run Gemini from a throwaway home dir
(overriding :meth:`_cwd`) and write the settings there, pinning each MCP
server's ``cwd`` to the repo root.

LiteLLM routing — IMPORTANT: Gemini CLI does not speak the OpenAI wire format.
It is pointed at a Gemini-API-compatible endpoint via ``GOOGLE_GEMINI_BASE_URL``
(LiteLLM exposes a ``/gemini`` passthrough). OpenRouter is OpenAI-only, so it is
excluded. Verify the env-var name and passthrough path for your versions.

stream-json events:
    {"type":"message","role":"assistant","messageType":"text","content":"..."}
    {"type":"tool_use","tool_name":"...","tool_id":"...","parameters":{...}}
    {"type":"tool_result","tool_id":"...","output":"...","error":...}
    {"type":"result","response":"<final>","stats":{...}}
"""

from __future__ import annotations

import json
from pathlib import Path

from ...models import Trajectory
from .._providers import ResolvedProvider
from ..base import CliCodingAgentRunner


class GeminiCliRunner(CliCodingAgentRunner):
    agent_name = "gemini-cli"
    default_model = "gemini-2.5-pro"
    # Gemini CLI speaks the Gemini API, so it needs a Gemini-compatible endpoint
    # (LiteLLM's /gemini passthrough, a configured TokenRouter, or direct Google
    # auth). OpenRouter is OpenAI-format only, so it's excluded.
    supported_providers = frozenset({"litellm", "tokenrouter", "direct"})

    def __init__(self, *args, gemini_bin: str = "gemini", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gemini_bin = gemini_bin
        self._tool_ids: dict[str, object] = {}

    def _cwd(self, home: Path) -> str:
        # Run from the config home so Gemini discovers .gemini/settings.json
        # there instead of in the repo.
        return str(home)

    def _write_config(self, home: Path, provider: ResolvedProvider) -> dict[str, str]:
        gem_dir = home / ".gemini"
        gem_dir.mkdir(parents=True, exist_ok=True)
        servers = {
            s.name: {"command": s.command, "args": s.args, "cwd": s.cwd, "timeout": 60000}
            for s in self._mcp_servers()
        }
        (gem_dir / "settings.json").write_text(
            json.dumps({"mcpServers": servers}, indent=2)
        )
        self._tool_ids.clear()

        env = {"NO_BROWSER": "1"}
        if provider.base_url:
            # LiteLLM serves a Gemini-compatible API under "<base>/gemini";
            # other proxies are assumed already Gemini-rooted.
            base = provider.base_url.rstrip("/")
            if provider.name == "litellm":
                base = f"{base}/gemini"
            env["GOOGLE_GEMINI_BASE_URL"] = base
            env["GEMINI_API_KEY"] = provider.api_key
        # direct: leave the user's native GEMINI_API_KEY in place.
        return env

    def _build_command(self, home: Path, system_prompt: str, question: str) -> list[str]:
        # Prompt comes from stdin via `-p -` (see _stdin_text).
        return [
            self._gemini_bin,
            "-p",
            "-",
            "--model",
            self._resolved_model,
            "--output-format",
            "stream-json",
            "--approval-mode",
            "yolo",
        ]

    def _stdin_text(self, system_prompt: str, question: str) -> str:
        return f"{system_prompt}\n\n---\n\nUser question: {question}"

    def _handle_event(self, event: dict, trajectory: Trajectory) -> str | None:
        etype = event.get("type", "")
        answer: str | None = None

        if etype == "message":
            if event.get("delta"):
                return None
            if event.get("role") == "assistant" and event.get("messageType") != "thinking":
                text = event.get("content", "") or ""
                if text:
                    self._add_message(trajectory, text)
                    answer = text

        elif etype == "tool_use":
            params = event.get("parameters", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {"raw": params}
            tc_id = event.get("tool_id", "") or ""
            self._add_tool_call(
                trajectory,
                name=event.get("tool_name", "") or "",
                input=params if isinstance(params, dict) else {"raw": params},
                id=tc_id,
            )
            if tc_id and trajectory.turns:
                self._tool_ids[tc_id] = trajectory.turns[-1].tool_calls[-1]

        elif etype == "tool_result":
            call = self._tool_ids.get(event.get("tool_id", ""))
            if call is not None:
                call.output = event.get("error") or event.get("output")

        elif etype == "result":
            response = event.get("response", "") or ""
            if response.strip():
                self._add_message(trajectory, response)
                answer = response
            stats = event.get("stats", {}) or {}
            self._apply_stats(stats, trajectory)

        return answer

    @staticmethod
    def _apply_stats(stats: dict, trajectory: Trajectory) -> None:
        """Best-effort token usage from the result event's stats."""
        if not isinstance(stats, dict) or not trajectory.turns:
            return

        def _toks(d: dict) -> tuple[int, int]:
            inp = d.get("input_tokens", d.get("inputTokens", d.get("input", 0)))
            out = d.get("output_tokens", d.get("outputTokens", d.get("candidates", 0)))
            return int(inp or 0), int(out or 0)

        models = stats.get("models", {})
        in_tot = out_tot = 0
        if isinstance(models, dict) and models:
            for m in models.values():
                if isinstance(m, dict):
                    src = m.get("tokens") if isinstance(m.get("tokens"), dict) else m
                    i, o = _toks(src)
                    in_tot += i
                    out_tot += o
        else:
            in_tot, out_tot = _toks(stats)

        if in_tot or out_tot:
            trajectory.turns[-1].input_tokens = in_tot
            trajectory.turns[-1].output_tokens = out_tot
