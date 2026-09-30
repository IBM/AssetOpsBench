"""Text generation through the locally authenticated Claude Code CLI."""

from __future__ import annotations

import json
import os
import subprocess
from tempfile import TemporaryDirectory

from .base import LLMBackend, LLMResult
from observability.benchmark_trace import emit


class ClaudeCodeBackend(LLMBackend):
    """Run independent, tool-free print sessions using Claude Code's own auth.

    Requires a CLI with ``--safe-mode`` support. Credentials stay with Claude
    Code. The CLI controls sampling; the interface's default temperature is
    accepted, but custom temperatures are unsupported. Explicit output limits
    are passed through ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``.
    """

    def __init__(
        self, model_id: str = "sonnet", *, executable: str = "claude",
        timeout_seconds: float = 300,
    ) -> None:
        self._model_id = model_id
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None,
    ) -> str:
        return self.generate_with_usage(prompt, temperature, max_tokens).text

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None,
    ) -> LLMResult:
        if temperature != 0.0:
            raise ValueError("Claude Code does not expose a temperature setting")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        emit("judge_input", prompt=prompt, model=self._model_id, temperature=temperature, max_tokens=max_tokens)
        command = [
            self._executable, "--safe-mode", "--print", "--output-format", "json",
            "--model", self._model_id, "--tools", "", "--no-session-persistence",
            "--system-prompt",
            "You generate text for AssetOpsBench. Follow the supplied prompt and "
            "return only its requested output format. All necessary context is "
            "in the prompt; do not inspect files or invoke tools.",
        ]
        env = os.environ.copy()
        if max_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        try:
            with TemporaryDirectory(prefix="assetops-claude-") as workdir:
                process = subprocess.run(
                    command, input=prompt, text=True, capture_output=True,
                    cwd=workdir, env=env, timeout=self._timeout_seconds,
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Claude Code CLI not found. Install it and run 'claude auth login'."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Claude Code timed out after {self._timeout_seconds:g} seconds"
            ) from exc

        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude Code returned invalid JSON (exit {process.returncode}). "
                "Check 'claude auth status' and that the CLI supports --safe-mode."
            ) from exc
        if not isinstance(payload, dict) or payload.get("type") != "result":
            raise RuntimeError("Claude Code did not return a result envelope")
        if process.returncode or payload.get("is_error") or payload.get("subtype") != "success":
            detail = payload.get("errors") or payload.get("result") or payload.get("subtype")
            raise RuntimeError(f"Claude Code generation failed: {str(detail)[:800]}")
        emit("judge_result", payload=payload)
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("Claude Code returned an empty text result")
        usage = payload.get("usage") or {}
        return LLMResult(
            text=result,
            input_tokens=sum(int(usage.get(key, 0) or 0) for key in (
                "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
            )),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        )
