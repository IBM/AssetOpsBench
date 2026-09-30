"""Text generation through the locally authenticated Codex CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from .base import LLMBackend, LLMResult


_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "multi_agent", "apps",
    "plugins", "remote_plugin", "hooks", "memories", "browser_use",
    "computer_use", "image_generation", "tool_suggest", "goals", "sleep_tool",
    "workspace_dependencies",
)


class CodexBackend(LLMBackend):
    """Use independent Codex exec sessions without changing stored CLI config.

    Authentication stays with Codex. User config and host skills are skipped,
    and each call runs in an empty read-only workspace. The CLI controls
    sampling and output limits; ``max_tokens`` is an advisory response budget
    in the instructions, not an API-enforced token cap.
    """

    def __init__(
        self, model_id: str | None = None, *, executable: str = "codex",
        timeout_seconds: float = 300,
    ) -> None:
        self._model_id = model_id or "codex-default"
        self._requested_model = model_id
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
            raise ValueError("Codex CLI does not expose a temperature setting")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        instructions = (
            "You generate text for AssetOpsBench. Follow the supplied prompt and "
            "return only its requested output format. All necessary context is "
            "in the prompt. Do not inspect files, invoke tools, or delegate work."
        )
        if max_tokens is not None:
            instructions += f" Aim to keep the final response within {max_tokens} tokens."
        try:
            with TemporaryDirectory(prefix="assetops-codex-") as workdir:
                root = Path(workdir)
                instruction_file = root / "instructions.txt"
                instruction_file.write_text(instructions, encoding="utf-8")
                output_file = root / "response.txt"
                command = [
                    self._executable, "exec", "--json", "--ephemeral",
                    "--ignore-user-config", "--skip-git-repo-check",
                    "--sandbox", "read-only", "--color", "never",
                    "--output-last-message", str(output_file),
                    "--enable", "skip_host_skill_discovery",
                    "-c", "approval_policy=\"never\"",
                    "-c", "web_search=\"disabled\"",
                    "-c", "tools.view_image=false",
                    "-c", "agents.enabled=false",
                    "-c", "project_doc_max_bytes=0",
                    "-c", f"model_instructions_file={json.dumps(str(instruction_file))}",
                ]
                for feature in _DISABLED_FEATURES:
                    command.extend(["--disable", feature])
                if self._requested_model:
                    command.extend(["--model", self._requested_model])
                command.append("-")
                process = subprocess.run(
                    command, input=prompt, text=True, capture_output=True,
                    cwd=workdir, timeout=self._timeout_seconds,
                )
                return self._read_result(process, output_file)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Codex CLI not found. Install it and run 'codex login'."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Codex timed out after {self._timeout_seconds:g} seconds"
            ) from exc

    @staticmethod
    def _read_result(process: subprocess.CompletedProcess, output_file: Path) -> LLMResult:
        try:
            events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise RuntimeError("Codex returned invalid JSON events") from exc
        if any(not isinstance(event, dict) for event in events):
            raise RuntimeError("Codex returned an invalid event envelope")
        completed = [event for event in events if event.get("type") == "turn.completed"]
        failed = [event for event in events if event.get("type") in {"error", "turn.failed"}]
        if process.returncode or failed or not completed:
            detail = (failed[-1].get("error") or failed[-1].get("message")) if failed else None
            raise RuntimeError(
                f"Codex generation failed (exit {process.returncode}): "
                f"{str(detail)[:800] if detail else 'no completed turn; check codex login status and CLI version'}"
            )
        if not output_file.is_file():
            raise RuntimeError("Codex completed without a final response file")
        result = output_file.read_text(encoding="utf-8")
        if not result.strip():
            raise RuntimeError("Codex returned an empty text result")
        usage = completed[-1].get("usage") or {}
        return LLMResult(
            text=result,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        )
