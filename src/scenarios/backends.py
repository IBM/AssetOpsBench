"""One registry for generator backends, defaults, and configuration help."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.cli import _DEFAULT_MODEL
from llm.base import LLMBackend
from llm.claude_code import ClaudeCodeBackend
from llm.codex import CodexBackend
from llm.glm import DEFAULT_GLM_MODEL, GLMBackend
from llm.litellm import LiteLLMBackend


@dataclass(frozen=True)
class BackendSpec:
    label: str
    default_model: str | None
    authentication: str
    factory: Callable[..., LLMBackend]


BACKENDS: dict[str, BackendSpec] = {
    "litellm": BackendSpec(
        "Watsonx / LiteLLM", _DEFAULT_MODEL,
        "WATSONX_APIKEY + WATSONX_PROJECT_ID, or LITELLM_API_KEY + LITELLM_BASE_URL",
        LiteLLMBackend,
    ),
    "claude-code": BackendSpec(
        "Claude Code", "sonnet", "Local login: claude auth login", ClaudeCodeBackend,
    ),
    "codex": BackendSpec("Codex", None, "Local login: codex login", CodexBackend),
    "glm": BackendSpec("GLM / Z.ai", DEFAULT_GLM_MODEL, "ZAI_API_KEY", GLMBackend),
}


def create_backend(backend: str, model_id: str | None = None) -> LLMBackend:
    try:
        spec = BACKENDS[backend]
    except KeyError:
        raise ValueError(f"Unknown scenario generation backend: {backend!r}") from None
    return spec.factory(model_id=model_id or spec.default_model)
