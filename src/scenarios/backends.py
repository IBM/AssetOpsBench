"""One registry for generator backends, defaults, and configuration help."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.cli import _DEFAULT_MODEL
from llm.base import LLMBackend, LLMResult
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


SCENARIO_MAX_TOKENS = 131072


def model_output_limit(backend: str, model_id: str) -> int | None:
    """Resolve output capacity, never substituting a context-window limit.

    Z.ai limits come from its API reference. Other exact model IDs use
    LiteLLM's local catalog; opaque proxies and CLI aliases may be unknown.
    """
    if backend == "glm":
        model = model_id.lower()
        if model.startswith(("glm-5", "glm-4.7", "glm-4.6")):
            if model.startswith("glm-4.6v"):
                return 32768
            return 131072
        if model.startswith("glm-4.5v"):
            return 16384
        if model.startswith("glm-4.5"):
            return 98304
    import litellm

    candidates = [model_id]
    if model_id.startswith("litellm_proxy/"):
        candidates.append(model_id.removeprefix("litellm_proxy/"))
    for candidate in candidates:
        limit = litellm.model_cost.get(candidate, {}).get("max_output_tokens")
        if isinstance(limit, int) and limit > 0:
            return limit
    return None


class ScenarioBackend(LLMBackend):
    """Apply the scenario budget to every stage, including implicit calls."""

    def __init__(self, delegate: LLMBackend, backend: str) -> None:
        self._delegate = delegate
        self._model_id = delegate.model_id
        self.output_limit = model_output_limit(backend, self.model_id)

    def _budget(self, max_tokens: int | None) -> int:
        requested = SCENARIO_MAX_TOKENS if max_tokens is None else max_tokens
        if requested <= 0:
            raise ValueError("max_tokens must be positive")
        return min(requested, SCENARIO_MAX_TOKENS, self.output_limit or SCENARIO_MAX_TOKENS)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        return self._delegate.generate(prompt, temperature, max_tokens=self._budget(max_tokens))

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None,
    ) -> LLMResult:
        return self._delegate.generate_with_usage(prompt, temperature, max_tokens=self._budget(max_tokens))


def create_backend(backend: str, model_id: str | None = None) -> LLMBackend:
    try:
        spec = BACKENDS[backend]
    except KeyError:
        raise ValueError(f"Unknown scenario generation backend: {backend!r}") from None
    return ScenarioBackend(spec.factory(model_id=model_id or spec.default_model), backend)
