"""LLM backend for AssetOpsBench MCP."""

from .base import LLMBackend, LLMResult
from .litellm import LiteLLMBackend
from .openai_compat import OpenAICompatBackend, is_openai_compat

__all__ = [
    "LLMBackend",
    "LLMResult",
    "LiteLLMBackend",
    "OpenAICompatBackend",
    "is_openai_compat",
    "make_backend",
]


def make_backend(model_id: str) -> LLMBackend:
    """Return the right backend for *model_id*.

    OpenAI-compatible routers (e.g. ``tokenrouter/...``) use the native
    ``openai`` SDK via :class:`OpenAICompatBackend`; everything else goes
    through :class:`LiteLLMBackend`, which covers watsonx and the LiteLLM
    proxy.
    """
    if is_openai_compat(model_id):
        return OpenAICompatBackend(model_id)
    return LiteLLMBackend(model_id)
