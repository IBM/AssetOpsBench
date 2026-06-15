"""Abstract base class for all agent runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from llm import LLMBackend
from llm.generation import GenerationParams, from_env

from .models import AgentResult

# Maps MCP-server names to either a uv entry-point name (str) or a script Path.
# Entry-point names are invoked as ``uv run <name>``; Paths fall back to
# ``uv run <path>``.  Subclassing runners receive a resolved copy via
# ``self._server_paths`` (defaulting to this dict when ``server_paths=None``).
DEFAULT_SERVER_PATHS: dict[str, Path | str] = {
    "iot": "iot-mcp-server",
    "utilities": "utilities-mcp-server",
    "fmsr": "fmsr-mcp-server",
    "tsfm": "tsfm-mcp-server",
    "wo": "wo-mcp-server",
    "vibration": "vibration-mcp-server",
}


class AgentRunner(ABC):
    """Abstract base class for all agent runners.

    Subclasses implement :meth:`run` to handle a natural-language question and
    return an :class:`AgentResult`.  After ``super().__init__``,
    ``self._server_paths`` is always a concrete ``dict`` — either the caller's
    override, or a copy of :data:`DEFAULT_SERVER_PATHS`.

    Args:
        llm: LLM backend (used by plan-execute; SDK-based runners accept
             ``None`` for interface compatibility).
        server_paths: MCP server specs.  Defaults to :data:`DEFAULT_SERVER_PATHS`.
        generation: Generation parameters applied to all LLM calls made by this
                    runner.  Defaults to :func:`~llm.generation.from_env` so
                    env vars take effect without explicit construction.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        *,
        generation: GenerationParams | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths: dict[str, Path | str] = (
            dict(DEFAULT_SERVER_PATHS) if server_paths is None else server_paths
        )
        self._generation: GenerationParams = (
            generation if generation is not None else from_env()
        )

    @abstractmethod
    async def run(self, question: str) -> AgentResult:
        """Run the agent on *question* and return a structured result."""
