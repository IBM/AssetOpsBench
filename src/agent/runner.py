"""Abstract base class for all agent runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from llm import LLMBackend

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
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths: dict[str, Path | str] = (
            dict(DEFAULT_SERVER_PATHS) if server_paths is None else server_paths
        )

    @abstractmethod
    async def run(self, question: str) -> AgentResult:
        """Run the agent on *question* and return a structured result."""
