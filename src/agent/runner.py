"""Abstract base class for all agent runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from llm import LLMBackend

from .mcp_servers import (
    DEFAULT_SERVER_PATHS as _DEFAULT_SERVER_PATHS,
    RemoteMCPServer,
    apply_env_overrides,
)
from .models import AgentResult

# Server specs live in :mod:`agent.mcp_servers` so a slot can point at a remote
# MCP service instead of a local stdio process. Re-exported here because every
# runner already imports it from this module.
DEFAULT_SERVER_PATHS = _DEFAULT_SERVER_PATHS


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
        server_paths: dict[str, Path | str | RemoteMCPServer] | None = None,
    ) -> None:
        self._llm = llm
        # ASSETOPS_MCP_URL_<NAME> redirects one slot to a remote MCP service,
        # which is how the "replace our tsfm with theirs" arm runs without a
        # fork. Applied to an explicit override too, so a caller that pins
        # server paths still honours the swap.
        self._server_paths: dict[str, Path | str | RemoteMCPServer] = (
            apply_env_overrides(
                dict(DEFAULT_SERVER_PATHS) if server_paths is None else server_paths
            )
        )

    @abstractmethod
    async def run(self, question: str) -> AgentResult:
        """Run the agent on *question* and return a structured result."""
