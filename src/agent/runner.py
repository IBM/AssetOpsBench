"""Abstract base class for all agent runners."""

from __future__ import annotations

import os
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

# Env var that pins the LLM used inside the FMSR MCP server
# (generate_failure_modes).  See :func:`mcp_server_env`.
FMSR_MODEL_ENV = "FMSR_MODEL_ID"


def resolve_fmsr_model_id(agent_model_id: str | None) -> str | None:
    """Return the model the FMSR server should use.

    An explicit ``FMSR_MODEL_ID`` (shell or ``.env``) always wins; otherwise
    the agent's own model id is used.  Empty strings count as unset.
    """
    explicit = (os.environ.get(FMSR_MODEL_ENV) or "").strip()
    agent = agent_model_id.strip() if isinstance(agent_model_id, str) else ""
    return explicit or agent or None


def fmsr_env_overrides(agent_model_id: str | None) -> dict[str, str]:
    """Env overrides that pin ``FMSR_MODEL_ID`` for spawned MCP servers.

    The value is always set explicitly, even when it equals the agent's model
    id, so the FMSR model is fixed and visible for every run.
    """
    fmsr_model = resolve_fmsr_model_id(agent_model_id)
    return {FMSR_MODEL_ENV: fmsr_model} if fmsr_model else {}


def mcp_server_env(agent_model_id: str | None) -> dict[str, str]:
    """Full environment for MCP servers spawned via the MCP SDK stdio client.

    That client passes only HOME/PATH/SHELL/... to child processes unless
    ``env`` is given, so the parent environment is forwarded explicitly.
    """
    return {**os.environ, **fmsr_env_overrides(agent_model_id)}


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
