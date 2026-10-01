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


# LLM provider credentials and endpoints. No MCP server calls a model, so
# mcp_server_env withholds all of them. Mirrors CREDENTIAL_ENV_VARS in
# src/assetops_harbor/stirrup.py; a test there keeps the two in step.
LLM_CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "TOKENROUTER_BASE_URL",
    "TOKENROUTER_API_KEY",
    "WATSONX_APIKEY",
    "WATSONX_URL",
    "WATSONX_PROJECT_ID",
    "WATSONX_DEPLOYMENT_SPACE_ID",
    "WATSONX_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_REGION_NAME",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GEMINI_API_KEY",
)


def mcp_server_env() -> dict[str, str]:
    """Environment for MCP servers spawned via the MCP SDK stdio client.

    That client passes only HOME/PATH/SHELL/... to child processes unless
    ``env`` is given, so the parent environment is forwarded explicitly, minus
    :data:`LLM_CREDENTIAL_ENV_VARS`.

    Those are blanked rather than dropped. Every server calls ``load_dotenv()``,
    which would refill a missing name from the repo's ``.env`` but never
    overwrites one that is already set, even to an empty string.
    """
    return {**os.environ, **dict.fromkeys(LLM_CREDENTIAL_ENV_VARS, "")}


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
