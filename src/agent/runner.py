"""Abstract base class for all agent runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from llm import LLMBackend

from .plan_execute.models import AgentResult


class AgentRunner(ABC):
    """Abstract base class for all agent runners.

    Subclasses implement :meth:`run` to handle a natural-language question and
    return an :class:`AgentResult`.  The ``llm`` and ``server_paths``
    attributes are available to all subclasses.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = server_paths

    @abstractmethod
    async def run(self, question: str) -> AgentResult:
        """Run the agent on *question* and return a structured result."""
