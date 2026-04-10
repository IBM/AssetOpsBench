"""Top-level data models for the agent orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResult:
    """Result returned by any AgentRunner."""

    question: str
    answer: str
    trajectory: Any
