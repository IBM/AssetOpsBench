"""Data models for the plan-execute orchestration client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


class StepFailureKind(StrEnum):
    """Machine-readable phase in which a plan step failed."""

    ARGUMENT_RESOLUTION = "argument_resolution"
    ARGUMENT_VALIDATION = "argument_validation"
    TOOL_ERROR = "tool_error"
    EMPTY_OUTPUT = "empty_output"
    FAILED_DEPENDENCY = "failed_dependency"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


class RetrySafety(StrEnum):
    """Why an automatic retry is allowed or prohibited."""

    SAFE_PRE_CALL = "safe_pre_call"
    READ_ONLY = "read_only"
    MUTATING = "mutating"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    step_number: int
    task: str
    server: str
    tool: str
    tool_args: dict
    dependencies: list[int]
    expected_output: str


@dataclass
class Plan:
    """An execution plan composed of ordered steps."""

    steps: list[PlanStep]
    raw: str  # Raw LLM output, preserved for debugging

    def get_step(self, number: int) -> Optional[PlanStep]:
        return next((s for s in self.steps if s.step_number == number), None)

    def resolved_order(self) -> list[PlanStep]:
        """Return steps in topological order (dependencies before dependents)."""
        seen: set[int] = set()
        ordered: list[PlanStep] = []

        def visit(n: int) -> None:
            if n in seen:
                return
            step = self.get_step(n)
            if step is None:
                return
            for dep in step.dependencies:
                visit(dep)
            seen.add(n)
            ordered.append(step)

        for step in self.steps:
            visit(step.step_number)
        return ordered


@dataclass
class StepResult:
    """Result of executing a single plan step."""

    step_number: int
    task: str
    server: str
    response: str
    error: Optional[str] = None
    tool: str = ""
    tool_args: dict = field(default_factory=dict)
    duration_ms: float | None = None
    failure_kind: StepFailureKind | None = None
    attempt_count: int = 1
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    initial_error: Optional[str] = None
    retry_safety: RetrySafety = RetrySafety.NOT_APPLICABLE
    retry_blocked: bool = False
    retry_exhausted: bool = False

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class OrchestratorResult:
    """Final result from the plan-execute orchestrator."""

    question: str
    answer: str
    plan: Plan
    trajectory: list[StepResult]
    escalation_action: str | None = None
    escalation_reasons: list[str] = field(default_factory=list)
