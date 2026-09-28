"""Deterministic escalation signal extraction for plan-execute runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

from .models import Plan, StepFailureKind, StepResult

DEFAULT_SPECIALIST_SERVERS = frozenset({"fmsr", "tsfm", "vibration", "wo"})

DEFAULT_ESCALATION_TERMS = (
    "work order",
    "work-order",
    "diagnostic",
    "diagnostics",
    "diagnosis",
    "failure",
    "failed",
    "fault",
    "alarm",
    "anomaly",
)


@dataclass
class EscalationSignals:
    """Signals that may later inform adaptive escalation decisions."""

    step_count: int
    dependency_depth: int
    uses_specialist_servers: bool
    specialist_servers_used: list[str] = field(default_factory=list)
    any_step_failed: bool = False
    failed_steps: list[int] = field(default_factory=list)
    servers_used: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    failure_kinds: list[str] = field(default_factory=list)
    recovery_attempted_steps: list[int] = field(default_factory=list)
    recovered_steps: list[int] = field(default_factory=list)
    retry_blocked_steps: list[int] = field(default_factory=list)
    retry_exhausted_steps: list[int] = field(default_factory=list)

    @property
    def has_domain_terms(self) -> bool:
        return bool(self.matched_terms)


class EscalationAction(StrEnum):
    """Bounded action selected after execution risk assessment."""

    NONE = "none"
    RETRY_STEP = "retry_step"
    VERIFY = "verify"
    REPORT_FAILURE = "report_failure"


@dataclass
class EscalationDecision:
    """Deterministic escalation decision for a plan-execute run."""

    action: EscalationAction = EscalationAction.NONE
    reasons: list[str] = field(default_factory=list)
    step_numbers: list[int] = field(default_factory=list)

    @property
    def should_escalate(self) -> bool:
        """Compatibility flag for callers that only need a routing boolean."""
        return self.action is not EscalationAction.NONE


def extract_escalation_signals(
    question: str,
    plan: Plan,
    trajectory: Iterable[StepResult] | None = None,
    specialist_servers: Iterable[str] = DEFAULT_SPECIALIST_SERVERS,
    escalation_terms: Iterable[str] = DEFAULT_ESCALATION_TERMS,
) -> EscalationSignals:
    """Extract deterministic escalation signals from a plan and trajectory.

    This function is intentionally side-effect free and makes no LLM calls.
    """
    results = list(trajectory or [])
    specialist_server_set = set(specialist_servers)

    servers_used = _unique_sorted(
        [step.server for step in plan.steps] + [result.server for result in results]
    )
    tools_used = _unique_sorted(
        [step.tool for step in plan.steps if step.tool]
        + [result.tool for result in results if result.tool]
    )
    specialist_servers_used = [
        server for server in servers_used if server in specialist_server_set
    ]
    failed_steps = [result.step_number for result in results if not result.success]
    recovered_steps = [
        result.step_number for result in results if result.recovery_succeeded
    ]

    return EscalationSignals(
        step_count=len(plan.steps),
        dependency_depth=_dependency_depth(plan),
        uses_specialist_servers=bool(specialist_servers_used),
        specialist_servers_used=specialist_servers_used,
        any_step_failed=bool(failed_steps),
        failed_steps=failed_steps,
        servers_used=servers_used,
        tools_used=tools_used,
        matched_terms=_matched_terms(question, plan, results, escalation_terms),
        failure_kinds=_unique_sorted(
            result.failure_kind.value for result in results if result.failure_kind
        ),
        recovery_attempted_steps=[
            result.step_number for result in results if result.recovery_attempted
        ],
        recovered_steps=recovered_steps,
        retry_blocked_steps=[
            result.step_number for result in results if result.retry_blocked
        ],
        retry_exhausted_steps=[
            result.step_number for result in results if result.retry_exhausted
        ],
    )


def should_escalate(signals: EscalationSignals) -> EscalationDecision:
    """Choose an action from direct execution evidence.

    Specialist-server use, dependency depth, and domain words remain observable
    diagnostics but do not route. They do not establish a correctable failure.
    """
    if signals.any_step_failed:
        hard_failure_kinds = {
            StepFailureKind.FAILED_DEPENDENCY.value,
            StepFailureKind.UNSUPPORTED_CAPABILITY.value,
        }
        hard_failure = bool(hard_failure_kinds.intersection(signals.failure_kinds))
        if hard_failure or signals.retry_exhausted_steps:
            return EscalationDecision(
                action=EscalationAction.REPORT_FAILURE,
                reasons=["required execution evidence unavailable"],
                step_numbers=signals.failed_steps,
            )
        if signals.retry_blocked_steps:
            return EscalationDecision(
                action=EscalationAction.VERIFY,
                reasons=["automatic retry prohibited by tool safety"],
                step_numbers=signals.retry_blocked_steps,
            )
        return EscalationDecision(
            action=EscalationAction.REPORT_FAILURE,
            reasons=["unresolved execution failure"],
            step_numbers=signals.failed_steps,
        )

    if signals.recovered_steps:
        return EscalationDecision(
            action=EscalationAction.RETRY_STEP,
            reasons=["bounded safe recovery succeeded"],
            step_numbers=signals.recovered_steps,
        )

    return EscalationDecision()


def _dependency_depth(plan: Plan) -> int:
    """Return the longest dependency chain length, counting the step itself."""
    steps_by_number = {step.step_number: step for step in plan.steps}
    visiting: set[int] = set()
    memo: dict[int, int] = {}

    def depth(step_number: int) -> int:
        if step_number in memo:
            return memo[step_number]
        if step_number in visiting:
            return 1

        step = steps_by_number.get(step_number)
        if step is None:
            return 0

        visiting.add(step_number)
        dep_depth = max((depth(dep) for dep in step.dependencies), default=0)
        visiting.remove(step_number)
        memo[step_number] = dep_depth + 1
        return memo[step_number]

    return max((depth(step.step_number) for step in plan.steps), default=0)


def _matched_terms(
    question: str,
    plan: Plan,
    trajectory: list[StepResult],
    escalation_terms: Iterable[str],
) -> list[str]:
    text = "\n".join(
        [
            question,
            plan.raw,
            *[
                "\n".join([step.task, step.expected_output, step.server, step.tool])
                for step in plan.steps
            ],
            *[
                "\n".join(
                    [
                        result.task,
                        result.server,
                        result.tool,
                        result.response,
                        result.error or "",
                    ]
                )
                for result in trajectory
            ],
        ]
    )
    matched = []
    seen = set()
    for term in escalation_terms:
        key = term.casefold()
        if key in seen:
            continue
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE):
            seen.add(key)
            matched.append(term)
    return matched


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


__all__ = [
    "DEFAULT_ESCALATION_TERMS",
    "DEFAULT_SPECIALIST_SERVERS",
    "EscalationAction",
    "EscalationDecision",
    "EscalationSignals",
    "extract_escalation_signals",
    "should_escalate",
]
