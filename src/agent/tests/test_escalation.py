"""Tests for deterministic escalation signal extraction."""

from agent.plan_execute.escalation import (
    DEFAULT_SPECIALIST_SERVERS,
    EscalationAction,
    EscalationDecision,
    extract_escalation_signals,
    should_escalate,
)
from agent.plan_execute.models import Plan, PlanStep, StepResult
from agent.runner import DEFAULT_SERVER_PATHS


def _step(
    n: int,
    server: str = "iot",
    tool: str = "sites",
    deps: list[int] | None = None,
    task: str | None = None,
    expected_output: str = "output",
) -> PlanStep:
    return PlanStep(
        step_number=n,
        task=task or f"Task {n}",
        server=server,
        tool=tool,
        tool_args={},
        dependencies=deps or [],
        expected_output=expected_output,
    )


def test_extracts_step_count_and_dependency_depth():
    plan = Plan(
        steps=[
            _step(1),
            _step(2, deps=[1]),
            _step(3, deps=[2]),
            _step(4, deps=[1]),
        ],
        raw="",
    )

    signals = extract_escalation_signals("Q", plan)

    assert signals.step_count == 4
    assert signals.dependency_depth == 3


def test_empty_plan_has_zero_dependency_depth():
    signals = extract_escalation_signals("Q", Plan(steps=[], raw=""))

    assert signals.step_count == 0
    assert signals.dependency_depth == 0


def test_detects_specialist_servers():
    plan = Plan(
        steps=[
            _step(1, server="iot"),
            _step(2, server="wo", tool="work_orders"),
            _step(3, server="vibration", tool="analyze"),
        ],
        raw="",
    )

    signals = extract_escalation_signals("Q", plan)

    assert signals.uses_specialist_servers is True
    assert signals.specialist_servers_used == ["vibration", "wo"]


def test_default_specialist_servers_match_registered_server_names():
    assert DEFAULT_SPECIALIST_SERVERS <= set(DEFAULT_SERVER_PATHS)


def test_collects_servers_and_tools_from_plan_and_trajectory():
    plan = Plan(
        steps=[
            _step(1, server="iot", tool="assets"),
            _step(2, server="utilities", tool="current_date_time"),
        ],
        raw="",
    )
    trajectory = [
        StepResult(
            step_number=1,
            task="Task 1",
            server="iot",
            response="ok",
            tool="assets",
        ),
        StepResult(
            step_number=2,
            task="Task 2",
            server="fmsr",
            response="ok",
            tool="diagnose_failure",
        ),
    ]

    signals = extract_escalation_signals("Q", plan, trajectory)

    assert signals.servers_used == ["fmsr", "iot", "utilities"]
    assert signals.tools_used == ["assets", "current_date_time", "diagnose_failure"]


def test_detects_failed_steps_from_trajectory():
    plan = Plan(steps=[_step(1), _step(2)], raw="")
    trajectory = [
        StepResult(step_number=1, task="Task 1", server="iot", response="ok"),
        StepResult(
            step_number=2,
            task="Task 2",
            server="iot",
            response="",
            error="timeout",
        ),
    ]

    signals = extract_escalation_signals("Q", plan, trajectory)

    assert signals.any_step_failed is True
    assert signals.failed_steps == [2]


def test_matches_domain_terms_across_question_plan_and_trajectory():
    plan = Plan(
        steps=[
            _step(
                1,
                task="Open work order history",
                expected_output="Recent maintenance records",
            )
        ],
        raw="#Task1: Run diagnostics",
    )
    trajectory = [
        StepResult(
            step_number=1,
            task="Task 1",
            server="iot",
            response="Asset reported a pump failure alarm",
        )
    ]

    signals = extract_escalation_signals("Any anomaly on CH-1?", plan, trajectory)

    assert signals.has_domain_terms is True
    assert signals.matched_terms == [
        "work order",
        "diagnostics",
        "failure",
        "alarm",
        "anomaly",
    ]


def test_custom_specialist_servers_and_terms_are_supported():
    plan = Plan(steps=[_step(1, server="custom", task="Check severe drift")], raw="")

    signals = extract_escalation_signals(
        "Q",
        plan,
        specialist_servers={"custom"},
        escalation_terms=["severe drift"],
    )

    assert signals.uses_specialist_servers is True
    assert signals.specialist_servers_used == ["custom"]
    assert signals.matched_terms == ["severe drift"]


def test_matched_terms_are_case_insensitive_and_deduplicated():
    plan = Plan(steps=[_step(1, task="Investigate FAILURE alarm")], raw="")

    signals = extract_escalation_signals(
        "Failure reported",
        plan,
        escalation_terms=["failure", "Failure", "alarm"],
    )

    assert signals.matched_terms == ["failure", "alarm"]


def test_escalation_decision_dataclass_is_available():
    decision = EscalationDecision(
        action=EscalationAction.REPORT_FAILURE, reasons=["failed step"]
    )

    assert decision.should_escalate is True
    assert decision.action is EscalationAction.REPORT_FAILURE
    assert decision.reasons == ["failed step"]


def test_policy_escalates_on_failed_steps():
    plan = Plan(steps=[_step(1)], raw="")
    trajectory = [
        StepResult(
            step_number=1,
            task="Task 1",
            server="iot",
            response="",
            error="timeout",
        )
    ]

    decision = should_escalate(extract_escalation_signals("Q", plan, trajectory))

    assert decision.should_escalate is True
    assert decision.action is EscalationAction.REPORT_FAILURE
    assert decision.reasons == ["unresolved execution failure"]


def test_policy_does_not_escalate_on_specialist_server_usage():
    plan = Plan(steps=[_step(1, server="vibration")], raw="")

    decision = should_escalate(extract_escalation_signals("Q", plan))

    assert decision.should_escalate is False
    assert decision.action is EscalationAction.NONE
    assert decision.reasons == []


def test_policy_does_not_escalate_on_domain_terms():
    plan = Plan(steps=[_step(1, task="Review work order history")], raw="")

    decision = should_escalate(extract_escalation_signals("Q", plan))

    assert decision.should_escalate is False
    assert decision.action is EscalationAction.NONE
    assert decision.reasons == []


def test_policy_does_not_escalate_simple_low_risk_plan():
    plan = Plan(steps=[_step(1, task="List sites", expected_output="Site list")], raw="")

    decision = should_escalate(extract_escalation_signals("Q", plan))

    assert decision.should_escalate is False
    assert decision.reasons == []
