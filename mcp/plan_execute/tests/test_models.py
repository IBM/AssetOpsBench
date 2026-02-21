"""Tests for Plan and StepResult data models."""

from plan_execute.models import Plan, PlanStep, StepResult


def _step(n: int, deps: list[int] | None = None) -> PlanStep:
    return PlanStep(
        step_number=n,
        task=f"Task {n}",
        agent="IoTAgent",
        tool="sites",
        tool_args={},
        dependencies=deps or [],
        expected_output="output",
    )


class TestPlanResolvedOrder:
    def test_no_dependencies_preserves_order(self):
        plan = Plan(steps=[_step(1), _step(2), _step(3)], raw="")
        assert [s.step_number for s in plan.resolved_order()] == [1, 2, 3]

    def test_linear_chain(self):
        # 1 → 2 → 3
        plan = Plan(steps=[_step(1), _step(2, [1]), _step(3, [2])], raw="")
        order = [s.step_number for s in plan.resolved_order()]
        assert order.index(1) < order.index(2) < order.index(3)

    def test_diamond_dependency(self):
        # 1 → 2, 1 → 3, {2,3} → 4
        plan = Plan(
            steps=[_step(1), _step(2, [1]), _step(3, [1]), _step(4, [2, 3])],
            raw="",
        )
        order = [s.step_number for s in plan.resolved_order()]
        assert order.index(1) < order.index(2)
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(4)
        assert order.index(3) < order.index(4)

    def test_parallel_independent_steps(self):
        # Steps 1 and 2 are independent; both must appear before step 3
        plan = Plan(steps=[_step(1), _step(2), _step(3, [1, 2])], raw="")
        order = [s.step_number for s in plan.resolved_order()]
        assert order.index(1) < order.index(3)
        assert order.index(2) < order.index(3)

    def test_duplicate_visit_skipped(self):
        # Shared dependency: both step 2 and 3 depend on step 1
        plan = Plan(steps=[_step(1), _step(2, [1]), _step(3, [1])], raw="")
        order = [s.step_number for s in plan.resolved_order()]
        # Step 1 should appear exactly once
        assert order.count(1) == 1

    def test_single_step(self):
        plan = Plan(steps=[_step(1)], raw="")
        assert [s.step_number for s in plan.resolved_order()] == [1]

    def test_empty_plan(self):
        plan = Plan(steps=[], raw="")
        assert plan.resolved_order() == []


class TestPlanGetStep:
    def test_found(self):
        plan = Plan(steps=[_step(1), _step(2)], raw="")
        assert plan.get_step(2).step_number == 2

    def test_not_found_returns_none(self):
        plan = Plan(steps=[_step(1)], raw="")
        assert plan.get_step(99) is None


class TestStepResult:
    def test_success_when_no_error(self):
        r = StepResult(step_number=1, task="t", agent="a", response="ok")
        assert r.success is True

    def test_failure_when_error_set(self):
        r = StepResult(step_number=1, task="t", agent="a", response="", error="oops")
        assert r.success is False
