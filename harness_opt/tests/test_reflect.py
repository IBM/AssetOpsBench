"""The paired accept rule, and the noise floor it is calibrated against."""

from __future__ import annotations

import pytest

from harness_opt.reflect import PairedOutcome, compare, flip_rate


def test_buckets_every_shared_task() -> None:
    outcome = compare(
        parent={"a": True, "b": True, "c": False, "d": False},
        patch={"a": True, "b": False, "c": True, "d": False},
    )

    assert outcome.still_passing == ("a",)
    assert outcome.regressed == ("b",)
    assert outcome.fixed == ("c",)
    assert outcome.still_failing == ("d",)
    assert outcome.n == 4


def test_tasks_only_one_side_scored_are_excluded_not_assumed_failed() -> None:
    outcome = compare(parent={"a": True, "b": True}, patch={"a": True})

    assert outcome.n == 1
    assert outcome.regressed == ()


def test_more_fixes_than_regressions_accepts() -> None:
    assert compare({"a": False, "b": True}, {"a": True, "b": True}).accepts()


def test_equal_fixes_and_regressions_does_not_accept() -> None:
    """A wash is not an improvement. Ties must not advance the frontier."""
    outcome = compare({"a": False, "b": True}, {"a": True, "b": False})

    assert outcome.net == 0
    assert not outcome.accepts()


def test_a_patch_that_only_regresses_is_rejected() -> None:
    assert not compare({"a": True}, {"a": False}).accepts()


def test_a_patch_that_changes_nothing_is_rejected() -> None:
    assert not compare({"a": True, "b": False}, {"a": True, "b": False}).accepts()


def test_the_mean_can_rise_while_the_paired_rule_refuses() -> None:
    """Why the gate is paired: unequal task counts make means misleading."""
    parent = {"a": True, "b": True, "c": False}
    patch = {"a": False, "b": True, "c": True}
    outcome = compare(parent, patch)

    assert outcome.parent_score == outcome.patch_score
    assert not outcome.accepts()


def test_fix_and_regression_rates_use_the_right_denominators() -> None:
    outcome = compare(
        parent={"p1": True, "p2": True, "f1": False, "f2": False, "f3": False},
        patch={"p1": True, "p2": False, "f1": True, "f2": False, "f3": False},
    )

    assert outcome.fix_rate == pytest.approx(1 / 3)
    assert outcome.regression_rate == pytest.approx(1 / 2)


def test_p_value_is_one_when_nothing_moved() -> None:
    assert compare({"a": True}, {"a": True}).p_value() == 1.0


def test_six_clean_fixes_reach_significance() -> None:
    """The bar a per-patch significance gate would set, which is why we don't."""
    parent = {f"t{i}": False for i in range(6)}
    patch = {f"t{i}": True for i in range(6)}

    assert compare(parent, patch).p_value() < 0.05


def test_a_realistic_patch_does_not_reach_significance_but_still_accepts() -> None:
    outcome = compare(
        parent={"a": False, "b": False, "c": True},
        patch={"a": True, "b": True, "c": False},
    )

    assert outcome.accepts()
    assert outcome.p_value() > 0.05


def test_flip_rate_measures_disagreement_between_identical_runs() -> None:
    runs = [
        {"a": True, "b": True, "c": False},
        {"a": True, "b": False, "c": False},
        {"a": True, "b": True, "c": False},
    ]

    assert flip_rate(runs) == pytest.approx(1 / 3)


def test_flip_rate_is_zero_when_runs_agree() -> None:
    assert flip_rate([{"a": True}, {"a": True}]) == 0.0


def test_flip_rate_needs_two_runs() -> None:
    with pytest.raises(ValueError, match="at least two runs"):
        flip_rate([{"a": True}])


def test_outcome_serializes_every_reported_field() -> None:
    body = compare({"a": False}, {"a": True}).to_json()

    for key in ("fixed", "net", "fix_rate", "regression_rate", "p_value"):
        assert key in body


def test_empty_outcome_has_no_division_errors() -> None:
    outcome = PairedOutcome()

    assert outcome.fix_rate == 0.0
    assert outcome.regression_rate == 0.0
    assert outcome.parent_score == 0.0
