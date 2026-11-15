"""The actor is explicitly told when no further feedback cycle remains."""

import pytest

from agent.reactxen_agent.prompts import actor_round_notice


def test_two_round_budget_marks_only_the_last_attempt_final():
    assert "ACTOR ROUND 1 OF 2" in actor_round_notice(0, 2)
    final = actor_round_notice(1, 2)
    assert "FINAL ACTOR ROUND — 2 OF 2" in final
    assert "no additional Review/Reflect correction cycle" in final
    assert "feedback already provided" in final
    assert "final Review may still assess" in final


def test_single_round_is_final_immediately():
    assert "FINAL ACTOR ROUND — 1 OF 1" in actor_round_notice(0, 1)


def test_no_review_means_one_final_actor_attempt():
    notice = actor_round_notice(0, 3, review_enabled=False)
    assert "FINAL ACTOR ROUND — 1 OF 1" in notice
    assert "Review and Reflect are disabled" in notice


def test_invalid_round_is_rejected():
    with pytest.raises(ValueError):
        actor_round_notice(2, 2)
