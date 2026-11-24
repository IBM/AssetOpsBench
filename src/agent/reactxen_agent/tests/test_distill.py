"""Conservative advisory clarification keeps the original question authoritative."""

import json

import pytest

from agent.reactxen_agent.distill import (
    DistillationProposal,
    clarification_block,
    distill_question,
    validate_edits,
)
from agent.reactxen_agent.review_plan import ReviewFinishParams

ORIGINAL = 'Which unit costs most? Exclude servicing. Return only "asset" and "cost".'


def proposal(**changes):
    value = dict(
        revision_needed=True,
        original_question=ORIGINAL,
        revised_question=ORIGINAL.replace("costs most", "has the highest cost"),
        objective="Identify the unit with the highest cost excluding servicing",
        explicit_constraints=["Exclude servicing.", 'Return only "asset" and "cost".'],
        edits=[
            dict(
                old_text="costs most",
                new_text="has the highest cost",
                reason="Clarify comparative wording",
            )
        ],
        unresolved_ambiguities=["What constitutes servicing is not defined"],
        rationale="Wording clarification only",
    )
    value.update(changes)
    return value


def test_minimal_source_grounded_edit_preserves_constraints():
    validate_edits(ORIGINAL, DistillationProposal(**proposal()))


def test_hidden_edits_new_numbers_and_changed_original_are_rejected():
    for change in [
        {"original_question": "different"},
        {"revised_question": "Choose unit A"},
        {
            "revised_question": ORIGINAL.replace("costs most", "costs most since 2020"),
            "edits": [
                dict(
                    old_text="costs most",
                    new_text="costs most since 2020",
                    reason="new window",
                )
            ],
        },
    ]:
        with pytest.raises(ValueError):
            validate_edits(ORIGINAL, DistillationProposal(**proposal(**change)))


@pytest.mark.anyio
async def test_proposal_requires_independent_semantic_audit():
    replies = [
        json.dumps(proposal()),
        json.dumps({"equivalent": False, "reason": "meaning changed"}),
    ]

    async def complete(system, user):
        return replies.pop(0), 10, 2

    record = await distill_question(complete, ORIGINAL, {})
    assert not record["applied"] and record["status"] == "rejected"
    assert record["input_tokens"] == 20


@pytest.mark.anyio
async def test_acceptance_is_advisory_and_never_replaces_original():
    replies = [
        json.dumps(proposal()),
        json.dumps({"equivalent": True, "reason": "same request"}),
    ]

    async def complete(system, user):
        return replies.pop(0), 10, 2

    record = await distill_question(complete, ORIGINAL, {})
    assert record["applied"] and record["original_question"] == ORIGINAL
    block = clarification_block(ORIGINAL, record["clarified_question"])
    assert "original user question remains authoritative" in block
    assert record["proposal"]["explicit_constraints"][0] in record["clarified_question"]


@pytest.mark.anyio
async def test_already_clear_question_needs_no_audit_or_change():
    calls = []

    async def complete(system, user):
        calls.append(system)
        return (
            json.dumps(
                proposal(revision_needed=False, revised_question=ORIGINAL, edits=[])
            ),
            10,
            2,
        )

    record = await distill_question(complete, ORIGINAL, {})
    assert record["status"] == "unchanged" and not record["applied"]
    assert len(calls) == 1


def test_misaligned_answer_cannot_be_marked_accomplished():
    with pytest.raises(ValueError):
        ReviewFinishParams(
            outcome="accomplished",
            arithmetic_verified=True,
            population_verified=True,
            question_alignment_verified=False,
            alignment_findings="Answers a broader question",
            critique="Wrong scope",
            alternative_search_summary="Compared scope",
            alternatives=[],
        )


@pytest.mark.anyio
async def test_distill_handoff_keeps_actor_task_original_and_accounts_usage(
    monkeypatch,
):
    from unittest.mock import AsyncMock, MagicMock

    import agent.reactxen_agent.runner as module
    from agent.models import AgentResult, Trajectory, TurnRecord
    from agent.reactxen_agent.models import Outcome, ReActXenTrajectory, ReviewVerdict

    runner = module.ReActXenAgentRunner(max_rounds=2, self_ask_enabled=False)
    prompts, tasks = [], []

    class Core:
        async def run(self, question):
            tasks.append(question)
            return AgentResult(
                question,
                '{"asset":"X","cost":1}',
                Trajectory(turns=[TurnRecord(0, "", input_tokens=5)]),
            )

    def core(prompt, _):
        prompts.append(prompt)
        return Core()

    runner._core = core

    async def review(question, record, workspace):
        record.review_plan = {
            "population_verified": record.index == 1,
            "question_alignment_verified": record.index == 1,
        }
        return ReviewVerdict(
            Outcome.PARTIAL if record.index == 0 else Outcome.ACCOMPLISHED
        )

    runner._react_review = review

    async def reflect(*args):
        return "Check the collected evidence.", 0, 0

    runner._react_reflect = reflect

    async def distill(complete, original, signals):
        assert original == ORIGINAL
        return {
            "status": "accepted",
            "applied": True,
            "original_question": original,
            "clarified_question": proposal()["revised_question"],
            "input_tokens": 20,
            "output_tokens": 4,
        }

    monkeypatch.setattr(module, "distill_question", distill)
    workspace = MagicMock(write_file_bytes=AsyncMock())
    result = await runner._run_loop(ORIGINAL, ReActXenTrajectory(), workspace)
    assert tasks == [ORIGINAL, ORIGINAL]
    assert "ADVISORY QUESTION CLARIFICATION" not in prompts[0]
    assert proposal()["revised_question"] in prompts[1]
    assert "FINAL ACTOR ROUND" in prompts[1]
    assert result.question == ORIGINAL
    assert result.trajectory.total_input_tokens == 30
    assert (
        result.trajectory.rounds[1].advisory_clarified_question
        == proposal()["revised_question"]
    )


def test_plausible_ledger_candidate_triggers_clarity_check_despite_true_flags():
    from agent.reactxen_agent.distill import needs_distillation

    review = dict(
        arithmetic_verified=True,
        population_verified=True,
        question_alignment_verified=True,
        alternatives=[],
        candidate_comparisons=[dict(candidate_id="prior_subtotal", status="plausible")],
    )
    assert needs_distillation(review, {})
    review["candidate_comparisons"][0]["status"] = "ruled_out"
    assert not needs_distillation(review, {})
    assert needs_distillation(
        review, {"candidate_assessments": [dict(decision="unresolved")]}
    )
