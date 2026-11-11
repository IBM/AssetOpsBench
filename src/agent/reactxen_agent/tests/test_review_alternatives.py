"""Typed review decisions keep semantic uncertainty and alternatives explicit."""

import pytest

from agent.reactxen_agent.models import Outcome
from agent.reactxen_agent.reflection_plan import ReflectionFinishParams
from agent.reactxen_agent.review import parse_review
from agent.reactxen_agent.review_plan import ReviewFinishParams


def report(**changes):
    value = dict(
        outcome="accomplished",
        arithmetic_verified=True,
        population_verified=True,
        question_alignment_verified=True,
        alignment_findings="Matches original request",
        critique="Calculation and population verified",
        alternative_search_summary="Compared event definitions",
        alternatives=[],
        evidence_paths=[],
    )
    value.update(changes)
    return value


def test_legacy_annotated_outcome_does_not_trigger_spurious_retry():
    outcome, _, _ = parse_review(
        "OUTCOME: accomplished — reproduced independently\nCRITIQUE: NONE\nALTERNATIVES: NONE"
    )
    assert outcome is Outcome.ACCOMPLISHED
    outcome, _, _ = parse_review(
        "OUTCOME: not_accomplished — filter contradicted\nCRITIQUE: wrong population"
    )
    assert outcome is Outcome.NOT_ACCOMPLISHED


def test_typed_verdict_rejects_annotated_enum():
    with pytest.raises(ValueError):
        ReviewFinishParams(**report(outcome="accomplished — verified"))


def test_correct_arithmetic_cannot_approve_unverified_population():
    with pytest.raises(ValueError, match="population"):
        ReviewFinishParams(**report(population_verified=False))
    result = ReviewFinishParams(
        **report(outcome="partially_accomplished", population_verified=False)
    )
    assert result.arithmetic_verified
    assert not result.population_verified


def test_unresolved_competing_option_must_remain_partial():
    alternate = dict(
        interpretation="Different event filter",
        candidate_answer={"machine": "B"},
        status="plausible",
        supporting_evidence="alternate records",
        counterevidence_or_gap="definition unavailable",
    )
    with pytest.raises(ValueError, match="plausible"):
        ReviewFinishParams(**report(alternatives=[alternate]))
    result = ReviewFinishParams(
        **report(outcome="partially_accomplished", alternatives=[alternate])
    )
    assert result.alternatives[0].candidate_answer == {"machine": "B"}


def test_reflector_schema_exposes_actual_field_limits():
    properties = ReflectionFinishParams.model_json_schema()["properties"]
    assert properties["selection_reason"]["maxLength"] == 2000
    assert "alternative_assessments" in properties
    assert "answer" not in properties
