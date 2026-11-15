"""Tests for the ReActXen loop.

The mechanical half (preconditions, verdict parsing, round accounting) is
tested without a model call, because that is the half whose whole purpose is
to be decidable without one.
"""

from __future__ import annotations

import json

import pytest

from agent.models import ToolCall, Trajectory, TurnRecord
from agent.reactxen_agent.models import (
    Outcome,
    ReActXenTrajectory,
    ReviewVerdict,
    RoundRecord,
)
from agent.reactxen_agent.reflect import as_prompt_block, summarise
from agent.reactxen_agent.review import check_preconditions, parse_review


def _traj(*calls: ToolCall, text: str = "", in_tok: int = 0, out_tok: int = 0):
    return Trajectory(
        turns=[
            TurnRecord(
                index=0,
                text=text,
                tool_calls=list(calls),
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        ]
    )


def _named(name: str, **params):
    return ToolCall(name=name, input=params)


# -- preconditions -----------------------------------------------------------


def test_empty_answer_fails():
    results = check_preconditions(
        question="q", answer="   ", trajectory=_traj(_named("get_sites"))
    )
    assert not next(r for r in results if r.name == "answer_nonempty").passed


def test_no_tool_call_fails():
    results = check_preconditions(question="q", answer="42", trajectory=_traj())
    assert not next(r for r in results if r.name == "domain_tool_called").passed


def test_malformed_json_is_caught_when_expected():
    results = check_preconditions(
        question="q",
        answer='{"mode": "bearing_wear",',
        trajectory=_traj(_named("get_sites")),
        expect_json=True,
    )
    assert not next(r for r in results if r.name == "answer_parses").passed


def test_valid_json_passes():
    results = check_preconditions(
        question="q",
        answer=json.dumps({"mode": "bearing_wear"}),
        trajectory=_traj(_named("get_sites")),
        expect_json=True,
    )
    assert next(r for r in results if r.name == "answer_parses").passed


def test_declared_evidence_not_consulted_is_caught():
    results = check_preconditions(
        question="q",
        answer="ok",
        trajectory=_traj(_named("get_sites", site="POKMAIN")),
        declared_evidence=["chiller6_vibration.csv"],
    )
    check = next(r for r in results if r.name == "declared_evidence_consulted")
    assert not check.passed
    assert "chiller6_vibration.csv" in check.detail


def test_declared_evidence_consulted_passes():
    results = check_preconditions(
        question="q",
        answer="ok",
        trajectory=_traj(_named("read_file", path="data/chiller6_vibration.csv")),
        declared_evidence=["chiller6_vibration.csv"],
    )
    assert next(r for r in results if r.name == "declared_evidence_consulted").passed


def test_spinning_is_caught():
    repeated = [_named("histories", sensor="S1") for _ in range(4)]
    results = check_preconditions(
        question="q", answer="ok", trajectory=_traj(*repeated)
    )
    assert not next(r for r in results if r.name == "no_spinning").passed


def test_distinct_calls_do_not_count_as_spinning():
    calls = [_named("histories", sensor=f"S{i}") for i in range(4)]
    results = check_preconditions(question="q", answer="ok", trajectory=_traj(*calls))
    assert next(r for r in results if r.name == "no_spinning").passed


# -- verdict parsing ---------------------------------------------------------


def test_parse_full_review():
    outcome, critique, alternatives = parse_review(
        "OUTCOME: partially_accomplished\n"
        "CRITIQUE: Pattern matching was used where normalisation belonged.\n"
        "ALTERNATIVES:\n"
        "1. Normalise then match against the catalogue.\n"
        "2. Train a classifier on the labelled register.\n"
    )
    assert outcome is Outcome.PARTIAL
    assert "normalisation" in critique
    assert len(alternatives) == 2
    assert alternatives[0].startswith("Normalise")


def test_parse_none_sections():
    outcome, critique, alternatives = parse_review(
        "OUTCOME: accomplished\nCRITIQUE: NONE\nALTERNATIVES: NONE\n"
    )
    assert outcome is Outcome.ACCOMPLISHED
    assert critique == ""
    assert alternatives == []


def test_parse_caps_alternatives_at_three():
    outcome, _, alternatives = parse_review(
        "OUTCOME: not_accomplished\nCRITIQUE: x\nALTERNATIVES:\n"
        + "\n".join(f"- option {i}" for i in range(6))
    )
    assert outcome is Outcome.NOT_ACCOMPLISHED
    assert len(alternatives) == 3


def test_unparseable_outcome_defaults_to_partial():
    outcome, _, _ = parse_review("OUTCOME: looks fine to me\nCRITIQUE: NONE\n")
    assert outcome is Outcome.PARTIAL


# -- accounting --------------------------------------------------------------


def test_should_continue_tracks_outcome():
    assert not ReviewVerdict(outcome=Outcome.ACCOMPLISHED).should_continue
    assert ReviewVerdict(outcome=Outcome.PARTIAL).should_continue


def test_token_totals_span_core_review_and_reflect():
    record = ReActXenTrajectory(
        rounds=[
            RoundRecord(
                index=0,
                answer="a",
                trajectory=_traj(_named("t"), in_tok=100, out_tok=10),
                review=ReviewVerdict(
                    outcome=Outcome.PARTIAL, input_tokens=50, output_tokens=5
                ),
                reflect_input_tokens=20,
                reflect_output_tokens=2,
            )
        ]
    )
    assert record.total_input_tokens == 170
    assert record.total_output_tokens == 17


def test_self_ask_rate_reports_zero_when_tool_unused():
    record = ReActXenTrajectory(
        rounds=[
            RoundRecord(
                index=0, answer="a", trajectory=_traj(_named("t")), self_ask_calls=0
            )
        ]
    )
    assert record.self_ask_rate() == 0.0


def test_self_ask_rate_is_calls_per_step():
    record = ReActXenTrajectory(
        rounds=[
            RoundRecord(
                index=0, answer="a", trajectory=_traj(_named("t")), self_ask_calls=2
            )
        ]
    )
    assert record.self_ask_rate() == pytest.approx(2.0)


# -- reflect formatting ------------------------------------------------------


def test_empty_guidance_injects_nothing():
    assert as_prompt_block("   ") == ""


def test_guidance_is_wrapped_once():
    block = as_prompt_block("Normalise the work-order text first.")
    assert "Normalise the work-order text first." in block
    assert block.count("Guidance from a review") == 1


def test_summary_lists_tools_in_order():
    summary = summarise(_traj(_named("a"), _named("b")), "answer text")
    assert "a -> b" in summary
    assert "answer text" in summary
