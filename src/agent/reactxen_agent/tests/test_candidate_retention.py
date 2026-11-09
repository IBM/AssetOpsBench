"""Retain tool-output calculations beyond a review shortlist and across phases."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

from agent.reactxen_agent.candidates import CandidateLedger, extract_calculations
from agent.reactxen_agent.reflection_plan import (
    ReflectionFinishParams,
    build_reflection_finish_tool,
)
from agent.reactxen_agent.review_plan import (
    ReviewFinishParams,
    build_review_finish_tool,
)


def workspace():
    return MagicMock(write_file_bytes=AsyncMock())


def make_ledger(count=7):
    ledger = CandidateLedger(workspace())
    for n in range(count):
        ledger.retain(
            {
                "kind": "answer",
                "interpretation": f"filter_{n}",
                "candidate_answer": {"entity": "X", "cost": n + 10},
            },
            {"phase": "actor", "source": "recorded-output.json"},
        )
    return ledger


def test_all_labelled_tables_retained_with_distinct_filter_definitions():
    text = "<stdout>all\nX 6 120.25\nY 3 80.50\n\nevent_marker_present\nX 2 30.75\nY 1 20.00\n</stdout>"
    observations = extract_calculations(text)
    assert [x["interpretation"] for x in observations] == [
        "all",
        "event_marker_present",
    ]
    assert observations[1]["results"][0] == {
        "entity": "X",
        "count": 2,
        "value": "30.75",
    }


def test_different_filters_with_same_answer_remain_distinct():
    ledger = make_ledger(0)
    value = {"entity": "X", "cost": 20}
    first = ledger.retain(
        {"kind": "answer", "interpretation": "filter_a", "candidate_answer": value},
        {"phase": "actor"},
    )
    second = ledger.retain(
        {"kind": "answer", "interpretation": "filter_b", "candidate_answer": value},
        {"phase": "review"},
    )
    duplicate = ledger.retain(
        {"kind": "answer", "interpretation": "filter_a", "candidate_answer": value},
        {"phase": "reflect"},
    )
    assert first != second and duplicate == first
    assert len(ledger.entries) == 2
    assert len(ledger.entries[0]["provenance"]) == 2


@pytest.mark.anyio
async def test_live_code_capture_writes_provenance_and_notifies_model():
    class Params(BaseModel):
        cmd: str

    async def execute(params):
        return ToolResult(
            content="<stdout>event_marker_present\nX 2 30.75\nY 1 20.00\n</stdout>",
            metadata=ToolUseCountMetadata(),
        )

    tool = Tool(
        name="code_exec", description="test", parameters=Params, executor=execute
    )
    ledger = CandidateLedger(workspace())
    ledger.phase = "actor"
    result = await ledger.wrap_code_tool(tool).executor(Params(cmd="calculate"))
    assert len(ledger.entries) == 1
    assert ledger.entries[0]["candidate_id"] in result.content
    assert ledger.entries[0]["provenance"][0]["source"].startswith(
        "candidate-ledger/sources/"
    )
    assert any(
        c.args[0] == "candidate-ledger/index.json"
        for c in ledger.workspace.write_file_bytes.call_args_list
    )


@pytest.mark.anyio
async def test_review_cannot_drop_seventh_candidate_from_five_item_shortlist():
    ledger = make_ledger()
    comparisons = [
        {
            "candidate_id": e["candidate_id"],
            "status": "plausible",
            "interpretation": e["interpretation"],
            "supporting_evidence": "recorded output",
            "counterevidence_or_gap": "definition unresolved",
        }
        for e in ledger.entries
    ]
    report = dict(
        outcome="partially_accomplished",
        arithmetic_verified=True,
        population_verified=False,
        question_alignment_verified=False,
        alignment_findings="Original scope unresolved",
        critique="definitions unresolved",
        alternative_search_summary="compared all observations",
    )
    tool = build_review_finish_tool(ledger)
    failed = await tool.executor(
        ReviewFinishParams(**report, candidate_comparisons=comparisons[:5])
    )
    assert not failed.success
    assert ledger.entries[6]["candidate_id"] in failed.content
    accepted = await tool.executor(
        ReviewFinishParams(**report, candidate_comparisons=comparisons)
    )
    assert accepted.success


@pytest.mark.anyio
async def test_reflect_must_address_unshortlisted_candidate_and_select_ledger_id():
    ledger = make_ledger()
    ids = [e["candidate_id"] for e in ledger.entries]
    assessments = [
        {
            "candidate_id": ident,
            "decision": "selected" if n == 6 else "unresolved",
            "reason": "computed evidence compared; semantics may need verification",
        }
        for n, ident in enumerate(ids)
    ]
    fields = dict(
        selected_alternative="new",
        selection_reason="selected a previously computed option",
        candidate_answer={"entity": "X", "cost": 16},
        selected_candidate_ids=[ids[6]],
        correction_steps=[
            "Verify the retained option against the requested population"
        ],
        candidate_assessments=assessments[:5],
    )
    tool = build_reflection_finish_tool([], ledger)
    failed = await tool.executor(ReflectionFinishParams(**fields))
    assert not failed.success
    fields["candidate_assessments"] = assessments
    accepted = await tool.executor(ReflectionFinishParams(**fields))
    assert accepted.success


def test_unknown_duplicate_or_missing_candidate_ids_fail_coverage():
    from agent.reactxen_agent.candidates import CandidateAssessment

    ledger = make_ledger(1)
    ident = ledger.entries[0]["candidate_id"]
    item = CandidateAssessment(
        candidate_id=ident, decision="unresolved", reason="needs evidence"
    )
    with pytest.raises(ValueError):
        ledger.require_coverage([])
    with pytest.raises(ValueError):
        ledger.require_coverage([item, item])
    with pytest.raises(ValueError):
        ledger.require_coverage(
            [
                CandidateAssessment(
                    candidate_id="unknown", decision="not_answer", reason="unknown"
                )
            ]
        )
