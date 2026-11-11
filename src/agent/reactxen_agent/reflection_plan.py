"""Validated final reflection decisions; never promote unfinished model text."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

from ..stirrup_agent.finish_tool import _finish_executor
from .candidates import CandidateAssessment
from .evidence_paths import EvidencePaths


class AlternativeAssessment(BaseModel):
    alternative_id: str = Field(min_length=1, max_length=100)
    decision: Literal["selected", "rejected", "unresolved"]
    reason: str = Field(min_length=1, max_length=1500)
    evidence_paths: EvidencePaths = Field(default_factory=list, max_length=10)


class ReflectionPlan(BaseModel):
    selected_alternative: str = Field(
        min_length=1,
        max_length=100,
        description="An offered alternative_id, 'new' for a new correction, or 'none' if evidence is insufficient.",
    )
    selection_reason: str = Field(min_length=1, max_length=2000)
    alternative_assessments: list[AlternativeAssessment] = Field(
        default_factory=list, max_length=10
    )
    evidence_paths: EvidencePaths = Field(default_factory=list, max_length=10)
    candidate_answer: dict | str | None = None
    correction_steps: list[str] = Field(min_length=1, max_length=5)
    candidate_assessments: list[CandidateAssessment] = Field(
        default_factory=list,
        description="Assess EVERY retained candidate_id in candidate-ledger/index.json; no shortlist limit. Give evidence for selection/rejection or preserve unresolved options.",
    )
    selected_candidate_ids: list[str] = Field(
        default_factory=list,
        description="Ledger IDs supporting the chosen candidate_answer; match assessments marked selected.",
    )


class ReflectionFinishParams(ReflectionPlan):
    """Expose every field and length limit directly in the model's tool schema."""

    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Optional individual existing output FILE paths only. Never list directories. "
            "Use [] if nothing needs exporting; evidence citations belong in evidence_paths."
        ),
    )

    @model_validator(mode="after")
    def reject_deliberation(self):
        value = self.model_dump_json().lower()
        if "<think" in value or "</think" in value:
            raise ValueError(
                "Return final conclusions and evidence without thinking tags"
            )
        return self

    @property
    def answer(self):
        return self.model_dump_json(exclude={"paths"})


def validate_selection(plan: ReflectionPlan, alternative_ids):
    expected = set(alternative_ids)
    assessed = [a.alternative_id for a in plan.alternative_assessments]
    if set(assessed) != expected or len(assessed) != len(expected):
        raise ValueError(
            "Assess every offered alternative exactly once, using its supplied alternative_id"
        )
    if plan.selected_alternative not in expected | {"new", "none"}:
        raise ValueError(
            "selected_alternative must identify an offered alternative, 'new', or 'none'"
        )
    selected = [
        a.alternative_id
        for a in plan.alternative_assessments
        if a.decision == "selected"
    ]
    wanted = (
        [plan.selected_alternative] if plan.selected_alternative in expected else []
    )
    if selected != wanted:
        raise ValueError(
            "The selected alternative must match the assessment marked selected"
        )


def build_reflection_finish_tool(alternative_ids, ledger=None):
    async def execute(params: ReflectionFinishParams):
        plan = ReflectionPlan.model_validate_json(params.answer)
        try:
            validate_selection(plan, alternative_ids)
            if ledger is not None:
                ledger.require_coverage(plan.candidate_assessments)
                selected = {
                    a.candidate_id
                    for a in plan.candidate_assessments
                    if a.decision == "selected"
                }
                if set(plan.selected_candidate_ids) != selected:
                    raise ValueError(
                        "selected_candidate_ids must match candidate assessments marked selected"
                    )
                if (
                    plan.candidate_answer is not None
                    and ledger.entries
                    and not selected
                ):
                    raise ValueError(
                        "Retain the proposed answer with retain_candidate and select its supporting ledger ID before finishing"
                    )
        except ValueError as exc:
            return ToolResult(
                content=str(exc), success=False, metadata=ToolUseCountMetadata()
            )
        return await _finish_executor(params)

    return Tool(
        name="finish",
        parameters=ReflectionFinishParams,
        executor=execute,
        description="Finish reflection using the typed fields shown in this tool schema; assess ALL offered alternatives. Respect each field length limit and call alone.",
    )


def format_guidance(plan: ReflectionPlan):
    lines = [
        f"Selected alternative: {plan.selected_alternative}",
        f"Selection reason: {plan.selection_reason}",
    ]
    if plan.candidate_answer is not None:
        import json

        lines.append(
            "Candidate answer to verify: "
            + json.dumps(plan.candidate_answer, ensure_ascii=False)
        )
    lines.append(
        "Evidence: "
        + (", ".join(plan.evidence_paths) or "Evidence gaps remain; see assessments.")
    )
    for assessment in plan.alternative_assessments:
        lines.append(
            f"{assessment.alternative_id}: {assessment.decision} — {assessment.reason} "
            f"[evidence: {', '.join(assessment.evidence_paths)}]"
        )
    if plan.candidate_assessments:
        lines.append(
            "All retained candidate decisions (full provenance: candidate-ledger/index.json):"
        )
        for assessment in plan.candidate_assessments:
            lines.append(
                f"{assessment.candidate_id}: {assessment.decision} — {assessment.reason}"
            )
    lines.extend(f"{i}. {step}" for i, step in enumerate(plan.correction_steps, 1))
    return "\n".join(lines)
