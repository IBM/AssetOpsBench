"""Typed review verdicts and explicit investigation of competing answers."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

from ..stirrup_agent.finish_tool import _finish_executor
from .candidates import CandidateComparison
from .evidence_paths import EvidencePaths
from .models import Outcome


class ReviewAlternative(BaseModel):
    interpretation: str = Field(min_length=1, max_length=1200)
    candidate_answer: dict | str | None = None
    status: Literal["supported", "plausible", "ruled_out"]
    supporting_evidence: str = Field(min_length=1, max_length=1500)
    counterevidence_or_gap: str = Field(min_length=1, max_length=1500)
    evidence_paths: EvidencePaths = Field(default_factory=list, max_length=10)


class ReviewFinishParams(BaseModel):
    outcome: Outcome
    arithmetic_verified: bool = Field(
        description="Whether the candidate's calculation was independently reproduced"
    )
    question_alignment_verified: bool = Field(
        description="Whether the generated answer precisely answers the ORIGINAL question: objective, scope, exclusions, units and output format; not merely a revised interpretation"
    )
    alignment_findings: str = Field(
        min_length=1,
        max_length=2000,
        description="Explain direct compliance or specific deviations from the original question. Advisory distillation never overrides it.",
    )
    population_verified: bool = Field(
        description="Whether evidence establishes that its filters, scope, units and exclusions match the request, separately from arithmetic"
    )
    critique: str = Field(min_length=1, max_length=4000)
    alternative_search_summary: str = Field(
        min_length=1,
        max_length=1500,
        description="Distinct interpretations and counterexamples investigated; explain if none are plausible",
    )
    alternatives: list[ReviewAlternative] = Field(
        default_factory=list,
        max_length=5,
        description="Meaningfully different filters, methods, populations or candidate answers, including plausible unresolved options. Do not fabricate options or just repeat the actor answer.",
    )
    candidate_comparisons: list[CandidateComparison] = Field(
        default_factory=list,
        description="Compare EVERY ID in candidate-ledger/index.json exactly once, including candidates found in tool outputs. This list is NOT limited to the five new alternatives.",
    )
    evidence_paths: EvidencePaths = Field(default_factory=list, max_length=10)
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Optional individual existing output FILE paths only. Never list directories. "
            "Use [] if nothing needs exporting; evidence citations belong in evidence_paths."
        ),
    )

    @model_validator(mode="after")
    def require_resolved_verification(self):
        if self.outcome is Outcome.ACCOMPLISHED and (
            not self.arithmetic_verified
            or not self.population_verified
            or not self.question_alignment_verified
            or any(a.status == "plausible" for a in self.alternatives)
            or any(c.status == "plausible" for c in self.candidate_comparisons)
        ):
            raise ValueError(
                "Use partially_accomplished when arithmetic/population/original-question alignment is unverified or a plausible competing interpretation is unresolved"
            )
        return self

    @property
    def answer(self):
        return self.model_dump_json(exclude={"paths"})


def build_review_finish_tool(ledger=None):
    async def execute(params: ReviewFinishParams):
        if ledger is not None:
            try:
                ledger.require_coverage(params.candidate_comparisons)
            except ValueError as exc:
                return ToolResult(
                    content=str(exc), success=False, metadata=ToolUseCountMetadata()
                )
        return await _finish_executor(params)

    return Tool(
        name="finish",
        parameters=ReviewFinishParams,
        executor=execute,
        description="Finish review with typed verification checks and a comparison for EVERY retained ledger candidate. Never omit a candidate because it was not shortlisted. Call alone.",
    )
