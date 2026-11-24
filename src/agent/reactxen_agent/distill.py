"""Optional, bounded question clarification after the first reflection.

The original question remains the task. A proposal is an advisory view only;
unknown business rules cannot be promoted to requirements by rewording.
"""

import json
import re

from pydantic import BaseModel, Field

DISTILL_SYSTEM = """You clarify an existing question; you do not solve it or change its meaning.
The original question is authoritative. Extract its objective, explicit constraints
and unresolved terms. Review/Reflect feedback identifies possible ambiguities,
not new user requirements. If a small wording change is genuinely needed, propose
minimal, non-overlapping substring edits grounded ONLY in the original question.
Do not add facts from data, a candidate answer, numbers, machines, dates, filters,
field predicates, units, scope, or business definitions not supplied by the user.
Do not resolve an ambiguous term by choosing the actor's or critic's preferred
interpretation. Preserve exclusions, negation, requested output keys and format.
If wording is already clear or the missing information is an unknown domain rule,
leave the question unchanged and list the unresolved ambiguity instead.
Never provide a solution, generalize the task, or add a Python algorithm.
Return ONLY JSON with: revision_needed (bool), original_question (verbatim),
revised_question, objective, explicit_constraints (verbatim excerpts), edits
(each old_text, new_text, reason), unresolved_ambiguities, rationale. Limit edits
to six and write concise explanations. For no revision, edits=[] and copy the
original into revised_question. No thinking tags or candidate answers."""

EQUIVALENCE_SYSTEM = """Check a question clarification for strict semantic preservation.
Compare only the immutable original question, proposed revised question and edits.
Do not answer the question. A clarification must preserve the original objective,
population, exclusions/negation, units, temporal scope, constraints, requested
output keys and output format. Reject added facts, candidate answers, data-specific
filters, assumed business definitions, expanded scope or narrowed interpretations.
Unknown terms must stay unresolved. Stylistic precision is allowed; task changes
are not. Return ONLY JSON: equivalent (bool), reason (brief)."""


class QuestionEdit(BaseModel):
    old_text: str = Field(min_length=1, max_length=500)
    new_text: str = Field(min_length=1, max_length=600)
    reason: str = Field(min_length=1, max_length=1000)


class DistillationProposal(BaseModel):
    revision_needed: bool
    original_question: str
    revised_question: str
    objective: str = Field(min_length=1)
    explicit_constraints: list[str] = Field(default_factory=list)
    edits: list[QuestionEdit] = Field(default_factory=list, max_length=6)
    unresolved_ambiguities: list[str] = Field(default_factory=list, max_length=10)
    rationale: str = Field(min_length=1)


class EquivalenceAudit(BaseModel):
    equivalent: bool
    reason: str = Field(min_length=1)


def validate_edits(original, proposal):
    if proposal.original_question != original:
        raise ValueError("Distiller altered the original question")
    if any(not item or item not in original for item in proposal.explicit_constraints):
        raise ValueError("Explicit constraints must quote the original question")
    if not proposal.revision_needed:
        if proposal.edits or proposal.revised_question != original:
            raise ValueError(
                "A no-revision proposal must preserve the question exactly"
            )
        return
    if not proposal.edits:
        raise ValueError("A revision requires explicit, source-grounded edits")
    spans = []
    for edit in proposal.edits:
        if original.count(edit.old_text) != 1:
            raise ValueError(
                "Each old_text must occur exactly once in the original question"
            )
        start = original.index(edit.old_text)
        spans.append((start, start + len(edit.old_text), edit.new_text))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Question edits must not overlap")
    reconstructed = original
    for start, end, replacement in reversed(spans):
        reconstructed = reconstructed[:start] + replacement + reconstructed[end:]
    if reconstructed != proposal.revised_question or reconstructed == original:
        raise ValueError("Revised question must match the declared edits exactly")

    def numbers(text: str) -> set[str]:
        return set(re.findall(r"(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)", text))

    if numbers(original) != numbers(reconstructed):
        raise ValueError(
            "Question revision must not add, remove or change numeric literals"
        )
    quoted = re.findall(r'"[A-Za-z_][A-Za-z0-9_]*"', original)
    if any(token not in reconstructed for token in quoted):
        raise ValueError(
            "Question revision removed or renamed a quoted identifier/output key"
        )


async def distill_question(complete, original, signals):
    record = {
        "status": "unchanged",
        "applied": False,
        "original_question": original,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    try:
        text, it, ot = await complete(
            DISTILL_SYSTEM,
            json.dumps(
                {
                    "original_question": original,
                    "ambiguity_signals_not_requirements": signals,
                },
                ensure_ascii=False,
            ),
        )
        record["input_tokens"] += it
        record["output_tokens"] += ot
        proposal = DistillationProposal.model_validate_json(text)
        record["proposal"] = proposal.model_dump()
        validate_edits(original, proposal)
        if not proposal.revision_needed:
            return record
        text, it, ot = await complete(
            EQUIVALENCE_SYSTEM,
            json.dumps(
                {
                    "original_question": original,
                    "proposed_revision": proposal.model_dump(),
                },
                ensure_ascii=False,
            ),
        )
        record["input_tokens"] += it
        record["output_tokens"] += ot
        audit = EquivalenceAudit.model_validate_json(text)
        record["audit"] = audit.model_dump()
        if audit.equivalent:
            record.update(
                status="accepted",
                applied=True,
                clarified_question=proposal.revised_question,
            )
        else:
            record.update(status="rejected", error=audit.reason)
    except Exception as exc:
        record.update(status="failed_or_rejected", error=f"{type(exc).__name__}: {exc}")
    return record


def clarification_block(original, clarified):
    if not clarified:
        return ""
    return (
        "\n\nADVISORY QUESTION CLARIFICATION\n"
        "The original user question remains authoritative and is unchanged in your task message. "
        "This wording clarification cannot add requirements or override the original objective, "
        "scope, exclusions, unresolved meanings, or requested output format. "
        "If there is any conflict, follow the original question.\n" + clarified + "\n"
    )


def needs_distillation(review_plan, reflection_plan):
    """Unresolved competing meanings matter even if overall flags were optimistic."""
    return (
        not review_plan.get("population_verified", True)
        or not review_plan.get("question_alignment_verified", True)
        or any(
            a.get("status") == "plausible" for a in review_plan.get("alternatives", [])
        )
        or any(
            c.get("status") == "plausible"
            for c in review_plan.get("candidate_comparisons", [])
        )
        or any(
            a.get("decision") == "unresolved"
            for a in reflection_plan.get("alternative_assessments", [])
        )
        or any(
            c.get("decision") == "unresolved"
            for c in reflection_plan.get("candidate_assessments", [])
        )
    )
