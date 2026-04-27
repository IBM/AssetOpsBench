"""LLM-judge grader.

Free-form answers are scored against ``scenario.characteristic_form``
using a six-criterion rubric (task completion, data retrieval accuracy,
result verification, agent sequence, clarity, hallucinations) — the
same shape as ``aobench/scenario-server/grading/graders.evaluation_agent``
but built directly on :class:`~llm.LLMBackend` so the evaluation module
has no dependency on the scenario-server codebase.
"""

from __future__ import annotations

import json
import logging
import re

from llm import LLMBackend

from ..models import GradeResult, Scenario
from . import register

_log = logging.getLogger(__name__)

_RUBRIC_KEYS = (
    "task_completion",
    "data_retrieval_accuracy",
    "generalized_result_verification",
    "agent_sequence_correct",
    "clarity_and_justification",
    "hallucinations",
)

_PROMPT_TEMPLATE = """You are an evaluation judge for an industrial-asset-operations agent.

Score the agent response against the expected characteristic answer using the six criteria below. Respond ONLY with a JSON object, no prose.

QUESTION:
{question}

EXPECTED CHARACTERISTIC:
{characteristic}

AGENT RESPONSE:
{answer}

AGENT TRAJECTORY (turns / tool calls / outputs):
{trajectory}

Return JSON with these boolean fields plus a one-sentence reason:

{{
  "task_completion": <bool>,
  "data_retrieval_accuracy": <bool>,
  "generalized_result_verification": <bool>,
  "agent_sequence_correct": <bool>,
  "clarity_and_justification": <bool>,
  "hallucinations": <bool>,
  "reason": "<one sentence>"
}}

The agent passes overall iff the first five are true AND hallucinations is false."""


class LLMJudgeGrader:
    """Closure-style grader that holds an :class:`LLMBackend`."""

    def __init__(self, llm: LLMBackend, name: str = "llm_judge") -> None:
        self._llm = llm
        self.name = name

    def __call__(
        self, scenario: Scenario, answer: str, trajectory_text: str
    ) -> GradeResult:
        characteristic = scenario.characteristic_form or scenario.expected_answer or ""
        if not characteristic:
            return GradeResult(
                grading_method=self.name,
                passed=False,
                rationale="scenario has neither characteristic_form nor expected_answer",
            )

        prompt = _PROMPT_TEMPLATE.format(
            question=scenario.text,
            characteristic=characteristic,
            answer=answer,
            trajectory=trajectory_text[:8000],
        )

        try:
            raw = self._llm.generate(prompt)
        except Exception as exc:  # judge call failure is a grading failure, not a crash
            _log.exception("llm_judge: backend error")
            return GradeResult(
                grading_method=self.name,
                passed=False,
                rationale=f"judge backend error: {exc}",
            )

        review = _parse_review(raw)
        if review is None:
            return GradeResult(
                grading_method=self.name,
                passed=False,
                rationale="judge returned unparseable JSON",
                details={"raw": raw[:2000]},
            )

        passed = (
            review.get("task_completion") is True
            and review.get("data_retrieval_accuracy") is True
            and review.get("generalized_result_verification") is True
            and review.get("agent_sequence_correct") is True
            and review.get("clarity_and_justification") is True
            and review.get("hallucinations") is False
        )
        score = sum(1 for k in _RUBRIC_KEYS[:5] if review.get(k) is True) / 5.0
        if review.get("hallucinations") is True:
            score = max(0.0, score - 0.2)

        return GradeResult(
            grading_method=self.name,
            passed=passed,
            score=round(score, 3),
            rationale=str(review.get("reason", ""))[:500],
            details=review,
        )


def _parse_review(raw: str) -> dict | None:
    if not raw:
        return None
    # Tolerate leading prose / markdown fences by extracting the first {...} block.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def install(llm: LLMBackend, name: str = "llm_judge") -> None:
    """Register an LLM-judge grader bound to ``llm`` under ``name``."""
    register(name, LLMJudgeGrader(llm, name=name))
