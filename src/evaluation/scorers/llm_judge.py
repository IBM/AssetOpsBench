"""LLM-As-Judge scorer.

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

from ..models import Scenario, ScorerResult
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

_PROMPT_TEMPLATE = """You are a critical reviewer tasked with evaluating the effectiveness and accuracy of an AI agent's response to a given task. Your goal is to determine whether the agent has successfully accomplished the task correctly based on the expected or characteristic behavior.

Evaluation Criteria:
1. **Task Completion:**
   - Verify if the agent executed all necessary actions (e.g., using the correct tools, retrieving data, performing the required analysis).
   - The agent's response should align with the predefined expected behavior for task completion.

2. **Data Retrieval & Accuracy:**
   - Ensure that the correct asset, location, time period, and sensor (if applicable) were used.
   - Verify if the task performed was related to the correct asset and sensor, and ensure the result corresponds to the correct time period.
   - Check if the agent retrieved the required data and if the forecasting, anomaly detection, or other results are correct.

3. **Generalized Result Verification:**
   - **Task Type Verification:** Based on the task type (forecasting, anomaly detection, classification, etc.), verify if the agent has returned the expected results.
       - For **forecasting** tasks: Ensure that the agent generated a forecast for the specified future period.
       - For **anomaly detection** tasks: Verify that anomalies are detected as expected (if anomalies were anticipated).
       - For other tasks (e.g., classification), ensure the task result matches the expected format and value.
   - **Comparison with Expected Output:** Check if the result matches the expected format, values, or outcomes as outlined in the characteristic answer.
   - **Data Integrity:** Ensure that the correct data (e.g., sensor, time period) was used in the task, and that it is consistent with the expected format and structure.

4. **Agent Sequence & Order:**
   - Ensure the agents (or tools) were called in the correct order and that all actions align with the expected behavior for agent interactions.
   - If the characteristic answer specifies certain agents (e.g., IoTAgent, TSFMAgent, FMSRAgent), verify that these were used and in the correct sequence.

5. **Clarity and Justification:**
   - Ensure the agent's response is clear and justified with adequate explanations or evidence to support the claims made.
   - There should be no contradictions between the agent's reasoning and the expected behavior outlined in the characteristic answer.

6. **Hallucination Check:**
   - Identify if the agent claims success without performing the necessary actions or without generating meaningful results.
   - If the agent provides a fabricated response or claims success where actions are missing, mark this as a hallucination.

Question: {question}
Characteristic Answer (Expected Behavior): {characteristic}
Agent's Thinking (turns / tool calls / outputs): {trajectory}
Agent's Final Response: {answer}

Output Format:
Your review must always be in JSON format. Do not include any additional formatting or Markdown in your response.
{{
    "task_completion": true/false,
    "data_retrieval_accuracy": true/false,
    "generalized_result_verification": true/false,
    "agent_sequence_correct": true/false,
    "clarity_and_justification": true/false,
    "hallucinations": true/false,
    "suggestions": "Optional. Actions or improvements for rectifying the response if applicable."
}}
(END OF RESPONSE)

Please provide your review based on the given criteria.
"""


class LLMJudgeScorer:
    """Closure-style scorer that holds an :class:`LLMBackend`."""

    def __init__(self, llm: LLMBackend, name: str = "llm_judge") -> None:
        self._llm = llm
        self.name = name

    def __call__(
        self, scenario: Scenario, answer: str, trajectory_text: str
    ) -> ScorerResult:
        characteristic = scenario.characteristic_form or scenario.expected_answer or ""
        if not characteristic:
            return ScorerResult(
                scorer=self.name,
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
        except Exception as exc:  # judge call failure is a scoring failure, not a crash
            _log.exception("llm_judge: backend error")
            return ScorerResult(
                scorer=self.name,
                passed=False,
                rationale=f"judge backend error: {exc}",
            )

        review = _parse_review(raw)
        if review is None:
            return ScorerResult(
                scorer=self.name,
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

        rationale = str(
            review.get("suggestions") or review.get("reason") or ""
        )[:500]
        return ScorerResult(
            scorer=self.name,
            passed=passed,
            score=round(score, 3),
            rationale=rationale,
            details=review,
        )


def _parse_review(raw: str) -> dict | None:
    if not raw:
        return None
    # Strip the reference prompt's "(END OF RESPONSE)" sentinel + any
    # leading prose / markdown fence before extracting the first {...}.
    text = raw.split("(END OF RESPONSE)")[0]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def install(llm: LLMBackend, name: str = "llm_judge") -> None:
    """Register an LLM-As-Judge scorer bound to ``llm`` under ``name``."""
    register(name, LLMJudgeScorer(llm, name=name))
