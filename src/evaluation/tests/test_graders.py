"""Tests for deterministic + LLM-judge graders."""

from __future__ import annotations

from evaluation import graders as registry
from evaluation.graders.deterministic import exact_string_match, numeric_match
from evaluation.graders.llm_judge import LLMJudgeGrader, install
from llm import LLMBackend


class _StubLLM(LLMBackend):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._response


class TestExactStringMatch:
    def test_match_case_insensitive(self, make_scenario):
        s = make_scenario(expected_answer="Hello World")
        r = exact_string_match(s, "hello world", "")
        assert r.passed and r.score == 1.0

    def test_mismatch(self, make_scenario):
        s = make_scenario(expected_answer="foo")
        r = exact_string_match(s, "bar", "")
        assert not r.passed
        assert r.details["expected"] == "foo"

    def test_missing_expected(self, make_scenario):
        s = make_scenario(expected_answer=None)
        r = exact_string_match(s, "anything", "")
        assert not r.passed
        assert "expected_answer" in r.rationale


class TestNumericMatch:
    def test_within_tolerance(self, make_scenario):
        s = make_scenario(expected_answer="3.14159")
        r = numeric_match(s, "3.141591", "")
        assert r.passed

    def test_unparseable(self, make_scenario):
        s = make_scenario(expected_answer="3.14")
        r = numeric_match(s, "not a number", "")
        assert not r.passed
        assert "could not parse" in r.rationale

    def test_custom_tolerance(self, make_scenario):
        s = make_scenario(expected_answer="100", tolerance=0.05)
        r = numeric_match(s, "104", "")
        assert r.passed


class TestRegistry:
    def test_deterministic_graders_registered(self):
        assert "exact_string_match" in registry.names()
        assert "numeric_match" in registry.names()

    def test_get_unknown_raises(self):
        try:
            registry.get("does_not_exist")
        except KeyError as e:
            assert "does_not_exist" in str(e)
        else:
            raise AssertionError("expected KeyError")


class TestLLMJudgeGrader:
    def _all_pass_response(self) -> str:
        return (
            '{"task_completion": true, "data_retrieval_accuracy": true, '
            '"generalized_result_verification": true, "agent_sequence_correct": true, '
            '"clarity_and_justification": true, "hallucinations": false, '
            '"reason": "Looks good."}'
        )

    def test_passes_when_all_criteria_true(self, make_scenario):
        grader = LLMJudgeGrader(_StubLLM(self._all_pass_response()))
        r = grader(make_scenario(), "answer", "trajectory")
        assert r.passed
        assert r.score == 1.0
        assert r.rationale == "Looks good."

    def test_fails_on_hallucination(self, make_scenario):
        resp = self._all_pass_response().replace(
            '"hallucinations": false', '"hallucinations": true'
        )
        grader = LLMJudgeGrader(_StubLLM(resp))
        r = grader(make_scenario(), "answer", "trajectory")
        assert not r.passed
        # Score is penalized but not zeroed when 5/5 criteria pass.
        assert r.score < 1.0

    def test_handles_unparseable_response(self, make_scenario):
        grader = LLMJudgeGrader(_StubLLM("not json at all"))
        r = grader(make_scenario(), "a", "t")
        assert not r.passed
        assert "unparseable" in r.rationale

    def test_handles_markdown_fenced_response(self, make_scenario):
        wrapped = "Here you go:\n```json\n" + self._all_pass_response() + "\n```"
        grader = LLMJudgeGrader(_StubLLM(wrapped))
        r = grader(make_scenario(), "a", "t")
        assert r.passed

    def test_missing_characteristic_short_circuits(self, make_scenario):
        grader = LLMJudgeGrader(_StubLLM(self._all_pass_response()))
        s = make_scenario(characteristic_form=None, expected_answer=None)
        r = grader(s, "a", "t")
        assert not r.passed
        assert "characteristic_form" in r.rationale

    def test_install_registers_under_default_name(self, make_scenario):
        install(_StubLLM(self._all_pass_response()))
        assert "llm_judge" in registry.names()
        grader = registry.get("llm_judge")
        r = grader(make_scenario(), "a", "t")
        assert r.passed
