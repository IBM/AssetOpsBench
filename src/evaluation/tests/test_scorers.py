"""Tests for the three scorer families: code-based, LLM-as-judge, semantic.

Code-Based scorers are skeletons only and have no behaviour tests yet.
"""

from __future__ import annotations

from evaluation import scorers as registry
from evaluation.scorers.code_based import exact_string_match, numeric_match
from evaluation.scorers.llm_judge import LLMJudgeScorer, install
from evaluation.scorers.semantic import semantic_similarity
from llm import LLMBackend


class _StubLLM(LLMBackend):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._response


class TestCodeBasedSkeletons:
    def test_exact_string_match_not_implemented(self, make_scenario):
        try:
            exact_string_match(make_scenario(expected_answer="x"), "x", "")
        except NotImplementedError:
            return
        raise AssertionError("expected NotImplementedError")

    def test_numeric_match_not_implemented(self, make_scenario):
        try:
            numeric_match(make_scenario(expected_answer="1.0"), "1.0", "")
        except NotImplementedError:
            return
        raise AssertionError("expected NotImplementedError")


class TestSemanticSkeleton:
    def test_semantic_similarity_not_implemented(self, make_scenario):
        try:
            semantic_similarity(make_scenario(), "a", "")
        except NotImplementedError:
            return
        raise AssertionError("expected NotImplementedError")


class TestRegistry:
    def test_skeleton_scorers_not_auto_registered(self):
        # code_based and semantic ship as skeletons; only llm_judge is
        # registered (lazily, via install()).
        assert "exact_string_match" not in registry.names()
        assert "numeric_match" not in registry.names()
        assert "semantic_similarity" not in registry.names()

    def test_get_unknown_raises(self):
        try:
            registry.get("does_not_exist")
        except KeyError as e:
            assert "does_not_exist" in str(e)
        else:
            raise AssertionError("expected KeyError")


class TestLLMJudgeScorer:
    def _all_pass_response(self) -> str:
        return (
            '{"task_completion": true, "data_retrieval_accuracy": true, '
            '"generalized_result_verification": true, "agent_sequence_correct": true, '
            '"clarity_and_justification": true, "hallucinations": false, '
            '"reason": "Looks good."}'
        )

    def test_passes_when_all_criteria_true(self, make_scenario):
        scorer = LLMJudgeScorer(_StubLLM(self._all_pass_response()))
        r = scorer(make_scenario(), "answer", "trajectory")
        assert r.passed
        assert r.score == 1.0
        assert r.rationale == "Looks good."

    def test_fails_on_hallucination(self, make_scenario):
        resp = self._all_pass_response().replace(
            '"hallucinations": false', '"hallucinations": true'
        )
        scorer = LLMJudgeScorer(_StubLLM(resp))
        r = scorer(make_scenario(), "answer", "trajectory")
        assert not r.passed
        # Score is penalized but not zeroed when 5/5 criteria pass.
        assert r.score < 1.0

    def test_handles_unparseable_response(self, make_scenario):
        scorer = LLMJudgeScorer(_StubLLM("not json at all"))
        r = scorer(make_scenario(), "a", "t")
        assert not r.passed
        assert "unparseable" in r.rationale

    def test_handles_markdown_fenced_response(self, make_scenario):
        wrapped = "Here you go:\n```json\n" + self._all_pass_response() + "\n```"
        scorer = LLMJudgeScorer(_StubLLM(wrapped))
        r = scorer(make_scenario(), "a", "t")
        assert r.passed

    def test_missing_characteristic_short_circuits(self, make_scenario):
        scorer = LLMJudgeScorer(_StubLLM(self._all_pass_response()))
        s = make_scenario(characteristic_form=None, expected_answer=None)
        r = scorer(s, "a", "t")
        assert not r.passed
        assert "characteristic_form" in r.rationale

    def test_install_registers_under_default_name(self, make_scenario):
        install(_StubLLM(self._all_pass_response()))
        assert "llm_judge" in registry.names()
        scorer = registry.get("llm_judge")
        r = scorer(make_scenario(), "a", "t")
        assert r.passed
