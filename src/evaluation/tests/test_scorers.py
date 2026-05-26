"""Tests for the three scorer families: code-based, LLM-as-judge, semantic.
"""

from __future__ import annotations

from evaluation import scorers as registry
from evaluation.scorers.code_based import exact_string_match, numeric_match, install
from evaluation.scorers.llm_judge import LLMJudgeScorer as LLMJudgeScorerClass
from evaluation.scorers.llm_judge import install as install_llm_judge
from evaluation.scorers.semantic import semantic_similarity
from llm import LLMBackend


class _StubLLM(LLMBackend):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._response


class TestCodeBasedScorers:
    def test_exact_string_match_exact_match(self, make_scenario):
        scenario = make_scenario(expected_answer="test answer")
        result = exact_string_match(scenario, "test answer", "")
        assert result.passed is True
        assert result.score == 1.0
        assert result.scorer == "exact_string_match"
    
    def test_exact_string_match_case_insensitive(self, make_scenario):
        scenario = make_scenario(expected_answer="Test Answer")
        result = exact_string_match(scenario, "test answer", "")
        assert result.passed is True
        assert result.score == 1.0
    
    def test_exact_string_match_whitespace_normalization(self, make_scenario):
        scenario = make_scenario(expected_answer="test  answer")
        result = exact_string_match(scenario, "test answer", "")
        assert result.passed is True
        assert result.score == 1.0
    
    def test_exact_string_match_no_match(self, make_scenario):
        scenario = make_scenario(expected_answer="correct")
        result = exact_string_match(scenario, "wrong", "")
        assert result.passed is False
        assert result.score == 0.0
        assert "Expected" in result.rationale
    
    def test_exact_string_match_none_expected(self, make_scenario):
        scenario = make_scenario(expected_answer=None)
        result = exact_string_match(scenario, "answer", "")
        assert result.passed is False
        assert result.score == 0.0
        assert "None" in result.rationale
    
    def test_numeric_match_exact_match(self, make_scenario):
        scenario = make_scenario(expected_answer="42.5")
        result = numeric_match(scenario, "42.5", "")
        assert result.passed is True
        assert result.score == 1.0
        assert result.scorer == "numeric_match"
    
    def test_numeric_match_within_tolerance(self, make_scenario):
        scenario = make_scenario(expected_answer="100.0")
        result = numeric_match(scenario, "100.5", "")
        # Default tolerance is 1% relative, so 100.5 should be within tolerance
        assert result.passed is True
        assert result.score == 1.0
    
    def test_numeric_match_with_units(self, make_scenario):
        scenario = make_scenario(expected_answer="42.5 kg")
        result = numeric_match(scenario, "42.5 kg", "")
        assert result.passed is True
        assert result.score == 1.0
    
    def test_numeric_match_custom_absolute_tolerance(self, make_scenario):
        scenario = make_scenario(expected_answer="100.0", tolerance=1.0)
        result = numeric_match(scenario, "100.5", "")
        assert result.passed is True
        assert result.score == 1.0
    
    def test_numeric_match_custom_relative_tolerance(self, make_scenario):
        scenario = make_scenario(expected_answer="100.0", tolerance={"relative": 0.001})
        result = numeric_match(scenario, "100.5", "")
        # 0.5% difference should fail with 0.1% tolerance
        assert result.passed is False
        assert result.score == 0.0
    
    def test_numeric_match_no_match(self, make_scenario):
        scenario = make_scenario(expected_answer="100.0")
        result = numeric_match(scenario, "200.0", "")
        assert result.passed is False
        assert result.score == 0.0
        assert "Expected" in result.rationale
    
    def test_numeric_match_none_expected(self, make_scenario):
        scenario = make_scenario(expected_answer=None)
        result = numeric_match(scenario, "42.5", "")
        assert result.passed is False
        assert result.score == 0.0
        assert "None" in result.rationale
    
    def test_numeric_match_invalid_answer(self, make_scenario):
        scenario = make_scenario(expected_answer="100.0")
        result = numeric_match(scenario, "not a number", "")
        assert result.passed is False
        assert result.score == 0.0
        assert "Failed to parse" in result.rationale


class TestSemanticSkeleton:
    def test_semantic_similarity_not_implemented(self, make_scenario):
        try:
            semantic_similarity(make_scenario(), "a", "")
        except NotImplementedError:
            return
        raise AssertionError("expected NotImplementedError")


class TestRegistry:
    def test_code_based_scorers_registered_after_install(self):
        # code_based scorers are registered when install() is called
        install()
        assert "exact_string_match" in registry.names()
        assert "numeric_match" in registry.names()
        # semantic_similarity is still a skeleton
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
        scorer = LLMJudgeScorerClass(_StubLLM(self._all_pass_response()))
        r = scorer(make_scenario(), "answer", "trajectory")
        assert r.passed
        assert r.score == 1.0
        assert r.rationale == "Looks good."

    def test_fails_on_hallucination(self, make_scenario):
        resp = self._all_pass_response().replace(
            '"hallucinations": false', '"hallucinations": true'
        )
        scorer = LLMJudgeScorerClass(_StubLLM(resp))
        r = scorer(make_scenario(), "answer", "trajectory")
        assert not r.passed
        # Score is penalized but not zeroed when 5/5 criteria pass.
        assert r.score < 1.0

    def test_handles_unparseable_response(self, make_scenario):
        scorer = LLMJudgeScorerClass(_StubLLM("not json at all"))
        r = scorer(make_scenario(), "a", "t")
        assert not r.passed
        assert "unparseable" in r.rationale

    def test_handles_markdown_fenced_response(self, make_scenario):
        wrapped = "Here you go:\n```json\n" + self._all_pass_response() + "\n```"
        scorer = LLMJudgeScorerClass(_StubLLM(wrapped))
        r = scorer(make_scenario(), "a", "t")
        assert r.passed

    def test_missing_characteristic_short_circuits(self, make_scenario):
        scorer = LLMJudgeScorerClass(_StubLLM(self._all_pass_response()))
        s = make_scenario(characteristic_form=None, expected_answer=None)
        r = scorer(s, "a", "t")
        assert not r.passed
        assert "characteristic_form" in r.rationale

    def test_install_registers_under_default_name(self, make_scenario):
        install_llm_judge(_StubLLM(self._all_pass_response()))
        assert "llm_judge" in registry.names()
        scorer = registry.get("llm_judge")
        r = scorer(make_scenario(), "a", "t")
        assert r.passed
