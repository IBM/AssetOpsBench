import json

import pytest

from evaluation.models import Scenario
from evaluation.scorers.fmea import fmea_score


def _scenario(gold, *, tag="asset_modes", weights=None):
    dimensions = weights or [
        {"name": "coverage", "weight": 0.35},
        {"name": "precision", "weight": 0.35},
        {"name": "instruction_following", "weight": 0.2},
        {"name": "grounding", "weight": 0.1},
    ]
    return Scenario(
        id="9001",
        text="",
        expected_answer=json.dumps(gold),
        evaluation_metadata={
            "scenario_meta": {"tag": tag},
            "rubric": {"dimensions": dimensions},
        },
    )


def test_asset_modes_awards_set_precision_and_recall():
    scenario = _scenario({"asset": "Electric Motor", "failure_modes": {"A": "", "B": ""}})
    answer = json.dumps({"asset": "electric motor", "failure_modes": {"A": "", "C": ""}})

    result = fmea_score(scenario, answer, "")

    assert result.passed is False
    assert result.details["dimensions"]["coverage"] == 0.5
    assert result.details["dimensions"]["precision"] == 0.5
    assert result.details["asset_match"] is True
    assert result.details["missing_modes"] == ["B"]
    assert result.details["extra_modes"] == ["C"]
    assert result.score == 0.5


def test_asset_modes_perfect_answer_passes_case_insensitive_asset():
    gold = {"asset": "Electric Motor", "failure_modes": {"A": ""}}
    result = fmea_score(
        _scenario(gold), '{"asset":"electric motor","failure_modes":{"A":""}}', ""
    )
    assert result.passed is True
    assert result.score == 1.0


def test_markdown_fence_gets_content_credit_but_fails_contract():
    gold = {"asset": "Electric Motor", "failure_modes": {"A": ""}}
    result = fmea_score(_scenario(gold), f"```json\n{json.dumps(gold)}\n```", "")
    assert result.passed is True
    assert result.details["semantic_exact"] is True
    assert result.details["strict_success"] is False
    assert result.details["json_only"] is False
    assert result.details["dimensions"]["instruction_following"] == 0.0


def test_analytics_scores_atomic_facts_and_ordered_arrays():
    scenario = _scenario(
        {
            "ranking": ["Seal", "Bearing", "Motor"],
            "counts": {"Seal": 16, "Bearing": 13, "Motor": 7},
        },
        tag="ranking_top3",
    )
    answer = json.dumps(
        {
            "ranking": ["Bearing", "Seal", "Motor"],
            "counts": {"Seal": 16, "Bearing": 13, "Motor": 7},
        }
    )
    result = fmea_score(scenario, answer, "")
    assert result.passed is False
    assert result.details["dimensions"]["coverage"] == pytest.approx(4 / 6)
    assert result.details["wrong_value_keys"] == ["answer.ranking[0]", "answer.ranking[1]"]


def test_duplicate_nested_mode_key_fails_json_contract():
    scenario = _scenario({"asset": "Volute", "failure_modes": {"Corrosion": ""}})
    answer = '{"asset":"Volute","failure_modes":{"Corrosion":"","Corrosion":""}}'
    result = fmea_score(scenario, answer, "")
    assert result.passed is True
    assert result.details["semantic_exact"] is True
    assert result.details["strict_success"] is False
    assert result.details["duplicate_keys"] == ["Corrosion"]


def test_trajectory_reports_catalog_and_out_of_scope_tool_use():
    scenario = _scenario({"asset": "Electric Motor", "failure_modes": {"A": ""}})
    trajectory = json.dumps(
        {"turns": [{"tool_calls": [
            {"name": "utilities__get_asset_catalog"},
            {"name": "utilities__get_failure_mode_catalog"},
            {"name": "fmsr__get_failure_modes"},
        ]}]}
    )
    result = fmea_score(scenario, scenario.expected_answer, trajectory)
    assert result.details["missing_catalog_tools"] == []
    assert result.details["out_of_scope_domain_tools"] == ["fmsr__get_failure_modes"]
    assert result.details["catalog_process_compliant"] is False


def test_pass_uses_strict_greater_than_point_seven_f1():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(7)}}
    )
    below = {
        "asset": "Electric Motor",
        "failure_modes": {
            **{str(i): "" for i in range(4)},
            **{f"extra-{i}": "" for i in range(3)},
        },
    }
    result = fmea_score(scenario, json.dumps(below), "")
    assert result.score == pytest.approx(4 / 7, abs=1e-6)
    assert result.passed is False
    assert result.details["f1_pass_threshold"] == 0.7


def test_f1_above_point_seven_passes_without_requiring_exact_match():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(8)}}
    )
    partial = {
        "asset": "Electric Motor",
        "failure_modes": {str(i): "" for i in range(6)},
    }
    result = fmea_score(scenario, json.dumps(partial), "")
    assert result.score == pytest.approx(6 / 7, abs=1e-6)
    assert result.passed is True
    assert result.details["strict_success"] is False
