import json

import pytest

from evaluation import scorers
from evaluation.evaluator import Evaluator
from evaluation.models import PersistedTrajectory, Scenario
from evaluation.scorers.rul import rul_score


def scenario(**meta):
    # Synthetic fixture, independent of benchmark targets and adopted tolerances.
    return Scenario(
        id="rul-test",
        text="Estimate remaining cycles.",
        expected_answer=json.dumps({"remaining_cycles": 100}),
        scoring_method="rul_numeric_v1",
        evaluation_metadata={
            "scenario_meta": {
                "relative_tolerance": 0.1,
                **meta,
            }
        },
    )


def test_registered():
    assert scorers.get("rul_numeric_v1") is rul_score


@pytest.mark.parametrize(
    ("prediction", "passed"),
    [
        (100, True),
        (90, True),
        (110, True),
        (89.9999, False),
        (110.0001, False),
        (0, False),
    ],
)
def test_inclusive_decimal_boundaries(prediction, passed):
    result = rul_score(scenario(), json.dumps({"remaining_cycles": prediction}), "")
    assert result.passed is passed
    assert result.score == float(passed)
    assert result.details["relative_tolerance"] == 0.1


def test_tolerance_is_metadata_not_code():
    answer = '{"remaining_cycles":108}'
    assert rul_score(scenario(), answer, "").passed
    assert not rul_score(scenario(relative_tolerance="0.05"), answer, "").passed


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "not JSON",
        "null",
        "[]",
        "100",
        "{}",
        '{"remaining_cycles":true}',
        '{"remaining_cycles":-1}',
        '{"remaining_cycles":"100"}',
        '{"remaining_cycles":NaN}',
        '{"remaining_cycles":Infinity}',
        json.dumps({"remaining_cycles": 10**400}),
    ],
)
def test_invalid_prediction_fails(answer):
    result = rul_score(scenario(), answer, "")
    assert not result.passed and result.score == 0.0
    assert result.details["valid_prediction"] is False
    assert result.details["relative_tolerance"] == 0.1


@pytest.mark.parametrize("tolerance", [None, True, -1, "bad", "NaN", "Infinity"])
def test_invalid_tolerance_is_configuration_error(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        rul_score(
            scenario(relative_tolerance=tolerance), '{"remaining_cycles":100}', ""
        )


@pytest.mark.parametrize("actual", [0, -1, True, float("nan"), float("inf")])
def test_invalid_gold_is_harness_error(actual):
    case = scenario().model_copy(
        update={"expected_answer": json.dumps({"remaining_cycles": actual})}
    )
    with pytest.raises(ValueError):
        rul_score(case, '{"remaining_cycles":100}', "")


def test_diagnostic_fields_do_not_change_numerical_pass():
    result = rul_score(scenario(), '{"remaining_cycles":100,"asset_id":"wrong"}', "")
    assert result.passed
    assert result.details["asset_correct"] is False
    assert result.details["answer_schema_valid"] is False


def test_native_evaluator_dispatches_without_wrapper_or_policy_file():
    trajectory = PersistedTrajectory(
        run_id="test",
        scenario_id="rul-test",
        runner="test",
        model="test",
        question="",
        answer='{"remaining_cycles":108}',
    )
    result = Evaluator()._score_one(scenario(), trajectory)
    assert result.score.scorer == "rul_numeric_v1"
    assert result.score.passed
