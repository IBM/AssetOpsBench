import json

import pytest

from evaluation import scorers
from evaluation.evaluator import Evaluator
from evaluation.models import PersistedTrajectory, Scenario
from evaluation.scorers.rul import rul_score

# Independent synthetic fixtures; no benchmark targets or adopted tolerance.
GOLD = {
    "asset_id": "asset-target",
    "cutoff_timestamp": "2040-01-01T00:00:00Z",
    "remaining_cycles": 100000,
    "failure_within_horizon": True,
    "documented_failed_assets": ["asset-A", "asset-B"],
    "ongoing_assets": ["asset-target"],
}


def scenario(horizon=150000, **meta):
    gold = dict(GOLD, failure_within_horizon=GOLD["remaining_cycles"] <= horizon)
    return Scenario(
        id="rul-test",
        text=(
            f"Predict failure within the next {horizon} operating cycles, including equality. "
            f"Return failure_within_horizon equal to remaining_cycles <= {horizon}."
        ),
        expected_answer=json.dumps(gold),
        scoring_method="rul_numeric_v1",
        evaluation_metadata={"scenario_meta": {"relative_tolerance": 0.1, **meta}},
    )


def answer(value=100000, horizon=150000, **changes):
    prediction = dict(
        GOLD, remaining_cycles=value, failure_within_horizon=value <= horizon
    )
    prediction.update(changes)
    return json.dumps(prediction)


def test_registered():
    assert scorers.get("rul_numeric_v1") is rul_score


@pytest.mark.parametrize(
    ("prediction", "passed"),
    [
        (100000, True),
        (90000, True),
        (110000, True),
        (89999.9999, False),
        (110000.0001, False),
        (0, False),
    ],
)
def test_inclusive_decimal_boundaries(prediction, passed):
    result = rul_score(scenario(), answer(prediction), "")
    assert result.passed is passed
    assert result.score == float(passed)
    assert result.details["numerical_pass"] is passed
    assert result.details["label_pass"] is True
    assert result.details["relative_tolerance"] == 0.1


def test_tolerance_is_metadata_not_code():
    assert rul_score(scenario(), answer(108000), "").passed
    assert not rul_score(scenario(relative_tolerance="0.05"), answer(108000), "").passed


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not JSON",
        "null",
        "[]",
        "100000",
        "{}",
        '{"remaining_cycles":true}',
        '{"remaining_cycles":-1}',
        '{"remaining_cycles":"100000"}',
        '{"remaining_cycles":NaN}',
        '{"remaining_cycles":Infinity}',
        json.dumps({"remaining_cycles": 10**400}),
    ],
)
def test_invalid_prediction_fails_both_checks(raw):
    result = rul_score(scenario(), raw, "")
    assert not result.passed and result.score == 0.0
    assert result.details["valid_prediction"] is False
    assert result.details["label_pass"] is False
    assert result.details["numerical_pass"] is False


@pytest.mark.parametrize("tolerance", [None, True, -1, "bad", "NaN", "Infinity"])
def test_invalid_tolerance_is_configuration_error(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        rul_score(scenario(relative_tolerance=tolerance), answer(), "")


@pytest.mark.parametrize("actual", [0, -1, True, float("nan"), float("inf")])
def test_invalid_gold_is_harness_error(actual):
    case = scenario().model_copy(
        update={"expected_answer": json.dumps(dict(GOLD, remaining_cycles=actual))}
    )
    with pytest.raises(ValueError):
        rul_score(case, answer(), "")


@pytest.mark.parametrize(
    "field,value",
    [
        ("asset_id", "wrong"),
        ("cutoff_timestamp", "2040-01-02T00:00:00Z"),
        ("documented_failed_assets", []),
        ("documented_failed_assets", ["asset-A", "asset-A", "asset-B"]),
        ("documented_failed_assets", ["asset-B", "asset-A"]),
        ("ongoing_assets", []),
        ("ongoing_assets", ["wrong"]),
        ("failure_within_horizon", False),
        ("failure_within_horizon", "true"),
        ("failure_within_horizon", 1),
        ("failure_within_horizon", 0),
    ],
)
def test_numeric_pass_cannot_override_wrong_labels(field, value):
    result = rul_score(scenario(), answer(**{field: value}), "")
    assert result.details["numerical_pass"] is True
    assert result.details["label_pass"] is False
    assert not result.passed and result.score == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "asset_id",
        "cutoff_timestamp",
        "documented_failed_assets",
        "ongoing_assets",
        "failure_within_horizon",
    ],
)
def test_missing_label_fails(field):
    prediction = dict(GOLD)
    prediction.pop(field)
    result = rul_score(scenario(), json.dumps(prediction), "")
    assert result.details["numerical_pass"] is True
    assert not result.details["label_pass"] and not result.passed


@pytest.mark.parametrize("horizon", [30000, 50000, 80000, 150000])
def test_horizon_comes_from_question_without_metadata(horizon):
    result = rul_score(scenario(horizon=horizon), answer(horizon=horizon), "")
    assert result.passed and result.details["horizon_cycles"] == horizon
    assert result.details["forecast_decision_consistent"] is True


def test_horizon_includes_equality():
    result = rul_score(scenario(horizon=100000), answer(horizon=100000), "")
    assert result.passed


@pytest.mark.parametrize("label", [True, False])
def test_tolerated_forecast_crossing_horizon_fails(label):
    result = rul_score(
        scenario(horizon=105000),
        answer(108000, horizon=105000, failure_within_horizon=label),
        "",
    )
    assert result.details["numerical_pass"] is True
    assert not result.details["label_pass"] and not result.passed
    assert result.details["horizon_correct"] is label
    assert result.details["forecast_decision_consistent"] is (not label)


def test_correct_labels_do_not_override_numeric_failure():
    result = rul_score(scenario(), answer(120000), "")
    assert result.details["label_pass"] is True
    assert result.details["numerical_pass"] is False
    assert not result.passed


@pytest.mark.parametrize(
    "text",
    [
        "Estimate remaining cycles.",
        "Within the next 150000 operating cycles; remaining_cycles <= 140000.",
        "Within the next 150000 operating cycles and next 140000 operating cycles; remaining_cycles <= 150000.",
    ],
)
def test_missing_or_conflicting_question_horizon_is_harness_error(text):
    with pytest.raises(ValueError, match="horizon"):
        rul_score(scenario().model_copy(update={"text": text}), answer(), "")


@pytest.mark.parametrize(
    "changes",
    [
        {"asset_id": None},
        {"cutoff_timestamp": ""},
        {"documented_failed_assets": ["asset-B", "asset-A"]},
        {"ongoing_assets": ["asset-target", "asset-target"]},
        {"documented_failed_assets": ["asset-target"]},
        {"ongoing_assets": []},
        {"failure_within_horizon": "true"},
        {"failure_within_horizon": False},
    ],
)
def test_invalid_gold_labels_are_harness_errors(changes):
    case = scenario().model_copy(
        update={"expected_answer": json.dumps(dict(GOLD, **changes))}
    )
    with pytest.raises(ValueError):
        rul_score(case, answer(), "")


def test_native_evaluator_dispatches_both_checks():
    trajectory = PersistedTrajectory(
        run_id="test",
        scenario_id="rul-test",
        runner="test",
        model="test",
        question="",
        answer=answer(108000),
    )
    result = Evaluator()._score_one(scenario(), trajectory)
    assert result.score.scorer == "rul_numeric_v1" and result.score.passed
    assert result.score.details["label_pass"] and result.score.details["numerical_pass"]
