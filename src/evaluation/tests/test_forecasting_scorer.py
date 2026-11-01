"""Synthetic fixtures only: no benchmark identifiers, dates, values or cutoffs."""

import copy
import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from evaluation import scorers
from evaluation.evaluator import Evaluator
from evaluation.models import PersistedTrajectory, Scenario
from evaluation.scorers.forecasting import forecasting_score

BASE = 2_000_000_000
LIMIT = 5_000_000


def payload(count=4):
    start = datetime(2099, 1, 1, tzinfo=timezone.utc)
    return {
        "forecasts": [
            {
                "timestamp": (start + timedelta(hours=3 * i)).isoformat(),
                "value": BASE + i * 100_000_000,
            }
            for i in range(count)
        ]
    }


def scenario(limit=LIMIT, count=4):
    return Scenario(
        id="synthetic-forecast",
        text="Forecast the requested timestamps.",
        expected_answer=json.dumps(payload(count)),
        scoring_method="forecasting_rmse_v1",
        evaluation_metadata={"scenario_meta": {"rmse_threshold": limit}},
    )


def shifted(delta, count=4):
    answer = payload(count)
    for row in answer["forecasts"]:
        row["value"] += delta
    return answer


def test_registered():
    assert scorers.get("forecasting_rmse_v1") is forecasting_score


@pytest.mark.parametrize(
    ("delta", "passed"), [(0, True), (LIMIT, True), (-LIMIT, True), (LIMIT + 1, False)]
)
def test_inclusive_rmse_boundary(delta, passed):
    result = forecasting_score(scenario(), json.dumps(shifted(delta)), "")
    assert result.passed is passed
    assert result.score == float(passed)
    assert result.details["RMSE"] == abs(delta)


def test_cutoff_is_metadata():
    answer = json.dumps(shifted(LIMIT))
    assert forecasting_score(scenario(), answer, "").passed
    assert not forecasting_score(scenario(limit=LIMIT // 2), answer, "").passed


@pytest.mark.parametrize("count", [1, 7])
def test_horizon_comes_from_gold(count):
    assert forecasting_score(
        scenario(count=count), json.dumps(payload(count)), ""
    ).passed


@pytest.mark.parametrize(
    "answer", ["", "not json", "null", "[]", "{}", '{"forecasts":[]}']
)
def test_malformed_answer_fails(answer):
    result = forecasting_score(scenario(), answer, "")
    assert not result.passed and result.score == 0.0
    assert not result.details["valid_forecast"]


@pytest.mark.parametrize(
    "value", [True, None, "2000000000", float("nan"), float("inf"), 10**400]
)
def test_invalid_values_fail(value):
    answer = payload()
    answer["forecasts"][0]["value"] = value
    assert not forecasting_score(scenario(), json.dumps(answer), "").passed


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "reversed", "shifted", "naive"]
)
def test_wrong_horizon_or_timestamps_fail(mutation):
    answer = payload()
    rows = answer["forecasts"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(copy.deepcopy(rows[-1]))
    elif mutation == "duplicate":
        rows[1]["timestamp"] = rows[0]["timestamp"]
    elif mutation == "reversed":
        rows.reverse()
    elif mutation == "shifted":
        rows[0]["timestamp"] = "2098-01-01T00:00:00Z"
    elif mutation == "naive":
        rows[0]["timestamp"] = "2099-01-01T00:00:00"
    assert not forecasting_score(scenario(), json.dumps(answer), "").passed


@pytest.mark.parametrize(
    "limit", [None, True, -LIMIT, "bad", float("nan"), float("inf")]
)
def test_invalid_configuration_is_harness_error(limit):
    with pytest.raises(ValueError, match="rmse_threshold"):
        forecasting_score(scenario(limit), json.dumps(payload()), "")


def test_invalid_gold_is_harness_error():
    case = scenario().model_copy(update={"expected_answer": '{"forecasts":[]}'})
    with pytest.raises(ValueError):
        forecasting_score(case, json.dumps(payload()), "")


def test_large_finite_errors_do_not_overflow_squares():
    case = scenario(limit=1e201)
    result = forecasting_score(case, json.dumps(shifted(1e200)), "")
    assert result.passed and math.isfinite(result.details["RMSE"])


def test_equivalent_timezone_notation():
    answer = payload()
    answer["forecasts"][0]["timestamp"] = "2098-12-31T19:00:00-05:00"
    assert forecasting_score(scenario(), json.dumps(answer), "").passed


def test_native_dispatch_without_wrapper():
    case = scenario()
    record = PersistedTrajectory(
        run_id="synthetic",
        scenario_id=case.id,
        runner="test",
        model="test",
        question=case.text,
        answer=case.expected_answer,
    )
    assert Evaluator()._score_one(case, record).score.passed
