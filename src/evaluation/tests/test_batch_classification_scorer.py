"""Invented large class codes and identifiers only; no private scenario fixtures."""

import json
import math

import pytest

from evaluation import scorers
from evaluation.models import Scenario
from evaluation.scorers.batch_classification import batch_classification_score, grade

CODES = [8_700_000_101, 8_700_000_202, 8_700_000_303]
GOLD = {"predictions": {f"fictional-observation-{i}": c for i, c in enumerate(CODES)}}
META = {"allowed_codes": CODES, "macro_f1_threshold": 0.72}


def evaluate(answer=GOLD, **overrides):
    return grade(json.dumps(GOLD), json.dumps(answer), {**META, **overrides})


def test_registration_and_native_scenario_dispatch():
    scenario = Scenario(
        id="fictional-batch",
        text="Classify assigned observations.",
        expected_answer=json.dumps(GOLD),
        scoring_method="batch_classification_macro_f1_v1",
        evaluation_metadata={"scenario_meta": META},
    )
    assert scorers.get(scenario.scoring_method) is batch_classification_score
    result = scorers.get(scenario.scoring_method)(scenario, json.dumps(GOLD), "")
    assert result.passed and result.score == 1.0


def test_fixed_class_macro_f1_and_inclusive_metadata_cutoff():
    answer = {"predictions": {key: CODES[0] for key in GOLD["predictions"]}}
    score = evaluate(answer)["score"]
    # One class has F1=1/2; the other two have zero, so fixed-class mean is 1/6.
    assert score == pytest.approx(1 / 6)
    assert evaluate(answer, macro_f1_threshold=score)["passed"]
    assert not evaluate(answer, macro_f1_threshold=math.nextafter(score, 1.0))["passed"]
    assert evaluate(answer, macro_f1_threshold=0.12)["passed"]


def test_absent_configured_class_has_zero_f1():
    one = {"predictions": {"fictional-only": CODES[0]}}
    result = grade(json.dumps(one), json.dumps(one), META)
    assert result["score"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "not json",
        "null",
        "[]",
        "{}",
        '{"predictions":{}}',
        '{"predictions":{},"predictions":{}}',
    ],
)
def test_malformed_answers_score_zero(answer):
    result = grade(json.dumps(GOLD), answer, META)
    assert not result["valid"] and not result["passed"] and result["score"] == 0


@pytest.mark.parametrize(
    "value",
    [
        True,
        None,
        "8700000101",
        8_700_000_101.0,
        9_999_999_999,
        float("nan"),
        float("inf"),
    ],
)
def test_non_integer_or_unknown_codes_rejected(value):
    assert not evaluate(
        {"predictions": {**GOLD["predictions"], "fictional-observation-0": value}}
    )["valid"]


@pytest.mark.parametrize("mode", ["missing", "extra", "unexpected_field"])
def test_exact_answer_coverage(mode):
    answer = json.loads(json.dumps(GOLD))
    if mode == "missing":
        answer["predictions"].pop("fictional-observation-0")
    elif mode == "extra":
        answer["predictions"]["fictional-extra"] = CODES[0]
    else:
        answer["explanation"] = "unsolicited"
    assert not evaluate(answer)["valid"]


@pytest.mark.parametrize(
    "override",
    [
        {"allowed_codes": []},
        {"allowed_codes": [True]},
        {"allowed_codes": [CODES[0], CODES[0]]},
        {"allowed_codes": "wrong"},
        {"macro_f1_threshold": True},
        {"macro_f1_threshold": -0.2},
        {"macro_f1_threshold": 1.2},
        {"macro_f1_threshold": float("nan")},
        {"max_sensor_channels": True},
        {"max_sensor_channels": 0},
        {"max_sensor_channels": 2.5},
        {"require_program": "true"},
        {"require_model_id": 1},
    ],
)
def test_invalid_configuration_raises_harness_error(override):
    with pytest.raises(ValueError):
        evaluate(**override)


def test_invalid_gold_is_not_a_submission_failure():
    with pytest.raises(ValueError):
        grade('{"predictions":{}}', json.dumps(GOLD), META)


def constrained_answer():
    return {
        **GOLD,
        "selected_sensors": ["CHANNEL-OMEGA"],
        "selected_features": ["CHANNEL-OMEGA__mean"],
        "model_id": "fictional-estimator",
        "program": 'print("fictional")',
    }


def test_optional_declarations_are_validated_without_execution():
    answer = constrained_answer()
    result = evaluate(
        answer, max_sensor_channels=2, require_model_id=True, require_program=True
    )
    assert result["passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_sensors", ["CHANNEL-OMEGA", "CHANNEL-OMEGA"]),
        ("selected_sensors", ["CHANNEL-OMEGA", "CHANNEL-SIGMA", "CHANNEL-TAU"]),
        ("selected_sensors", []),
        ("selected_sensors", [None]),
        ("selected_features", ["CHANNEL-SIGMA__mean"]),
        ("selected_features", ["CHANNEL-OMEGA__mean", "CHANNEL-OMEGA__mean"]),
        ("selected_features", []),
        ("model_id", ""),
        ("program", ""),
        ("program", None),
    ],
)
def test_invalid_declarations_rejected(field, value):
    answer = constrained_answer()
    answer[field] = value
    assert not evaluate(
        answer, max_sensor_channels=2, require_model_id=True, require_program=True
    )["valid"]


def test_duplicate_episode_keys_rejected():
    text = '{"predictions":{"fictional-duplicate":8700000101,"fictional-duplicate":8700000202}}'
    assert not grade(json.dumps(GOLD), text, META)["valid"]
