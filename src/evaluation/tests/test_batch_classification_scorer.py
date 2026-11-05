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


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_exact_answer_coverage(mode):
    answer = json.loads(json.dumps(GOLD))
    if mode == "missing":
        answer["predictions"].pop("fictional-observation-0")
    elif mode == "extra":
        answer["predictions"]["fictional-extra"] = CODES[0]
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
    ],
)
def test_invalid_configuration_raises_harness_error(override):
    with pytest.raises(ValueError):
        evaluate(**override)


def test_invalid_gold_is_not_a_submission_failure():
    with pytest.raises(ValueError):
        grade('{"predictions":{}}', json.dumps(GOLD), META)


@pytest.mark.parametrize(
    "extra",
    [
        {"selected_sensors": ["FICTIONAL-X", "FICTIONAL-Y"]},
        {"selected_features": ["UNRELATED__value"]},
        {"program": "not executable"},
        {"program": None},
        {"model_id": ""},
        {"explanation": "ignored"},
    ],
)
def test_extra_submission_fields_do_not_affect_classification(extra):
    result = evaluate({**GOLD, **extra})
    assert result["valid"] and result["passed"] and result["score"] == 1.0


def test_duplicate_episode_keys_rejected():
    text = '{"predictions":{"fictional-duplicate":8700000101,"fictional-duplicate":8700000202}}'
    assert not grade(json.dumps(GOLD), text, META)["valid"]
