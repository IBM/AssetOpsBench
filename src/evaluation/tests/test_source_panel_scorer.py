"""Entirely fictional fixtures; no private identifiers, labels, values or policies."""

import copy
import json
import math

import pytest

from evaluation import scorers
from evaluation.models import Scenario
from evaluation.scorers.source_panel import NAME, grade, source_panel_score

A, B = "FICTIONAL-NEBULA", "FICTIONAL-QUASAR"
CODE = 9_600_000_777
C_META = {
    "kind": "classification",
    "targets": ["target_alpha", "target_beta"],
    "allowed_codes": {"target_alpha": [A, B], "target_beta": [CODE, CODE + 1]},
}
C_GOLD = {
    "predictions": {
        "fictional-item-x": {"target_alpha": A, "target_beta": CODE},
        "fictional-item-y": {"target_alpha": B, "target_beta": CODE + 1},
    }
}
N_META = {
    "kind": "numeric",
    "targets": ["target_alpha", "target_beta"],
    "target_scales": [400_000_000, 800_000_000],
}
N_GOLD = {
    "predictions": {
        "fictional-item-x": {
            "target_alpha": 9_000_000_000_000,
            "target_beta": 7_000_000_000_000,
        },
        "fictional-item-y": {
            "target_alpha": 11_000_000_000_000,
            "target_beta": 5_000_000_000_000,
        },
    }
}


def run(answer=C_GOLD, meta=C_META, gold=C_GOLD):
    return grade(json.dumps(gold), json.dumps(answer), meta)


def scenario(meta):
    return Scenario(
        id="fictional-task",
        text="Predict assigned targets.",
        expected_answer=json.dumps(
            C_GOLD if meta["kind"] == "classification" else N_GOLD
        ),
        scoring_method=NAME,
        evaluation_metadata={"scenario_meta": meta},
    )


def test_perfect_multi_output_classification():
    result = run()
    assert result["score"] == result["complete_row_accuracy"] == 1.0
    assert result["calibrated_pass"] is None


def test_classification_mean_and_joint_accuracy_differ():
    answer = copy.deepcopy(C_GOLD)
    answer["predictions"]["fictional-item-y"]["target_alpha"] = A
    result = run(answer)
    assert result["per_target"]["target_alpha"]["macro_f1"] == pytest.approx(1 / 3)
    assert result["mean_component_macro_f1"] == pytest.approx(2 / 3)
    assert result["complete_row_accuracy"] == 0.5


def test_absent_configured_class_is_zero():
    meta = copy.deepcopy(C_META)
    meta["allowed_codes"]["target_alpha"].append("FICTIONAL-PULSAR")
    assert run(meta=meta)["per_target"]["target_alpha"]["macro_f1"] == pytest.approx(
        2 / 3
    )


def test_numeric_metrics_and_adapter():
    answer = copy.deepcopy(N_GOLD)
    for row in answer["predictions"].values():
        row["target_alpha"] += 200_000_000
        row["target_beta"] += 800_000_000
    result = run(answer, N_META, N_GOLD)
    assert result["mean_normalized_rmse"] == pytest.approx(0.75)
    assert result["score"] == pytest.approx(1 / 1.75)
    assert result["per_target"]["target_alpha"]["mae"] == 200_000_000


def test_large_finite_errors_do_not_overflow_squares():
    meta = {
        "kind": "numeric",
        "targets": ["fictional_target"],
        "target_scales": [1e201],
    }
    gold = {"predictions": {"fictional-item": {"fictional_target": 0}}}
    answer = {"predictions": {"fictional-item": {"fictional_target": 1e200}}}
    assert run(answer, meta, gold)["per_target"]["fictional_target"]["rmse"] == 1e200


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json",
        "null",
        "[]",
        "{}",
        '{"predictions":{}}',
        '{"predictions":{},"predictions":{}}',
    ],
)
def test_bad_json_or_structure_fails(text):
    result = grade(json.dumps(C_GOLD), text, C_META)
    assert not result["valid"] and result["score"] == 0


@pytest.mark.parametrize(
    "change", ["missing_item", "extra_item", "missing_target", "extra_target"]
)
def test_exact_coverage(change):
    answer = copy.deepcopy(C_GOLD)
    if change == "missing_item":
        answer["predictions"].pop("fictional-item-x")
    elif change == "extra_item":
        answer["predictions"]["fictional-extra"] = answer["predictions"][
            "fictional-item-x"
        ]
    elif change == "missing_target":
        answer["predictions"]["fictional-item-x"].pop("target_alpha")
    else:
        answer["predictions"]["fictional-item-x"]["extra"] = A
    assert not run(answer)["valid"]


@pytest.mark.parametrize(
    "value", [True, None, 9_600_000_777.0, "UNKNOWN", float("nan")]
)
def test_bad_class_code(value):
    answer = copy.deepcopy(C_GOLD)
    answer["predictions"]["fictional-item-x"]["target_beta"] = value
    assert not run(answer)["valid"]


@pytest.mark.parametrize(
    "value", [True, None, "9000000000000", float("nan"), float("inf"), 10**400]
)
def test_bad_numeric_value(value):
    answer = copy.deepcopy(N_GOLD)
    answer["predictions"]["fictional-item-x"]["target_alpha"] = value
    assert not run(answer, N_META, N_GOLD)["valid"]


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {"kind": "unknown"},
        {**C_META, "targets": []},
        {**C_META, "targets": ["duplicate", "duplicate"]},
        {**C_META, "allowed_codes": {}},
        {**C_META, "allowed_codes": {"target_alpha": [A], "target_beta": [True]}},
        {**N_META, "target_scales": [0, 1]},
        {**N_META, "target_scales": [True, 1]},
        {**N_META, "target_scales": [float("inf"), 1]},
        {**N_META, "target_scales": [1]},
    ],
)
def test_bad_configuration_is_harness_error(meta):
    with pytest.raises(ValueError):
        run(meta=meta)


def test_bad_gold_is_harness_error():
    with pytest.raises(ValueError):
        grade("{}", json.dumps(C_GOLD), C_META)


def test_optional_submission_fields_ignored():
    assert (
        run(
            {**C_GOLD, "program": None, "selected_sensors": ["IGNORED"], "model_id": ""}
        )["score"]
        == 1
    )


def test_native_registration_and_no_implicit_pass_policy():
    assert scorers.get(NAME) is source_panel_score
    with pytest.raises(ValueError, match="Explicit acceptance policy"):
        source_panel_score(scenario(C_META), json.dumps(C_GOLD), "")


def test_classification_inclusive_policy_boundary():
    answer = copy.deepcopy(C_GOLD)
    answer["predictions"]["fictional-item-y"]["target_alpha"] = A
    value = run(answer)["mean_component_macro_f1"]
    meta = {
        **C_META,
        "acceptance": {"metric": "mean_component_macro_f1", "threshold": value},
    }
    assert source_panel_score(scenario(meta), json.dumps(answer), "").passed
    meta["acceptance"]["threshold"] = math.nextafter(value, 1)
    assert not source_panel_score(scenario(meta), json.dumps(answer), "").passed


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"metric": "rmse", "threshold": 5_000_000},
        {"metric": "mean_component_macro_f1", "threshold": True},
        {"metric": "mean_component_macro_f1", "threshold": 1.3},
    ],
)
def test_invalid_policy_is_harness_error(policy):
    with pytest.raises(ValueError):
        source_panel_score(
            scenario({**C_META, "acceptance": policy}), json.dumps(C_GOLD), ""
        )


def test_numeric_inclusive_policy_boundary():
    meta = {
        **N_META,
        "acceptance": {"metric": "mean_normalized_rmse", "threshold": 0.75},
    }
    answer = copy.deepcopy(N_GOLD)
    for row in answer["predictions"].values():
        row["target_alpha"] += 200_000_000
        row["target_beta"] += 800_000_000
    assert source_panel_score(scenario(meta), json.dumps(answer), "").passed
    value = run(answer, N_META, N_GOLD)["mean_normalized_rmse"]
    meta["acceptance"]["threshold"] = value
    assert source_panel_score(scenario(meta), json.dumps(answer), "").passed
    meta["acceptance"]["threshold"] = math.nextafter(value, 0)
    assert not source_panel_score(scenario(meta), json.dumps(answer), "").passed


def test_single_target_source_unit_rmse_policy():
    meta = {
        "kind": "numeric",
        "targets": ["fictional_response"],
        "target_scales": [20_000_000],
        "acceptance": {"metric": "rmse", "threshold": 4_000_000},
    }
    gold = {
        "predictions": {"fictional-item": {"fictional_response": 9_000_000_000_000}}
    }
    s = Scenario(
        id="fictional-single",
        text="Predict assigned response.",
        expected_answer=json.dumps(gold),
        scoring_method=NAME,
        evaluation_metadata={"scenario_meta": meta},
    )
    answer = {
        "predictions": {"fictional-item": {"fictional_response": 9_000_004_000_000}}
    }
    assert source_panel_score(s, json.dumps(answer), "").passed
    answer["predictions"]["fictional-item"]["fictional_response"] += 1
    assert not source_panel_score(s, json.dumps(answer), "").passed
