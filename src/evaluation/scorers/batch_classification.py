"""Metadata-driven fixed-class macro-F1; no family or scenario-ID dependency."""

import json
import math

from ..models import Scenario, ScorerResult
from . import register


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=_object)


def predictions(payload, codes, extra_keys=()):
    if not isinstance(payload, dict) or set(payload) != {"predictions", *extra_keys}:
        raise ValueError("One top-level predictions object required")
    result = payload["predictions"]
    if not isinstance(result, dict) or not result:
        raise ValueError("Nonempty predictions object required")
    if any(type(v) is not int or v not in codes for v in result.values()):
        raise ValueError("Integer allowed condition codes required")
    return result


def grade(expected, answer, meta):
    if not isinstance(meta, dict):
        raise ValueError("scenario_meta must be an object")
    codes = meta.get("allowed_codes")
    threshold = meta.get("macro_f1_threshold")
    if (
        not isinstance(codes, list)
        or not codes
        or any(type(c) is not int for c in codes)
        or len(codes) != len(set(codes))
    ):
        raise ValueError("Invalid allowed_codes configuration")
    if (
        type(threshold) not in (float, int)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid macro_f1_threshold configuration")
    for flag in ["require_model_id", "require_program"]:
        if flag in meta and type(meta[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")
    budget = meta.get("max_sensor_channels")
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("max_sensor_channels must be a positive integer")
    # Invalid gold/configuration raises a harness error; never blame the submission.
    gold = predictions(loads(expected), codes)
    try:
        payload = loads(answer)
        extra_keys = set()
        if meta.get("require_model_id"):
            extra_keys.add("model_id")
        if meta.get("require_program"):
            extra_keys.add("program")
        if meta.get("max_sensor_channels") is not None:
            extra_keys.update(["selected_sensors", "selected_features"])
        pred = predictions(payload, codes, extra_keys)
        if meta.get("require_model_id") and (
            not isinstance(payload.get("model_id"), str)
            or not payload["model_id"].strip()
        ):
            raise ValueError("Nonempty model_id required")
        if meta.get("require_program") and (
            not isinstance(payload.get("program"), str)
            or not payload["program"].strip()
        ):
            raise ValueError("Executable program submission required")
        budget = meta.get("max_sensor_channels")
        if budget is not None:
            sensors = payload.get("selected_sensors")
            features = payload.get("selected_features")
            if (
                not isinstance(sensors, list)
                or not 1 <= len(sensors) <= budget
                or any(not isinstance(v, str) or not v.strip() for v in sensors)
                or len(set(sensors)) != len(sensors)
            ):
                raise ValueError(
                    "One shared distinct sensor set within budget required"
                )
            if (
                not isinstance(features, list)
                or not features
                or any(not isinstance(v, str) or "__" not in v for v in features)
                or len(set(features)) != len(features)
            ):
                raise ValueError("Distinct selected_features required")
            if {f.split("__")[0] for f in features} != set(sensors):
                raise ValueError("Selected features must match shared sensor set")
        if set(pred) != set(gold):
            raise ValueError("Exact target episode coverage required")
    except (ValueError, TypeError) as error:
        return dict(valid=False, passed=False, score=0.0, error=str(error))
    f1 = []
    for code in codes:
        tp = sum(gold[k] == code and pred[k] == code for k in gold)
        den = sum(v == code for v in gold.values()) + sum(
            v == code for v in pred.values()
        )
        f1.append(2 * tp / den if den else 0.0)
    score = sum(f1) / len(codes)
    correct = sum(gold[k] == pred[k] for k in gold)
    return dict(
        valid=True,
        passed=score >= threshold,
        score=score,
        macro_F1=score,
        macro_f1_threshold=threshold,
        correct=correct,
        n=len(gold),
        accuracy=correct / len(gold),
    )


def batch_classification_score(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    meta = (scenario.evaluation_metadata or {}).get("scenario_meta", {})
    r = grade(scenario.expected_answer, answer, meta)
    return ScorerResult(
        scorer="batch_classification_macro_f1_v1",
        passed=r["passed"],
        score=r["score"],
        rationale=r.get(
            "error",
            "Pass iff complete valid predictions and macro-F1 >= metadata cutoff",
        ),
        details=r,
    )


def install():
    register("batch_classification_macro_f1_v1", batch_classification_score)
