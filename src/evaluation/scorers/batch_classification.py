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


def predictions(payload, codes):
    if not isinstance(payload, dict) or "predictions" not in payload:
        raise ValueError("A predictions object is required")
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
    # Invalid gold/configuration raises a harness error; never blame the submission.
    gold = predictions(loads(expected), codes)
    try:
        payload = loads(answer)
        pred = predictions(payload, codes)
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
