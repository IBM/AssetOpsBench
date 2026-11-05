"""Generic nested prediction metrics; verifier data comes only from the scenario."""

import json
import math

from ..models import Scenario, ScorerResult
from . import register

NAME = "source_panel_continuous_v1"


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _configuration(meta):
    if not isinstance(meta, dict) or meta.get("kind") not in (
        "classification",
        "numeric",
    ):
        raise ValueError("Valid scenario kind required")
    targets = meta.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(t, str) or not t for t in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("Distinct nonempty target names required")
    if meta["kind"] == "classification":
        codes = meta.get("allowed_codes")
        if not isinstance(codes, dict) or set(codes) != set(targets):
            raise ValueError("Class inventory required for every target")
        for values in codes.values():
            if (
                not isinstance(values, list)
                or not values
                or any(type(v) not in (int, str) for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Distinct integer/string class codes required")
    else:
        scales = meta.get("target_scales")
        if (
            not isinstance(scales, list)
            or len(scales) != len(targets)
            or any(not _finite(v) or v <= 0 for v in scales)
        ):
            raise ValueError("Positive finite normalization scale per target required")


def _predictions(text, meta):
    payload = json.loads(text, object_pairs_hook=_unique)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("predictions"), dict)
        or not payload["predictions"]
    ):
        raise ValueError("Nonempty predictions object required")
    for row in payload["predictions"].values():
        if not isinstance(row, dict) or set(row) != set(meta["targets"]):
            raise ValueError("Exact target fields required")
        for target, value in row.items():
            if meta["kind"] == "classification":
                if (
                    type(value) not in (int, str)
                    or value not in meta["allowed_codes"][target]
                ):
                    raise ValueError("Allowed integer/string class code required")
            elif not _finite(value):
                raise ValueError("Finite numeric prediction required")
    return payload["predictions"]


def grade(expected, answer, meta):
    """Compute continuous metrics without inventing an acceptance policy."""
    _configuration(meta)
    gold = _predictions(expected, meta)  # Bad gold/configuration is a harness error.
    try:
        pred = _predictions(answer, meta)
        if set(pred) != set(gold):
            raise ValueError("Exact requested episode coverage required")
        per = {}
        for index, target in enumerate(meta["targets"]):
            y = [row[target] for row in gold.values()]
            p = [pred[key][target] for key in gold]
            if meta["kind"] == "classification":
                f1 = []
                for code in meta["allowed_codes"][target]:
                    tp = sum(a == code and b == code for a, b in zip(y, p))
                    den = sum(v == code for v in y) + sum(v == code for v in p)
                    f1.append(2 * tp / den if den else 0.0)
                per[target] = {
                    "macro_f1": sum(f1) / len(f1),
                    "accuracy": sum(a == b for a, b in zip(y, p)) / len(y),
                }
            else:
                errors = [b - a for a, b in zip(y, p)]
                rmse = math.hypot(*(v / math.sqrt(len(y)) for v in errors))
                mae = math.fsum(abs(v) / len(y) for v in errors)
                scale = meta["target_scales"][index]
                normalized = rmse / scale
                if not all(math.isfinite(v) for v in (rmse, mae, normalized)):
                    raise ValueError("Nonfinite numeric error")
                per[target] = {
                    "rmse": rmse,
                    "mae": mae,
                    "reference_scale": scale,
                    "normalized_rmse": normalized,
                }
    except (ValueError, TypeError, OverflowError) as error:
        return {
            "valid": False,
            "score": 0.0,
            "calibrated_pass": None,
            "error": str(error),
        }
    if meta["kind"] == "classification":
        mean = sum(v["macro_f1"] for v in per.values()) / len(per)
        result = {
            "mean_component_macro_f1": mean,
            "complete_row_accuracy": sum(pred[k] == gold[k] for k in gold) / len(gold),
            "score": mean,
        }
    else:
        mean = math.fsum(v["normalized_rmse"] / len(per) for v in per.values())
        result = {"mean_normalized_rmse": mean, "score": 1 / (1 + mean)}
    return {
        "valid": True,
        "query_count": len(gold),
        "per_target": per,
        "calibrated_pass": None,
        **result,
    }


def _acceptance(meta):
    policy = meta.get("acceptance") if isinstance(meta, dict) else None
    if not isinstance(policy, dict):
        raise ValueError(
            "Explicit acceptance policy required for native pass/fail; use grade() for uncalibrated continuous analysis"
        )
    metric, limit = policy.get("metric"), policy.get("threshold")
    valid = (
        {"mean_component_macro_f1"}
        if meta.get("kind") == "classification"
        else {"mean_normalized_rmse"}
        | ({"rmse"} if len(meta.get("targets", [])) == 1 else set())
    )
    if (
        metric not in valid
        or not _finite(limit)
        or limit < 0
        or (metric == "mean_component_macro_f1" and limit > 1)
    ):
        raise ValueError("Invalid acceptance metric/threshold")
    return metric, limit


def source_panel_score(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    meta = (scenario.evaluation_metadata or {}).get("scenario_meta", {})
    _configuration(meta)
    metric, limit = _acceptance(meta)
    result = grade(scenario.expected_answer, answer, meta)
    passed = False
    if result["valid"]:
        value = (
            next(iter(result["per_target"].values()))["rmse"]
            if metric == "rmse"
            else result[metric]
        )
        passed = (
            value >= limit if metric == "mean_component_macro_f1" else value <= limit
        )
        result["calibrated_pass"] = passed
    return ScorerResult(
        scorer=NAME,
        passed=passed,
        score=result["score"],
        rationale=result.get("error", "Metadata acceptance policy applied, inclusive"),
        details=result,
    )


def install():
    register(NAME, source_panel_score)
