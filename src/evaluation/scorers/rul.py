"""Deterministic RUL scoring with required categorical and numerical checks.

Select rul_numeric_v1 and provide relative_tolerance in
scenario_meta.json. Agent JSON supplies remaining_cycles;
asset, cutoff, fleet and horizon labels must also be correct. The horizon comes
from the question. No dataset, fixed tolerance or policy file is bundled.
"""

import json
import math
import re
from decimal import Decimal, InvalidOperation

from ..models import Scenario, ScorerResult
from . import register


def horizon_from_question(text):
    stated = {
        int(v)
        for v in re.findall(
            r"\bnext\s+([0-9]+)\s+operating\s+cycles\b", text, re.IGNORECASE
        )
    }
    contract = {
        int(v) for v in re.findall(r"\bremaining_cycles\s*<=\s*([0-9]+)\b", text)
    }
    if len(stated) != 1 or contract != stated:
        raise ValueError(
            "Question requires one unambiguous horizon and matching output comparison"
        )
    return stated.pop()


def _valid_fleet(value):
    return (
        isinstance(value, list)
        and all(type(v) is str and v for v in value)
        and value == sorted(set(value))
    )


def validate_gold(gold, horizon):
    if not isinstance(gold, dict):
        raise ValueError("Ground truth must be a JSON object")
    value = gold.get("remaining_cycles")
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Positive finite ground-truth RUL required")
    for key in ("asset_id", "cutoff_timestamp"):
        if type(gold.get(key)) is not str or not gold[key]:
            raise ValueError(f"Ground truth requires {key}")
    for key in ("documented_failed_assets", "ongoing_assets"):
        if not _valid_fleet(gold.get(key)):
            raise ValueError(f"Ground truth requires sorted unique {key}")
    if set(gold["documented_failed_assets"]) & set(gold["ongoing_assets"]):
        raise ValueError("Ground-truth fleet labels must not overlap")
    if gold["asset_id"] not in gold["ongoing_assets"]:
        raise ValueError("Positive-RUL target must be in ground-truth ongoing assets")
    if type(gold.get("failure_within_horizon")) is not bool or gold[
        "failure_within_horizon"
    ] != (value <= horizon):
        raise ValueError(
            "Ground-truth horizon label must match its RUL and question horizon"
        )


def score(record, prediction, gold, *, settings=None):
    settings = settings_from_metadata(record) if settings is None else settings
    tolerance = Decimal(str(settings["relative_tolerance"]))
    h = record["horizon_cycles"]
    validate_gold(gold, h)
    if not isinstance(prediction, dict):
        raise ValueError("Answer must be a JSON object")
    value = prediction.get("remaining_cycles")
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError("remaining_cycles must be a finite nonnegative number")
    if type(gold.get("remaining_cycles")) not in (int, float):
        raise ValueError("Positive numeric ground-truth RUL required")
    y = Decimal(str(gold["remaining_cycles"]))
    if not y.is_finite() or y <= 0:
        raise ValueError("Positive ground-truth RUL required")
    error = abs(Decimal(str(value)) - y)
    numerical_pass = error <= tolerance * y
    keys = {
        "asset_id",
        "cutoff_timestamp",
        "remaining_cycles",
        "failure_within_horizon",
        "documented_failed_assets",
        "ongoing_assets",
    }

    def fleet(name):
        v = prediction.get(name)
        return (
            isinstance(v, list)
            and all(type(i) is str for i in v)
            and v == sorted(set(v))
            and v == gold.get(name)
        )

    labels = dict(
        asset_correct=prediction.get("asset_id") == gold["asset_id"],
        cutoff_timestamp_correct=prediction.get("cutoff_timestamp")
        == gold["cutoff_timestamp"],
        horizon_correct=type(prediction.get("failure_within_horizon")) is bool
        and prediction["failure_within_horizon"] == gold["failure_within_horizon"],
        forecast_decision_consistent=type(prediction.get("failure_within_horizon"))
        is bool
        and prediction["failure_within_horizon"] == (value <= h),
        failed_assets_exact=fleet("documented_failed_assets"),
        ongoing_assets_exact=fleet("ongoing_assets"),
    )
    label_pass = all(labels.values())
    return dict(
        numerical_pass=numerical_pass,
        label_pass=label_pass,
        overall_pass=label_pass and numerical_pass,
        horizon_cycles=h,
        relative_tolerance=float(tolerance),
        absolute_error=float(error),
        relative_error=float(error / y),
        squared_error=float(error * error),
        answer_schema_valid=set(prediction) == keys,
        **labels,
    )


def settings_from_metadata(record):
    """Resolve verifier-owned per-scenario settings; required in native AOB/Harbor."""
    value = record.get("relative_tolerance")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("scenario_meta requires relative_tolerance")
    try:
        tolerance = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid scenario relative_tolerance") from error
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("Scenario tolerance must be finite and nonnegative")
    return {"relative_tolerance": str(tolerance)}


def rul_score(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    metadata = (scenario.evaluation_metadata or {}).get("scenario_meta", {})
    try:
        settings = settings_from_metadata(metadata)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(
            f"Invalid engine scorer configuration for scenario {scenario.id}: {error}"
        ) from error
    configuration = dict(
        relative_tolerance=float(Decimal(str(settings["relative_tolerance"]))),
    )
    gold = json.loads(scenario.expected_answer)
    horizon = horizon_from_question(scenario.text)
    validate_gold(gold, horizon)
    try:
        metrics = score(
            dict(metadata, horizon_cycles=horizon),
            json.loads(answer),
            gold,
            settings=settings,
        )
    except (ValueError, TypeError, KeyError) as error:
        return ScorerResult(
            scorer="rul_numeric_v1",
            passed=False,
            score=0.0,
            rationale=str(error),
            details=dict(
                valid_prediction=False,
                label_pass=False,
                numerical_pass=False,
                overall_pass=False,
                **configuration,
            ),
        )
    return ScorerResult(
        scorer="rul_numeric_v1",
        passed=metrics["overall_pass"],
        score=float(metrics["overall_pass"]),
        rationale=(
            "Pass iff categorical labels are correct and forecast-consistent, and "
            "finite nonnegative RUL has absolute relative error <= "
            f"{settings['relative_tolerance']}, inclusive."
        ),
        details=dict(valid_prediction=True, **metrics),
    )


def install():
    register("rul_numeric_v1", rul_score)
