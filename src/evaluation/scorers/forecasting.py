"""Deterministic RMSE scoring of timestamped forecasts.

Select forecasting_rmse_v1 and set rmse_threshold in private scenario metadata.
The answer and gold contain a forecasts array of timestamp/value objects.
The gold defines the horizon; no dataset, fixed horizon or threshold is bundled.
"""

import json
import math
from datetime import datetime, timezone

from ..models import Scenario, ScorerResult
from . import register

_NAME = "forecasting_rmse_v1"


def _rows(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("forecasts"), list):
        raise ValueError("A JSON object with a forecasts array is required")
    if not payload["forecasts"]:
        raise ValueError("Forecasts must not be empty")
    times, values = [], []
    for row in payload["forecasts"]:
        if not isinstance(row, dict):
            raise ValueError("Each forecast must be an object")
        stamp, value = row.get("timestamp"), row.get("value")
        if not isinstance(stamp, str):
            raise ValueError("Each timestamp must be an ISO 8601 string")
        instant = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if instant.utcoffset() is None:
            raise ValueError("Timezone-aware timestamps required")
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Forecast values must be finite JSON numbers")
        times.append(instant.astimezone(timezone.utc))
        values.append(value)
    if len(set(times)) != len(times) or times != sorted(times):
        raise ValueError("Forecast timestamps must be unique and chronological")
    return times, values


def forecasting_score(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    meta = (scenario.evaluation_metadata or {}).get("scenario_meta", {})
    limit = meta.get("rmse_threshold")
    try:
        valid_limit = (
            type(limit) in (int, float) and math.isfinite(limit) and limit >= 0
        )
    except OverflowError:
        valid_limit = False
    if not valid_limit:
        raise ValueError(
            "Finite nonnegative rmse_threshold required in scenario metadata"
        )
    # Invalid verifier settings/gold are harness errors, not model failures.
    expected_times, actual = _rows(json.loads(scenario.expected_answer))
    details = {"rmse_threshold": limit, "horizon_steps": len(actual)}
    try:
        times, predicted = _rows(json.loads(answer))
        if times != expected_times:
            raise ValueError(
                "Forecast timestamps must match the complete requested horizon"
            )
        errors = [p - y for p, y in zip(predicted, actual)]
        # Scale before taking the norm so large finite errors do not overflow squares.
        scale = math.sqrt(len(errors))
        rmse = math.hypot(*(e / scale for e in errors))
        mae = math.fsum(abs(e) / len(errors) for e in errors)
        if not math.isfinite(rmse) or not math.isfinite(mae):
            raise ValueError("Forecast error is outside the finite numeric range")
    except (ValueError, TypeError, OverflowError) as error:
        return ScorerResult(
            scorer=_NAME,
            passed=False,
            score=0.0,
            rationale=str(error),
            details={"valid_forecast": False, **details},
        )
    passed = rmse <= limit
    return ScorerResult(
        scorer=_NAME,
        passed=passed,
        score=float(passed),
        rationale=f"Pass iff timestamps match gold and RMSE <= {limit}, inclusive.",
        details={"valid_forecast": True, "RMSE": rmse, "MAE": mae, **details},
    )


def install():
    register(_NAME, forecasting_score)
