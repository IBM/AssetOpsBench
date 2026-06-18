"""Operational metric extraction and aggregation."""

from __future__ import annotations

import statistics
from typing import Any

from .models import AggregateOps, OpsMetrics, PersistedTrajectory, ScenarioResult

# USD per 1M tokens, rough public list-prices.  None when unknown.  Used
# only for the optional ``est_cost_usd`` rollup; consumers should treat
# it as an estimate, not a billing source of truth.
_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5": (10.0, 30.0),
    "gpt-4.1": (3.0, 12.0),
    "gpt-4o": (2.5, 10.0),
    "llama-4-maverick": (0.27, 0.85),
}


def metrics_from_trajectory(record: PersistedTrajectory) -> OpsMetrics:
    """Extract per-task ops metrics from a persisted trajectory record."""
    traj = record.trajectory
    if traj is None:
        return OpsMetrics()

    if isinstance(traj, dict) and "turns" in traj:
        return _from_sdk_trajectory(traj, record.model)
    if isinstance(traj, list):
        return _from_plan_execute(traj, record.model)
    return OpsMetrics()


def _from_sdk_trajectory(traj: dict, model: str) -> OpsMetrics:
    turns = traj.get("turns", []) or []
    tokens_in = sum(int(t.get("input_tokens") or 0) for t in turns)
    tokens_out = sum(int(t.get("output_tokens") or 0) for t in turns)

    durations_ms = [t.get("duration_ms") for t in turns if t.get("duration_ms") is not None]
    duration_ms = sum(durations_ms) if durations_ms else None

    tool_names: list[str] = []
    for t in turns:
        for tc in t.get("tool_calls") or []:
            name = tc.get("name")
            if name:
                tool_names.append(name)

    return OpsMetrics(
        turn_count=len(turns),
        tool_call_count=len(tool_names),
        unique_tools=sorted(set(tool_names)),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=duration_ms,
        est_cost_usd=_estimate_cost(model, tokens_in, tokens_out),
    )


def _from_plan_execute(steps: list[Any], model: str) -> OpsMetrics:
    # plan-execute persists ``list[StepResult]``; the dataclass exposes
    # ``server`` / ``tool`` / ``response`` fields but no per-step token
    # counts, so we surface what is available and leave the rest at zero.
    tool_names = [
        s.get("tool")
        for s in steps
        if isinstance(s, dict) and s.get("tool")
    ]
    return OpsMetrics(
        turn_count=len(steps),
        tool_call_count=len(tool_names),
        unique_tools=sorted(set(tool_names)),
        est_cost_usd=_estimate_cost(model, 0, 0),
    )


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    if not model or (tokens_in == 0 and tokens_out == 0):
        return None
    key = _normalize_model(model)
    rate = _PRICE_PER_1M.get(key)
    if rate is None:
        return None
    in_rate, out_rate = rate
    return round((tokens_in * in_rate + tokens_out * out_rate) / 1_000_000, 6)


def _normalize_model(model: str) -> str:
    # Strip provider prefixes like ``litellm_proxy/anthropic/`` and
    # version suffixes like ``-20250101``.
    tail = model.rsplit("/", 1)[-1].lower()
    parts = tail.split("-")
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 6:
        parts = parts[:-1]
    return "-".join(parts)


def aggregate_ops(results: list[ScenarioResult]) -> AggregateOps:
    if not results:
        return AggregateOps()

    durations = [r.ops.duration_ms for r in results if r.ops.duration_ms is not None]
    costs = [r.ops.est_cost_usd for r in results if r.ops.est_cost_usd is not None]

    return AggregateOps(
        tokens_in_total=sum(r.ops.tokens_in for r in results),
        tokens_out_total=sum(r.ops.tokens_out for r in results),
        duration_ms_p50=_percentile(durations, 50),
        duration_ms_p95=_percentile(durations, 95),
        tool_calls_total=sum(r.ops.tool_call_count for r in results),
        est_cost_usd_total=round(sum(costs), 6) if costs else None,
    )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return float(quantiles[int(pct) - 1])
