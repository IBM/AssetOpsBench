"""Per-run trajectory persistence for offline evaluation.

Separation of concerns:
  * **Spans** (``observability.tracing`` + ``runspan``) carry *metadata* —
    runner, model, run/scenario IDs, question/answer lengths, span timing.
  * **Trajectories** (this module) carry *content* — per-turn text, tool
    call inputs/outputs, per-turn token usage.  Persisted alongside spans,
    joined by ``run_id``.

When ``AGENT_TRAJECTORY_DIR`` is set, each runner's :meth:`run` appends a
``{run_id}.json`` file to that directory.  When unset, the module is a
no-op and trajectories live only in-process on the returned
:class:`~agent.models.AgentResult`.

Design choices:
  * The output is a single JSON file per run keyed by ``run_id``.  Joining
    with the OTLP-JSON trace file uses the ``agent.run_id`` span attribute.
  * ``dataclasses.asdict`` is used for both the SDK runners' ``Trajectory``
    dataclass and plan-execute's ``list[StepResult]``, which keeps the
    serializer runner-agnostic.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

from .runspan import _run_id_var, _scenario_id_var

_log = logging.getLogger(__name__)

_TRAJECTORY_DIR_ENV = "AGENT_TRAJECTORY_DIR"


def persist_trajectory(
    *,
    runner_name: str,
    model: str,
    question: str,
    answer: str,
    trajectory: Any,
) -> Path | None:
    """Write a per-run evaluation record when ``AGENT_TRAJECTORY_DIR`` is set.

    Reads ``run_id`` / ``scenario_id`` from the same contextvars used by
    :func:`agent_run_span`, so CLI-level wiring doesn't have to touch the
    runner's public signature.

    Returns the output path, or ``None`` when persistence is disabled.
    """
    dir_env = os.environ.get(_TRAJECTORY_DIR_ENV)
    if not dir_env:
        return None

    run_id = _run_id_var.get()
    if not run_id:
        _log.warning(
            "%s is set but no run_id in context; skipping trajectory persist",
            _TRAJECTORY_DIR_ENV,
        )
        return None

    out_dir = Path(dir_env)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    record = {
        "run_id": run_id,
        "scenario_id": _scenario_id_var.get(),
        "runner": runner_name,
        "model": model,
        "question": question,
        "answer": answer,
        "trajectory": _serialize_trajectory(trajectory),
    }

    try:
        out_path.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        _log.exception("persist_trajectory: write failed at %s", out_path)
        return None
    return out_path


def _serialize_trajectory(trajectory: Any) -> Any:
    """Reduce a trajectory to plain JSON-friendly Python.

    Handles the two shapes in this repo: SDK runners' ``Trajectory``
    dataclass, and plan-execute's ``list[StepResult]`` (also dataclasses).
    """
    if trajectory is None:
        return None
    if dataclasses.is_dataclass(trajectory):
        return dataclasses.asdict(trajectory)
    if isinstance(trajectory, list):
        return [
            dataclasses.asdict(item) if dataclasses.is_dataclass(item) else item
            for item in trajectory
        ]
    return trajectory
