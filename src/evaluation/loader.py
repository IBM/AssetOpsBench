"""Load trajectories and scenarios, then join them by ``scenario_id``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from .models import PersistedTrajectory, Scenario

_log = logging.getLogger(__name__)


def load_trajectories(path: Path) -> list[PersistedTrajectory]:
    """Load every ``*.json`` trajectory under ``path``.

    ``path`` may be a directory or a single JSON file. If a trajectory has
    ``scenario_id`` set to null, the filename stem is used as a fallback.
    This supports layouts such as ``traces/trajectories/34.json`` where
    ``34`` is the scenario id.
    """
    p = Path(path)
    if p.is_file():
        return [_load_one_trajectory(p)] if p.suffix == ".json" else []

    out: list[PersistedTrajectory] = []
    for child in sorted(p.glob("*.json")):
        try:
            out.append(_load_one_trajectory(child))
        except Exception:
            _log.exception("loader: failed to parse %s", child)
    return out


def _load_one_trajectory(path: Path) -> PersistedTrajectory:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if raw.get("scenario_id") is None:
        raw["scenario_id"] = path.stem

    return PersistedTrajectory.from_raw(raw)


def load_scenarios(paths: Iterable[Path] | Path) -> list[Scenario]:
    """Load scenarios from one or more files or directories.

    Supported inputs:

    1. Existing JSON / JSONL scenario files.
    2. A directory containing scenario subdirectories, each with
       ``groundtruth.txt``. For example:

       scenarios_data/
  scenario_11/
    groundtruth.txt
  scenario_12/
    groundtruth.txt

    For folder-based scenarios, the folder name becomes the scenario id and
    ``groundtruth.txt`` becomes ``expected_answer``.
    """
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]

    out: list[Scenario] = []
    for p in paths:
        p = Path(p)

        if p.is_dir():
            out.extend(_load_scenario_dir(p))
        else:
            out.extend(_load_scenario_file(p))

    return out


def _load_scenario_dir(path: Path) -> list[Scenario]:
    """Load scenario folders from a directory containing groundtruth.txt files.

    Supports folder names like:
        scenario_34/
          groundtruth.txt

    The scenario id becomes "34" so it can join with trajectory files such as:
        traces/trajectories/34.json
    """
    scenarios: list[Scenario] = []

    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue

        groundtruth_path = child / "groundtruth.txt"
        if not groundtruth_path.exists():
            continue

        scenario_id = child.name
        if scenario_id.startswith("scenario_"):
            scenario_id = scenario_id.removeprefix("scenario_")

        expected_answer = groundtruth_path.read_text(encoding="utf-8").strip()

        question_path = child / "question.txt"
        text = (
            question_path.read_text(encoding="utf-8").strip()
            if question_path.exists()
            else ""
        )

        scenarios.append(
            Scenario.from_raw(
                {
                    "id": scenario_id,
                    "text": text,
                    "type": "structured",
                    "expected_answer": expected_answer,
                    "scoring_method": "static_json",
                }
            )
        )

    return scenarios


def _load_scenario_file(path: Path) -> list[Scenario]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        return [
            Scenario.from_raw(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]

    raw = json.loads(text)
    if isinstance(raw, list):
        return [Scenario.from_raw(item) for item in raw]
    if isinstance(raw, dict):
        return [Scenario.from_raw(raw)]
    raise ValueError(f"unexpected scenario JSON shape in {path}: {type(raw).__name__}")


def join_records(
    scenarios: list[Scenario],
    trajectories: list[PersistedTrajectory],
) -> Iterator[tuple[Scenario, PersistedTrajectory]]:
    """Yield (scenario, trajectory) pairs joined on ``scenario_id``."""
    by_id: dict[str, Scenario] = {s.id: s for s in scenarios}
    for traj in trajectories:
        if traj.scenario_id is None:
            continue
        scenario = by_id.get(traj.scenario_id)
        if scenario is not None:
            yield scenario, traj