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

    ``path`` may be a directory (the ``AGENT_TRAJECTORY_DIR`` layout) or
    a single JSON file.  Files that fail to parse are logged and
    skipped — a partial directory should still yield a usable batch.
    """
    p = Path(path)
    if p.is_file():
        return [_load_one(p)] if p.suffix == ".json" else []

    out: list[PersistedTrajectory] = []
    for child in sorted(p.glob("*.json")):
        try:
            out.append(_load_one(child))
        except Exception:
            _log.exception("loader: failed to parse %s", child)
    return out


def _load_one(path: Path) -> PersistedTrajectory:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PersistedTrajectory.from_raw(raw)


def load_scenarios(paths: Iterable[Path] | Path) -> list[Scenario]:
    """Load scenarios from one or more files.

    Each file may be a JSON list, a single JSON object, or JSONL.
    Scenario IDs are coerced to strings to make the join key uniform
    (CouchDB-style trajectories use string IDs; local JSON files use
    ints).
    """
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]

    out: list[Scenario] = []
    for p in paths:
        out.extend(_load_scenario_file(Path(p)))
    return out


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
    """Yield (scenario, trajectory) pairs joined on ``scenario_id``.

    Scenarios with no matching trajectory and trajectories with no
    matching scenario are silently dropped — the caller can compute the
    diff from the input lists if reporting is needed.
    """
    by_id: dict[str, Scenario] = {s.id: s for s in scenarios}
    for traj in trajectories:
        if traj.scenario_id is None:
            continue
        scenario = by_id.get(traj.scenario_id)
        if scenario is not None:
            yield scenario, traj
