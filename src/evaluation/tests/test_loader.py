"""Tests for the trajectory + scenario loader."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.loader import (
    join_records,
    load_scenarios,
    load_trajectories,
)
from evaluation.models import Scenario


def test_load_trajectories_from_dir(trajectory_dir: Path):
    records = load_trajectories(trajectory_dir)
    assert len(records) == 1
    assert records[0].run_id == "run-1"
    assert records[0].scenario_id == "1"


def test_load_trajectories_skips_unparseable(tmp_path: Path, make_persisted_record):
    (tmp_path / "good.json").write_text(json.dumps(make_persisted_record()), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    records = load_trajectories(tmp_path)
    assert len(records) == 1


def test_load_scenarios_json_list(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            [{"id": 1, "text": "Q1"}, {"id": "2", "text": "Q2"}]
        ),
        encoding="utf-8",
    )
    out = load_scenarios(p)
    assert [s.id for s in out] == ["1", "2"]


def test_load_scenarios_jsonl(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"id": 1, "text": "Q1"}\n{"id": 2, "text": "Q2"}\n',
        encoding="utf-8",
    )
    out = load_scenarios(p)
    assert [s.id for s in out] == ["1", "2"]


def test_load_scenarios_single_object(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"id": 7, "text": "Q"}), encoding="utf-8")
    out = load_scenarios(p)
    assert [s.id for s in out] == ["7"]


_SCENARIOS_LOCAL = Path(__file__).resolve().parents[2] / "scenarios" / "local"


def test_workorder_scenarios_load_and_conform():
    """The bundled work order scenarios parse and carry the expected schema."""
    path = _SCENARIOS_LOCAL / "workorder_utterance.json"
    scenarios = load_scenarios(path)

    assert len(scenarios) >= 5
    assert all(isinstance(s, Scenario) for s in scenarios)
    # Every scenario is a work order scenario with a non-empty question and rubric.
    for s in scenarios:
        assert s.type == "WorkOrder"
        assert s.text.strip()
        assert s.category.strip()
        assert s.characteristic_form and s.characteristic_form.strip()
    # IDs are unique and at least one scenario targets failure-code categorization.
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids))
    assert any(s.category == "Categorization" for s in scenarios)


def test_join_drops_orphans(make_persisted_record):
    from evaluation.models import PersistedTrajectory

    scenarios = [
        Scenario.from_raw({"id": 1, "text": "Q1"}),
        Scenario.from_raw({"id": 2, "text": "Q2"}),
    ]
    trajs = [
        PersistedTrajectory.from_raw(make_persisted_record(scenario_id=1)),
        PersistedTrajectory.from_raw(make_persisted_record(run_id="r2", scenario_id=99)),
    ]
    pairs = list(join_records(scenarios, trajs))
    assert len(pairs) == 1
    assert pairs[0][0].id == "1"
