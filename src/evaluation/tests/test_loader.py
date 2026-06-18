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


def test_load_trajectory_uses_filename_as_scenario_id_when_missing(tmp_path):
    trajectory_path = tmp_path / "34.json"
    trajectory_path.write_text(
        """
        {
          "run_id": "24",
          "scenario_id": null,
          "runner": "stirrup-agent",
          "model": "litellm_proxy/aws/claude-opus-4-8",
          "question": "Question text",
          "answer": "{\\"Repair\\": 13, \\"Replace\\": 20, \\"Service\\": 3}",
          "trajectory": {"turns": []}
        }
        """,
        encoding="utf-8",
    )

    trajectories = load_trajectories(tmp_path)

    assert len(trajectories) == 1
    assert trajectories[0].scenario_id == "34"


def test_load_scenarios_from_groundtruth_folders(tmp_path):
    scenario_dir = tmp_path / "scenario_11"
    scenario_dir.mkdir()
    (scenario_dir / "groundtruth.txt").write_text(
        "{'energy': 14, 'material': 48}",
        encoding="utf-8",
    )

    scenarios = load_scenarios(tmp_path)

    assert len(scenarios) == 1
    assert scenarios[0].id == "11"
    assert scenarios[0].expected_answer == "{'energy': 14, 'material': 48}"
    assert scenarios[0].scoring_method == "static_json"