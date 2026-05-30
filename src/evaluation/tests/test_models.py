"""Tests for evaluation Pydantic models."""

from evaluation.models import PersistedTrajectory, Scenario


def test_scenario_from_raw_coerces_int_id_to_str():
    s = Scenario.from_raw({"id": 301, "text": "Q"})
    assert s.id == "301"
    assert isinstance(s.id, str)


def test_scenario_preserves_extra_fields():
    s = Scenario.from_raw({"id": "1", "text": "Q", "characteristic_form": "X", "tolerance": 0.01})
    extra = s.model_extra or {}
    assert extra.get("tolerance") == 0.01


def test_persisted_trajectory_coerces_scenario_id():
    t = PersistedTrajectory.from_raw(
        {
            "run_id": "r",
            "scenario_id": 42,
            "runner": "plan-execute",
            "model": "m",
            "question": "q",
            "answer": "a",
            "trajectory": None,
        }
    )
    assert t.scenario_id == "42"


def test_persisted_trajectory_allows_none_scenario_id():
    t = PersistedTrajectory.from_raw(
        {
            "run_id": "r",
            "scenario_id": None,
            "runner": "plan-execute",
            "model": "m",
            "question": "q",
            "answer": "a",
            "trajectory": None,
        }
    )
    assert t.scenario_id is None
