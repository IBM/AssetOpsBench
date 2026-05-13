"""Shared fixtures for evaluation unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.models import Scenario


@pytest.fixture
def make_scenario():
    def _factory(**overrides) -> Scenario:
        defaults = {
            "id": "1",
            "text": "What sensors are on Chiller 6?",
            "type": "iot",
            "category": "Knowledge Query",
            "characteristic_form": "Should list temperature, pressure, vibration sensors.",
        }
        defaults.update(overrides)
        return Scenario.from_raw(defaults)

    return _factory


@pytest.fixture
def make_persisted_record():
    def _factory(**overrides) -> dict:
        defaults = {
            "run_id": "run-1",
            "scenario_id": "1",
            "runner": "plan-execute",
            "model": "watsonx/ibm/granite",
            "question": "Q?",
            "answer": "A.",
            "trajectory": {
                "turns": [
                    {
                        "index": 0,
                        "text": "thinking",
                        "tool_calls": [{"name": "sites", "input": {}}],
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "duration_ms": 100.0,
                    },
                    {
                        "index": 1,
                        "text": "answer",
                        "tool_calls": [],
                        "input_tokens": 12,
                        "output_tokens": 7,
                        "duration_ms": 200.0,
                    },
                ],
                "started_at": "2026-04-27T00:00:00Z",
            },
        }
        defaults.update(overrides)
        return defaults

    return _factory


@pytest.fixture
def trajectory_dir(tmp_path: Path, make_persisted_record):
    """A directory pre-populated with one trajectory JSON file."""
    rec = make_persisted_record()
    (tmp_path / f"{rec['run_id']}.json").write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path
