from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition.dataset_utils import build_dataset, load_public_scenarios
from competition.eval_framework import CompetitionKit


def test_load_public_scenarios_supports_jsonl(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "text": "Question text", "type": "Vibration"}) + "\n",
        encoding="utf-8",
    )

    scenarios = load_public_scenarios(dataset)

    assert len(scenarios) == 1
    assert scenarios[0].id == "q1"
    assert scenarios[0].text == "Question text"
    assert scenarios[0].metadata["type"] == "Vibration"


def test_load_public_scenarios_supports_mcqa_schema(tmp_path: Path):
    dataset = tmp_path / "mcqa.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "question_type": "open_ended_multi_choice",
                "passage": "This is a passage.",
                "question": "Which option should be selected?",
                "options": {"A": "Option A", "B": "Option B"},
                "metadata": {"asset_class": "toy", "family": "format_example"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    scenarios = load_public_scenarios(dataset)

    assert len(scenarios) == 1
    assert scenarios[0].id == "q1"
    assert "This is a passage" in scenarios[0].text
    assert "A. Option A" in scenarios[0].text
    assert scenarios[0].metadata["asset_class"] == "toy"


def test_build_dataset_alias(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"id": "q1", "text": "Question text"}]), encoding="utf-8")

    scenarios = build_dataset(dataset)

    assert scenarios[0].id == "q1"


def test_load_public_scenarios_requires_prompt_content(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"id": "q1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required prompt content"):
        load_public_scenarios(dataset)


def test_competition_kit_packages_submission(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id": "q1", "text": "Question text", "type": "Vibration"}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_name": "unit-model",
                    "track": "agentic_reasoning",
                    "base_model_type": "API",
                    "base_model_name": "unit-base",
                    "dataset": "unit_dataset",
                },
                "dataset": {
                    "dataset_name": "unit_dataset",
                    "dataset_path": str(dataset),
                    "description": "test",
                },
                "submission_columns": ["id", "answer"],
                "output_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )

    kit = CompetitionKit(str(config))
    result = kit.run_predictions(lambda scenario: {"answer": "A", "prediction": "A"})
    package = kit.save_submission(result)

    assert package.exists()
    with zipfile.ZipFile(package) as zf:
        assert sorted(zf.namelist()) == ["meta_data.json", "submission.csv"]
        with zf.open("submission.csv") as f:
            rows = list(csv.DictReader(line.decode("utf-8") for line in f))
        metadata = json.loads(zf.read("meta_data.json").decode("utf-8"))

    assert rows == [{"id": "q1", "answer": "A"}]
    assert metadata["meta_data"]["dataset"] == "unit_dataset"
