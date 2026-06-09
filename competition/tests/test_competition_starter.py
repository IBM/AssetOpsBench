from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition.dataset_utils import load_public_scenarios, write_public_dataset
from competition.eval_framework import CompetitionKit


def test_load_public_scenarios_rejects_private_fields(tmp_path: Path):
    dataset = tmp_path / "private.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": 1,
                "text": "Q",
                "characteristic_form": "private rubric",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="private evaluation field"):
        load_public_scenarios(dataset)


def test_write_public_dataset_strips_private_fields(tmp_path: Path):
    private_dataset = tmp_path / "private.json"
    public_dataset = tmp_path / "public.jsonl"
    private_dataset.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "text": "Q",
                    "type": "Vibration",
                    "expected_answer": "A",
                    "characteristic_form": "rubric",
                }
            ]
        ),
        encoding="utf-8",
    )

    write_public_dataset(private_dataset, public_dataset)
    scenarios = load_public_scenarios(public_dataset)

    assert len(scenarios) == 1
    assert scenarios[0].id == "1"
    assert scenarios[0].metadata["type"] == "Vibration"


def test_competition_kit_packages_submission(tmp_path: Path):
    dataset = tmp_path / "public.jsonl"
    dataset.write_text(
        '{"id": "301", "text": "What vibration tools are available?", "type": "Vibration"}\n',
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
                "output_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )

    kit = CompetitionKit(str(config))
    result = kit.run_predictions(
        lambda scenario: {
            "prediction": f"answer for {scenario.id}",
            "reasoning": {"used": "unit"},
            "trajectory": [{"tool": "none"}],
        }
    )
    package = kit.save_submission(result)

    assert package.exists()
    with zipfile.ZipFile(package) as zf:
        assert sorted(zf.namelist()) == ["meta_data.json", "submission.csv"]
        with zf.open("submission.csv") as f:
            rows = list(csv.DictReader(line.decode("utf-8") for line in f))
        metadata = json.loads(zf.read("meta_data.json").decode("utf-8"))

    assert rows == [
        {
            "id": "301",
            "prediction": "answer for 301",
            "reasoning": '{"used": "unit"}',
            "trajectory": '[{"tool": "none"}]',
        }
    ]
    assert metadata["meta_data"]["dataset"] == "unit_dataset"
