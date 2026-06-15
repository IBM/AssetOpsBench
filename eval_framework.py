"""Industrial Automation Challenge submission framework.

Runs participant prediction code over challenge scenarios and writes a Kaggle-ready
CSV submission.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from .dataset_utils import AssetOpsScenario, load_public_scenarios
except ImportError:
    from dataset_utils import AssetOpsScenario, load_public_scenarios

logger = logging.getLogger(__name__)
PredictionFunc = Callable[[AssetOpsScenario], Any]


@dataclass
class SubmissionResult:
    dataset_name: str
    predictions: list[dict[str, str]]


def _clean_cell(value: Any, fallback: str = "NOTAVALUE") -> str:
    if value is None:
        return fallback
    text = str(value).replace("\r", " ").strip()
    return text if text else fallback


def _normalize_prediction(raw: Any) -> dict[str, str]:
    """Normalize predictor output into the `answer` submission column."""

    if isinstance(raw, dict):
        answer = raw.get("answer", raw.get("choice", raw.get("prediction", raw.get("response", ""))))
    else:
        answer = raw
    return {"answer": _clean_cell(answer, "NOTAVALUE")}


def _load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_predictor(spec: str) -> PredictionFunc:
    """Load ``module:function`` or a Python-file path plus function."""

    if ":" not in spec:
        raise ValueError("Predictor must be formatted as 'module:function'.")

    module_name, function_name = spec.rsplit(":", 1)
    module_path = Path(module_name)
    if module_path.suffix == ".py" or module_path.exists():
        module = _load_module_from_file(module_path.resolve())
    else:
        module = importlib.import_module(module_name)

    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"Predictor target is not callable: {spec}")
    return func


def command_predictor(command_template: str) -> PredictionFunc:
    """Create a predictor that invokes a participant-controlled command.

    Available template fields:
    - ``{id}``
    - ``{question}``
    - ``{question_json}``
    - ``{scenario_json}``

    If stdout is JSON with an ``answer``, ``choice``, or ``prediction`` field,
    that value is used. Otherwise stdout becomes the answer value.
    """

    def predict(scenario: AssetOpsScenario) -> dict[str, str]:
        command = command_template.format(
            id=scenario.id,
            question=scenario.text,
            question_json=json.dumps(scenario.text),
            scenario_json=json.dumps(scenario.to_dict(), ensure_ascii=False),
        )
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return {"answer": "NOTAVALUE"}

        stdout = completed.stdout.strip()
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return {"answer": stdout}
        if isinstance(parsed, dict):
            return parsed
        return {"answer": stdout}

    return predict


class CompetitionKit:
    """Small starter-kit class for generating CSV submissions."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.config = load_config_file(config_path) if config_path else {}
        self.output_dir = Path(self.config.get("output_dir", "competition_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_config = self.config.get("dataset", {})
        self.submission_columns = self.config.get("submission_columns", ["id", "answer"])

    def list_datasets(self) -> None:
        name = self.dataset_config.get("dataset_name", "industrial_automation_challenge")
        description = self.dataset_config.get("description", "")
        print(f"{name}: {description}")

    def run_predictions(
        self,
        predictor: PredictionFunc,
        *,
        subset_size: int | None = None,
        dataset_path: str | None = None,
    ) -> SubmissionResult:
        dataset_path = dataset_path or self.dataset_config.get("dataset_path")
        if not dataset_path:
            raise ValueError("Dataset path is required. Set dataset.dataset_path or pass --dataset-path.")

        dataset_name = self.dataset_config.get("dataset_name", Path(dataset_path).stem)
        scenarios = load_public_scenarios(dataset_path)
        if subset_size is not None and subset_size > 0:
            scenarios = scenarios[:subset_size]

        predictions: list[dict[str, str]] = []
        for index, scenario in enumerate(scenarios, start=1):
            logger.info("Predicting %s/%s: %s", index, len(scenarios), scenario.id)
            try:
                normalized = _normalize_prediction(predictor(scenario))
            except Exception:
                logger.exception("Predictor failed for scenario %s", scenario.id)
                normalized = {"answer": "NOTAVALUE"}
            predictions.append({"id": scenario.id, **normalized})

        return SubmissionResult(dataset_name=dataset_name, predictions=predictions)

    def save_submission(self, result: SubmissionResult, *, filename: str = "submission.csv") -> Path:
        csv_path = self.output_dir / filename
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.submission_columns,
                quoting=csv.QUOTE_ALL,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(result.predictions)
        return csv_path


def create_metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial Automation Challenge submission starter kit")
    parser.add_argument("--config", type=str, help="Path to dataset/predictor JSON config.")
    parser.add_argument("--dataset-path", type=str, help="Override dataset path from config.")
    parser.add_argument("--predictor", type=str, help="Python predictor as module:function.")
    parser.add_argument(
        "--agent-command",
        type=str,
        help="Shell command template for an existing agent. Use {question_json} for the prompt.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default="submission.csv")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def load_config_file(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_merge_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config_file(args.config) if args.config else {}

    if args.output_dir is None and config.get("output_dir"):
        args.output_dir = config["output_dir"]
    if args.output_file == "submission.csv" and config.get("output_file"):
        args.output_file = config["output_file"]

    predictor = config.get("predictor", {})
    if args.predictor is None and predictor.get("path"):
        args.predictor = predictor["path"]
    if args.agent_command is None and predictor.get("agent_command"):
        args.agent_command = predictor["agent_command"]

    dataset = config.get("dataset", {})
    if args.dataset_path is None and dataset.get("dataset_path"):
        args.dataset_path = dataset["dataset_path"]

    return args


def build_predictor_from_args(args: argparse.Namespace) -> PredictionFunc:
    if args.predictor:
        return load_predictor(args.predictor)
    if args.agent_command:
        return command_predictor(args.agent_command)
    raise ValueError("Provide either --predictor module:function or --agent-command.")
