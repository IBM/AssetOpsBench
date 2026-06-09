"""AssetOpsBench public competition submission framework.

It runs participant prediction code over public scenarios and packages a
Kaggle/offline submission zip. AssetOpsBench scoring remains organizer-side.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from .dataset_utils import AssetOpsScenario, load_public_scenarios
except ImportError:
    from dataset_utils import AssetOpsScenario, load_public_scenarios


logger = logging.getLogger(__name__)

PredictionFunc = Callable[[AssetOpsScenario], Any]


REQUIRED_METADATA_FIELDS = (
    "model_name",
    "track",
    "base_model_type",
    "base_model_name",
    "dataset",
)


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
    """Normalize predictor output into submission columns."""

    if isinstance(raw, dict):
        prediction = raw.get("prediction", raw.get("answer", raw.get("response", "")))
        reasoning = raw.get("reasoning", raw.get("rationale", ""))
        trajectory = raw.get("trajectory", raw.get("trace", ""))
    else:
        prediction = raw
        reasoning = ""
        trajectory = ""

    if isinstance(reasoning, (dict, list)):
        reasoning = json.dumps(reasoning, ensure_ascii=False)
    if isinstance(trajectory, (dict, list)):
        trajectory = json.dumps(trajectory, ensure_ascii=False)

    return {
        "prediction": _clean_cell(prediction, "No prediction available"),
        "reasoning": _clean_cell(reasoning, "No reasoning provided"),
        "trajectory": _clean_cell(trajectory, "No trajectory provided"),
    }


def _load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_predictor(spec: str) -> PredictionFunc:
    """Load ``module:function`` or an absolute Python-file path plus function."""

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

    If stdout is JSON with an ``answer`` or ``prediction`` field, that field is
    used. Otherwise stdout becomes the prediction text.
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
            return {
                "prediction": "Error occurred",
                "reasoning": completed.stderr.strip(),
                "trajectory": "",
            }

        stdout = completed.stdout.strip()
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return {"prediction": stdout, "reasoning": "", "trajectory": ""}
        if isinstance(parsed, dict):
            return parsed
        return {"prediction": stdout, "reasoning": "", "trajectory": ""}

    return predict


class CompetitionKit:
    """Small public starter-kit class for generating submissions."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.config = load_config_file(config_path) if config_path else {}
        self.output_dir = Path(self.config.get("output_dir", "competition_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_config = self.config.get("dataset", {})
        self.metadata_config = self.config.get("metadata", {})

    def list_datasets(self) -> None:
        name = self.dataset_config.get("dataset_name", "assetopsbench")
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
            except Exception as exc:
                logger.exception("Predictor failed for scenario %s", scenario.id)
                normalized = {
                    "prediction": "Error occurred",
                    "reasoning": str(exc),
                    "trajectory": "No trajectory provided",
                }
            predictions.append({"id": scenario.id, **normalized})

        return SubmissionResult(dataset_name=dataset_name, predictions=predictions)

    def save_submission(
        self,
        result: SubmissionResult,
        *,
        filename: str = "submission.csv",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        metadata = self.get_metadata(result.dataset_name, metadata)
        self._validate_metadata(metadata)

        csv_path = self.output_dir / filename
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "prediction", "reasoning", "trajectory"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(result.predictions)

        metadata_path = self.output_dir / "meta_data.json"
        metadata_path.write_text(
            json.dumps({"meta_data": metadata}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        zip_path = self.output_dir / filename.replace(".csv", ".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, filename)
            zipf.write(metadata_path, "meta_data.json")

        return zip_path

    def get_metadata(
        self,
        dataset_name: str,
        fallback_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "model_name": "unknown",
            "track": "agentic_reasoning",
            "base_model_type": "API",
            "base_model_name": "unknown",
            "dataset": dataset_name,
            "additional_info": "",
        }
        metadata.update(self.metadata_config)
        if fallback_metadata:
            metadata.update({k: v for k, v in fallback_metadata.items() if v is not None})
        return metadata

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        missing = [field for field in REQUIRED_METADATA_FIELDS if not metadata.get(field)]
        if missing:
            raise ValueError(f"Missing required metadata field(s): {', '.join(missing)}")


def create_metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AssetOpsBench submission starter kit")
    parser.add_argument("--config", type=str, help="Path to metadata/dataset JSON config.")
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
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--track", type=str, choices=["internal_reasoning", "agentic_reasoning"])
    parser.add_argument("--base-model-type", type=str, choices=["API", "OpenWeighted", "Hybrid"])
    parser.add_argument("--base-model-name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--additional-info", type=str)
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
    if args.dataset is None and dataset.get("dataset_name"):
        args.dataset = dataset["dataset_name"]

    metadata = config.get("metadata", {})
    for field in (
        "model_name",
        "track",
        "base_model_type",
        "base_model_name",
        "additional_info",
    ):
        arg_name = field.replace("-", "_")
        if getattr(args, arg_name, None) is None and field in metadata:
            setattr(args, arg_name, metadata[field])

    return args


def metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "track": args.track,
        "base_model_type": args.base_model_type,
        "base_model_name": args.base_model_name,
        "dataset": args.dataset,
        "additional_info": args.additional_info,
    }


def build_predictor_from_args(args: argparse.Namespace) -> PredictionFunc:
    if args.predictor:
        return load_predictor(args.predictor)
    if args.agent_command:
        return command_predictor(args.agent_command)
    raise ValueError("Provide either --predictor module:function or --agent-command.")
