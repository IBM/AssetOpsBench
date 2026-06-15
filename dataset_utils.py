"""Dataset loading utilities for the Industrial Automation Challenge starter kit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PUBLIC_EXTRA_FIELDS = {
    "question_type",
    "passage",
    "question",
    "options",
    "type",
    "category",
    "asset_class",
    "family",
    "domain",
    "phase",
    "difficulty",
}


@dataclass(frozen=True)
class AssetOpsScenario:
    """One Industrial Automation Challenge scenario."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def options(self) -> Any:
        return self.metadata.get("options")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, **self.metadata}


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON list, single JSON object, or JSONL file."""

    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    raw = json.loads(text)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            return raw["data"]
        return [raw]
    raise ValueError(f"Unsupported JSON shape in {p}: {type(raw).__name__}")


def load_public_scenarios(path: str | Path) -> list[AssetOpsScenario]:
    """Load scenarios from a JSON/JSONL dataset file."""

    scenarios: list[AssetOpsScenario] = []
    for index, raw in enumerate(read_json_records(path)):
        if not isinstance(raw, dict):
            raise ValueError(f"Record {index} must be an object, got {type(raw).__name__}")

        scenario_id = raw.get("id", raw.get("scenario_id"))
        text = raw.get("text", raw.get("prompt"))
        if not text:
            text = _compose_question_text(raw)
        if scenario_id is None:
            raise ValueError(f"Record {index} is missing required field 'id'.")
        if not text:
            raise ValueError(
                f"Record {index} is missing required prompt content. Expected "
                "either 'text'/'prompt' or the MCQA fields 'passage' and/or 'question'."
            )

        metadata: dict[str, Any] = {}
        if isinstance(raw.get("metadata"), dict):
            metadata.update(raw["metadata"])
        metadata.update({k: raw[k] for k in PUBLIC_EXTRA_FIELDS if k in raw})
        scenarios.append(AssetOpsScenario(id=str(scenario_id), text=str(text), metadata=metadata))

    return scenarios


def build_dataset(dataset_path: str | Path | None = None) -> list[AssetOpsScenario]:
    """Dataset builder used by the starter framework."""

    if dataset_path is None:
        raise ValueError("dataset_path is required")
    return load_public_scenarios(dataset_path)


def _compose_question_text(raw: dict[str, Any]) -> str:
    """Build a prompt from the MCQA schema."""

    parts: list[str] = []
    passage = raw.get("passage")
    question = raw.get("question")
    options = raw.get("options")
    if passage:
        parts.append(str(passage).strip())
    if question:
        parts.append(str(question).strip())
    if options:
        parts.append(_format_options(options))
    return "\n\n".join(part for part in parts if part)


def _format_options(options: Any) -> str:
    if isinstance(options, dict):
        items = options.items()
    elif isinstance(options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        items = [(letters[i], value) for i, value in enumerate(options)]
    else:
        return str(options).strip()
    lines = ["Options:"]
    for key, value in items:
        lines.append(f"{key}. {value}")
    return "\n".join(lines)
