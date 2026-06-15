"""Public dataset loading helpers for AssetOpsBench competition submissions.

The public competition dataset must not contain ground truth or rubric fields.
These helpers intentionally reject private/evaluation fields by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PRIVATE_FIELD_NAMES = {
    "answer",
    "answers",
    "correct_answer",
    "expected_answer",
    "ground_truth",
    "label",
    "labels",
    "reference_answer",
    "rubric",
    "scoring_method",
    "target",
    "characteristic_form",
    # Kaggle solution files use ``usage`` as the Public/Private split marker.
    # It is organizer-side metadata and must not be present in participant
    # datasets. If the competition later scores token/cost efficiency, use a
    # distinct public column name such as ``token_usage`` to avoid ambiguity.
    "usage",
}

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
    """One public AssetOpsBench competition scenario."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

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


def load_public_scenarios(
    path: str | Path,
    *,
    allow_private_fields: bool = False,
) -> list[AssetOpsScenario]:
    """Load scenarios from a public dataset file.

    By default this raises if any record includes ground-truth-like fields.
    Set ``allow_private_fields=True`` only for local organizer-side conversion
    scripts, never for a public Kaggle data artifact.
    """

    scenarios: list[AssetOpsScenario] = []
    for index, raw in enumerate(read_json_records(path)):
        if not isinstance(raw, dict):
            raise ValueError(f"Record {index} must be an object, got {type(raw).__name__}")

        private = sorted(PRIVATE_FIELD_NAMES.intersection(raw))
        if private and not allow_private_fields:
            joined = ", ".join(private)
            raise ValueError(
                f"Record {index} contains private evaluation field(s): {joined}. "
                "Remove ground truth before publishing or submitting."
            )

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
        scenarios.append(
            AssetOpsScenario(id=str(scenario_id), text=str(text), metadata=metadata)
        )

    return scenarios


def _compose_question_text(raw: dict[str, Any]) -> str:
    """Build a model prompt from the FailureSensorIQ/MCQA public schema."""

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


def strip_private_fields(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return public-safe copies of private scenario records."""

    public_records: list[dict[str, Any]] = []
    for raw in records:
        cleaned = {k: v for k, v in raw.items() if k not in PRIVATE_FIELD_NAMES}
        public_records.append(cleaned)
    return public_records


def write_public_dataset(source_path: str | Path, output_path: str | Path) -> Path:
    """Create a public-safe JSONL dataset by removing private fields."""

    records = strip_private_fields(read_json_records(source_path))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
