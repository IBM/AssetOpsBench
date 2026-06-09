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
}

PUBLIC_EXTRA_FIELDS = {
    "type",
    "category",
    "asset_class",
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
        text = raw.get("text", raw.get("question", raw.get("prompt")))
        if scenario_id is None:
            raise ValueError(f"Record {index} is missing required field 'id'.")
        if not text:
            raise ValueError(f"Record {index} is missing required field 'text'.")

        metadata = {k: raw[k] for k in PUBLIC_EXTRA_FIELDS if k in raw}
        scenarios.append(
            AssetOpsScenario(id=str(scenario_id), text=str(text), metadata=metadata)
        )

    return scenarios


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

