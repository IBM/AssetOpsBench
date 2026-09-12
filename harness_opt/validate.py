"""Gates a candidate must clear before it costs a benchmark run.

Four cheap checks, all of them before any agent executes. StarHarness's four
forbidden classes, plus a fifth this corpus needs.

The fifth gate distinguishes **schema identifiers** from **data values**. An
adapter that narrows a work-order schema has to name the field ``siteid``; that
is the interface. Writing ``MAIN`` into the harness is memorizing the corpus.
Banning both would reject every legitimate patch, which is why the corpus is
built from the *values* columns take, never from their names.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Field and tool names are the interface and are always allowed, even when a
#: value elsewhere in the corpus happens to match.
SCHEMA_IDENTIFIERS = frozenset(
    {
        "siteid", "assetnum", "status", "wonum", "failurecode", "location",
        "description", "worktype", "wopriority", "parent", "taskid", "asset_id",
        "timestamp", "sensors", "orgid", "assettype", "reportedby",
    }
)

#: Values shorter than this are too generic to attribute to the corpus.
MIN_VALUE_LENGTH = 4


@dataclass
class Verdict:
    """Why a candidate was rejected, or that it was not."""

    ok: bool
    gate: str = ""
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def passed(cls) -> "Verdict":
        return cls(ok=True)

    @classmethod
    def failed(cls, gate: str, reason: str) -> "Verdict":
        return cls(ok=False, gate=gate, reason=reason)


@dataclass
class EntityCorpus:
    """The literal values that appear in the scenario data."""

    values: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_shared(cls, shared_dir: Path | str) -> "EntityCorpus":
        """Harvest distinct values from the CSV and JSON files under shared/."""
        root = Path(shared_dir)
        found: set[str] = set()
        for path in sorted(root.rglob("*.csv")):
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    found.update(_usable(v) for v in row.values())
        for path in sorted(root.rglob("*.json")):
            try:
                found.update(_walk_json(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        found.discard("")
        return cls(values=frozenset(v for v in found if v))

    def hits(self, text: str) -> list[str]:
        """Corpus values quoted verbatim in the text, longest first."""
        lowered = text.lower()
        return sorted(
            {
                v
                for v in self.values
                if v.lower() in lowered and v.lower() not in SCHEMA_IDENTIFIERS
            },
            key=lambda v: (-len(v), v),
        )


def check_scope(changed_paths: list[str], editable: list[str]) -> Verdict:
    """The diff touches only the declared-editable directories."""
    outside = sorted(
        p for p in changed_paths if not any(p.startswith(e) for e in editable)
    )
    if outside:
        return Verdict.failed(
            "scope", f"touches {outside} outside the editable surface {editable}"
        )
    return Verdict.passed()


def check_spec(spec_raw: dict) -> Verdict:
    """The spec parses, which is this adapter's equivalent of "it imports"."""
    from agent.stirrup_agent.adapter import AdapterError, AdapterSpec

    try:
        AdapterSpec.from_raw(spec_raw)
    except AdapterError as exc:
        return Verdict.failed("spec", str(exc))
    return Verdict.passed()


def check_no_leak(text: str, answers: list[str], window: int = 8) -> Verdict:
    """No eight-word sequence shared with the answer set.

    The same rule ``skills/tools/validate_skills.py`` already applies to skill
    libraries, pointed at the patch instead.
    """
    patch_grams = _ngrams(text, window)
    if not patch_grams:
        return Verdict.passed()
    for answer in answers:
        shared = patch_grams & _ngrams(answer, window)
        if shared:
            return Verdict.failed(
                "no_leak",
                f"shares a {window}-word sequence with an answer: "
                f"{next(iter(sorted(shared)))!r}",
            )
    return Verdict.passed()


def check_no_entity_values(text: str, corpus: EntityCorpus) -> Verdict:
    """No literal from the scenario data, schema identifiers excepted."""
    hits = corpus.hits(text)
    if hits:
        return Verdict.failed(
            "no_entity_values",
            f"quotes corpus values {hits[:5]}; adapt the interface, not the data",
        )
    return Verdict.passed()


def validate(
    spec_raw: dict,
    changed_paths: list[str],
    editable: list[str],
    corpus: EntityCorpus | None = None,
    answers: list[str] | None = None,
) -> Verdict:
    """Every gate, cheapest first. The first failure is the reported one."""
    for verdict in (
        check_scope(changed_paths, editable),
        check_spec(spec_raw),
    ):
        if not verdict:
            return verdict

    text = json.dumps(spec_raw)
    if answers:
        verdict = check_no_leak(text, answers)
        if not verdict:
            return verdict
    if corpus is not None:
        verdict = check_no_entity_values(text, corpus)
        if not verdict:
            return verdict
    return Verdict.passed()


def _ngrams(text: str, window: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if len(words) < window:
        return set()
    return {tuple(words[i : i + window]) for i in range(len(words) - window + 1)}


def _usable(value) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if len(value) < MIN_VALUE_LENGTH or value.lower() in SCHEMA_IDENTIFIERS:
        return ""
    # Long free text is prose, not an identifier worth banning.
    return value if len(value.split()) <= 6 else ""


def _walk_json(obj, depth: int = 0) -> set[str]:
    if depth > 8:
        return set()
    found: set[str] = set()
    if isinstance(obj, dict):
        for value in obj.values():
            found |= _walk_json(value, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:500]:
            found |= _walk_json(item, depth + 1)
    elif isinstance(obj, str):
        usable = _usable(obj)
        if usable:
            found.add(usable)
    return found
