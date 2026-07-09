"""Static JSON scorer for structured AssetOpsBench answers.

This scorer is deterministic and is intended for structured outputs such as:

- JSON objects
- JSON arrays
- Python-style dictionaries
- Python-style lists/tuples
- nested structures
- integer/count-only answers
- noisy outputs with markdown fences or answer prefixes

It plugs into the existing evaluation pipeline as a scorer named
``static_json`` and can be invoked with:

    uv run evaluate --scorer-default static_json ...
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import Scenario, ScorerResult
from . import register


@dataclass
class KeyComparison:
    """Per-key comparison between gold and model output."""

    key: str
    gold_value: str
    model_value: str
    exact: bool
    match_type: str
    similarity: float
    accepted: bool
    numeric: bool = False
    range_match: bool | None = None
    delta_1_match: bool | None = None


@dataclass
class StaticJsonScore:
    """Structured score for one gold/model answer pair."""

    partial_match_accuracy: float
    partial_exact_match_accuracy: float
    strict_exact_match_accuracy: float
    partial_similarity_score: float
    partial_numeric_match_accuracy: float
    range_match_accuracy: float
    delta_1_match_accuracy: float
    precision: float
    recall: float
    f1: float
    total_gold_keys: int
    total_model_keys: int
    matched_keys: int
    accepted_value_matches: int
    exact_value_matches: int
    numeric_gold_keys: int
    numeric_value_matches: int
    range_eligible_keys: int
    range_value_matches: int
    delta_1_eligible_keys: int
    delta_1_value_matches: int
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    details: list[KeyComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        data = asdict(self)
        data["details"] = [asdict(item) for item in self.details]
        return data


def extract_answer_text(text: Any) -> str:
    """Extract likely final answer text from raw model output."""
    if not isinstance(text, str):
        return str(text)

    content = text.strip()

    patterns = [
        r"<Answer>\s*:?\s*(.*)$",
        r"Final Answer\s*:?\s*(.*)$",
        r"Answer\s*:?\s*(.*)$",
        r"Output\s*:?\s*(.*)$",
        r"Result\s*:?\s*(.*)$",
        r"Response\s*:?\s*(.*)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    return content


def _strip_markdown_fence(content: str) -> str:
    """Strip markdown code fences if present."""
    content = content.strip()

    match = re.search(
        r"```(?:json|python|py)?\s*(.*?)```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    return content


def _extract_balanced_structure(content: str) -> str:
    """Extract first balanced {...}, [...], or (...) from noisy text."""
    content = content.strip()

    candidates = [
        (content.find("{"), "{", "}"),
        (content.find("["), "[", "]"),
        (content.find("("), "(", ")"),
    ]
    candidates = [
        (idx, open_ch, close_ch)
        for idx, open_ch, close_ch in candidates
        if idx != -1
    ]

    if not candidates:
        return content

    start, open_ch, close_ch = min(candidates, key=lambda item: item[0])

    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for index in range(start, len(content)):
        ch = content[index]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in {"'", '"'}:
            in_string = True
            quote_char = ch
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return content[start : index + 1].strip()

    return content[start:].strip()


def _extract_count_from_text(content: str) -> int | float | None:
    """Extract a count when the answer is count-only or nearly count-only."""
    stripped = content.strip()

    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)

    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return float(stripped)

    numbers = re.findall(
        r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])",
        stripped,
    )
    if len(numbers) == 1:
        number = numbers[0]
        return float(number) if "." in number else int(number)

    return None


def parse_structured_answer(value: Any) -> Any:
    """Parse JSON/Python-like structured output into a Python object."""
    if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
        return value

    content = extract_answer_text(value)
    content = _strip_markdown_fence(content)

    count = _extract_count_from_text(content)
    if count is not None:
        return count

    content = _extract_balanced_structure(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        pass

    try:
        return json.loads(content.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    count = _extract_count_from_text(content)
    if count is not None:
        return count

    return content.strip()


def normalize_value(value: Any) -> str:
    """Normalize scalar values for stable comparison."""
    parsed = parse_structured_answer(value)

    if isinstance(parsed, bool):
        return str(parsed).lower()

    if parsed is None:
        return "none"

    if isinstance(parsed, float):
        return f"{parsed:.6f}".rstrip("0").rstrip(".")

    if isinstance(parsed, int):
        return str(parsed)

    return str(parsed).strip().lower()


def flatten_answer(value: Any, prefix: str = "answer") -> dict[str, str]:
    """Flatten nested structures into comparable key-value pairs."""
    parsed = parse_structured_answer(value)

    if isinstance(parsed, dict):
        flat: dict[str, str] = {}
        for key, item in parsed.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_answer(item, new_prefix))
        return flat

    if isinstance(parsed, (list, tuple)):
        flat = {}
        for index, item in enumerate(parsed):
            flat.update(flatten_answer(item, f"{prefix}[{index}]"))
        return flat

    return {prefix: normalize_value(parsed)}


def similarity_score(gold_value: str, model_value: str) -> float:
    """Return a partial similarity score in [0, 1]."""
    if gold_value == model_value:
        return 1.0

    try:
        gold_num = float(gold_value)
        model_num = float(model_value)

        if gold_num == 0:
            return 1.0 if model_num == 0 else 0.0

        relative_error = abs(gold_num - model_num) / abs(gold_num)

        if relative_error < 0.01:
            return 0.9
        if relative_error < 0.05:
            return 0.7
        if relative_error < 0.10:
            return 0.5
        return 0.0

    except (TypeError, ValueError):
        pass

    gold_chars = set(gold_value)
    model_chars = set(model_value)
    union = gold_chars | model_chars

    if not union:
        return 1.0

    score = len(gold_chars & model_chars) / len(union)

    if gold_value in model_value or model_value in gold_value:
        score = max(score, 0.6)

    return score


_NUMBER_RE = r"[+-]?\d+(?:\.\d+)?"
_RANGE_FIELD_PAIRS = (
    ("start_point", "end_point"),
    ("start", "end"),
    ("start_time", "end_time"),
    ("start_timestamp", "end_timestamp"),
    ("min", "max"),
    ("minimum", "maximum"),
    ("lower", "upper"),
    ("lower_bound", "upper_bound"),
    ("range_start", "range_end"),
)


@dataclass(frozen=True)
class _ValueComparison:
    exact: bool
    accepted: bool
    match_type: str
    similarity: float
    numeric: bool
    numeric_match: bool
    range_eligible: bool
    range_match: bool | None
    delta_1_eligible: bool
    delta_1_match: bool | None


def _as_float(value: str) -> float | None:
    """Parse a normalized scalar value as a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _normalize_range(left: float, right: float) -> tuple[float, float]:
    return (left, right) if left <= right else (right, left)


def _extract_numeric_range(value: str) -> tuple[float, float] | None:
    """Parse simple range strings such as ``240-511`` or ``between 3 and 7``."""
    value = value.strip().lower()

    patterns = [
        rf"^({_NUMBER_RE})\s*(?:\.\.|-|to|through|,|–|—)\s*({_NUMBER_RE})$",
        rf"^between\s+({_NUMBER_RE})\s+and\s+({_NUMBER_RE})$",
        rf"^(?:min|minimum|lower|start)\s*[:=]\s*({_NUMBER_RE}).*"
        rf"(?:max|maximum|upper|end)\s*[:=]\s*({_NUMBER_RE})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue
        left = _as_float(match.group(1))
        right = _as_float(match.group(2))
        if left is not None and right is not None:
            return _normalize_range(left, right)

    return None


def _split_flat_key(key: str) -> tuple[str, str] | None:
    if "." not in key:
        return None
    parent, field_name = key.rsplit(".", 1)
    return parent, field_name


def _build_context_ranges(gold_flat: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Infer sibling numeric ranges from flattened gold fields.

    For anomaly detection outputs, this maps both ``answer.start_point`` and
    ``answer.end_point`` to the interval bounded by those gold values.
    """
    by_parent: dict[str, dict[str, str]] = {}
    for key in gold_flat:
        split = _split_flat_key(key)
        if split is None:
            continue
        parent, field_name = split
        by_parent.setdefault(parent, {})[field_name] = key

    ranges: dict[str, tuple[float, float]] = {}
    for fields in by_parent.values():
        for lower_field, upper_field in _RANGE_FIELD_PAIRS:
            lower_key = fields.get(lower_field)
            upper_key = fields.get(upper_field)
            if lower_key is None or upper_key is None:
                continue

            lower = _as_float(gold_flat[lower_key])
            upper = _as_float(gold_flat[upper_key])
            if lower is None or upper is None:
                continue

            value_range = _normalize_range(lower, upper)
            ranges[lower_key] = value_range
            ranges[upper_key] = value_range

    return ranges


def _compare_value(
    key: str,
    gold_value: str,
    model_value: str,
    context_ranges: dict[str, tuple[float, float]],
    *,
    similarity_threshold: float,
) -> _ValueComparison:
    """Compare one flattened value with exact, range, and delta-1 checks."""
    base_similarity = similarity_score(gold_value, model_value)
    exact = gold_value == model_value

    gold_num = _as_float(gold_value)
    model_num = _as_float(model_value)
    gold_range = _extract_numeric_range(gold_value)
    context_range = context_ranges.get(key)
    value_range = context_range or gold_range

    numeric = (
        gold_num is not None
        or gold_range is not None
        or context_range is not None
    )

    delta_1_eligible = gold_num is not None and model_num is not None
    delta_1_match = (
        abs(model_num - gold_num) <= 1
        if delta_1_eligible and model_num is not None and gold_num is not None
        else None
    )

    range_eligible = value_range is not None and model_num is not None
    range_match = (
        value_range[0] <= model_num <= value_range[1]
        if range_eligible and model_num is not None and value_range is not None
        else None
    )

    numeric_match = bool(exact or range_match or delta_1_match) if numeric else False
    accepted = bool(exact or numeric_match)

    if exact:
        match_type = "exact"
        similarity = 1.0
    elif delta_1_match:
        match_type = "partial_delta_1"
        similarity = 1.0
    elif range_match:
        match_type = "partial_range"
        similarity = 1.0
    elif base_similarity > similarity_threshold:
        match_type = f"partial ({base_similarity:.2f})"
        similarity = base_similarity
    else:
        match_type = "mismatch"
        similarity = base_similarity

    return _ValueComparison(
        exact=exact,
        accepted=accepted,
        match_type=match_type,
        similarity=similarity,
        numeric=numeric,
        numeric_match=numeric_match,
        range_eligible=range_eligible,
        range_match=range_match,
        delta_1_eligible=delta_1_eligible,
        delta_1_match=delta_1_match,
    )


def evaluate_static_json(
    gold_answer: Any,
    model_answer: Any,
    *,
    similarity_threshold: float = 0.0,
) -> StaticJsonScore:
    """Evaluate one structured gold answer against one model answer."""
    gold_flat = flatten_answer(gold_answer)
    model_flat = flatten_answer(model_answer)
    context_ranges = _build_context_ranges(gold_flat)

    gold_keys = set(gold_flat)
    model_keys = set(model_flat)
    common_keys = gold_keys & model_keys

    details: list[KeyComparison] = []
    exact_matches = 0
    accepted_matches = 0
    numeric_gold_keys = 0
    numeric_matches = 0
    range_eligible_keys = 0
    range_matches = 0
    delta_1_eligible_keys = 0
    delta_1_matches = 0
    total_similarity = 0.0

    for key in sorted(common_keys):
        gold_value = gold_flat[key]
        model_value = model_flat[key]

        comparison = _compare_value(
            key,
            gold_value,
            model_value,
            context_ranges,
            similarity_threshold=similarity_threshold,
        )
        total_similarity += comparison.similarity

        if comparison.exact:
            exact_matches += 1

        if comparison.accepted:
            accepted_matches += 1

        if comparison.numeric:
            numeric_gold_keys += 1
            if comparison.numeric_match:
                numeric_matches += 1

        if comparison.range_eligible:
            range_eligible_keys += 1
            if comparison.range_match:
                range_matches += 1

        if comparison.delta_1_eligible:
            delta_1_eligible_keys += 1
            if comparison.delta_1_match:
                delta_1_matches += 1

        details.append(
            KeyComparison(
                key=key,
                gold_value=gold_value,
                model_value=model_value,
                exact=comparison.exact,
                match_type=comparison.match_type,
                similarity=comparison.similarity,
                accepted=comparison.accepted,
                numeric=comparison.numeric,
                range_match=comparison.range_match,
                delta_1_match=comparison.delta_1_match,
            )
        )

    missing_keys = sorted(gold_keys - model_keys)
    extra_keys = sorted(model_keys - gold_keys)

    for key in missing_keys:
        details.append(
            KeyComparison(
                key=key,
                gold_value=gold_flat[key],
                model_value="MISSING",
                exact=False,
                match_type="missing",
                similarity=0.0,
                accepted=False,
            )
        )

    for key in extra_keys:
        details.append(
            KeyComparison(
                key=key,
                gold_value="NOT_IN_GOLD",
                model_value=model_flat[key],
                exact=False,
                match_type="extra",
                similarity=0.0,
                accepted=False,
            )
        )

    total_gold_keys = len(gold_flat)
    total_model_keys = len(model_flat)

    precision = exact_matches / total_model_keys if total_model_keys else 0.0
    recall = exact_matches / total_gold_keys if total_gold_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    partial_match = accepted_matches / total_gold_keys if total_gold_keys else 0.0
    partial_exact = exact_matches / total_gold_keys if total_gold_keys else 0.0
    partial_similarity = total_similarity / total_gold_keys if total_gold_keys else 0.0
    partial_numeric = (
        numeric_matches / numeric_gold_keys if numeric_gold_keys else 0.0
    )
    range_accuracy = (
        range_matches / range_eligible_keys if range_eligible_keys else 0.0
    )
    delta_1_accuracy = (
        delta_1_matches / delta_1_eligible_keys if delta_1_eligible_keys else 0.0
    )
    strict_exact = 1.0 if gold_flat == model_flat else 0.0

    return StaticJsonScore(
        partial_match_accuracy=partial_match,
        partial_exact_match_accuracy=partial_exact,
        strict_exact_match_accuracy=strict_exact,
        partial_similarity_score=partial_similarity,
        partial_numeric_match_accuracy=partial_numeric,
        range_match_accuracy=range_accuracy,
        delta_1_match_accuracy=delta_1_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        total_gold_keys=total_gold_keys,
        total_model_keys=total_model_keys,
        matched_keys=len(common_keys),
        accepted_value_matches=accepted_matches,
        exact_value_matches=exact_matches,
        numeric_gold_keys=numeric_gold_keys,
        numeric_value_matches=numeric_matches,
        range_eligible_keys=range_eligible_keys,
        range_value_matches=range_matches,
        delta_1_eligible_keys=delta_1_eligible_keys,
        delta_1_value_matches=delta_1_matches,
        missing_keys=missing_keys,
        extra_keys=extra_keys,
        details=details,
    )

def evaluate_static_json_batch(
    pairs: list[tuple[Any, Any]],
    *,
    similarity_threshold: float = 0.0,
) -> dict[str, Any]:
    """Evaluate multiple gold/model answer pairs and aggregate metrics."""
    scores = [
        evaluate_static_json(
            gold,
            model,
            similarity_threshold=similarity_threshold,
        )
        for gold, model in pairs
    ]

    if not scores:
        return {
            "num_examples": 0,
            "partial_exact_match_accuracy": 0.0,
            "strict_exact_match_accuracy": 0.0,
            "partial_similarity_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "examples": [],
        }

    return {
        "num_examples": len(scores),
        "partial_exact_match_accuracy": sum(
            score.partial_exact_match_accuracy for score in scores
        )
        / len(scores),
        "strict_exact_match_accuracy": sum(
            score.strict_exact_match_accuracy for score in scores
        )
        / len(scores),
        "partial_similarity_score": sum(
            score.partial_similarity_score for score in scores
        )
        / len(scores),
        "precision": sum(score.precision for score in scores) / len(scores),
        "recall": sum(score.recall for score in scores) / len(scores),
        "f1": sum(score.f1 for score in scores) / len(scores),
        "examples": [score.to_dict() for score in scores],
    }

class StaticJsonScorer:
    """Evaluation scorer wrapper for the trajectory-based pipeline."""

    def __init__(self, name: str = "static_json") -> None:
        self.name = name

    def __call__(
        self,
        scenario: Scenario,
        answer: str,
        trajectory_text: str,
    ) -> ScorerResult:
        gold_answer = scenario.expected_answer or scenario.characteristic_form

        if gold_answer is None or str(gold_answer).strip() == "":
            return ScorerResult(
                scorer=self.name,
                passed=False,
                score=0.0,
                rationale=(
                    "scenario has neither expected_answer nor characteristic_form "
                    "for static_json scoring"
                ),
            )

        static_score = evaluate_static_json(gold_answer, answer)
        passed = static_score.strict_exact_match_accuracy == 1.0

        return ScorerResult(
            scorer=self.name,
            passed=passed,
            score=round(static_score.f1, 3),
            rationale=(
                "strict structured match"
                if passed
                else "structured answer differs from ground truth"
            ),
            details=static_score.to_dict(),
        )


def install(name: str = "static_json") -> None:
    """Register the static JSON scorer."""
    register(name, StaticJsonScorer(name=name))
