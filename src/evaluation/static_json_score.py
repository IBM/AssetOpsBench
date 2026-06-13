"""Static structured-answer evaluator for AssetOpsBench.

This module provides a deterministic evaluator for structured answers such as:

- JSON objects: {"energy": 14, "material": 48}
- JSON arrays: [{"equipment_group": "pumps", "count": 3}]
- Python literals: {'energy': 14, 'material': 48}
- Python-style tuple lists: [("Engines & motors", 5), ("Lines & drives", 2)]
- Count-only answers: 34
- Noisy model outputs with prefixes, markdown fences, or extra text.

The implementation is inspired by DeepSynth-style static scoring, but is
AssetOpsBench-specific and does not depend on DeepSynth reference files.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KeyComparison:
    """Per-key comparison between gold and model answer."""

    key: str
    gold_value: str
    model_value: str
    exact: bool
    match_type: str
    similarity: float


@dataclass
class StaticJsonScore:
    """Aggregate static score for one structured answer pair."""

    partial_exact_match_accuracy: float
    strict_exact_match_accuracy: float
    partial_similarity_score: float
    precision: float
    recall: float
    f1: float
    total_gold_keys: int
    total_model_keys: int
    matched_keys: int
    exact_value_matches: int
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    details: list[KeyComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary."""
        data = asdict(self)
        data["details"] = [asdict(item) for item in self.details]
        return data


def extract_answer_text(text: Any) -> str:
    """Extract likely final answer text from raw model output.

    Examples:
        "Answer: {\"a\": 1}" -> "{\"a\": 1}"
        "<Answer>: 34" -> "34"
        "Final Answer: [('x', 2)]" -> "[('x', 2)]"
    """
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
    """Extract first balanced {...}, [...], or (...) structure from noisy text.

    This helps with outputs like:
        "Here is the answer: {\"a\": 1}. Thanks."
    """
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

    # Handle simple noisy count answers such as "The answer is 34."
    numbers = re.findall(r"-?\d+(?:\.\d+)?", stripped)
    if len(numbers) == 1:
        number = numbers[0]
        if "." in number:
            return float(number)
        return int(number)

    return None


def parse_structured_answer(value: Any) -> Any:
    """Parse a structured gold/model answer into a Python object.

    Supports JSON, Python literals, markdown-fenced JSON/Python,
    tuple lists, count-only strings, and noisy answer-prefixed text.
    """
    if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
        return value

    content = extract_answer_text(value)
    content = _strip_markdown_fence(content)

    # First handle count-only or simple noisy count answer.
    count = _extract_count_from_text(content)
    if count is not None:
        return count

    content = _extract_balanced_structure(content)

    # Strict JSON.
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Python literals: dicts, lists, tuples, strings, numbers.
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        pass

    # Common repair: single quotes to double quotes.
    try:
        return json.loads(content.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    # Try count extraction again after balanced extraction.
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
    """Flatten nested structures into comparable key-value pairs.

    Examples:
        {"a": {"b": 2}} -> {"answer.a.b": "2"}

        [("x", 1), ("y", 2)] ->
        {
            "answer[0][0]": "x",
            "answer[0][1]": "1",
            "answer[1][0]": "y",
            "answer[1][1]": "2",
        }
    """
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

    # Numeric similarity.
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

    # Simple character-set similarity for non-numeric values.
    gold_chars = set(gold_value)
    model_chars = set(model_value)
    union = gold_chars | model_chars

    if not union:
        return 1.0

    score = len(gold_chars & model_chars) / len(union)

    if gold_value in model_value or model_value in gold_value:
        score = max(score, 0.6)

    return score


def evaluate_static_json(
    gold_answer: Any,
    model_answer: Any,
    *,
    similarity_threshold: float = 0.0,
) -> StaticJsonScore:
    """Evaluate one structured gold answer against one model answer."""
    gold_flat = flatten_answer(gold_answer)
    model_flat = flatten_answer(model_answer)

    gold_keys = set(gold_flat)
    model_keys = set(model_flat)
    common_keys = gold_keys & model_keys

    details: list[KeyComparison] = []
    exact_matches = 0
    total_similarity = 0.0

    for key in sorted(common_keys):
        gold_value = gold_flat[key]
        model_value = model_flat[key]

        score = similarity_score(gold_value, model_value)
        total_similarity += score

        exact = score == 1.0
        if exact:
            exact_matches += 1
            match_type = "exact"
        elif score > similarity_threshold:
            match_type = f"partial ({score:.2f})"
        else:
            match_type = "mismatch"

        details.append(
            KeyComparison(
                key=key,
                gold_value=gold_value,
                model_value=model_value,
                exact=exact,
                match_type=match_type,
                similarity=score,
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

    partial_exact = exact_matches / total_gold_keys if total_gold_keys else 0.0
    partial_similarity = total_similarity / total_gold_keys if total_gold_keys else 0.0
    strict_exact = 1.0 if gold_flat == model_flat else 0.0

    return StaticJsonScore(
        partial_exact_match_accuracy=partial_exact,
        strict_exact_match_accuracy=strict_exact,
        partial_similarity_score=partial_similarity,
        precision=precision,
        recall=recall,
        f1=f1,
        total_gold_keys=total_gold_keys,
        total_model_keys=total_model_keys,
        matched_keys=len(common_keys),
        exact_value_matches=exact_matches,
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


def evaluate_static_json_files(
    gold_path: str | Path,
    model_path: str | Path,
    *,
    gold_key: str = "answer",
    model_key: str = "answer",
    id_key: str = "scenario_id",
    similarity_threshold: float = 0.0,
) -> dict[str, Any]:
    """Evaluate two JSON files containing answer records.

    Example record format:
        {"scenario_id": "11", "answer": {"energy": 14, "material": 48}}

    The two files are aligned by ``id_key``.
    """
    gold_data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    model_data = json.loads(Path(model_path).read_text(encoding="utf-8"))

    if not isinstance(gold_data, list) or not isinstance(model_data, list):
        return evaluate_static_json_batch(
            [(gold_data, model_data)],
            similarity_threshold=similarity_threshold,
        )

    gold_by_id = {
        str(item[id_key]): item
        for item in gold_data
        if isinstance(item, dict) and id_key in item
    }
    model_by_id = {
        str(item[id_key]): item
        for item in model_data
        if isinstance(item, dict) and id_key in item
    }

    common_ids = sorted(set(gold_by_id) & set(model_by_id))

    pairs = [
        (gold_by_id[item_id][gold_key], model_by_id[item_id][model_key])
        for item_id in common_ids
    ]

    result = evaluate_static_json_batch(
        pairs,
        similarity_threshold=similarity_threshold,
    )
    result["matched_ids"] = common_ids
    result["missing_ids"] = sorted(set(gold_by_id) - set(model_by_id))
    result["extra_ids"] = sorted(set(model_by_id) - set(gold_by_id))

    return result