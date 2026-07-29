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
    mode_key_match: float | None = None
    mode_exactly_one_key: float | None = None
    mode_required_terms: list[str] = field(default_factory=list)
    mode_matched_terms: list[str] = field(default_factory=list)
    mode_term_coverage: float | None = None

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
        r"(?:^|\n)\s*<Answer>\s*:?\s*(.*)$",
        r"(?:^|\n)\s*Final Answer\s*:?\s*(.*)$",
        r"(?:^|\n)\s*Answer\s*:?\s*(.*)$",
        r"(?:^|\n)\s*Output\s*:?\s*(.*)$",
        r"(?:^|\n)\s*Result\s*:?\s*(.*)$",
        r"(?:^|\n)\s*Response\s*:?\s*(.*)$",
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


def _balanced_from_index(
    content: str, start: int, open_ch: str, close_ch: str
) -> str | None:
    """Return the balanced structure starting at ``start``, if present."""
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

    return None


def _extract_balanced_structures(content: str) -> list[str]:
    """Extract parseable-looking {...}, [...], and (...) candidates from noisy text."""
    content = content.strip()
    candidates: list[tuple[int, int, str]] = []

    for priority, (open_ch, close_ch) in enumerate(
        [("{", "}"), ("[", "]"), ("(", ")")]
    ):
        start = content.find(open_ch)
        while start != -1:
            candidate = _balanced_from_index(content, start, open_ch, close_ch)
            if candidate is not None:
                candidates.append((priority, start, candidate))
                break
            start = content.find(open_ch, start + 1)

    return [candidate for _, _, candidate in sorted(candidates)]


def _extract_balanced_structure(content: str) -> str:
    """Extract the first balanced {...}, [...], or (...) candidate from noisy text."""
    candidates = _extract_balanced_structures(content)
    return candidates[0] if candidates else content.strip()


_PARSE_MISSING = object()


def _parse_json_or_python(content: str) -> Any:
    """Parse JSON/Python literal text, returning a sentinel on failure."""
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

    return _PARSE_MISSING


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


def _extract_final_count_from_text(content: str) -> int | float | None:
    """Extract a final standalone count from a noisy scalar answer."""
    stripped = content.strip()
    count = _extract_count_from_text(stripped)
    if count is not None:
        return count

    final_number = re.compile(
        r"^\s*(?:"
        r"(?:final\s+answer|answer|count|result)\s*(?:is|:)?\s*"
        r")?(-?\d+(?:\.\d+)?)\s*\.?\s*$",
        flags=re.IGNORECASE,
    )
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        match = final_number.fullmatch(line)
        if match:
            number = match.group(1)
            return float(number) if "." in number else int(number)
        break

    return None


_CHOICE_LETTER_RE = re.compile(r"^[A-Za-z]$")
_CHOICE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*)?"
    r"[\(\[]?([A-Za-z])[\)\].:]?"
    r"\s*$",
    flags=re.IGNORECASE,
)
_CHOICE_PHRASE_RE = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*"
    r"[\(\[]?([A-Za-z])[\)\].:]?\s*$",
    flags=re.IGNORECASE,
)
_TRAILING_CHOICE_RE = re.compile(
    r"(?:^|[\s,;:])[\(\[]?([A-Za-z])[\)\].:]?\s*$"
)


def _is_choice_scalar(value: Any) -> bool:
    """Return true for ground-truth answers like ``C``."""
    parsed = parse_structured_answer(value)
    return isinstance(parsed, str) and bool(
        _CHOICE_LETTER_RE.fullmatch(parsed.strip())
    )


def _extract_final_choice_from_text(value: Any) -> str | None:
    """Extract a final multiple-choice letter from noisy scalar answers."""
    if not isinstance(value, str):
        return None

    content = _strip_markdown_fence(extract_answer_text(value)).strip()
    if not content:
        return None

    direct = _CHOICE_LINE_RE.fullmatch(content)
    if direct:
        return direct.group(1).upper()

    for line in reversed([line.strip() for line in content.splitlines()]):
        if not line:
            continue
        line_match = _CHOICE_LINE_RE.fullmatch(line)
        if line_match:
            return line_match.group(1).upper()
        phrase_match = _CHOICE_PHRASE_RE.search(line)
        if phrase_match:
            return phrase_match.group(1).upper()
        trailing_match = _TRAILING_CHOICE_RE.search(line)
        if trailing_match:
            return trailing_match.group(1).upper()
        break

    return None


def _normalize_choice_answer(value: Any) -> Any:
    """Return a final choice letter when one can be safely extracted."""
    choice = _extract_final_choice_from_text(value)
    return choice if choice is not None else value


def parse_structured_answer(value: Any) -> Any:
    """Parse JSON/Python-like structured output into a Python object."""
    if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
        return value

    content = extract_answer_text(value)
    content = _strip_markdown_fence(content)

    parsed = _parse_json_or_python(content)
    if parsed is not _PARSE_MISSING:
        return parsed

    for candidate in _extract_balanced_structures(content):
        parsed = _parse_json_or_python(candidate)
        if parsed is not _PARSE_MISSING:
            return parsed

    count = _extract_final_count_from_text(content)
    if count is not None:
        return count

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


_MODE_KEYS = frozenset({"response", "clarification", "abstain"})
_IMPORTANT_MODE_TERMS = (
    "air conditioner",
    "cannot determine",
    "date",
    "dataset",
    "exhaust leak",
    "handrail",
    "lately",
    "main unit",
    "pressure vessel",
    "pump",
    "steering",
    "tag",
    "time",
    "unreliable",
    "usual suspect",
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "because",
        "by",
        "do",
        "does",
        "for",
        "has",
        "have",
        "is",
        "it",
        "of",
        "or",
        "the",
        "this",
        "to",
        "what",
        "which",
        "why",
        "with",
        "you",
    }
)


def _normalize_text_for_terms(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_article(term: str) -> str:
    parts = term.split()
    if parts and parts[0] in {"a", "an", "the"}:
        return " ".join(parts[1:])
    return term


def _dedupe_terms(terms: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _normalize_text_for_terms(term)
        normalized = _strip_leading_article(normalized)
        if not normalized or normalized in _STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _extract_required_mode_terms(value: Any) -> list[str]:
    """Extract lightweight required terms for mode-selection scenarios.

    The mode scenarios are not exact-string tasks: answers can be phrased
    differently as long as they choose the correct mode and mention the
    important ambiguity/evidence. We therefore extract only high-signal terms:
    quoted phrases, asset/fault identifiers, yes/no stance, and a small
    domain-term allowlist.
    """
    text = str(value)
    terms: list[str] = []

    quoted = re.findall(r"""["'“”‘’]([^"'“”‘’]{2,80})["'“”‘’]""", text)
    quoted = [
        phrase
        for phrase in quoted
        if not re.fullmatch(r"[A-Z]{2,4}", phrase.strip())
    ]
    terms.extend(quoted)

    identifiers = re.findall(r"\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b", text)
    terms.extend(identifiers)

    normalized = _normalize_text_for_terms(text)
    for term in _IMPORTANT_MODE_TERMS:
        if f" {term} " in f" {normalized} ":
            terms.append(term)

    if normalized.startswith("no "):
        terms.append("no")
    elif normalized.startswith("yes "):
        terms.append("yes")

    return _dedupe_terms(terms)


def _is_mode_gold_answer(value: Any) -> bool:
    parsed = parse_structured_answer(value)
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return False
    key = str(next(iter(parsed))).strip().lower()
    return key in _MODE_KEYS


def _evaluate_mode_json(gold_answer: Any, model_answer: Any) -> StaticJsonScore:
    gold = parse_structured_answer(gold_answer)
    model = parse_structured_answer(model_answer)

    gold_key = str(next(iter(gold))).strip().lower()
    gold_value = next(iter(gold.values()))

    model_is_dict = isinstance(model, dict)
    model_keys = [str(key).strip().lower() for key in model.keys()] if model_is_dict else []
    model_exactly_one_key = len(model_keys) == 1
    model_key = model_keys[0] if model_exactly_one_key else "INVALID"
    model_value = next(iter(model.values())) if model_is_dict and model_exactly_one_key else ""

    key_match = model_exactly_one_key and model_key == gold_key
    required_terms = _extract_required_mode_terms(gold_value)
    model_text = _normalize_text_for_terms(model_value)
    matched_terms = [
        term for term in required_terms if f" {term} " in f" {model_text} "
    ]
    term_coverage = (
        len(matched_terms) / len(required_terms) if required_terms else 1.0
    )

    details = [
        KeyComparison(
            key="answer.mode",
            gold_value=gold_key,
            model_value=model_key,
            exact=key_match,
            match_type="exact" if key_match else "mode_mismatch",
            similarity=1.0 if key_match else similarity_score(gold_key, model_key),
            accepted=key_match,
        )
    ]

    for term in required_terms:
        matched = term in matched_terms
        details.append(
            KeyComparison(
                key=f"answer.required_term.{term}",
                gold_value=term,
                model_value=term if matched else "MISSING",
                exact=matched,
                match_type="term_present" if matched else "term_missing",
                similarity=1.0 if matched else 0.0,
                accepted=matched,
            )
        )

    missing_keys = [] if key_match else [f"answer.{gold_key}"]
    extra_keys = []
    if model_is_dict:
        extra_keys = [
            f"answer.{key}"
            for key in model_keys
            if key not in {gold_key} or not model_exactly_one_key
        ]
    else:
        extra_keys = ["answer"] if model is not None else []

    total_gold_keys = 1 + len(required_terms)
    total_model_keys = 1 + len(required_terms) + len(extra_keys)
    exact_matches = (1 if key_match else 0) + len(matched_terms)

    precision = exact_matches / total_model_keys if total_model_keys else 0.0
    recall = exact_matches / total_gold_keys if total_gold_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    strict_exact = 1.0 if key_match and term_coverage == 1.0 and not extra_keys else 0.0

    return StaticJsonScore(
        partial_match_accuracy=recall,
        partial_exact_match_accuracy=recall,
        strict_exact_match_accuracy=strict_exact,
        partial_similarity_score=sum(item.similarity for item in details)
        / total_gold_keys,
        partial_numeric_match_accuracy=0.0,
        range_match_accuracy=0.0,
        delta_1_match_accuracy=0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        total_gold_keys=total_gold_keys,
        total_model_keys=total_model_keys,
        matched_keys=1 if key_match else 0,
        accepted_value_matches=exact_matches,
        exact_value_matches=exact_matches,
        numeric_gold_keys=0,
        numeric_value_matches=0,
        range_eligible_keys=0,
        range_value_matches=0,
        delta_1_eligible_keys=0,
        delta_1_value_matches=0,
        missing_keys=missing_keys,
        extra_keys=extra_keys,
        details=details,
        mode_key_match=1.0 if key_match else 0.0,
        mode_exactly_one_key=1.0 if model_exactly_one_key else 0.0,
        mode_required_terms=required_terms,
        mode_matched_terms=matched_terms,
        mode_term_coverage=term_coverage,
    )


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
    if _is_mode_gold_answer(gold_answer):
        return _evaluate_mode_json(gold_answer, model_answer)

    if _is_choice_scalar(gold_answer):
        model_answer = _normalize_choice_answer(model_answer)

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
