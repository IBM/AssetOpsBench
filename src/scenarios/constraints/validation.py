"""Deterministic scenario validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
import re

from ..models import AssetProfile
from ..text import normalize_for_fuzzy_dedup

_TOOL_PATTERN_TEMPLATE = r"(?<![a-z0-9_]){tool}(?![a-z0-9_])"
_DUPLICATE_THRESHOLD = 0.90
_REQUIRED_TEXT_FIELDS = ("text", "category", "characteristic_form")
_NEGATIVE_EXPECTATION_HINTS = (
    "cannot be answered",
    "cannot answer",
    "insufficient data",
    "insufficiency",
    "refusal",
    "refuse",
    "missing data",
    "not available",
    "unavailable",
    "not present",
    "do not hallucinate",
)
_UNSUPPORTED_DEPENDENCY_PATTERNS = (
    r"\bweather\b",
    r"\bweather api\b",
    r"\bsap\b",
    r"\berp\b",
    r"\bsatellite\b",
    r"\bmarket price\b",
    r"\belectricity price\b",
    r"\bcontract data\b",
    r"\binvoice\b",
    r"\bprocurement\b",
    r"\bgis\b",
)
_LONG_HORIZON_PATTERN = re.compile(
    r"\b(?:next|coming|over the next)\s+(?:\d+\s+)?(?:year|years|month|months)\b|\b\d{1,2}-year\b|\bdecade\b",
    re.IGNORECASE,
)
_ISO_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)?\b"
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_EXPLICIT_IDENTIFIER_PATTERNS = (
    re.compile(r"\basset(?:\s+id)?\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})", re.IGNORECASE),
    re.compile(r"\bequipment\s+id\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})", re.IGNORECASE),
    re.compile(r"\bsite\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})", re.IGNORECASE),
)
_ASSET_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b")


@dataclass(frozen=True)
class ScenarioValidationFailure:
    scenario: dict
    reasons: tuple[str, ...]

    def to_prompt_dict(self, index: int) -> dict:
        return {
            "index": index,
            "text": str(self.scenario.get("text", "")),
            "category": str(self.scenario.get("category", "")),
            "reasons": list(self.reasons),
        }


def failure_payload(failures: list[ScenarioValidationFailure]) -> str:
    data = [
        failure.to_prompt_dict(index=index) for index, failure in enumerate(failures)
    ]
    return json.dumps(data, indent=2)


def validate_scenario_batch(
    focus: str,
    scenarios: list[dict],
    accepted_scenarios: Iterable[dict] | None = None,
    profile: AssetProfile | None = None,
    generation_mode: str = "closed_form",
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[list[dict], list[ScenarioValidationFailure]]:
    prior_texts = [
        str(scenario.get("text", ""))
        for scenario in (accepted_scenarios or [])
        if str(scenario.get("text", "")).strip()
    ]
    valid: list[dict] = []
    failures: list[ScenarioValidationFailure] = []

    for scenario in scenarios:
        reasons = validate_scenario(
            focus,
            scenario,
            accepted_texts=prior_texts + [str(item.get("text", "")) for item in valid],
            profile=profile,
            generation_mode=generation_mode,
            tool_names_by_focus=tool_names_by_focus,
        )
        if reasons:
            failures.append(
                ScenarioValidationFailure(scenario=scenario, reasons=tuple(reasons))
            )
            continue
        valid.append(scenario)

    return valid, failures


def validate_negative_scenario_batch(
    focus: str,
    scenarios: list[dict],
    accepted_scenarios: Iterable[dict] | None = None,
    profile: AssetProfile | None = None,
    generation_mode: str = "closed_form",
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[list[dict], list[ScenarioValidationFailure]]:
    prior_texts = [
        str(scenario.get("text", ""))
        for scenario in (accepted_scenarios or [])
        if str(scenario.get("text", "")).strip()
    ]
    valid: list[dict] = []
    failures: list[ScenarioValidationFailure] = []

    for scenario in scenarios:
        reasons = validate_negative_scenario(
            focus,
            scenario,
            accepted_texts=prior_texts + [str(item.get("text", "")) for item in valid],
            profile=profile,
            generation_mode=generation_mode,
            tool_names_by_focus=tool_names_by_focus,
        )
        if reasons:
            failures.append(
                ScenarioValidationFailure(scenario=scenario, reasons=tuple(reasons))
            )
            continue
        valid.append(scenario)

    return valid, failures


def _flatten_tool_names(
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> tuple[str, ...]:
    if not tool_names_by_focus:
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for tools in tool_names_by_focus.values():
        for t in tools:
            if t and t not in seen:
                seen.add(t)
                ordered.append(t)
    return tuple(ordered)


def _validate_text_excludes_tool_names(
    text: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    if not tool_names_by_focus:
        return []
    all_tools = _flatten_tool_names(tool_names_by_focus)
    if not all_tools:
        return []
    mentioned = _mentioned_tools(text, all_tools)
    if not mentioned:
        return []
    return [
        "scenario text must not name MCP tools or API functions "
        f"(move tool names to characteristic_form only); found in text: {', '.join(mentioned)}"
    ]


def validate_scenario(
    focus: str,
    scenario: dict,
    accepted_texts: Iterable[str] | None = None,
    profile: AssetProfile | None = None,
    generation_mode: str = "closed_form",
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None = None,
) -> list[str]:
    reasons = _validate_required_fields(scenario)
    if reasons:
        return reasons

    text = str(scenario["text"]).strip()
    characteristic_form = str(scenario["characteristic_form"]).strip()
    combined = f"{text}\n{characteristic_form}"

    reasons.extend(_validate_text_excludes_tool_names(text, tool_names_by_focus))

    if _is_duplicate_text(text, accepted_texts or ()):
        reasons.append(
            f"text is a duplicate or near-duplicate of an already accepted scenario (threshold {_DUPLICATE_THRESHOLD:.2f})"
        )

    if focus == "multiagent":
        reasons.extend(
            _validate_multiagent_constraints(characteristic_form, tool_names_by_focus)
        )
    else:
        reasons.extend(
            _validate_primary_focus(focus, characteristic_form, tool_names_by_focus)
        )

    if generation_mode == "open_form":
        reasons.extend(_validate_open_form_grounding(focus, combined, profile))

    return reasons


def validate_negative_scenario(
    focus: str,
    scenario: dict,
    accepted_texts: Iterable[str] | None = None,
    profile: AssetProfile | None = None,
    generation_mode: str = "closed_form",
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None = None,
) -> list[str]:
    reasons = _validate_required_fields(scenario)
    if reasons:
        return reasons

    text = str(scenario["text"]).strip()
    characteristic_form = str(scenario["characteristic_form"]).strip()
    combined = f"{text}\n{characteristic_form}"

    reasons.extend(_validate_text_excludes_tool_names(text, tool_names_by_focus))

    reasons.extend(_validate_negative_expectation(characteristic_form))
    reasons.extend(
        _validate_negative_answerability(
            focus,
            combined,
            profile=profile,
            generation_mode=generation_mode,
        )
    )
    return reasons


def _validate_required_fields(scenario: dict) -> list[str]:
    reasons: list[str] = []
    for field in _REQUIRED_TEXT_FIELDS:
        value = scenario.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"field '{field}' must be a non-empty string")
    return reasons


def _validate_negative_expectation(characteristic_form: str) -> list[str]:
    lowered = characteristic_form.lower()
    if any(hint in lowered for hint in _NEGATIVE_EXPECTATION_HINTS):
        return []
    return [
        "negative scenarios must require an explicit insufficiency/refusal response in characteristic_form"
    ]


def _validate_primary_focus(
    focus: str,
    characteristic_form: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    focus_tools = (
        tuple(tool_names_by_focus.get(focus, ())) if tool_names_by_focus else ()
    )
    if not focus_tools:
        return []
    if _mentioned_tools(characteristic_form, focus_tools):
        return []
    return [
        f"{focus} scenarios must explicitly mention at least one concrete {focus} tool in characteristic_form: "
        + ", ".join(focus_tools)
    ]


def _validate_multiagent_constraints(
    characteristic_form: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    if not tool_names_by_focus:
        return []

    mentioned_focuses = {
        focus
        for focus, tool_names in tool_names_by_focus.items()
        if focus != "multiagent" and _mentioned_tools(characteristic_form, tool_names)
    }
    if len(mentioned_focuses) >= 2:
        return []
    return [
        "multiagent scenarios must reference concrete tools from at least two distinct focuses in characteristic_form"
    ]


def _validate_open_form_grounding(
    focus: str,
    combined: str,
    profile: AssetProfile | None,
) -> list[str]:
    if profile is None:
        return ["open-form validation requires an Asset Profile"]

    allowed_identifiers = (
        profile.grounded_sites()
        + profile.grounded_asset_ids(focus)
        + profile.grounded_sensor_names(focus)
        + profile.grounded_timestamps(focus)
    )
    if not allowed_identifiers:
        return [
            "open-form scenarios require grounded identifiers, but none were available in the profile"
        ]

    lowered = combined.lower()
    if any(
        identifier.lower() in lowered
        for identifier in allowed_identifiers
        if identifier
    ):
        return []
    return [
        "open-form scenarios must use grounded site names, asset ids, sensors, or timestamps from the Asset Profile"
    ]


def _validate_negative_answerability(
    focus: str,
    combined: str,
    *,
    profile: AssetProfile | None,
    generation_mode: str,
) -> list[str]:
    if _has_unsupported_external_dependency(combined):
        return []
    if _has_out_of_range_time_reference(combined, profile, focus):
        return []
    if _has_unknown_identifier_reference(combined, profile, focus):
        return []
    if generation_mode == "closed_form" and _LONG_HORIZON_PATTERN.search(combined):
        return []
    return [
        "negative scenario appears answerable; include a concrete unsupported condition such as an unknown identifier, out-of-range time reference, or unavailable external dependency"
    ]


def _has_unsupported_external_dependency(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _UNSUPPORTED_DEPENDENCY_PATTERNS)


def _parse_timestamp(value: str) -> datetime | None:
    candidate = value.strip()
    if not candidate:
        return None
    try:
        if "T" in candidate:
            normalized = candidate.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _grounded_time_bounds(profile: AssetProfile | None, focus: str) -> tuple[datetime, datetime] | None:
    if profile is None:
        return None
    parsed = [
        timestamp
        for timestamp in (
            _parse_timestamp(value) for value in profile.grounded_timestamps(focus)
        )
        if timestamp is not None
    ]
    if not parsed:
        return None
    return min(parsed), max(parsed)


def _has_out_of_range_time_reference(
    text: str,
    profile: AssetProfile | None,
    focus: str,
) -> bool:
    bounds = _grounded_time_bounds(profile, focus)
    if bounds is None:
        return False
    start, end = bounds
    for raw in _ISO_TIMESTAMP_PATTERN.findall(text):
        parsed = _parse_timestamp(raw)
        if parsed is None:
            continue
        if parsed < start or parsed > end:
            return True
    for raw_year in _YEAR_PATTERN.findall(text):
        year = int(raw_year)
        if year < start.year or year > end.year:
            return True
    return False


def _extract_identifier_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for pattern in _EXPLICIT_IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append(match.group(1).strip(" ,.;:!?"))
    mentions.extend(_ASSET_TOKEN_PATTERN.findall(text))
    deduped: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        lowered = mention.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(mention)
    return deduped


def _has_unknown_identifier_reference(
    text: str,
    profile: AssetProfile | None,
    focus: str,
) -> bool:
    if profile is None:
        return False
    allowed = {
        identifier.lower()
        for identifier in (
            profile.grounded_sites()
            + profile.grounded_asset_ids(focus)
            + profile.grounded_sensor_names(focus)
        )
        if identifier
    }
    if not allowed:
        return False
    for mention in _extract_identifier_mentions(text):
        lowered = mention.lower()
        if lowered in allowed:
            continue
        if any(char.isdigit() for char in mention) or "site " in text.lower():
            return True
    return False


def _is_duplicate_text(text: str, accepted_texts: Iterable[str]) -> bool:
    normalized = normalize_for_fuzzy_dedup(text)
    if not normalized:
        return False
    for accepted in accepted_texts:
        accepted_normalized = normalize_for_fuzzy_dedup(accepted)
        if not accepted_normalized:
            continue
        if (
            SequenceMatcher(None, normalized, accepted_normalized).ratio()
            >= _DUPLICATE_THRESHOLD
        ):
            return True
    return False


def _mentioned_tools(text: str, tools: Iterable[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for tool in tools:
        pattern = _TOOL_PATTERN_TEMPLATE.format(tool=re.escape(tool.lower()))
        if re.search(pattern, lowered):
            hits.append(tool)
    return hits
