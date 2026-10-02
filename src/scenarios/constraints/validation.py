"""Deterministic scenario validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
import json

from ..models import AssetProfile
from ..text import normalize_for_fuzzy_dedup
from .tool_references import find_tool_references

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


def _validate_text_excludes_tool_names(
    text: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    mentioned = [ref.spelling for ref in find_tool_references(text, tool_names_by_focus)]
    if not mentioned:
        return []
    return [
        "scenario text must not contain explicit MCP tool or API function references "
        f"(move tool names to characteristic_form only); found in text: {', '.join(mentioned)}"
    ]


def _validate_reference_names(
    characteristic_form: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    return [
        ref.error
        for ref in find_tool_references(characteristic_form, tool_names_by_focus)
        if ref.error
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
    reasons.extend(_validate_reference_names(characteristic_form, tool_names_by_focus))

    reasons.extend(_validate_unique_text(text, accepted_texts or ()))

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
    """Check negative scenario structure; answerability is reviewed by the LLM."""
    reasons = _validate_required_fields(scenario)
    if reasons:
        return reasons

    text = str(scenario["text"]).strip()
    characteristic_form = str(scenario["characteristic_form"]).strip()

    reasons.extend(_validate_text_excludes_tool_names(text, tool_names_by_focus))
    reasons.extend(_validate_reference_names(characteristic_form, tool_names_by_focus))
    reasons.extend(_validate_unique_text(text, accepted_texts or ()))
    reasons.extend(_validate_negative_expectation(characteristic_form))
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
    references = find_tool_references(characteristic_form, tool_names_by_focus)
    if any(ref.canonical and ref.canonical.startswith(f"{focus}.") for ref in references):
        return []
    return [
        f"{focus} scenarios must explicitly mention at least one concrete {focus} tool in characteristic_form: "
        + ", ".join(f"{focus}.{tool}" for tool in focus_tools)
    ]


def _validate_multiagent_constraints(
    characteristic_form: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[str]:
    if not tool_names_by_focus:
        return []

    mentioned_focuses = {
        ref.canonical.split(".", 1)[0]
        for ref in find_tool_references(characteristic_form, tool_names_by_focus)
        if ref.canonical and not ref.canonical.startswith("multiagent.")
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


def _validate_unique_text(text: str, accepted_texts: Iterable[str]) -> list[str]:
    if _is_duplicate_text(text, accepted_texts):
        return [
            f"text is a duplicate or near-duplicate of an already accepted scenario (threshold {_DUPLICATE_THRESHOLD:.2f})"
        ]
    return []


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
