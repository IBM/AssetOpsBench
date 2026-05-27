"""Prompt construction and normalization helpers for scenario generation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import math
import json
import logging
from pathlib import Path
import re
from typing import TypeVar

from ..constraints import FOCUS_ORDER
from ..models import AssetProfile, GroundingBundle
from ..text import slugify_asset_name


def _quiet_litellm_logging() -> None:
    for name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
        logging.getLogger(name).setLevel(logging.WARNING)


_MAX_SCENARIO_ATTEMPTS = 4
_MAX_FEWSHOT_EXAMPLES = 6


def _multiagent_budget_cap(total: int) -> int:
    if total <= 1:
        return total
    return (total * 3) // 4


_PROFILE_MAX_TOKENS = 4096
_MULTIAGENT_MAX_TOKENS = 4096
_BUDGET_MAX_TOKENS = 4096
_SCENARIO_LLM_MAX_TOKENS = 8192
DEFAULT_GENERATED_SCENARIOS_DIR = Path("generated/scenarios")


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_step(
    phase: str,
    info: str,
    details: str | None = None,
    tool_info: str | None = None,
) -> None:
    print(f"  [OK ] Step ({phase}): {info}")
    if tool_info:
        print(f"       {tool_info}")
    if details:
        indented = "\n".join("        " + line for line in details.splitlines())
        print(indented)


def _print_live_step(phase: str, info: str, details: str | None = None) -> None:
    print(f"  [....] Step ({phase}): {info}")
    if details:
        indented = "\n".join("        " + line for line in details.splitlines())
        print(indented)


def _log_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_scenario_output_path(asset_name: str) -> Path:
    timestamp = _log_timestamp()
    run_dir = DEFAULT_GENERATED_SCENARIOS_DIR / f"{slugify_asset_name(asset_name)}_scenarios_{timestamp}"
    return run_dir / "scenarios.json"


def negative_scenario_output_path(scenarios_path: Path) -> Path:
    return scenarios_path.with_name("negative_scenarios.json")


def hard_scenario_target(count: int, ratio: float = 0.7) -> int:
    if count <= 0:
        return 0
    return max(1, math.ceil(count * ratio))


def _asset_profile_json(profile: AssetProfile) -> str:
    return profile.model_dump_json(indent=2)


def _few_shot_examples_section(few_shots: list[dict]) -> str:
    if not few_shots:
        return (
            "No benchmark few-shot examples were available for this focus. "
            "Do not assume any hidden fallback examples."
        )

    return (
        "Benchmark few-shot examples (style, complexity, and specificity references only; "
        "follow Generation Mode for closed-form vs open-form grounding, and always prefer "
        "direct operator/manager wording over any legacy benchmark phrasing):\n"
        f"{json.dumps(few_shots, indent=2)}"
    )


def _tool_names_from_description(description: object) -> list[str]:
    if not isinstance(description, str):
        return []
    names: list[str] = []
    for line in description.splitlines():
        match = re.search(r"-\s*([a-zA-Z0-9_]+)\(", line)
        if match:
            names.append(match.group(1))
    return list(dict.fromkeys(names))


def _normalize_string_list(raw_values: object | None) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    return list(dict.fromkeys(str(value).strip() for value in raw_values if str(value).strip()))


def _normalize_failure_sensor_mapping(raw_mapping: object | None) -> dict[str, list[str]]:
    if not isinstance(raw_mapping, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for raw_failure_mode, raw_sensors in raw_mapping.items():
        if raw_failure_mode is None:
            continue
        failure_mode = str(raw_failure_mode).strip()
        if not failure_mode or not isinstance(raw_sensors, list):
            continue
        sensors = sorted(dict.fromkeys(str(sensor).strip() for sensor in raw_sensors if str(sensor).strip()))
        if sensors:
            normalized[failure_mode] = sensors
    return normalized


_TDesc = TypeVar("_TDesc")


def _label_desc_dict_from_list(raw: object, *, primary: str, alternate: str) -> dict[str, str]:
    if not isinstance(raw, list):
        return {}
    out: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_label = item.get(primary)
        if raw_label is None:
            raw_label = item.get(alternate)
        if raw_label is None:
            continue
        label = str(raw_label).strip()
        if not label:
            continue
        desc = str(item.get("description", "")).strip()
        if desc:
            out[label] = desc
    return out


def _ordered_descriptions(
    combined: dict[str, str],
    ordered_names: list[str],
    *,
    field: str,
    build: Callable[[str, str], _TDesc],
    sort_when_ordered_empty: bool = False,
) -> list[_TDesc]:
    if not ordered_names:
        if sort_when_ordered_empty:
            if not combined:
                return []
            return [build(k, str(combined[k]).strip()) for k in sorted(combined.keys())]
        raise ValueError(f"Asset profile field '{field}' requires a non-empty ordered name list")
    missing = [k for k in ordered_names if k not in combined or not str(combined[k]).strip()]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"Asset profile field '{field}' is missing names: {preview}{suffix}")
    return [build(k, str(combined[k]).strip()) for k in ordered_names]


def _require_nonempty_str(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Asset profile is missing required non-empty field: '{field}'")
    return text


def _require_nonempty_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Asset profile field '{field}' must be a list")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError(f"Asset profile is missing required non-empty list field: '{field}'")
    return list(dict.fromkeys(items))


def _invert_failure_mapping(failure_sensor_mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    sensor_failure_mapping: dict[str, list[str]] = {}
    for failure_mode, sensors in failure_sensor_mapping.items():
        for sensor_name in sensors:
            sensor_failure_mapping.setdefault(sensor_name, []).append(failure_mode)
    return {
        sensor_name: sorted(dict.fromkeys(failure_modes))
        for sensor_name, failure_modes in sensor_failure_mapping.items()
    }


def _grounding_summary_for_prompt(grounding: GroundingBundle) -> str:
    summary = {
        "requested_open_form": grounding.requested_open_form,
        "open_form_eligible": grounding.open_form_eligible,
        "iot_sensors": grounding.iot_sensors,
        "vibration_sensors": grounding.vibration_sensors,
        "asset_instances": [
            {
                "site_name": instance.site_name,
                "asset_id": instance.asset_id,
                "has_iot": instance.has_iot,
                "has_vibration": instance.has_vibration,
                "iot_time_range": instance.iot_time_range.model_dump() if instance.iot_time_range else None,
                "vibration_time_range": (
                    instance.vibration_time_range.model_dump() if instance.vibration_time_range else None
                ),
            }
            for instance in grounding.asset_instances
        ],
        "failure_modes": grounding.failure_modes,
        "failure_sensor_mapping": grounding.failure_sensor_mapping,
        "sensor_failure_mapping": grounding.sensor_failure_mapping,
    }
    return json.dumps(summary, indent=2)


def _tool_summary_for_prompt(server_desc: dict) -> str:
    compact = {
        focus: _tool_names_from_description(description)
        for focus, description in server_desc.items()
        if focus in {"iot", "fmsr", "tsfm", "wo", "vibration"}
    }
    return json.dumps(compact, indent=2)


def _validation_tool_names_by_focus(server_desc: dict) -> dict[str, tuple[str, ...]]:
    return {
        focus: tuple(_tool_names_from_description(server_desc.get(focus, "")))
        for focus in FOCUS_ORDER
        if focus != "multiagent"
    }


def _redact_logged_prompt(prompt: str, asset_profile_json: str) -> str:
    if asset_profile_json and asset_profile_json in prompt:
        return prompt.replace(asset_profile_json, ".redacted", 1)
    return prompt
